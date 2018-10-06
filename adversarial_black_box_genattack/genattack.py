# encoding: utf-8
# gen_attack.py
"""
Implementation of GenAttack from https://arxiv.org/abs/1805.11090 for Machines Can See contest
(https://competitions.codalab.org/competitions/19090#learn_the_details) held by organizers of
Machines Can See 2018 (http://machinescansee.com/)

From https://github.com/maremun/GenAttackMCS2018/blob/master/genattack.py
"""
import numpy as np
import torch
import torch.distributions as td

from skimage.measure import compare_ssim as ssim
from tqdm import trange
from torch.autograd import Variable

import sys
sys.path.append('modules/pytorch_mask_rcnn')


import matplotlib.pyplot as plt
import visualize


# Show detections (For debug purpose)
VISUALIZE_DETECTIONS = False

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def get_mutation(shape, alpha, delta, bernoulli):
    '''
    Sample mutation. Mutation happens on positions sampled from Bernoulli distribution.
        The amount of mutation is controlled by alpha and delta parameters that scale the
        samples from uniform distribution on [0,1] support.

    Args:
        shape: the shape for mutation tensor (mutation is additive to the original example)
        alpha: the parameter controlling mutation amount (step-size in the original paper)
        delta: the parametr controlling mutation amount (norm threshold in the original paper)
        bernoulli: the distribution to sample positions of mutation
    '''
    N, nchannels, h, w = shape  # N - population size
    U = torch.cuda.FloatTensor(N * nchannels * h * w).uniform_() * 2 * alpha * delta - alpha * delta
    mask = bernoulli.sample_n(N * nchannels * h * w).squeeze()
    mutation = mask * U
    mutation = mutation.view(N, nchannels, h, w)
    return mutation


def where(cond, x_1, x_2):
    '''
    Pytorch 0.3.1 does not have torch.where
    '''
    return (cond.float() * x_1) + ((1 - cond).float() * x_2)


def crossover(parents, fitness, population):
    """
    Crossover parents to get next generation. Parents are coupled and the features are sampled
        according to parents' fitness scores.

    Args:
        parents (1D array): indexes of sampled parents, [2 * (population_size - 1)]
        fitness (1D array): population fitness scores, [population_size]
        population (4D array): current generation features, [population_size x n_channels x h x w]

    Returns:
        children (4D array): next generation features, [population_size - 1 x n_features]
    """
    _, nchannels, h, w = population.shape
    fitness_pairs = fitness[parents.long()].view(-1, 2)
    prob = fitness_pairs[:, 0] / fitness_pairs.sum(1)
    parental_bernoulli = td.Bernoulli(prob)
    inherit_mask = parental_bernoulli.sample_n(nchannels * h * w)  # [N-1, nchannels * h * w]
    inherit_mask = inherit_mask.view(-1, nchannels, h, w)
    parent_features = population[parents.long()]
    children = torch.cuda.FloatTensor(inherit_mask.shape)
    children = where(inherit_mask, parent_features[::2], parent_features[1::2])
    return children


def get_fitness(population, target, net):
    '''
    Compute the fitness score for each example in population.

    Args:
        population (4D array): current generation features, [population_size x n_channels x h x w]
        target (1D array): target descriptor
        net: black box model to attack
        mse: torch.nn.MSELoss instance to compute MSE

    Returns:
        fitness (1D array): fitness scores as measured by MSE between descriptors
    '''
    # measure fitness with MSE between descriptors
    N = population.shape[0]
    dim = target.shape[0]
    descP = torch.cuda.FloatTensor(N, dim)

    for i in range(N):
        # obtain candidate descriptors from the black box [N x ddim]

        try:
            img = population[i].permute(1, 2, 0)
            img = img.clamp(0, 1).cpu().numpy()

            img = img * 255  # Convert back to 0, 255 range
            img = img.astype(np.uint8)
            result = net.detect([img])[0]
            res = result['scores'].sum().item()

            # Visualize results
            if VISUALIZE_DETECTIONS:
                visualize.display_instances(img, result['rois'], result['masks'], result['class_ids'],
                                            class_names, result['scores'])
                plt.show()

        except IndexError:
            res = 0
            print("No detections found")

        descP[i] = torch.cuda.FloatTensor([res])
    t = target.expand(N, -1)  # [N x ddim]

    return descP[:, 0]


def attack(x, target, delta, alpha, p, N, G, net):
    '''
    Attacks the black box model in `net` by generating and evolving population
        of attacking examples.

    Args:
        x (4D array): Original example of size [1, nchannels, h, w]
        target (1D array): target descriptor
        alpha (float): the parameter controlling mutation amount
        delta (float): the parametr controlling mutation amount
        p (float): the parameter for Bernoulli distribution used in mutation
        N (integer): the size of population
        G (integer): the number of generations to evolve through
        net: black box model to attack
    Returns:
        Pcurrent: evolved population of adversarial examples of the original `x`
            targeted with `target` descriptor to attack black box model with.
    '''
    # mse = torch.nn.MSELoss(reduce=False).cuda()
    bernoulli = td.Bernoulli(p)
    softmax = torch.nn.Softmax(0).cuda()
    # generate starting population
    nchannels, h, w = x.shape
    mutation = get_mutation([N, nchannels, h, w], alpha, delta, bernoulli)
    # init current population
    Pcurrent = x[None, :, :, :].expand(N, -1, -1, -1) + mutation
    Pnext = torch.zeros_like(Pcurrent)
    # init previous population with original example
    Pprev = x[None, :, :, :].expand(N, -1, -1, -1)
    # compute constraints to ensure permissible distance from the original example
    lo = x.min() - alpha[0] * delta[0]
    hi = x.max() + alpha[0] * delta[0]

    # init log
    log = []

    # start evolution
    t = trange(G)
    for g in t:
        # measure fitness with MSE between descriptors
        fitness = get_fitness(Pcurrent, target, net)  # [N]

        # Log fitness
        log.append(fitness)

        # check SSIM
        ssimm = np.zeros(N)
        for i in range(N):
            ssimm[i] = ssim(x.squeeze().permute(1, 2, 0).cpu().numpy(),
                            Pcurrent[i].permute(1, 2, 0).cpu().numpy(),
                            multichannel=True)  # [N]
        #survivors = ssimm >= 0.95  # [N]
        survivors = ssimm >= 0.40  # [N]

        # Update description
        t.set_description("Avg fitness: %f; avg ssimm %f" % (fitness.mean(), ssimm.mean()))
        t.refresh()

        # "Save" first output in console to compare
        if g == 1:
            print("\n")

        if survivors.sum() == 0:
            print('All candidates died.')
            return Pprev, log

        # End
        if g == G - 1:
            return Pcurrent, log

        # choose the best fit candidate among population
        _, best = torch.min(fitness, 0)  # get idx of the best fitted candidate
        # ensure the best candidate gets a place in the next population
        Pnext[0] = Pcurrent[best]

        # generate next population
        # TODO: not survivors are ignored atm
        #probs = softmax(Variable(torch.cuda.FloatTensor(survivors)) * Variable(fitness)).data
        probs = softmax(Variable(-fitness)).data
        cat = td.Categorical(probs[None, :].expand(2 * (N - 1), -1))
        parents = cat.sample()  # sample 2 parents per child, total number of children is N-1
        children = crossover(parents, fitness, Pcurrent)  # [(N-1) x nchannels x h x w]
        mutation = get_mutation([N - 1, nchannels, h, w], alpha, delta, bernoulli)
        children = children + mutation
        Pnext[1:] = children
        Pprev = Pcurrent  # update previous generation
        Pcurrent = Pnext  # update current generation
        # clip to ensure the distance constraints
        Pcurrent = torch.clamp(Pcurrent, lo, hi)
    return Pcurrent, log




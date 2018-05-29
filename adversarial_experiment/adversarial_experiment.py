# Sources:
# http://blog.outcome.io/pytorch-quick-start-classifying-an-image/
# https://github.com/Lextal/adv-attacks-pytorch-101
import argparse
import io
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
import time
import torch

import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision import models, transforms


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'


def variable_to_img(var):
    return T.ToPILImage()(var.data)


def show_variable_as_img(var):
    variable_to_img(var).show()


class AdversarialExperiment:

    def __init__(self):
        # Start time
        self.start_time = time.time()

        # Set manual seed for stable results
        torch.manual_seed(1)

        # Load model
        self.model = models.vgg16(pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.model.eval()

        # Get Image Net labels
        self.labels = {int(key): value for (key, value)
                       in requests.get(LABELS_URL).json().items()}

        # From https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L108-L113
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # List of possible attacks
        self.attacks = {"random_noise": self.attack_random_noise,
                        "FGSM": self.attack_FGSM,
                        "FGSM_target": self.attack_FGSM_target}

    def main(self, attack="random_noise"):
        # Get cat image
        response = requests.get(IMG_URL)
        img_pil = Image.open(io.BytesIO(response.content))

        # Preprocess image
        img_tensor = self.preprocess(img_pil)
        img_tensor.unsqueeze_(0)

        # Create
        perturbation = self.attacks[attack](img_tensor)

        self.evaluate_results(img_tensor, perturbation)

    def attack_random_noise(self, img):
        """
        Creates random noise as attack to img
        :param img: input image
        :return: random noise
        """
        noise = torch.rand(img.size())
        noise = noise * 0.07
        return noise

    def attack_FGSM(self, img, target=-1):
        """
        Using the Fast Gradient Sign Method (FGSM) to attack image img by adding stepwise
        the sign of the image gradient.

        For target attack the gradient gets subtracted to reach the desired class.

        :param img: input image
        :param target: target class. -1 for non target attack
        :return: perturbation
        """

        # Definitions
        steps = 10
        step_alpha = 0.001
        eps = 2 * 8 / 225.

        label = torch.zeros(1, 1)
        img = Variable(img)
        label = Variable(label)

        # Define label variable. Target for target attack.
        if target == -1:
            # Perform forwardpass to get prediction of the original image
            output = self.model(img)

            label.data = output.max(1)[1].data
        else:
            label.data = torch.Tensor([target]).type(torch.LongTensor)

        # Clone so img gets not manipulated
        img_adv = img.clone()
        img_adv.requires_grad = True

        for step in range(steps):
            zero_gradients(img_adv)
            out = self.model(img_adv)

            # Calculate loss, gradient and normed gradient based on the sign of the gradient
            _loss = self.loss(out, label)
            _loss.backward()
            normed_grad = step_alpha * torch.sign(img_adv.grad.data)

            # Add/Subtract perturbation
            if target == -1:
                step_adv = img_adv.data + normed_grad
            else:
                step_adv = img_adv.data - normed_grad

            # Postprocessing perturbation
            adv = step_adv - img.data
            adv = torch.clamp(adv, -eps, eps)  # Max eps range
            result = img.data + adv
            result = torch.clamp(result, 0.0, 1.0)  # Image range
            adv = result - img.data

            # Set adversarial image as new input
            img_adv.data = result

            print("Step: {0}, Loss: {1:.2f}, Top1: {2}".format(step, _loss.data[0], self.labels[out.data.numpy().argmax()]))

        return adv

    def attack_FGSM_target(self, img):
        return self.attack_FGSM(img, 875)

    def print_top_k(self, output, k):
        top_k = output.topk(k)
        for i in range(0, k):
            label_id = top_k[1].data[0][i]
            confidence = top_k[0].data[0][i]
            print("{0: >5.2f}% - {1}".format(confidence * 100, self.labels[label_id]))

    def get_label(self, output):
        label = output.data.max(1)[1].cpu().numpy()[0]
        return self.labels[label]

    @staticmethod
    def clamp_image(img):
        img = torch.clamp(img, 0.0, 1.0)
        return img

    def evaluate_results(self, img_org, perturbation):
        # Create adversarial image
        img_adv = Variable(img_org + perturbation)
        img_org = Variable(img_org)
        perturbation = Variable(perturbation)

        # factor to scale up perturbation
        self.clamp_image(perturbation[0])
        scale_factor = 1 / perturbation[0].data.numpy().max()

        # Forwardpass + softmax
        output_org = self.model(img_org)
        output_org = F.softmax(output_org, dim=1)
        output_adv = self.model(img_adv)
        output_adv = F.softmax(output_adv, dim=1)

        print("\n=== Top 5 original image ===")
        self.print_top_k(output_org, 5)
        print("\n=== Top 5 adversarial image ===")
        self.print_top_k(output_adv, 5)

        # Timing
        end_time = time.time()
        minutes = int((end_time - self.start_time) / 60)
        seconds = (end_time - self.start_time) - minutes * 60
        print('\n' + '=' * 31)
        print("Execution time: {0} min {1:.0f} sec".format(minutes, seconds))

        # Create plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        label_org, label_adv = self.get_label(output_org), self.get_label(output_adv)
        ax[0].imshow(variable_to_img(self.clamp_image(img_org[0])))
        ax[0].set_title('Original image: {}'.format(label_org))
        ax[1].imshow(variable_to_img(perturbation[0] * scale_factor))
        ax[1].set_title('Attacking noise (x{0:4.2f})'.format(scale_factor))
        ax[2].imshow(variable_to_img(self.clamp_image(img_adv[0])))
        ax[2].set_title('Adversarial example: {}'.format(label_adv))

        for i in range(3):
            ax[i].set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Init
    adv_experiment = AdversarialExperiment()

    # Arguments
    parser = argparse.ArgumentParser(description='Adversarial Experiments')
    parser.add_argument('--attack', type=str, default="random_noise", metavar='ATT',
                        help='Attacks: '
                             + ' | '.join(adv_experiment.attacks)
                             + ' (default: random_noise)')

    args = parser.parse_args()

    # Execute attack
    adv_experiment.main(args.attack)

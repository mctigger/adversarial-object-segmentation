# Sources:
# http://blog.outcome.io/pytorch-quick-start-classifying-an-image/
# https://github.com/Lextal/adv-attacks-pytorch-101

import io
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models, transforms


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'


def tensor_to_img(tensor):
    return np.asarray(T.ToPILImage()(tensor))


def show_tensor_as_img(tensor):
    to_pil = T.ToPILImage()
    to_pil(tensor).show()


class AdversarialExperiment:

    def __init__(self):
        # Set manual seed for stable results
        torch.manual_seed(1)

        # Load model
        self.model = models.vgg16(pretrained=True)

        # Get Image Net labels
        self.labels = {int(key): value for (key, value)
                       in requests.get(LABELS_URL).json().items()}

        # From https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L108-L113
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def main(self):
        # Get cat image
        response = requests.get(IMG_URL)
        img_pil = Image.open(io.BytesIO(response.content))

        # Preprocess image
        img_tensor = self.preprocess(img_pil)
        img_tensor.unsqueeze_(0)

        noise = self.attack_random_noise(img_tensor)

        self.evaluate_results(img_tensor, noise)

    def attack_random_noise(self, img):
        noise = torch.rand(img.size())
        noise = noise * 0.07
        return noise

    def print_top_k(self, output, k):
        top_k = output.topk(k)
        for i in range(0, k):
            label_id = top_k[1].data[0][i].item()
            confidence = top_k[0].data[0][i].item()
            print("{0: >5.2f}% - {1}".format(confidence * 100, self.labels[label_id]))

    def get_label(self, output):
        label = output.data.max(1)[1].cpu().numpy()[0]
        return self.labels[label]

    # As accuracy is not high enough unnormalization creates some minor artifacts
    def clamp_image(self, img):
        img = torch.clamp(img, 0.0, 1.0)
        return img

    def evaluate_results(self, img_org, noise):
        # Create adversarial image
        img_adv = img_org + noise

        # factor to scale up noise
        self.clamp_image(noise[0])
        scale_factor = 1 / noise[0].numpy().max()

        # Forwardpass + softmax
        output_org = self.model(img_org)
        output_org = F.softmax(output_org, dim=1)
        output_adv = self.model(img_adv)
        output_adv = F.softmax(output_adv, dim=1)

        print("\n=== Top 5 original image ===")
        self.print_top_k(output_org, 5)
        print("\n=== Top 5 adversarial image ===")
        self.print_top_k(output_adv, 5)

        # Create plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        label_org, label_adv = self.get_label(output_org), self.get_label(output_adv)
        ax[0].imshow(tensor_to_img(self.clamp_image(img_org[0])))
        ax[0].set_title('Original image: {}'.format(label_org))
        ax[1].imshow(tensor_to_img(noise[0] * scale_factor))
        ax[1].set_title('Attacking noise (x{0:4.2f})'.format(scale_factor))
        ax[2].imshow(tensor_to_img(self.clamp_image(img_adv[0])))
        ax[2].set_title('Adversarial example: {}'.format(label_adv))

        for i in range(3):
            ax[i].set_axis_off()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    adv_experiment = AdversarialExperiment()
    adv_experiment.main()

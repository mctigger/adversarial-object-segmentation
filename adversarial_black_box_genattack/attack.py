
import os
import sys

sys.path.append('modules/pytorch_mask_rcnn')

from genattack import *

import skimage.io
import coco
from pycococreatortools import pycococreatortools
import model as modellib
import json
import torch
import matplotlib.pyplot as plt
import visualize
from torchvision import transforms
import skimage
import pathlib
import time
import csv


# Root directory of the project
ROOT_DIR = "./"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/val2014")

# Show detections (For debug purpose)
VISUALIZE_DETECTIONS = False


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

# Prepare image for genattack
transform = transforms.Compose([
        transforms.ToTensor(),
])


# Convert result torch to mupy image
def torch2numpy(img):
    img = img.permute(1, 2, 0)
    img = img.clamp(0, 1).cpu().numpy()

    img = img * 255  # Convert back to 0, 255 range
    img = img.astype(np.uint8)
    return img


# Load and transform image
#image = skimage.io.imread(os.path.join(IMAGE_DIR, "COCO_val2014_000000083277.jpg"))
#image = skimage.io.imread(os.path.join(IMAGE_DIR, "COCO_val2014_000000289343.jpg"))
img_path = os.path.join(IMAGE_DIR, sys.argv[1])
image = skimage.io.imread(img_path)
input_img = transform(image)
input_img = input_img.cuda(async=True)

N = 6  # size of population to evolve
G = 40000  # number of generations to evolve through
#p = torch.cuda.FloatTensor([0.0001])
p = torch.cuda.FloatTensor([float(sys.argv[2])])
alpha = torch.cuda.FloatTensor([1.])
delta = torch.cuda.FloatTensor([0.05])

target = torch.cuda.FloatTensor([0])


# Show detections of original image
if VISUALIZE_DETECTIONS:
    result = model.detect([image])[0]
    visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                class_names, result['scores'])
    plt.show()

# Generate adversarials
image_adversarials, fitness = attack(input_img, target, delta, alpha, p, N, G, model)

# Create folder
path = os.path.join(DEFAULT_LOGS_DIR, "adversarial_examples")
path = os.path.join(path, time.strftime("%Y%m%d_%H%M%S"))
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# Save images
for i in range(len(image_adversarials)):
    path_img = os.path.join(path, "adv_example_" + str(i) + ".jpg")
    skimage.io.imsave(path_img, torch2numpy(image_adversarials[i]))
    print("Adversarial example saved to: " + path_img)

# Save fitness
path_fitness = os.path.join(path, "fitness.csv")
with open(path_fitness, 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header
    writer.writerow(["adv_example_" + str(x) for x in range(6)])

    # Write data
    for row in fitness:
        writer.writerow(row.cpu().numpy().tolist())

print("Saved fitness log to : " + path_fitness)


# Save Config
path_conf = os.path.join(path, "conf.cfg")
with open(path_conf, 'w') as f:
    f.write("input_img: %s\ndelta %f\nalpha %f\np %f\nN %f\nG %f" % (img_path, delta, alpha, p, N, G))

print("Saved cfg  to : " + path_conf)

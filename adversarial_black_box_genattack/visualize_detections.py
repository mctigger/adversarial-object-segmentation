import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import pathlib

import coco
import utils
import model as modellib
import visualize

import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

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



# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = "/disk/vanishing_data/mlprak4/mask-r-cnn_logs/adversarial_examples/20180826_132929/"

# Path to save detections
DETECTIONS_PATH = os.path.join(IMAGE_DIR, "detections")
pathlib.Path(DETECTIONS_PATH).mkdir(parents=True, exist_ok=True)


def show_and_save_detection(file_path):
    image = skimage.io.imread(file_path)
    print(file_path)

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]

    print(str(len(r['scores'])) + " objects detected")
    fig = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    detections_name = file_path.split("/")[-1][:-4] + "_detections.png"
    detections_path = file_path = os.path.join(DETECTIONS_PATH, detections_name)
    fig.savefig(detections_path, bbox_inches='tight')


# Open original image
with open(os.path.join(IMAGE_DIR, "conf.cfg")) as f:
    line = f.readline()

original_file_path = line.split(": ")[-1].replace("\n", "")
show_and_save_detection(original_file_path)


# All images from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
for file_name in file_names:
    if not file_name.endswith(".jpg"):
        continue

    file_path = os.path.join(IMAGE_DIR, file_name)
    show_and_save_detection(file_path)

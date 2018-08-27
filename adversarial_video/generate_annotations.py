import os
import sys

sys.path.append('modules/pytorch_mask_rcnn')

import numpy as np
import skimage.io
from skimage.transform import rescale
from skimage.draw import circle
from scipy.misc import imshow

import coco
from pycococreatortools import pycococreatortools
import model as modellib
import json

import torch

# Root directory of the project
ROOT_DIR = "./"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/video/video2014")


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

detections = []

template_path = os.path.join("./data/annotations/instances_adversarial_attack_target_XXX2014.json.template")
with open(template_path) as f:
    data = json.load(f)

    for i, image_file in enumerate(sorted(os.listdir(IMAGE_DIR))):
        print("{}: working on image {}".format(i, image_file))
        image = skimage.io.imread(os.path.join(IMAGE_DIR, image_file))
        a, b = image.shape[0:2]
        image_size = b, a

        image_id = int(image_file.split('-')[1][:-4])
        img_data = pycococreatortools.create_image_info(image_id, image_file, image_size)
        data["images"].append(img_data)


        # Run detection
        results = model.detect([image])

        # Visualize results
        result = results[0]

        # Loop for each annotation
        for i in range(len(result["class_ids"])):
            # Get mask
            masks = result["masks"]
            m = masks[:, :, i]

            # https://patrickwasp.com/create-your-own-coco-style-dataset/
            category_info = {'id': result["class_ids"][i].item(), 'is_crowd': 0}

            # Create annotation info
            ann_data = pycococreatortools.create_annotation_info(
                image_id * 10000 + i, image_id, category_info, m, image_size, tolerance=2)

            data["annotations"].append(ann_data)




# Write output file
with open("./data/video/annotations/instances_video2014.json", "w") as outfile:
    json.dump(data, outfile, indent=4)

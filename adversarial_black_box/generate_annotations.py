import os
import sys

sys.path.append('modules/pytorch_mask_rcnn')
#sys.path.append('modules/cocoapi/PythonAPI/pycocotools')

import skimage.io
import coco
from pycococreatortools import pycococreatortools
import model as modellib
import json
import torch
import matplotlib.pyplot as plt
import visualize


# Show detections (For debug purpose)
VISUALIZE_DETECTIONS = False

# Root directory of the project
ROOT_DIR = "./"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/test2014")

# Log file for annotation errors
LOG_FILE = os.path.join(ROOT_DIR, "logs/annotation_errors.log")

# Split 0.8 = 80/20 train/val split
DATA_SPLIT = 0.8

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
annotation_errors = {}

template_path = os.path.join(ROOT_DIR, "data/annotations/instances_.template.json")
with open(template_path) as f:
    data_train = json.load(f)

with open(template_path) as f:
    data_val = json.load(f)

    annotation_id = 0
    num_images = len([name for name in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, name))])

    print(str(num_images) + " images found")

    for i, image_file in enumerate(sorted(os.listdir(IMAGE_DIR))):
        print("{}: working on image {}".format(i, image_file))
        image = skimage.io.imread(os.path.join(IMAGE_DIR, image_file))
        a, b = image.shape[0:2]
        image_size = b, a

        image_id = int(image_file.split('_')[-1][:-4])
        img_data = pycococreatortools.create_image_info(image_id, image_file, image_size)
        
        if i < num_images * DATA_SPLIT:
            data_train["images"].append(img_data)
        else:
            data_val["images"].append(img_data)

        # Run detection
        try:
            results = model.detect([image])
        # Continue if no detection was made
        except IndexError:
            print("No detection found")
            annotation_errors[image_file] = "No detection found"
            continue
        except:
            print("Error on image annotation")
            annotation_errors[image_file] = "Error on image annotation"
            continue

        result = results[0]

        # Visualize results
        if VISUALIZE_DETECTIONS:
            visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                        class_names, result['scores'])
            plt.show()
            
        # Loop for each annotation
        for i in range(len(result["class_ids"])):
            # Get mask
            masks = result["masks"]
            m = masks[:, :, i]

            # https://patrickwasp.com/create-your-own-coco-style-dataset/
            category_info = {'id': result["class_ids"][i].item(), 'is_crowd': 0}

            # Create annotation info
            ann_data = pycococreatortools.create_annotation_info(
                annotation_id, image_id, category_info, m, image_size, tolerance=2)
            annotation_id = annotation_id + 1


            if i < num_images * DATA_SPLIT:
                 data_train["annotations"].append(ann_data)
            else:
                 data_val["annotations"].append(ann_data)


# Write output files
out_path = os.path.join(ROOT_DIR, "data/annotations/instances_test_train2014.json")
with open(out_path, "w+") as outfile:
    json.dump(data_train, outfile, indent=4)

out_path = os.path.join(ROOT_DIR, "data/annotations/instances_test_val2014.json")
with open(out_path, "w+") as outfile:
    json.dump(data_val, outfile, indent=4)

# Write error file
with open(LOG_FILE, "w+") as outfile:
    for k, v in annotation_errors.items():
        outfile.write(str(k) + ': ' + str(v) + '\n')

print("Errors logged to: '" + LOG_FILE + "'")

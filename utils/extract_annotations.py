#!/bin/python

## Script to copy all images used in the minival2014 to a seperate folder (5k images ~800MB)

import json
import shutil
import os
from pprint import pprint

PATH = "/disk/no_backup/mlprak4/adverserial-object-segmentation/data/"
ANNOTATIONS = "annotations/instances_minival2014.json"
FILE_NAME = "COCO_val2014_000000289343.jpg"

# Open minival annotations
with open(PATH + ANNOTATIONS) as f:
	data = json.load(f)

# Copy info / licenses / type
extract = json.loads("{}")
extract["info"] = data["info"]
extract["licenses"] = data["licenses"]
extract["type"] = data["type"]

# Extract all image information
extract["images"] = []
image_ids = []
for img in data["images"]:
	if img["file_name"] == FILE_NAME:
		extract["images"].append(img)
		image_ids.append(img["id"])

# Extract all annotations for the images
extract["annotations"] = []
for ann in data["annotations"]:
	if ann["image_id"] in image_ids:
		extract["annotations"].append(ann)

# Add Categories at bottom
extract["categories"] = data["categories"]


# Write output file
with open("instances_extract.json", "w") as outfile:
	json.dump(extract, outfile, indent=4)



print("Annotations for " + str(len(extract["images"])) + " images extracted")



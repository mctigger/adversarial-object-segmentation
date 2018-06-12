#!/bin/python

## Script to copy all images used in the minival2014 to a seperate folder (5k images ~800MB)

import json
import shutil
import os

PATH = "/disk/no_backup/mlprak4/adverserial-object-segmentation/data/"
ANNOTATIONS = "annotations/instances_minival2014.json"
DEST = "/fzi/ids/mlprak4/val2014_sample/"

with open(PATH + ANNOTATIONS) as f:
	data = json.load(f)


if not os.path.exists(DEST):
	os.makedirs(DEST)

for img in data["images"]:
	print(img["file_name"])
	shutil.copyfile(PATH + "val2014/" + img["file_name"], DEST + img["file_name"])

print(str(len(data["images"])) + " images copied")



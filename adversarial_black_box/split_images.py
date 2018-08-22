import os
import sys

import shutil
import json


# Root directory of the project
ROOT_DIR = "./"

SOURCE_DIR = os.path.join(ROOT_DIR, "data/test2014")

DEST_TRAIN = os.path.join(ROOT_DIR, "data/test_train2014")
DEST_VAL = os.path.join(ROOT_DIR, "data/test_val2014")

ANNOTATIONS_TRAIN = os.path.join(ROOT_DIR, "data/annotations/instances_test_train2014.json")
ANNOTATIONS_VAL = os.path.join(ROOT_DIR, "data/annotations/instances_test_val2014.json")

def copy_images(annotation_file, source, destination):
    print("Open annotation files")
    with open(annotation_file) as f:
        annotations = json.load(f)

    print("Clear previous image folders")
    shutil.rmtree(destination + "/*", ignore_errors=True)

    for image in annotations["images"]:
        print("Copying '" + image["file_name"] +"' to '" + destination + "'")
        src = os.path.join(source, image["file_name"])
        shutil.copy(src, destination)

# Execute
copy_images(ANNOTATIONS_TRAIN, SOURCE_DIR, DEST_TRAIN)
copy_images(ANNOTATIONS_VAL, SOURCE_DIR, DEST_VAL)

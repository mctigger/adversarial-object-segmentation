#!/bin/python

## Change segmentation and annotation coordinates of samle annotation

import json
import math

PATH = "/disk/no_backup/mlprak4/adverserial-object-segmentation/data/annotations/"
TEMPLATE = "instances_adversarial_attack_target_XXX2014.json.template"


# Open File
def load_template():
    with open(PATH + TEMPLATE) as f:
        data_template = json.load(f)

    return data_template

def change_localisation(data, annotation, offset_x, offset_y):
    # Add offset to all segmentation coordinates of annotation
    for coords in data["annotations"][annotation]["segmentation"]:
        coords[0::2] = [x + offset_x for x in coords[0::2]]
        coords[1::2] = [y + offset_y for y in coords[1::2]]

    # Add offset to BBox
    bbox = data["annotations"][annotation]["bbox"]
    bbox[0] = bbox[0] + offset_x
    bbox[1] = bbox[1] + offset_y

    return data


def change_class(data, annotation, class_id):
    data["annotations"][annotation]["category_id"] = class_id

    return data


def bloat_segmentation(data, annotation, factor=2):
    # Add offset to all segmentation coordinates of annotation
    mid_x = 0
    mid_y = 0
    for coords in data["annotations"][annotation]["segmentation"]:
        mid_x = sum(coords[0::2]) / len(coords[0::2])
        mid_y = sum(coords[1::2]) / len(coords[1::2])

    # Define max/min values
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for coords in data["annotations"][annotation]["segmentation"]:
        coords[0::2] = [x + (x - mid_x)*(factor - 1) for x in coords[0::2]]
        coords[1::2] = [y + (y - mid_y)*(factor - 1) for y in coords[1::2]]

        # Set max/min values for B
        min_x = min(min_x, min(coords[0::2]))
        min_y = min(min_y, min(coords[1::2]))
        max_x = max(max_x, max(coords[0::2]))
        max_y = max(max_y, max(coords[1::2]))

    bbox = data["annotations"][annotation]["bbox"]
    bbox[0] = min_x
    bbox[1] = min_y
    bbox[1] = max_x - min_x
    bbox[2] = max_y - min_y

    return data


def save_as(data, path):
    with open(path, "w+") as outfile:
        json.dump(data, outfile, indent=4)

    print("Created '" + path + "' from template")


# Change first annotation to airplane
data = load_template()
data = change_class(data, 0, 5)  # 5 = airplane
save_as(data, PATH + "instances_adversarial_attack_target_class2014.json")

# Change first annotation offset
data = load_template()
data = change_localisation(data, 0, offset_x=0, offset_y=-200)
save_as(data, PATH + "instances_adversarial_attack_target_localisation2014.json")

# Change first annotation segmentation size
data = load_template()
data = bloat_segmentation(data, 0, factor=4)
save_as(data, PATH + "instances_adversarial_attack_target_segmentation2014.json")

# Change first annotation class, offset and size
data = load_template()
data = change_class(data, 0, 5)  # 5 = airplane
data = change_localisation(data, 0, offset_x=0, offset_y=-200)
data = bloat_segmentation(data, 0, factor=4)
save_as(data, PATH + "instances_adversarial_attack_target_combined2014.json")


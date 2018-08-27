#!/bin/python

## Change segmentation and annotation coordinates of samle annotation

import json
import math
import random

PATH = "./data/video/annotations/"
TEMPLATE = "instances_video2014.json"


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

    bbox = data["annotations"][annotation][m
    bbox[1] = max_x - min_x
    bbox[2] = max_y - min_y

    return data


def save_as(data, path):
    with open(path, "w+") as outfile:
        json.dump(data, outfile, indent=4)

    print("Created '" + path + "' from template")


if __name__ == '__main__':
    data = load_template()
    for i in range(len(data["annotations"])):
        image_id = data["annotations"][i]['image_id']
        if image_id < 200:
            data = change_class(data, i, image_id // 10 % 80)

        if image_id > 200:
            data = change_localisation(data, i, math.sin(image_id / 20) * 100, math.sin(image_id / 10) * 100)

    index = 0
    for i in range(len(data["annotations"])):
        image_id = data["annotations"][index]['image_id']
        if image_id < 100 and random.random() > 0.5:
            del data["annotations"][index]
            print("deleted")
        else:
            index += 1

    save_as(data, PATH + "instances_adversarial_video2014.json")


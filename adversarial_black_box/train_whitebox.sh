#!/bin/sh
cd ..
python ./modules/pytorch_mask_rcnn/coco.py trainwhitebox --dataset=./data --model=imagenet


#!/bin/sh
cd ..
python ./modules/pytorch_mask_rcnn/coco.py evaluate --dataset=./data --model=last


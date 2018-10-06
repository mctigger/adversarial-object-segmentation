# Blackbox - GenAttack

## How it works
This experiment creates an gradient free blackbox attack on the pretrained COCO Mask R-CNN model. It is based on the paper [GenAttack: Practical Black-box Attacks with Gradient-Free Optimization](https://arxiv.org/abs/1805.11090) and a implementation by [maremun](https://github.com/maremun/GenAttackMCS2018).  

## Requirements
Install environment of root project.
Also the COCO 2014 test dataset has to be downloaded an put into data folder.

## Usage
```bash
$ source activate adverserial-object-segmentation
$ python attack.py COCO_val2014_000000289343.jpg 0.00001 # attack.py IMAGENAME MUATTIONPROBABLILITY
$ python visualize_detections.py # Saves detections for original image and adversarial attacks in a subfolder
```

## Note
On a GTX 1080 Ti this runs approx. 22 hours per image. It can be optimized by not copying the memory from cpu/gpu for calculating the fitness function. Also 20k generations should be sufficient. 
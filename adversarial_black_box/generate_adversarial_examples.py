import os
import sys

sys.path.insert(0, './adversarial_mask_rcnn')
import adversarial_attack

import numpy as np

import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from model import Dataset, unmold_image, MaskRCNN, compute_losses
from visualize import display_instances
from coco import CocoDataset, CocoConfig


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DATASET_DIR = os.path.join(ROOT_DIR, "data")
TARGET = "whitebox"
SHOW_PERTURBATION = True
USE_MASK = False

config = CocoConfig()
config.display()

# Create model
model = MaskRCNN(config=config, model_dir="logs/")
if config.GPU_COUNT:
    model = model.cuda()

# Select weights file to load
model_path = model.find_last()[1]

# Load weights
print("Loading weights ", model_path)
model.load_weights(model_path)

dataset_train = CocoDataset()
dataset_train.load_coco(DATASET_DIR, "adversarial_attack_target_" + TARGET, year=2014, auto_download=False)
dataset_train.prepare()

adversarial_attack.train_adversarial(
    model,
    dataset_train,
    epochs=161,
    layers='all',
    target_attack=TARGET,
    show_perturbation=SHOW_PERTURBATION,
    use_mask=USE_MASK,
    save_adversarials_to_logs=True
)


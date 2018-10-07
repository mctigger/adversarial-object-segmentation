# adversarial-attacks-on-mask-rcnn

Sourcecode to the ML-Praktikum at KIT (SoSe 2018) by Tim Joseph, Adrian Vetter and Qi Zhao.

## Installation

Problems first: May need CUDA installation for `sh install.sh` because CUDA kernels get compiled
1. Install anaconda with Python 3.6
2. Use `git clone --recursive git@github.com:Mctigger/adverserial-object-segmentation.git` to clone with all submodules.
3. Run `conda env create -f environment.yml` and activate environment with `source activate adverserial-object-segmentation`
4. Run `sh install.sh` and follow command line instructions
5. Verify that mask-rcnn is working by `cd ./modules/pytorch-mask_rcnn; python demo.py;`


## Project structure
* **.** root folder with environment information
* **adversarial_black_box** experiment for a blackbox substitute network
* **adversarial_black_box_genattack** experiment for a blackbox GenAttack
* **adversarial_experiment** basic adversarial experiments on image recognition
* **adversarial_mask_rcnn** whitebox attacks on Mask R-CNN
* **data** the place for all training data
* **modules** all used modules (cocoapi/pytorch Mask R-CNN)
* **utils** handy utils created while working on this project.

# adversarial-attacks-on-mask-rcnn

## Installation

Problems first: May need CUDA installation for `sh install.sh` because CUDA kernels get compiled
1. Install anaconda with Python 3.6
2. Use `git clone --recursive git@github.com:Mctigger/adverserial-object-segmentation.git` to clone with all submodules.
3. Run `conda env create -f environment.yml` and activate environment with `source activate adverserial-object-segmentation`
4. Run `sh install.sh` and follow command line instructions
5. Verify that mask-rcnn is working by `cd ./modules/pytorch-mask_rcnn; python demo.py;`

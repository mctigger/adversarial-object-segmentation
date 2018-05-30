#!/usr/bin/env bash
(
 arch=sm_61
 cd ./modules/pytorch-mask_rcnn/
 cd nms/src/cuda/
 nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=$arch
 cd ../../
 python build.py
 cd ../

 cd roialign/roi_align/src/cuda/
 nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=$arch
 cd ../../
 python build.py
 cd ../../
 ln -s ../cocoapi/PythonAPI/pycocotools/ pycocotools
 )

(cd ./modules/cocoapi/PythonAPI && make;)

echo "================================================================= \n";
echo "\033[33mNow you only need to visit https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc and download mask_rcnn_coco.pth into ./modules/pytorch_mask_rcnn \033[0m";
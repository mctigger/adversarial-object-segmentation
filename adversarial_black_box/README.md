# Blackbox attack

## How it works
This experiment creates an blackbox attack on the pretrained COCO Mask R-CNN model.
1. Download the COCO 2014 testset. The testset itself does not come with annotations.
2. Annotate all testset images and split them into test_train and test_val data with the pretrained model.
3. Train a whitebox on basis of pretrained imagenet model.
4. Create adversarial example(s) in whitebox.
5. Test adversarial example(s) on blackbox.


## Requirements
Install environment of root project.
Also the COCO 2014 test dataset has to be downloaded an put into data folder.

## Usage
```bash
$ source activate adverserial-object-segmentation
$ python generate_annotations.py # Creates train and validation annotation files
$ python split_images.py # Split imageset based on annoations
$ ./train_whitebox.sh # Trains new model based on previously annotated images
$ # tbc..
```


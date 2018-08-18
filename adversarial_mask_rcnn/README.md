# Mask R-CNN Adversarial Attacks





## Requirements
Install environment of root project.


## Usage
Execute script from root project.
```bash
$ source activate adverserial-object-segmentation
$ cd ..
$ python adversarial_mask_rcnn/adversarial_attack.py --help
usage: adversarial_attack.py [-h] --dataset /path/to/coco/ [--year <year>]
                             [--model /path/to/weights.pth]
                             [--logs /path/to/logs/]
                             [--target <class|localisation|segmentation|combined>]
                             [--use-mask <True|False>]
                             [--show-perturbation <True|False>]

Train Mask R-CNN on MS COCO.

optional arguments:
  -h, --help            show this help message and exit
  --dataset /path/to/coco/
                        Directory of the MS-COCO dataset
  --year <year>         Year of the MS-COCO dataset (2014 or 2017)
                        (default=2014)
  --model /path/to/weights.pth
                        Path to weights .pth file or 'coco'
  --logs /path/to/logs/
                        Logs and checkpoints directory (default=logs/)
  --target <class|localisation|segmentation|combined>
                        Perform a target attack on class, localisation,
                        segmentation or a combined attack (default=class)
  --use-mask <True|False>
                        Use bbox of first annotation as mask (default=False)
  --show-perturbation <True|False>
                        Shows scaled perturbation (default=False)

```

### Example
```
$ source activate adverserial-object-segmentation
$ cd ..
$ python adversarial_mask_rcnn/adversarial_attack.py --dataset=./data --target class
```


## Important Hints
 * If ```--show-perturbation True``` and perturbation image shows a lot of red pixels, its due to rounding errors on mould/unmolding images. 
 **WORKAROUND**: Change ```MEAN_PIXEL``` in ```adverserial-object-segmentation/modules/pytorch_mask_rcnn/config.py``` to ```MEAN_PIXEL = np.array([123, 116.8, 103.9])```
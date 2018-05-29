# Adversarial Experiment

Steadily creating experiments towards an adversarial attack against a mask r-cnn net. 


# Experiments
1. **Random Noise Image Classification:** Add random noise to *Image Net* input image to attack image classification with VGG16
2. **FGSM Image Classification:** Backpropagate gradient to image and add it to the image to decrease confidence for the original class
3. **Target FGSM Image Classification:** Backpropagate gradient to image and subtract it to get more confidence towards the desired target class


## Requirements
Install environment of root project.
Additionally torchvision for pytorch 0.3.1 is needed:

```bash
$ conda install cudatoolkit requests
$ conda install torchvision=0.2.0 -c pytorch --no-deps

```

## Usage
```bash
$ source activate adverserial-object-segmentation
$ python adversarial_experiment.py --help
usage: adversarial_experiment.py [-h] [--attack ATT]

Adversarial Experiments

optional arguments:
  -h, --help    show this help message and exit
  --attack ATT  Attacks: random_noise | FGSM | FGSM_target (default:
                random_noise)

```


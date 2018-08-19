import os
import sys
sys.path.append('modules/pytorch_mask_rcnn')

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

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


def img_to_np(img):
    img = img.cpu().numpy()
    img = np.copy(img)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    return img


def train_adversarial(model, train_dataset, epochs, layers, target_attack=False, show_perturbation=False, use_mask=False,
                      change_object_location=False):
    """Train the model.
    train_dataset, val_dataset: Training and validation Dataset objects.
    learning_rate: The learning rate to train with
    epochs: Number of training epochs. Note that previous training epochs
            are considered to be done alreay, so this actually determines
            the epochs to train in total rather than in this particaular
            call.
    layers: Allows selecting wich layers to train. It can be:
        - A regular expression to match layer names to train
        - One of these predefined values:
          heaads: The RPN, classifier and mask heads of the network
          all: All the layers
          3+: Train Resnet stage 3 and up
          4+: Train Resnet stage 4 and up
          5+: Train Resnet stage 5 and up
    """

    # Pre-defined layer regular expressions
    layer_regex = {
        # all layers but the backbone
        "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # From a specific Resnet stage and up
        "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # All layers
        "all": ".*",
    }
    if layers in layer_regex.keys():
        layers = layer_regex[layers]

    # Data generators
    train_set = Dataset(train_dataset, model.config, augment=True)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)

    model.set_trainable(layers)

    for epoch in range(model.epoch + 1, epochs + 1):
        # Training
        train_adversarial_batch(model, train_generator, target_attack=target_attack,
                                show_perturbation=show_perturbation, use_mask=use_mask,
                                change_object_location=change_object_location)


def train_adversarial_batch(model, datagenerator, target_attack=False, show_perturbation=False, use_mask=False,
                            change_object_location=False):
    for inputs in datagenerator:
        images = inputs[0]
        image_metas = inputs[1]
        rpn_match = inputs[2]
        rpn_bbox = inputs[3]
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        # image_metas as numpy array
        image_metas = image_metas.numpy()

        # To GPU
        if model.config.GPU_COUNT:
            images = images.cuda()
            rpn_match = rpn_match.cuda()
            rpn_bbox = rpn_bbox.cuda()
            gt_class_ids = gt_class_ids.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_masks = gt_masks.cuda()

        # SETTINGS
        steps = 40
        max_perturbation = 20

        # Wrap in variables
        images_orig = images.clone()
        images = Variable(images, requires_grad=True)
        rpn_match = Variable(rpn_match)
        rpn_bbox = Variable(rpn_bbox)
        gt_class_ids = Variable(gt_class_ids)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        # Create mask of
        mask = create_mask(images_orig.shape, gt_boxes[0][0])

        for step in range(steps):
            model.zero_grad()
            zero_gradients(images)
            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(
                rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss
            loss.backward()
            print("step {}: loss={}".format(step, loss.data.cpu().numpy()[0]))

            # Calculate gradient
            #grad = images.grad * 10000
            grad = torch.sign(images.grad)

            # Change part of the image in mask only if enabled
            if use_mask:
                grad.data = grad.data * mask

            # Clamp max perturbation per step
            grad = torch.clamp(grad, -max_perturbation/steps, max_perturbation/steps)

            # Add/Subtract perturbation based on attack
            if target_attack:
                images_tmp = unmold_image_tensor(images.data - grad.data, model.config)
            else:
                images_tmp = unmold_image_tensor(images.data + grad.data, model.config)

            # Clamp to reasonable image values
            images_tmp = torch.clamp(images_tmp, 0, 255)
            images_data = mold_image_tensor(images_tmp, model.config)

            # Set adversarial image as new input
            images.data = images_data

        a = images[0].data.clone()

        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        # Run detection
        image_org = unmold_image(img_to_np(images_orig[0]), model.config)
        #results = model.detect([image_org])

        # Visualize results
        #r = results[0]
        #display_instances(image_org, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        # Run detection
        image_adv = unmold_image(img_to_np(a), model.config)
        results = model.detect([image_adv])

        # Visualize results
        r = results[0]
        display_instances(image_adv, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        # Visualize perturbation
        if show_perturbation:
            image_tmp = image_adv.astype(np.int16)
            perturbation = image_tmp - image_org
            perturbation.clip(0, 255)
            scale_factor = 255/perturbation.max()

            # Show Plot
            _, ax = plt.subplots(1, figsize=(16, 16))
            height, width = perturbation.shape[:2]
            ax.set_ylim(height + 10, -10)
            ax.set_xlim(-10, width + 10)
            ax.axis('off')
            ax.set_title('Attacking noise (x{0:4.2f})'.format(scale_factor))
            ax.imshow(perturbation * scale_factor)
            plt.show()

    # Move perturbation created for annotation 1 to different location in image
    # TODO: Currently its added to plain image for testing purpose .
    # TODO: And does not even work when original image is a plain grey image too.
    if change_object_location:
        image_tmp = image_adv.astype(np.int16)
        perturbation = image_tmp - image_org

        # Cut fake object from BBox size of first annotation
        bbox = gt_boxes[0][0].data.cpu().numpy().astype(int)
        fake_object = perturbation[bbox[0]:bbox[2], bbox[1]:bbox[3], :]

        # Create plain grey = 127 image
        plain = np.zeros([512, 512, 3]) + 127

        # Add fake_object to plain image
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        start_x = bbox[0]
        start_y = bbox[1]
        plain[start_x:start_x + width, start_y:start_y + height, :] = fake_object
        plain = plain.astype(np.uint8)

        # Run detection
        try:
            results = model.detect([plain])

            # Visualize results
            r = results[0]
            display_instances(plain, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        except IndexError:
            print("No detection found")


def create_mask(shape, bbox):
    '''

    :param shape: mask shape
    :param bbox: (x, y, width, height)
    :return:
    '''

    bbox = bbox.data.cpu().numpy().astype(int)
    mask = torch.zeros(shape)
    for i in range(3):
        for j in range(bbox[0], bbox[2]):
            for k in range(bbox[1], bbox[3]):
                mask[0][i][j][k] = 1

    return mask.cuda()


def mold_image_tensor(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images - torch.from_numpy(config.MEAN_PIXEL).float().cuda().unsqueeze(1).unsqueeze(2).unsqueeze(0).expand_as(images)


def unmold_image_tensor(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return normalized_images + torch.from_numpy(config.MEAN_PIXEL).float().cuda().unsqueeze(1).unsqueeze(2).unsqueeze(0).expand_as(normalized_images)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--download', required=False,
    #                     default=False,
    #                     metavar="<True|False>",
    #                     help='Automatically download and unzip MS-COCO files (default=False)',
    #                     type=bool)
    parser.add_argument('--target', required=False,
                        default="class",
                        metavar="<class|localisation|segmentation|combined>",
                        help='Perform a target attack on class, localisation, segmentation '
                             'or a combined attack (default=class)')
    parser.add_argument('--use-mask', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Use bbox of first annotation as mask (default=False)',
                        type=bool)
    parser.add_argument('--show-perturbation', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Shows scaled perturbation (default=False)',
                        type=bool)
    parser.add_argument('--change-object-location', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Moves perturbation created for first annotation to a different place in the image'
                             ' (default=False)',
                        type=bool)


    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Target: ", args.target)
    print("Show Perturbation: ", args.show_perturbation)
    print("Use Mask: ", args.use_mask)
    print("Change object location: ", args.change_object_location)
    # print("Auto Download: ", args.download)

    config = CocoConfig()
    config.display()

    # Create model
    model = MaskRCNN(config=config, model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    model_path = COCO_MODEL_PATH

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    dataset_train = CocoDataset()
    #dataset_train.load_coco(args.dataset, "minival", year=args.year, auto_download=args.download)  # Uncomment to get all coco images
    if args.target is not None and args.target != "":
        dataset_train.load_coco(args.dataset, "adversarial_attack_target_" + args.target, year=2014, auto_download=False)
    else:
        dataset_train.load_coco(args.dataset, "adversarial_attack", year=2014, auto_download=False)
    dataset_train.prepare()

    train_adversarial(
        model,
        dataset_train,
        epochs=1,
        layers='all',
        target_attack=args.target,
        show_perturbation=args.show_perturbation,
        use_mask=args.use_mask
    )


"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------ 

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 cockheads.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 cockheads.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 cockheads.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 cockheads.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
    # working example
    python3 cockheads.py detect --dataset=/home/lab/Documents/python/Mask_RCNN/datasets/ --subset=stage1_test --weights=/home/lab/Documents/python/Mask_RCNN/logs/nucleus20180816T2105/mask_rcnn_nucleus_0040.h5

python AP.py detect --weights=/home/lab/Documents/python/ClumsyChickens/logs/chicken20200313T2051/mask_rcnn_chicken_0120.h5 --dataset=/home/lab/Documents/python/ClumsyChickens/datasets --subset=val
```
python AP.py detect --weights=/home/lab/Documents/python/ClumsyChickens/logs/chicken20200820T1955/mask_rcnn_chicken_0098.h5 --dataset=/home/lab/Documents/python/ClumsyChickens/datasets --subset=val
"""
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os, warnings
from distutils.version import LooseVersion
import sys
import json
import random
import datetime
import numpy as np
import cv2
import skimage.io
from imgaug import augmenters as iaa

# suppress skimage 0.14.0 UserWarning about anti-aliasing options (you can also revert to 0.13)
if LooseVersion(skimage.__version__) > LooseVersion('0.13.0'):
    warnings.filterwarnings('ignore',category=UserWarning)

try:
    import mrcnn
except ImportError as err:
    MRCNN_ROOT = os.path.join(os.environ['HOME'], 'Documents/python/Mask_RCNN')
    # add Mask_RCNN to path
    sys.path.append(MRCNN_ROOT)
finally:
    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
    from mrcnn import visualize
    from mrcnn.model import log

filepath = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(filepath)
# image file extension
IMAGE_EXT = 'jpg'
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/chicken/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to serve as a validation set.
# TODO: use train_test_split from sklearn
VAL_IMAGE_IDS = [
    '20200311_151734','20200311_151736','20200311_151738','20200311_151740',
    '20200311_151802','20200311_151804','20200313_140419','20200313_140428',
    '20200324_143715','20200324_143735','20200324_143724','20200324_143723',
    '20200324_143742','20200324_143743','20200324_143741','20200324_143720',
    '20200412_141119','20200412_141129','20200412_141130','20200412_141141',
    '20200412_141153','20200412_141154','20200412_152624','20200412_152630',
    '20200425_141324','20200425_141325','20200425_141336','20200425_141337',
    '20200425_141339','20200425_141341','20200425_141331','20200425_141331',
    '20200506_151600','20200506_151601','20200506_151602','20200508_152747',
    '20200508_152753','20200508_152717','20200506_151538','20200506_151548',
    '20200523_152911','20200523_152907','20200523_152855','20200523_152859',
    '20200523_152903','20200523_152843','20200523_152845','20200523_152825'
]
############################################################
#  Configurations
############################################################

class ChickenConfig(Config):
    """Configuration for training on the chicken segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "chicken"  
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + chicken_heads
    # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)
    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.8
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"
    # Input image resizing
    # Random crops of size 512x512
    # IMAGE_RESIZE_MODE = "crop"
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    # USE_MINI_MASK = True
    USE_MINI_MASK = False    
    MINI_MASK_SHAPE = (56, 56) #(height, width) of the mini-mask 
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128
    # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 200
    MAX_GT_INSTANCES = 50
    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 50

inference_config = ChickenConfig()

class ChickenInferenceConfig(ChickenConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    # IMAGES_PER_GPU = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
############################################################
#  Dataset
############################################################
class ChickenDataset(utils.Dataset):

    def load_chickens(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset `chicken`, and the class `head`
        self.add_class("chicken", 1, "head")
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            # print(image_ids)
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))
            # if subset == "stage1_test":
            #     image_ids = list(set(image_ids))  
        # Add images
        for image_id in image_ids:
            self.add_image(
                "chicken",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{fname}.{ext}".format(fname=image_id, ext=IMAGE_EXT)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        # Read mask files from .jpg image
        # TODO: sopport multiple formats
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            # if f.endswith(".png"):
            # i don't think this is needed!!!!????
            m = skimage.io.imread(os.path.join(mask_dir, f), as_grey = True).astype(np.bool)
            mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "chicken":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = ChickenDataset()
    dataset_train.load_chickens(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ChickenDataset()
    dataset_val.load_chickens(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((1, 3), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))      
    ])
    # *** This training schedule is an example. Update to your needs ***
    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)
############################################################
#  Detection
############################################################
def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = ChickenDataset()
    dataset.load_chickens(dataset_dir, subset)
    dataset.prepare()

    dataset_val = ChickenDataset()
    dataset_val.load_chickens(dataset_dir, "stage1_test")
    dataset_val.load_chickens(dataset_dir, "val")
    dataset_val.prepare()

    # Load over images
    submission = []
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    num = 0
    count = 0

    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        count = count+1

        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

        r = model.detect([image], verbose=0)[0]

        APss, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(APss)
        num = num + APss
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks

        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            #  show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.jpg".format(submit_dir, dataset.image_info[image_id]["id"]))
    
    print(APs)       
    print(num/count)   
    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ChickenConfig()
    else:
        config = ChickenInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
   # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

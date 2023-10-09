import os
import sys
import json
import numpy as np
import skimage.draw
import cv2
import imgaug

# Root directory of the project
ROOT_DIR = "C:/Users/parvd/OneDrive/Desktop/Project/Project"

# Import Mask RCNN (assuming Mask R-CNN is in the ROOT_DIR)
sys.path.append(ROOT_DIR)  

from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 4  # Background + Apple_Black_rot, Apple_healthy, Apple_Rust, Apple_Scrab
    STEPS_PER_EPOCH = 5
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        # Add classes
        self.add_class("object", 1, "Apple_Black_Rot")
        self.add_class("object", 2, "Apple_Healthy")
        self.add_class("object", 3, "Apple_Rust")
        self.add_class("object", 4, "Apple_Scab")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations1 = json.load(open(os.path.join(dataset_dir, "Apple.json")))
        annotations = list(annotations1.values())

        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['names'] for s in a['regions']]
            name_dict = {"Apple_Black_Rot": 1, "Apple_Healthy": 2, "Apple_Rust": 3, "Apple_Scab": 4}
            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        num_ids = image_info['num_ids']
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(image_info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    dataset_train = CustomDataset()
    dataset_train.load_custom("C:/Users/parvd/OneDrive/Desktop/Project/Project/Data Set/Train", "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom("C:/Users/parvd/OneDrive/Desktop/Project/Project/Data Set/Val", "val")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='heads',
                augmentation=imgaug.augmenters.Sequential([
                    imgaug.augmenters.Fliplr(1),
                    imgaug.augmenters.Flipud(1),
                    imgaug.augmenters.Affine(rotate=(-45, 45)),
                    imgaug.augmenters.Affine(rotate=(-90, 90)),
                    imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                    imgaug.augmenters.Crop(px=(0, 10)),
                    imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                    imgaug.augmenters.AddToHueAndSaturation((-20, 20)),
                    imgaug.augmenters.Add((-10, 10), per_channel=0.5),
                    imgaug.augmenters.Invert(0.05, per_channel=True),
                    imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                ]))

config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

train(model)

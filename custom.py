import os
import sys
import random
import math
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import json
from tensorflow.keras.layers import Layer
from mrcnn.config import Config
from mrcnn import model as modellib,utils
from mrcnn.model import log

# Set the ROOT_DIR to the root directory of the Mask R-CNN repository
ROOT_DIR = "C:/Users/parvd/OneDrive/Desktop/Instance Segmentation"

# Import Mask RCNN
sys.path.append(ROOT_DIR)

# Path to trained weights file
COCO_MODEL_PATH = "C:/Users/parvd/OneDrive/Desktop/Instance Segmentation/mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to the training and validation datasets (COCO format)
TRAIN_ANNOTATIONS_PATH = "C:/Users/parvd/OneDrive/Desktop/Instance Segmentation/Data Set/Train/Apple.json"
VAL_ANNOTATIONS_PATH = "C:/Users/parvd/OneDrive/Desktop/Instance Segmentation/Data Set/Val/Val_json.json"

class PlantDiseaseConfig(Config):
    NAME = "plant_disease"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + class for plant disease
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

config = PlantDiseaseConfig()
config.display()

class PlantDiseaseDataset(utils.Dataset):
    def load_plantdisease(self, dataset_dir, subset):
        """Load a subset of the plant disease dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load (train or val)
        """
        # Add classes (only one class for plant disease detection)
        self.add_class("plant_disease", 1, "disease")

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, f"{subset}.json")))
        annotations = annotations['_via_img_metadata']

        # Add images and their annotations to the dataset
        for image_id, image_info in annotations.items():
            if 'regions' in image_info:
                polygons = [r['shape_attributes'] for r in image_info['regions']]
                image_path = os.path.join(dataset_dir, image_info['filename'])
                self.add_image(
                    "plant_disease",
                    image_id=image_id,
                    path=image_path,
                    width=image_info['width'],
                    height=image_info['height'],
                    polygons=polygons
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "plant_disease":
            return super(PlantDiseaseDataset, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape [height, width, instances]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

# Create training and validation dataset
dataset_train = PlantDiseaseDataset()
dataset_train.load_plantdisease("C:/Users/parvd/OneDrive/Desktop/Instance Segmentation/Data Set", "Train")
dataset_train.prepare()

dataset_val = PlantDiseaseDataset()
dataset_val.load_plantdisease("C:/Users/parvd/OneDrive/Desktop/Instance Segmentation/Data Set", "Val")
dataset_val.prepare()

# Load and create model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Data augmentation
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-45, 45)),
])

# Train the model
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads',
            augmentation=augmentation)

# Save weights (optional)
model.keras_model.save_weights("C:/Users/parvd/OneDrive/Desktop/Instance Segmentation/plant_disease_weights.h5")

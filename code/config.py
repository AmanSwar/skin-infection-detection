import torch.nn as nn
import os
import torch
import torchvision.transforms.v2 as v2
import random
import torchvision.transforms.functional as F
from torchvision import transforms



CLASSES = []
DATA_DIR = "/home/aman/code/CV/skin_infec_detect/data/IMG_CLASSES"
DEVICE = "cuda"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
CLASSES = [i for i in os.listdir(DATA_DIR)]
CLASS_TO_LABELS = {class_name : label for label , class_name in enumerate(CLASSES)}
DEPTH = 121
GROWTH_RATE = 32
REDUCTION_RATE = 0.5
DROP_RATE = 0.3
LOSS_FN = nn.BCEWithLogitsLoss()
CLASS_WEIGHTS = []
DATA_SIZE = sum([len(os.listdir(os.path.join(DATA_DIR , i))) for i in os.listdir(DATA_DIR)])


# tranformation

# image to tensor
transform_general = transforms.Compose(
    [
        transforms.Resize((256 , 256)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32)
    ]
)


# base transformation

base_transform = v2.Compose(
    [
        v2.Resize((256,256))
    ]
)

# flipping 
trans_flip = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.8)
    ]
)

new_h = random.randrange(200 ,256)
new_w = random.randrange(200 , 256)

trans_zoom = v2.Compose(
    [
        v2.RandomCrop(size=(new_h , new_w))
    ]
)

# rotation 
trans_rotate = v2.Compose(
    [
        v2.RandomRotation(degrees=(-10 , 10))
    ]
)

def adjust_brightness(image):

    factor = random.uniform(0.5 , 1.9)

    return F.adjust_brightness(image , factor)

TRANS_ALL = [trans_flip , trans_zoom , trans_rotate , adjust_brightness]


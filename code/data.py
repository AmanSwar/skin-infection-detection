import torch
from torch.utils.data import Dataset , DataLoader  , Subset
from config import CLASS_TO_LABELS , DATA_DIR
import os
from config import transform_general , BATCH_SIZE , DEVICE
import random
from PIL import Image
from utils import get_random_subset



class SkinDiseasDataset(Dataset):

    def __init__(self , data_dir , transforms=transform_general):
        self.data_dir = data_dir
        self.transform = transform_general
        self.image_path = []
        self.labels = []

        for class_name , labels in CLASS_TO_LABELS.items():

            class_dir = os.path.join(data_dir ,class_name)


            for image_path in os.listdir(class_dir):
                self.image_path.append(os.path.join(class_dir , image_path))
                self.labels.append(labels)

        random.shuffle(self.image_path)
        # print(len(CLASS_TO_LABELS))

    def __len__(self):
        return len(self.image_path)
    

    def __getitem__(self , idx):
        random.shuffle(self.image_path)
        image_path = self.image_path[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]

        return image , label
    

DATASET = SkinDiseasDataset(data_dir=DATA_DIR , transforms=transform_general)

# print(len(DATASET))

train_dataset , valid_dataset = get_random_subset(DATASET , 110212)


TRAIN_DL = DataLoader(train_dataset , batch_size=BATCH_SIZE , shuffle=True , pin_memory=True)
VALID_DL = DataLoader(valid_dataset , batch_size=BATCH_SIZE , shuffle=False , pin_memory=True)


def train_valid():
    return TRAIN_DL , VALID_DL



        
import os
from config import CLASSES , CLASS_WEIGHTS , TRANS_ALL , DATA_SIZE , base_transform , DATA_DIR
import random
from torch.utils.data import Subset


def get_classes(data_dir):
    
    for i in os.listdir(data_dir):
        if i not in CLASSES:
            CLASSES.append(i)

def apply_transformations(img , image_dir , image_name):
    print("I am here")
    img = base_transform(img)
    for trans in TRANS_ALL:

        trans_img = trans(img)
        filename = f"{image_dir}_{trans}_{image_name}"
        filepath = os.path.join(image_dir , filename)
        print(filepath)
        trans_img.save(filepath)
        
def custom_weights():
    # weights = total samples / no. of samples in that grp
    
    for i in range(len(CLASSES)):
        weight_ = DATA_SIZE / len(os.listdir(os.path.join(DATA_DIR , CLASSES[i])))
        CLASS_WEIGHTS.append(weight_)


def get_random_subset(ds , num_sample):
    all_indices = list(range(len(ds)))

    random.shuffle(all_indices)
    random.shuffle(all_indices)
    random.shuffle(all_indices)

    train_indices = all_indices[:num_sample]
    valid_indices = all_indices[num_sample:]

    train_sub = Subset(ds , train_indices)
    valid_sub = Subset(ds , valid_indices)


    return train_sub , valid_sub


def check_requires_grad(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name} - Requires Grad: True")
        else:
            print(f"Parameter Name: {name} - Requires Grad: False")
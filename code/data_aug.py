import os
from PIL import Image

from config import CLASSES , DATA_DIR
from utils import apply_transformations

def data_basic_transform():
    for i in CLASSES:
  
        image_dir = os.path.join(DATA_DIR , i)

        for image_name in os.listdir(image_dir):

            image_path = os.path.join(image_dir , image_name)

            img = Image.open(image_path)

            apply_transformations(img , image_dir=image_dir , image_name=image_name)


import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2
import csv

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, train=True):
        self.image_dir = image_dir
        self.transform = transform
        self.image_name_list = self.get_image_name_list()
        if train:
            random.shuffle(self.image_name_list)
        else:
            pass
        assert len(self.image_name_list) > 0

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index):
        assert index < len(self.image_name_list), 'index range error'
        image_name = self.image_name_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        image_label = int(image_name[-5])
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((224, 224), Image.ANTIALIAS)

        if self.transform is not None:
            image = self.transform(image)
        return image, image_label

    def get_image_name_list(self):
        image_name_list = []
        for dir, subdirs, subfiles in os.walk(self.image_dir):
            for file in subfiles:
                if os.path.splitext(file)[1] == '.png':
                    image_name_list.append(file)
        return image_name_list


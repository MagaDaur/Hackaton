from transformers import CLIPTokenizerFast, CLIPModel, CLIPProcessor

import cv2
from tqdm import tqdm
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from config import nn_classes

import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from PIL import Image

train_folder_path = "train_dataset/train"

class TrainLoader(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, sep=';')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        object_id = self.data.iloc[index]['object_id']
        img_name = self.data.iloc[index]['img_name']

        image_path = f'{train_folder_path}/{object_id}/{img_name}'
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        category_index = nn_classes.index(self.data.iloc[index]['group'])
            
        return image_path, category_index

device = 'cuda:0'

train_data = TrainLoader('train_dataset/train.csv')

image_embeddings = []
list_of_files = []

model_id = 'openai/clip-vit-base-patch32'
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

for image_path, label in tqdm(train_data):
    list_of_files.append(image_path)
    image = Image.open(image_path)

    inputs = processor(text=None, images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = image_features.squeeze(0)
    image_features = image_features.cpu().detach().numpy()

    image_embeddings.append(image_features)

image_arr = np.vstack(image_embeddings)

np.savez('image_embeddings.npz', embeddings=image_arr, file_names=list_of_files)
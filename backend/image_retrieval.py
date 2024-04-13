import numpy as np
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

device = 'cpu'

data = np.load('image_embeddings.npz')

image_filepaths = data['file_names']
image_embeddings = data['embeddings']

model_id = 'openai/clip-vit-base-patch32'
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def get_similar_images(file_path : str):
    search_image = Image.open(file_path)
    search_inputs = processor(text=None, images=search_image, return_tensors="pt")
    search_pixel_values = search_inputs["pixel_values"].to(device)
    search_features = model.get_image_features(pixel_values=search_pixel_values).squeeze(0).cpu().detach().numpy()

    distances = 1 - cosine_similarity([search_features], image_embeddings).flatten()

    path_indexes = distances.argsort()[:10]
    distances = sorted(distances)[:10]


    return [image_filepaths[path_indexes[i]] for i in range(10)]

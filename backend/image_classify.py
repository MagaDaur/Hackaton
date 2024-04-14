from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn

from config import nn_classes

device = 'cpu'

# Загрузка модели
loaded_model = models.resnet152()
num_ftrs = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_ftrs, 15)  # Указываем ту же архитектуру, что и при обучении
loaded_model.load_state_dict(torch.load('ResNet152.pth', map_location=torch.device(device)))
loaded_model = loaded_model.to(device)
loaded_model.eval()  # Переводим модель в режим оценки
# Трансформация изображения для предсказания

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Добавляем размерность батча
    return image

# Функция для получения предсказания
def predict_image(image_path):
    loaded_model.eval()
    image = transform_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        output = loaded_model(image)
        _, predicted = torch.max(output, 1)
        idx = predicted.item()

    return nn_classes[idx]
# utils.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from typing import Tuple

MODEL_NAME = "resnet50"
NUM_CLASSES = 43
IMAGE_SIZE = (64, 64)
MODEL_PATH = "models/best_model.pth"

# Определяем устройство для инференса
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(num_classes: int) -> nn.Module:
    """Создает модель, но БЕЗ загрузки предобученных весов ImageNet."""
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,  # Мы загрузим свои веса, предобучение не нужно
        num_classes=num_classes
    )
    return model

def load_trained_model() -> nn.Module:
    """Загружает модель и веса с диска."""
    model = create_model(num_classes=NUM_CLASSES)
    # Загружаем веса, убедившись, что они грузятся на правильное устройство
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Переводим модель в режим инференса (важно!)
    print(f"Model '{MODEL_NAME}' loaded from '{MODEL_PATH}' on device '{DEVICE}'")
    return model

def get_inference_transforms(img_size: Tuple[int, int]) -> A.Compose:
    """Трансформации для одного изображения на этапе инференса."""
    # Эти трансформации ДОЛЖНЫ быть такими же, как get_val_test_transforms в обучении
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Принимает изображение в виде байтов, преобразует и возвращает готовый тензор.
    """
    # 1. Читаем байты в numpy массив
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # Albumentations ожидает RGB

    # 2. Применяем трансформации
    transform = get_inference_transforms(IMAGE_SIZE)
    processed_image = transform(image=image_np)['image']
    
    # 3. Добавляем batch-измерение и отправляем на устройство
    input_tensor = processed_image.unsqueeze(0).to(DEVICE)
    
    return input_tensor

def get_class_name_by_id(class_id: int, lang: str) -> str:
    '''
    Возвращает имя класса по его id
    '''
    class_names = pd.read_csv('class_names.csv')
    return class_names.loc[class_names['class'] == class_id, lang].values[0]

def get_classes_names(language: str) -> dict:
    '''
    Возвращает словарь соответствий номера и имени доступных классов
    '''
    class_names = pd.read_csv('class_names.csv')
    return class_names[language].to_dict()


if __name__ == '__main__':
    print(get_classes_names('rus'))

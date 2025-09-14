"""
PyTorch Dataset class for Kanji-English pairs
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms

class KanjiDataset(Dataset):
    def __init__(self, dataset_path, image_size=256, transform=None):
        """
        Args:
            dataset_path (str): Path to the JSON dataset file
            image_size (int): Size to resize images to
            transform: Optional transform to be applied on images
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_size = image_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get text description
        text = item['text']
        
        return {
            'image': image,
            'text': text,
            'kanji': item['kanji']
        }

class KanjiDatasetHF(Dataset):
    """HuggingFace compatible dataset class for Stable Diffusion training"""
    def __init__(self, dataset_path, image_size=256, tokenizer=None):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_size = image_size
        self.tokenizer = tokenizer
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1] for RGB
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        pixel_values = self.transform(image)
        
        # Get text
        text = item['text']
        
        # Tokenize text if tokenizer is provided
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs.input_ids[0]  # Remove batch dimension
        else:
            input_ids = None
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'text': text,
            'kanji': item['kanji']
        }

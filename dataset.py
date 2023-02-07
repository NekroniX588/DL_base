import os
import math
import torch
import torchvision.transforms as transforms


from PIL import Image
from typing import List, Dict


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.paths = []
        self.transform = transform
        
        labels = []
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                continue
            self.paths.append(full_path)
            labels.append(file.split(".")[0])
            
        self.vocab = self.__create_vocab(labels)
        self.inv_vocab = {item: key for key, item in self.vocab.items()}
        
        self.labels = []
        for label in labels:
            self.labels.append(self.encode(label))
        
        
    def __create_vocab(self, labels: List[str]) -> Dict[str, int]:
        vocab = {}
        current = 0
        for label in labels:
            for char in label:
                if char not in vocab:
                    vocab[char] = current
                    current += 1
        return vocab
    
    def encode(self, label: str) -> List[int]:
        result = []
        for char in label:
            if char in self.vocab:
                result.append(self.vocab[char])
        return result
    
    def decode(self, encoded: List[int]) -> str:
        result = ""
        for item in encoded:
            if item in self.inv_vocab:
                result += self.inv_vocab[item]
        return result
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("L")
        target = torch.tensor(self.labels[index]).long()
        if self.transform is not None:
            img = self.transform(img)
        return img, target
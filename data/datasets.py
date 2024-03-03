import os
from   PIL import Image

import torch
from   torchvision import transforms

from   base import BaseDataset


class GeometricShapeMathematics(BaseDataset):
    def __init__(self, topology: str):
        self.files = os.path.join(self.images_path, 'train') if self.is_training else os.path.join(self.images_path, 'test')
        self.top = os.path.join(self.topology_path, 'train') if self.is_training else os.path.join(self.topology_path, 'test')
        self.topvecs = os.path.join(self.top, topology) if self.is_training else os.path.join(self.test_top, topology)
        
        self.file_names = self.meta['file_name']
        self.targets = self.meta['shape']
        
        self.prerpocess = transforms.ToTensor()
        target_transform = {
            'circle': 0,
            'kite': 1,
            'parallelogram': 2,
            'rectangle': 3,
            'rhombus': 4,
            'square': 5,
            'trapezoid': 6,
            'triangle': 7
            }
        

    def load_image(self, idx):
        file = self.file_names.iloc[idx]
        path = os.path.join(self.files, file)
        img = Image.open(path).convert('L')
        return img
        
    def load_topology(self, idx):
        file = self.files.iloc[idx] + '.npy'
        path = os.path.join(self.topvecs, file)
        vec = np.load(path)
        return torch.from_numpy(vec).float()
    
    def load_target(self, idx):
        target = self.meta.iloc[idx]
        return target_transform[target]
        
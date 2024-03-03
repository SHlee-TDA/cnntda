from abc import ABC, abstractmethod
import os
from typing import Tuple, Union

import pandas as pd
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(
            self,
            root: str,
            preprocess=None,
            multimodal_learning: bool = True,
            is_training: str = True
            ):
        """
            Initialize the dataset with a directory structure, preprocessing function, and a flag indicating the use of multimodal data.
            
            Parameters:
                root (str): The root directory where the dataset is stored.
                preprocess (callable, optional): A function for preprocessing the images.
                multimodal_learning (bool): Indicates whether to use multimodal data including topology.
        """
        self.root = root
        self.meta_path = os.path.join(self.root, 'metadata')  # Path to the metadata directory.
        self.images_path = os.path.join(self.root, 'images')  # Path to the images directory.
        self.topology_path = os.path.join(self.root, 'topology')  # Path to the topology data directory.
        self.transforms = preprocess  # Preprocessing/transformation function for images.
        self.use_topology = multimodal_learning  # Flag to use topology data alongside images.
        self.is_training = is_training
        
        self.meta = self.load_metadata()  # Load metadata from the metadata directory.
        
    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.meta)
    
    def __getitem__(self, idx):
        """
        Get an item by index, including its image, (optional) topology, and target.
        
        Parameters:
            idx (int): The index of the item.
            
        Returns:
            Tuple or Tuple of Tuples: Depending on the use of topology, returns either an image and its target or a tuple containing both an image and topology data along with the target.
        """
        img = self.load_image(idx)  # Load image
        if self.transforms:
            img = self.transforms(img)  # Apply transformations if any
        
        if not self.use_topology:
            if self.is_training:
                target = self.load_target(idx)  # Load target
                return img, target
            else:
                return img
        else:
            top = self.load_topology(idx)  # Load topological data if multimodal learning is enabled
            if self.is_training:
                target = self.load_target(idx)  # Load target
                return (img, top), target
            else:
                return (img, top)

    @abstractmethod
    def load_image(self, idx):
        """Abstract method for loading an image. Must be implemented by subclasses."""
        raise NotImplementedError
        
    @abstractmethod
    def load_topology(self, idx):
        """Abstract method for loading topological data. Must be implemented by subclasses."""
        raise NotImplementedError
    
    @abstractmethod
    def load_target(self, idx):
        """Abstract method for loading the target. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def load_metadata(self):
        """Load metadata from the first CSV file found in the metadata directory."""
        csv_files = [f for f in os.listdir(self.meta_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in metadata directory.")
        full_path = os.path.join(self.meta_path, csv_files[0])  # Assume the first CSV file is the metadata file
        return pd.read_csv(full_path)
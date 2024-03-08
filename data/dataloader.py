"""A script for creating DataLoader instances for training and inference, based on configurable dataset and transformations.

Author: Seong-Heon Lee (POSTECH MINDS)
Date: 2024-03-08
E-mail: shlee0125@postech.ac.kr
License: MIT 

This script provides utilities for creating DataLoader objects for training, validation, and inference purposes. 
It supports automatic detection of dataset classes within a specified module, applying transformations to the datasets, and generating DataLoader instances based on user-defined configurations.

Features:
- Detects and maps dataset classes derived from a BaseDataset class for dynamic dataset loading.
- Converts a list of transformation configurations into a torchvision.transforms.Compose object for data preprocessing.
- Creates DataLoader objects for training and validation sets with options for stratification and random splitting.
- Generates an inference DataLoader with a flexible configuration for model evaluation or real-time predictions.

The script is designed to be flexible and extensible, allowing for easy integration with custom dataset classes and preprocessing steps.

Usage:
1. Implement your dataset classes inheriting from BaseDataset in `datasets.py`.
2. Specify your dataset and DataLoader configurations in `config.json`.
3. Use create_data_loaders() for training/validation purposes or create_inference_loader() for inference.
"""

import inspect
import importlib
from   typing import Any, Dict, Optional, Tuple, Type

from   sklearn.model_selection import train_test_split
from   torch.utils.data import Subset, DataLoader
from   torchvision.transforms import Compose, transforms

from   .base import BaseDataset


def detect_datasets(module_path: str) -> Dict[str, Type[BaseDataset]]:
    """
    Detect dataset classes in a given module that are subclasses of BaseDataset.

    Args:
    - module_path (str): The module path to search for dataset classes.

    Returns:
    - Dict[str, Type[BaseDataset]]: A dictionary mapping dataset class names to their class types.
    
    Raises:
    - ImportError: If the specified module cannot be found.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        raise ImportError(f"Module {module_path} not found. Ensure the module path is correct.")
    
    dataset_mapping = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseDataset) and obj is not BaseDataset:
            dataset_mapping[name] = obj
    return dataset_mapping


def get_transforms(transforms_configs: Optional[dict]) -> Optional[Compose]:
    """
    Convert a list of transform configurations to a torchvision.transforms.Compose object.
    
    Args:
    - transforms_configs (list of dict): List of transform configurations.

    Returns:
    - torchvision.transforms.Compose: Composed transforms based on the configurations.
    
    Raises:
    - ValueError: If an unsupported transform is specified in the configurations.
    """
    if transforms_configs is None:
        return None
    else:
        transforms_mapping = {
            "Resize": transforms.Resize,
            "ToTensor": transforms.ToTensor,
            "Normalize": transforms.Normalize,
            # Add more transforms as needed
        }

        # Fetch the actual transform objects from the mapping using the given parameters
        try:
            transforms_objs = [transforms_mapping[tc["name"]](**tc["params"]) for tc in transforms_configs]
        except KeyError as e:
            raise ValueError(f"Transform {e} not supported. Check the transform configurations.")
        return transforms.Compose(transforms_objs)


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders based on the configuration.
    
    Args:
    - config (Dict[str, Any]): Configuration dictionary containing options for dataset and dataloaders.

    Returns:
    - Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoader objects.
    """
    
    # Determine the dataset class from the dataset name
    dataset_options = config.get('dataset_options')
    dataset_mapping = detect_datasets('data.datasets')
    DatasetClass = dataset_mapping[dataset_options.get("dataset")]
    
    # Fetch the transforms
    transforms_configs = dataset_options.get("preprocess")
    transforms = get_transforms(transforms_configs)

    # Initialize dataset with training flag set for train and validation
    dataset = DatasetClass(
        root=dataset_options.get('root'),
        transforms=transforms,
        multimodal_learning=config.get('multimodal_learning'),
        is_training=True
        )

    # Prepare dataset indices for splitting
    indices = list(range(len(dataset)))

    if dataset_options.get('task') == 'classification':
        # Apply stratification for classification tasks
        labels = [dataset.load_target(idx) for idx in indices]
        train_indices, valid_indices = train_test_split(
            indices, stratify=labels, test_size=float(dataset_options.get("val_size")))
    else:
        # Random split for other tasks
        train_indices, valid_indices = train_test_split(
            indices, test_size=float(dataset_options.get("val_size")))
        
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)

    train_loader = DataLoader(
            train_subset, 
            batch_size=dataset_options.get("batch_size"), 
            shuffle=True, 
            num_workers=dataset_options.get("num_workers")
            )
    val_loader = DataLoader(
            valid_subset, 
            batch_size=dataset_options.get("batch_size"), 
            shuffle=False, 
            num_workers=dataset_options.get("num_workers")
            )
    
    return train_loader, val_loader


def create_inference_loader(config: Dict[str, Any]) -> DataLoader:
    """
    Create an inference DataLoader based on the provided configuration.

    This function initializes a DataLoader for inference purposes, utilizing the dataset class specified
    in the configuration.

    Parameters:
    - config (Dict[str, Any]): A configuration dictionary that includes all necessary information
      such as dataset options, preprocessing transforms, and DataLoader settings.

    Returns:
    - DataLoader: A DataLoader object ready for inference, configured according to the provided settings.

    Raises:
    - ValueError: If the specified dataset class is not found within the available datasets.
    """
    
    # Determine the dataset class from the dataset name
    dataset_options = config.get('dataset_options')
    dataset_mapping = detect_datasets('data.datasets')
    DatasetClass = dataset_mapping.get(dataset_options.get("dataset"))

    # Fetch the transforms
    transforms_configs = dataset_options.get("preprocess")
    transforms = get_transforms(transforms_configs)

    # Initialize dataset with training flag set for train and validation
    dataset = DatasetClass(
        root=dataset_options.get('root'),
        transforms=transforms,
        multimodal_learning=config.get('multimodal_learning'),
        is_training=False  # Set to False for inference mode
    )

    # Initialize the DataLoader for the entire dataset to be used for inference
    inference_loader = DataLoader(
        dataset, 
        batch_size=dataset_options.get("batch_size"), 
        shuffle=False,
        num_workers=dataset_options.get("num_workers")
    )
    
    return inference_loader

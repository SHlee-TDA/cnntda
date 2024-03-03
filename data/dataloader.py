import inspect
import importlib
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import transforms

from .base import BaseDataset


def detect_datasets(module_path):
    module = importlib.import_module(module_path)
    dataset_mapping = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseDataset) and obj is not BaseDataset:
            dataset_mapping[name] = obj
    return dataset_mapping


def get_transforms(transforms_configs):
    """Convert a list of transform configurations to a torchvision.transforms.Compose object."""
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
        transforms_objs = [transforms_mapping[tc["name"]](**tc["params"]) for tc in transforms_configs]
        return transforms.Compose(transforms_objs)



def create_data_loaders(config):
    """Create train and validation dataloaders based on the configuration."""
    
    # 1. Determine the dataset class from the dataset name
    dataset_options = config.get('dataset_options')
    dataset_mapping = detect_datasets('datasets')
    DatasetClass = dataset_mapping[dataset_options.get("dataset")]
    
    # 2. Fetch the transforms
    transforms_configs = dataset_options.get("preprocess")
    transforms = get_transforms(transforms_configs)

    # 3. Initialize dataset with training flag set to True and False for train and validation respectively
    dataset = DatasetClass(
        root=dataset_options.get('root'),
        transforms=transforms,
        multimodal_learning=config.get('multimodal_learning'),
        is_training=True
        )

    # 첫 번째 분할: 훈련 및 검증/테스트 세트
    # 데이터셋 인덱스 준비
    indices = list(range(len(dataset)))

    if dataset_options.get('task') == 'classification':
        # 분류 작업인 경우에만 stratify 적용
        labels = [dataset.load_target(idx) for idx in indices]  # 레이블 로딩을 위한 메서드 또는 콜백 사용
        train_indices, valid_indices = train_test_split(
            indices, stratify=labels, test_size=float(dataset_options.get("val_size")))
    else:
        # 기타 작업인 경우 무작위 분할
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

def create_data_loaders(config):
    """Create train and validation dataloaders based on the configuration."""
    
    # 1. Determine the dataset class from the dataset name
    dataset_options = config.get('dataset_options')
    dataset_mapping = detect_datasets('data.datasets')
    DatasetClass = dataset_mapping[dataset_options.get("dataset")]
    
    # 2. Fetch the transforms
    transforms_configs = dataset_options.get("preprocess")
    transforms = get_transforms(transforms_configs)

    # 3. Initialize dataset with training flag set to True and False for train and validation respectively
    dataset = DatasetClass(
        topology=dataset_options.get('topology'),
        root=dataset_options.get('root'),
        preprocess=transforms,
        multimodal_learning=config.get('multimodal_learning'),
        is_training=True
        )

    # 첫 번째 분할: 훈련 및 검증/테스트 세트
    # 데이터셋 인덱스 준비
    indices = list(range(len(dataset)))

    if dataset_options.get('task') == 'classification':
        # 분류 작업인 경우에만 stratify 적용
        labels = [dataset.load_target(idx) for idx in indices]  # 레이블 로딩을 위한 메서드 또는 콜백 사용
        train_indices, valid_indices = train_test_split(
            indices, stratify=labels, test_size=float(dataset_options.get("val_size")))
    else:
        # 기타 작업인 경우 무작위 분할
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
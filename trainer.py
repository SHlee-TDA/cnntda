"""
Deep Learning Model Trainer

This script defines a Trainer class that encapsulates the entire process of training a deep learning model. It handles
the setup of the model for training on either CPU or GPU, supports single and multi-GPU setups, manages data loading,
executes the training and validation loops, implements early stopping, saves checkpoints for the best-performing models,
and logs training metrics using Weights & Biases (wandb).

Usage:
    To use this Trainer class, you need to have a PyTorch model, a configuration dictionary that defines your training,
    dataset, and experiment options, and the necessary data loaders.

    1. Define your PyTorch model.
    2. Prepare your configuration dictionary with the following keys:
        - 'dataset_options': Information about how to load the data, including task type and vectorization options.
        - 'training_options': Training parameters such as the number of epochs, loss function, optimizer settings, and
          whether to use early stopping.
        - 'experiment_options': Experiment settings for logging and checkpoint directory, including Weights & Biases
          (wandb) integration options.
    3. Instantiate the Trainer class with your model and config dictionary.
    4. Call the `train` method to start training.

Dependencies:
    - PyTorch: The script assumes the use of PyTorch for defining and training the model.
    - Weights & Biases: For experiment tracking and logging, ensure you have an account and project set up in wandb.

Example:
    ```
    model = YourModel()
    config = {
        'dataset_options': {...},
        'training_options': {...},
        'experiment_options': {...},
    }

    trainer = Trainer(model, config)
    trainer.train()
    ```

Note:
    This script is designed to be generic and may require customization to fit the specific needs of your project, such
    as adjusting the data loading process, modifying the training loop, or changing the logging and checkpointing mechanism.

Author: Seong-Heon Lee (POSTECH MINDS)
E-mail: shlee0125@postech.ac.kr
Date: 2024-02-23
License: MIT    
"""

import os
from glob import glob

from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset, DataLoader, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from data.datasets import *
from utils.metrics import MetricLogger


class Trainer:
    """
    A class for managing the training process of a deep learning model, including training, validation,
    checkpointing, and logging.

    Attributes:
        model (torch.nn.Module): The deep learning model to be trained.
        config (dict): Configuration dictionary containing dataset options, training options, and experiment options.
    """
    def __init__(self, model, config):
        """
        Initializes the Trainer class with model, configuration, and setup for training.

        Args:
            model (torch.nn.Module): The model to train.
            config (dict): A configuration dictionary with keys including 'dataset_options', 'training_options',
                           and 'experiment_options'.
        """       
        # Setting up the device for GPU usage if available, else CPU
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self.device)

        # Determine if DataParallel should be used based on config and available GPUs
        self.use_data_parallel = config.get('use_data_parallel', False)  # Default to False if not specified
        if self.use_data_parallel:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"Number of GPUs: {num_gpus}. Using DataParallel.")
                self._model = nn.DataParallel(self._model, device_ids=config.get('data_parallel_device_ids', None))
            else:
                print(f"Number of GPU: {num_gpus}. Not using DataParallel because only 1 GPU is available.")
        else:
            print("DataParallel usage is disabled in config.")

        # Load data loaders based on the given dataset options
        self.dataset_options = config["dataset_options"]
        self.task = self.dataset_options["task"]
        self.metric_logger = MetricLogger(self.task, default_metrics=True)
        self.multimodal_learning = True if self.dataset_options.get("vectorization", None) else False
        self.train_loader, self.val_loader = create_data_loaders(self.dataset_options)

        # Extracting training options from the config
        self.training_options = config["training_options"]
        self.n_epochs = self.training_options["n_epochs"]
        self.criterion = getattr(nn, self.training_options["loss_function"])()
        self.optimizer = get_optimizer(self.model, self.training_options)
        self.scheduler = get_scheduler(self.optimizer, self.training_options)
        
        # Experiment setting
        self.experiment_options = config["experiment_options"]
        wandb.init(**self.experiment_options['wandb'], config=config)
        self.experiment_name = self.experiment_options["title"]
        ckpt_dir = self.experiment_options["ckpt_dir"]
        self.ckpt_dir = os.path.join(ckpt_dir, self.experiment_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
                   

    def train(self):
        """
        Conducts the training process over a specified number of epochs, including training and validation steps,
        checkpoint saving, early stopping, and logging metrics.
        """
        # Initialize
        best_val_loss = float('inf')
        no_improve_count = 0
        n_samples = len(self.train_loader)

        for epoch in range(self.n_epochs):
            # Start an epoch
            self.model.train()
            self.metric_logger.reset_metrics()
            running_loss = 0.0

            for batch in self.train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                if self.multimodal_learning:
                    (images, vectors), targets = batch
                    images, vectors, targets = images.to(self.device), vectors.to(self.device), targets.to(self.device)
                    inputs = {"image_data": images, "topology_data": vectors}
                    outputs = self.model(**inputs)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    inputs = images
                    outputs = self.model(inputs)

                # Compute metrics
                loss = self.criterion(outputs, targets)
                metrics = self.metric_logger.compute_metrics(outputs, targets)
                running_loss += loss.item()
                running_metrics = self.metric_logger.update_metrics(metrics)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            # Terminate an epoch
            
            # Scheduler update
            if self.scheduler:
                    self.scheduler.step()

            # Log after each epoch
            avg_loss = running_loss / n_samples
            avg_metrics = {key: value / n_samples for key, value in running_metrics.items()}
            metrics_string = ", ".join([f"{key}: {value:.4f}" for key, value in avg_metrics.items()])

            # Validation step
            val_loss, val_metrics, fusion_weights = self.valid()
            val_metrics_string =  ", ".join([f"{key}: {value:.4f}" for key, value in val_metrics.items()])
            
            print(f"Epoch {epoch + 1}/{self.n_epochs}")
            print('-' * 20)

            # Print logs
            print(f"Train Loss: {avg_loss:.4f}, " + metrics_string)
            print(f"Val Loss: {val_loss:.4f}, " + val_metrics_string)

            # logging via wandb
            wandb.log({
                "Train": {
                    "Loss": avg_loss, **avg_metrics
                },
                "Validation": {
                    "Loss": val_loss, **val_metrics
                },
                "Learning Rate": self.optimizer.param_groups[0]['lr'],
                "Epoch": epoch,
                "Modal Importance (CNN)": fusion_weights[0],
                "Modal Importance (TDA)": fusion_weights[1]
            })

            # Save checkpoint & Early Stopping
            self.save_checkpoint(f'last_ckpt.bin')
            if val_loss < best_val_loss:
                no_improve_count = 0
                self.best_val_loss = val_loss
                self.save_checkpoint(f'best_ckpt_{epoch+1:04}.bin')
                # Keep top 3 models
                for path in sorted(glob(os.path.join(self.ckpt_dir, 'best_ckpt_*.bin')))[:-3]:
                    os.remove(path)
            else:
                no_improve_count += 1

            if no_improve_count >= self.training_options["early_stopping"]:
                print("Early stopping!")
                self.save_checkpoint(f'early_stopping_ckpt_{epoch+1:04}.bin')
                break

    def valid(self):
        # Initialize
        self.model.eval()
        self.metric_logger.reset_metrics()
        running_loss = 0.0

        # Modal Fusion Unit
        modal_importance = getattr(self.model, 'get_modal_importance', None)
        if self.multimodal_learning:
            fusion_weights = modal_importance().detach().cpu().numpy()
        else:
            fusion_weights = np.array([0.0, 0.0])
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.multimodal_learning:
                    (images, vectors), targets = batch
                    images, vectors, targets = images.to(self.device), vectors.to(self.device), targets.to(self.device)
                    inputs = {"image_data": images, "topology_data": vectors}
                    outputs = self.model(**inputs)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    inputs = images
                    outputs = self.model(inputs)
                
                # Compute metrics
                loss = self.criterion(outputs, targets)
                metrics = self.metric_logger.compute_metrics(outputs, targets)
                running_loss += loss.item()
                running_metrics = self.metric_logger.update_metrics(metrics)

        n_samples = len(self.val_loader)
        avg_loss = running_loss / n_samples
        avg_metrics = {key: value / n_samples for key, value in running_metrics.items()}

        return avg_loss, avg_metrics, fusion_weights
    
    def save_checkpoint(self, name):
        path = os.path.join(self.ckpt_dir, name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @property
    def model(self):
        """
        Returns the actual model, handling the case where the model is wrapped in nn.DataParallel.

        Returns:
            torch.nn.Module: The deep learning model.
        """
        if isinstance(self._model, nn.DataParallel):
            return self._model.module
        else:
            return self._model


def get_transforms(transforms_configs):
    """Convert a list of transform configurations to a torchvision.transforms.Compose object."""
    transforms_mapping = {
        "Resize": transforms.Resize,
        "ToTensor": transforms.ToTensor,
        "Normalize": transforms.Normalize,
        # Add more transforms as needed
    }

    # Fetch the actual transform objects from the mapping using the given parameters
    transforms_objs = [transforms_mapping[tc["name"]](**tc["params"]) for tc in transforms_configs]
    return transforms.Compose(transforms_objs)


def get_optimizer(model, training_options):
    optimizer_class = getattr(optim, training_options['optimizer']['name'])
    optimizer = optimizer_class(model.parameters(), **training_options['optimizer']['params'])
    return optimizer


def get_scheduler(optimizer, training_options):
    if ('scheduler' in training_options) and (training_options['scheduler']):
        scheduler_class = getattr(lr_scheduler, training_options['scheduler']['name'])
        scheduler = scheduler_class(optimizer, **training_options['scheduler']['params'])
        return scheduler
    return None


def create_data_loaders(dataset_options):
    """Create train and validation dataloaders based on the configuration."""
    
    # 1. Determine the dataset class from the dataset name
    dataset_mapping = {
        "HAM10000": HAM10000,
        "PSF_Injection": PSF_Injection,
        "GWUniverse": GWUniverse
        # ... Add more datasets as needed
    }
    DatasetClass = dataset_mapping[dataset_options["dataset"]]
    
    # 2. Fetch the transforms
    transforms_configs = dataset_options["transforms"]
    transforms = get_transforms(transforms_configs)

    # 3. Initialize dataset with training flag set to True and False for train and validation respectively
    train_dataset = DatasetClass(
        root=dataset_options["root"],
        transforms=transforms,
        is_training=True,
        task=dataset_options["task"],
        topological_vector="vectorization" in dataset_options,
        vectors_dir=dataset_options.get("vectorization", None),
        #modal_test=dataset_options["modal_test"]
    )

    test_dataset = DatasetClass(
        root=dataset_options["root"],
        transforms=transforms,
        is_training=False,
        task=dataset_options["task"],
        topological_vector="vectorization" in dataset_options,
        vectors_dir=dataset_options.get("vectorization", None)
    )
    
    # 4. Split dataset and create dataloaders
       # 4. Split dataset and create dataloaders
    # 데이터셋 인덱스와 레이블 준비
    indices = list(range(len(train_dataset)))
    labels = train_dataset.labels

    # 첫 번째 분할: 훈련 및 검증/테스트 세트
    train_indices, valid_indices, _, _ = train_test_split(
    indices, labels, stratify=labels, test_size=float(dataset_options["val_size"])) 
    
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(train_dataset, valid_indices)

    train_loader = DataLoader(train_subset, batch_size=dataset_options["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_subset, batch_size=dataset_options["batch_size"], shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=dataset_options["batch_size"], shuffle=False, num_workers=4)

    return train_loader, val_loader
# Deep Learning Model Trainer Documentation
Author: Seong-Heon Lee (POSTECH MINDS)
E-mail: shlee0125@postech.ac.kr
Date: 2024-02-23
License: MIT

## Overview

This document provides an overview and usage guide for the Deep Learning Model Trainer script. The Trainer class encapsulates the entire process of training a deep learning model, including setup, training, validation, early stopping, checkpointing, and logging.

## Features

- **Device Setup**: Automatically configures the model to train on GPU if available, with an option for multi-GPU training using DataParallel.
- **Data Loading**: Supports loading and vectorization of data for training and validation.
- **Training Loop**: Executes training over a specified number of epochs, including forward and backward passes.
- **Validation Loop**: Evaluates the model on a validation set to track performance improvements.
- **Early Stopping**: Implements early stopping to prevent overfitting.
- **Checkpointing**: Saves checkpoints for the last and best-performing models and maintains only the top 3 models based on validation loss.
- **Logging**: Integrates with Weights & Biases (wandb) for experiment tracking and metric logging.

## Requirements

- PyTorch
- Weights & Biases account for logging

## Configuration

The Trainer class requires a configuration dictionary with the following structure:

```python
config = {
    'dataset_options': {
        'task': 'classification',  # Example task
        'vectorization': True,     # Example vectorization flag
        # Other dataset-specific options
    },
    'training_options': {
        'n_epochs': 10,            # Number of training epochs
        'loss_function': 'CrossEntropyLoss',  # Loss function
        'early_stopping': 5,       # Early stopping patience
        # Other training-specific options
    },
    'experiment_options': {
        'ckpt_dir': './checkpoints',  # Directory for saving checkpoints
        'title': 'MyExperiment',      # Experiment title
        'wandb': {
            'project': 'my_project',  # Weights & Biases project name
            # Other wandb-specific options
        }
    }
}
```

## Usage

1. **Define Your Model**: Create a PyTorch model that you wish to train.
2. **Prepare Your Config**: Fill in the configuration dictionary as shown above.
3. **Instantiate the Trainer**:
```python
trainer = Trainer(model, config)
```
4. **Start Training**:
```python
trainer.train()
```

## Customization

The Trainer class is designed to be generic but may require customization to fit specific project needs. Consider modifying the data loading process, adjusting the training loop, or changing the logging and checkpointing mechanisms as needed.

## Example

```python
model = YourModel()
config = {
    'dataset_options': {...},
    'training_options': {...},
    'experiment_options': {...},
}

trainer = Trainer(model, config)
trainer.train()
```

## Note

This script and documentation are provided as a guide. Ensure to adapt and test it according to the specific requirements of your deep learning project.
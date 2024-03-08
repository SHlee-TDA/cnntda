# DataLoader Creation Script Documentation
- Author: Seong-Heon Lee (POSTECH MINDS)
- E-mail: shlee0125@postech.ac.kr
- Date: 2024-03-08
- License: MIT

## Overview

`dataloader.py` is a script designed to facilitate the creation of `DataLoader` instances for training, validation, and inference. 
It enables dynamic dataset loading and the application of transformations based on user-defined configurations.

## Features

- **Dynamic Dataset Loading**: Automatically detects and maps dataset classes that are derived from the `BaseDataset` class.
- **Data Transformations**: Supports converting a list of transformation configurations into a `torchvision.transforms.Compose` object for preprocessing.
- **DataLoader Instances**: Creates DataLoader objects for training, validation, and inference, with support for stratification and random splitting for training/validation and a straightforward configuration for inference.

## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed and set up:

- `Python >= 3.6`
- `PyTorch` and `torchvision`
- `scikit-learn`

### Dataset Class Implementation

Your dataset classes should inherit from `BaseDataset` and be placed within a module that `dataloader.py` can access. 
For example, you might have a file `datasets.py` containing your custom dataset classes.
Please check that you prepared your `data` directory structure as follows:


```bash
📁 data/
    ├──📁 data_directory/
    ├──📄 __init__.py
    ├──📄 base.py
    ├──📄 dataloader.py
    └──📄: datasets.py
```


### Configuration

Prepare a JSON configuration file that specifies the dataset options, preprocessing transformations, and DataLoader settings.
Here is a sample structure for the configuration file:

```json
{"multimodal_learning": false,
    ...
  "dataset_options": {
    "dataset": "YourDatasetClassName",
    "root": "path/to/your/dataset",
    "preprocess": [
      {"name": "Resize", "params": {"size": 256}},
      {"name": "ToTensor", "params": {}},
      {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    ],
    "batch_size": 32,
    "num_workers": 4,
    "val_size": 0.2
  },
  ...
}
```
Please make sure to familiarize yourself with the guidance on the config (`docs/configs/config.md`) before using this script.


## Usage

To create DataLoader instances for training and validation:

```python
from dataloader import create_data_loaders

config = load_your_config()  # Implement this function to load the JSON config
train_loader, val_loader = create_data_loaders(config)
```

For inference DataLoader creation:
```python
from dataloader import create_inference_loader

config = load_your_config()  
inference_loader = create_inference_loader(config)
```

## Extending the Script

- To support new dataset classes, ensure they inherit from `BaseDataset` and are discoverable by the script.
- To add support for additional transformations, update the `get_transforms` function with new entries in the `transforms_mapping` dictionary.


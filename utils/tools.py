"""
This script provides a collection of utility functions designed to support the development
and training of deep learning models. It includes functionality for reading and writing JSON files,
seeding all random number generators for reproducibility, printing the current experimental environment
including system and hardware specifications, and computing the output dimension of a model given a
specific input shape. These tools are intended to facilitate the management of model configurations,
ensure consistent experimental setups, and aid in the debugging and optimization of models.


Functions:
- read_json: Load a JSON file into an ordered dictionary.
- write_json: Write data to a JSON file, maintaining the order of keys.
- seed_all: Seed all random number generators to ensure experiment reproducibility.
- print_env: Print detailed information about the current experimental environment, including hardware and software configurations.
- compute_output_dim: Calculate the output dimensions of a model given a specific input size, useful for designing network architectures.
"""
import datetime
import math
import platform
import random
import json
from   pathlib import Path
from   itertools import repeat
from   collections import OrderedDict

import torch
import pandas as pd
import numpy as np
import psutil
import pynvml


def read_json(fname):
    """
    Reads a JSON file and returns its content as an ordered dictionary.

    Parameters:
        fname (str or Path): The file name or path to the JSON file.

    Returns:
        OrderedDict: The content of the JSON file as an ordered dictionary.
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
    
def write_json(content, fname):
    """
    Writes the given content to a JSON file.

    Parameters:
        content (dict): The content to write to the JSON file.
        fname (str or Path): The file name or path where the JSON file will be saved.

    This function writes the content with an indentation of 4 spaces and does not sort the keys,
    ensuring that the output file is both human-readable and retains the original ordering of keys.
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def seed_all(seed):
    """
    Seeds all random number generators with the same seed value to ensure reproducibility.

    Parameters:
        seed (int): The seed value to use for all random number generators.

    This function seeds the random number generators for the Python `random` module, NumPy, and PyTorch
    (both CPU and CUDA devices), and sets PyTorch's cuDNN to operate deterministically.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Fix all random seeds as {seed}')

def print_env():
    """
    Prints information about the current experimental environment, including the date, Python and PyTorch
    versions, operating system, CPU, RAM, and GPU specifications. This function is useful for documenting
    the environment in which experiments are run, aiding reproducibility and debugging.
    """
    print('========== System Information ==========')
    current_date = datetime.date.today()
    print(f'DATE : {current_date}')

    python_version = platform.python_version()
    print(f'Pyton Version : {python_version}')

    try:
        import torch
        pytorch_version = torch.__version__

    except ImportError:
        pytorch_version = "PyTorch not installed"

    print(f'PyTorch Version : {pytorch_version}')
    os_info = platform.system() + " " + platform.release()
    print(f'OS : {os_info}')

    cpu_info = platform.processor()
    print(f'CPU spec : {cpu_info}')

    mem_info = psutil.virtual_memory().total
    print(f'RAM spec : {mem_info / (1024**3):.2f} GB')

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print("="*30)
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()

        print(f"Device {i}:")
        print(f"Name: {name}")
        print(f"Total Memory: {memory_info.total / 1024**2} MB")
        print(f"Driver Version: {driver_version}")
        print("="*30)

    pynvml.nvmlShutdown()


def compute_output_dim(model, input_shape=(3, 51, 51)):
    """
    Computes the output dimension of a model's convolutional layers given a specific input shape.

    This function is useful for understanding the output size of a model's layers, which can be
    particularly handy when designing the architecture of a neural network and planning subsequent
    layers. By providing a dummy input based on the specified input shape, it allows for the examination
    of the output dimensions without needing to run actual data through the model.

    Parameters:
        model (torch.nn.Module): The PyTorch model for which to compute the output dimension.
        input_shape (tuple): The shape of the input tensor, typically in the form (C, H, W) where
                             C is the number of channels, H is the height, and W is the width of the input.

    Returns:
        torch.Size: The size of the output as a torch.Size object, which includes the batch size as its
                    first dimension and the output dimensions as the subsequent dimensions.
    """
    # Create a dummy input with the given shape
    dummy_input = torch.randn(1, *input_shape)
    # Forward pass the dummy input through the model
    dummy_output = model(dummy_input)
    # Return the output shape
    return dummy_output.shape[-1]
"""Script for dynamically selecting models based on configurable parameters and optional checkpoint loading.

Author: Seong-Heon Lee (POSTECH MINDS)
Date: 2024-03-14
E-mail: shlee0125@postech.ac.kr
License: MIT 

This script provides the infrastructure for dynamically selecting, building, and loading models based on a configuration. 
It supports various types of models including baseline, custom, and pretrained models, allowing for flexible experimentation and development within a deep learning project.

Key Components:
- ModelSelector: A class that selects and builds models based on the provided configuration. It supports multimodal learning configurations and handles different model types such as baseline, custom, and pretrained models.
- load_model: A function that dynamically loads a model class from a specified module and initializes it with given parameters. It also supports loading model weights from a specified checkpoint file.

Usage:
The script is designed to be used within a deep learning project where model selection and initialization need to be dynamic and configurable. 
It allows for easy switching between different model architectures and training configurations, facilitating rapid experimentation and development.

Features:
- Supports dynamic import of custom model classes.
- Allows for flexible model initialization through configuration dictionaries.
- Facilitates the use of pretrained models by enabling checkpoint loading.

Note:
This script is part of a larger project and relies on the project's structure and conventions, such as the organization of model classes in a 'custom_models' module and the use of a configuration dictionary for specifying model options.
"""
import importlib
import json
import os
import warnings

import torch
import torchvision.models as models

from   .baseline import *
from   .cnntdanet import *
from   .custom_models import *
from   utils import compute_output_dim


class ModelSelector:
    """
    A class for selecting and building models based on a given configuration.

    This class supports selecting and constructing various types of models including
    baseline, custom, and pretrained models. The choice of model and its initialization
    parameters are determined by the provided configuration dictionary.

    Attributes:
        multimodal_learning (bool): Indicates if multimodal learning is enabled.
        model_options (dict): A dictionary containing options for model construction.
        model_type (str): The type of the model to be built ('baseline', 'custom', 'pretrained').
        ckpt_dir (str): Directory path for saving/loading model checkpoints. Relevant for 'pretrained' models.
        ckpt_fname (str): Checkpoint filename for loading pretrained models.
        init_params (dict): Parameters for initializing the model.
        input_shapes (dict): Input shapes for the model. Necessary for building the model architecture.
        target_shape (int): The shape of the model's output. Critical for defining the model's final layer.

    Raises:
        ValueError: If 'ckpt_fname' is not specified for pretrained models or if an unknown model type is provided.
        Warning: If essential model options like 'type', 'input_shapes', or 'target_shape' are not specified.
    """

    def __init__(self, config):
        """Initializes the ModelSelector with a given configuration."""
        self.multimodal_learning = config.get('multimodal_learning', False)
        self.model_options = config.get('model_options')

        self.model_type = self.model_options.get('type', None)
        if self.model_type is None:
            self.model_type = 'baseline'
            warnings.warn("You may not input 'type' in 'model_options'. Defaulting to 'baseline'.", UserWarning)
        elif self.model_type == 'pretrained':
            self.ckpt_dir = config['experiment_options'].get('ckpt_dir')
            self.ckpt_fname = self.model_options.get('ckpt_fname', None)
            if self.ckpt_fname is None:
                raise ValueError("No 'ckpt_fname' specified in 'model_options' for pretrained model. This is required.")

        self.init_params = self.model_options.get('init_params', None)
        if self.init_params is not None:
            self.input_shapes = self.init_params.get('input_shapes', None)
            self.target_shape = self.init_params.get('target_shape', None)
        else:
            self.input_shapes = self.target_shape = None

        if self.input_shapes is None:
            warnings.warn("No 'input_shapes' specified in 'init_params'. This is required for building the model.", UserWarning)
        
        if self.target_shape is None:
            warnings.warn("No 'target_shape' specified in 'init_params'. This is required for defining the model output.", UserWarning)

    def build(self):
        """
        Constructs and returns a model instance based on the model type specified in the configuration.

        This method selects the appropriate model-building strategy based on the `model_type` attribute.
        It supports constructing 'baseline', 'custom', and 'pretrained' models. For 'baseline' models,
        it directly invokes a method to construct the model using the provided input and target shapes.
        For 'custom' models, it dynamically loads a model class from a specified module and initializes
        it with given parameters. For 'pretrained' models, it additionally loads model weights from a
        specified checkpoint file.

        Returns:
            torch.nn.Module: An instance of the specified model, ready for training or inference.
        """
        if self.model_type == 'baseline':
            # Constructs a baseline model using predefined architecture components
            return self._build_baseline_model()
        elif self.model_type == 'custom':
            # Dynamically constructs a custom model based on the model definition in custom_models.py
            return self._build_custom_model()
        elif self.model_type == 'pretrained':
            # Constructs a model and loads pretrained weights from a checkpoint
            return self._build_pretrained_model()
        elif self.model_type in ['transfer_learning']:
            # Placeholder for future implementation of transfer learning models
            raise NotImplementedError(f"{self.model_type} model is not implemented yet. Stay tuned for future updates.")
        else:
            # Handles the case where the model type is not recognized
            raise ValueError(f"Unknown model build strategy: {self.model_type}")
            
    def _build_baseline_model(self):
        if self.multimodal_learning == True:
            # For multimodal learning, construct CNNTDAPlus model
            projection_dim = 256
            cnn = BaselineImgConv2d(self.input_shapes['cnn'])
            tda = BaselineTopoConv1d(self.input_shapes['tda'])
            mlp = BaselineMlpClassifier(projection_dim, self.target_shape)
            cnntda = CNNTDAPlus(
                self.input_shapes['cnn'],
                self.input_shapes['tda'],
                cnn,
                tda,
                mlp,
                projection_dim
            )
            return cnntda
        else:
            # For non-multimodal learning, construct a simpler CNN model
            model = BaselineCNN(self.input_shapes, self.target_shape)
            return model
        
    def _build_pretrained_model(self):
        return load_model(self.model_options, ckpt_path=os.path.join(self.ckpt_dir, self.ckpt_fname))

    def _build_custom_model(self):
        return load_model(self.model_options, ckpt_path=None)


def load_model(model_options, module_name="custom_models", ckpt_path=None):
    """
    Dynamically loads and returns a model instance based on the specified options.

    This function dynamically imports a module containing model definitions and attempts
    to initialize a model class from this module based on the provided model options. It
    supports loading custom model classes and initializing them with specified parameters.
    Additionally, if a checkpoint path is provided, it loads model weights from the checkpoint.

    Parameters:
        model_options (dict): A dictionary containing options for model selection and initialization.
                              This includes the model name and initialization parameters.
        module_name (str): The name of the module from which to import the model class.
        ckpt_path (str, optional): The path to a checkpoint file from which to load model weights.
                                   If None, no weights are loaded.

    Returns:
        torch.nn.Module: An instance of the specified model class, optionally loaded with weights from a checkpoint.
    """
    # Attempt to dynamically import the specified module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_name}. Please check if the module exists and is correctly named.") from e

    # Retrieve the model class from the imported module
    model_name = model_options.get('model_name')
    try:
        model_class = getattr(module, model_name)
    except AttributeError:
        raise AttributeError(f"Model class {model_name} not found in module {module_name}. Please check if the class name is correct.")

    # Unpack initialization parameters and attempt to instantiate the model
    init_params = model_options.get('init_params', {})
    try:
        model = model_class(**init_params)  # Unpacking init_params for model initialization
    except TypeError as e:
        raise TypeError(f"Error initializing model {model_name} with provided parameters. Error: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid parameter value encountered when initializing model {model_name}. Error: {e}")

    # If a checkpoint path is provided, attempt to load model weights from the checkpoint
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        try:
            checkpoint = torch.load(ckpt_path)
            if 'model_state_dict' not in checkpoint:
                raise KeyError("Checkpoint does not contain 'model_state_dict'")
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}") from e
    
    return model

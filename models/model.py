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
    def __init__(self, config):
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
        self.input_shapes = self.init_params.get('input_shapes', None)
        if self.input_shapes is None:
            warnings.warn("No 'input_shapes' specified in 'model_options'. This is required for building the model.", UserWarning)
        
        self.target_shape = self.init_params.get('target_shape', None)
        if self.target_shape is None:
            warnings.warn("No 'target_shape' specified in 'model_options'. This is required for defining the model output.", UserWarning)

    def build(self):
        if self.model_type == 'baseline':
            return self._build_baseline_model()
        elif self.model_type == 'custom':
            return self._build_custom_model()
        elif self.model_type == 'pretrained':
            return self._build_pretrained_model()
        elif self.model_type in ['transfer_learning']:
            raise NotImplementedError(f"{self.model_type} model is not implemented yet. Stay tuned for future updates.")
        else:
            raise ValueError(f"Unknown model build strategy: {self.model_type}")
            
    def _build_baseline_model(self):
        if self.multimodal_learning == True:
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
            model = BaselineCNN(self.input_shapes, self.target_shape)
            return model
        
    def _build_pretrained_model(self):
        return load_model(self.model_options, ckpt_path=os.path.join(self.ckpt_dir, self.ckpt_fname))

    def _build_custom_model(self):
        return load_model(self.model_options, ckpt_path=None)


def load_model(model_options, module_name="custom_models", ckpt_path=None):
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_name}. Please check if the module exists and is correctly named.") from e

    model_name = model_options.get('model_name')
    try:
        model_class = getattr(module, model_name)
    except AttributeError:
        raise AttributeError(f"Model class {model_name} not found in module {module_name}. Please check if the class name is correct.")

    
    init_params = model_options.get('init_params', {})
    
    model = model_class(**init_params)  # Unpacking init_params for model initialization
    
    if ckpt_path:
        # Load model checkpoint if specified
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    
    
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

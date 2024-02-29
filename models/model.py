import warnings
import importlib

import torch
import torchvision.models as models
import json

from   baseline import *
from   cnntdanet import *
from   custom_models import *
from   utils import compute_output_dim


class ModelSelector:
    def __init__(self, config):
        self.multimodal_learning = config.get('multimodal_learning', False)
        self.model_options = config.get('model_options')
        self.model_type = self.model_options.get('type', None)
        if self.model_type == None:
            self.model_type = 'baseline'
            warnings.warn("You may not input 'type' in 'model_options'. Defaulting to 'baseline'.", UserWarning)
            
        self.input_shapes = self.model_options.get('input_shapes', None)
        if self.input_shapes is None:
            warnings.warn("No 'input_shapes' specified in 'model_options'. This is required for building the model.", UserWarning)
        
        self.target_shape = self.model_options.get('target_shape', None)
        if self.target_shape is None:
            warnings.warn("No 'target_shape' specified in 'model_options'. This is required for defining the model output.", UserWarning)


    def build(self):
        if self.model_type == 'baseline':
            return self._build_baseline_model()
        elif self.model_type in ['transfer_learning', 'pretrained', 'custom']:
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


def load_model(model_options, module_name="custom_models"):
    # 모듈 동적 불러오기
    module = importlib.import_module(module_name)
    # 클래스 동적 인스턴스화
    model_name = model_options.get('model_name')

    model_class = getattr(module, model_name)
    model = model_class()  # 모델 인스턴스 생성
    return model
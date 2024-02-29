from baseline import *
from cnntdanet import *
from utils import compute_output_dim

import torch
import torchvision.models as models
import json


class ModelSelector:
    def __init__(self, config):
        self.multimodal_learning = config.get('multimodal_learning', False)
        self.model_options = config['model_options']
        self.model_type = self.model_options['type']
        self.input_shapes = self.model_options['input_shapes']
        self.target_shape = self.model_options['target_shape']

    def build(self):
        if self.model_type == 'baseline':
            return self._build_baseline_model()
        elif self.model_type in ['transfer_learning', 'pretrained', 'custom']:
            # 아직 구현되지 않은 모델 구성 전략에 대한 처리
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
            cnn = BaselineImgConv2d(self.input_shapes['cnn'])
            mlp = BaselineMlpClassifier(
                compute_output_dim(cnn, self.input_shapes['cnn']),
                self.target_shape
                )
            model = BaselineCNN(cnn, mlp)
            return model


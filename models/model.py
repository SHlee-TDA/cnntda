import torch
import torchvision.models as models
import json

class ModelSelector:
    def __init__(self, config):
        self.model_options = config['model_options']
        self.strategy = self.model_options['strategy']
        
        # self.models = {
        #     'resnet18': models.resnet18,
        #     'vgg16': models.vgg16,
        #     # 필요에 따라 다른 모델을 추가할 수 있습니다.
        # }

    def load_pretrained_model(self, model_name):
        """사전 학습된 모델 로드"""
        if model_name in self.models:
            return self.models[model_name](pretrained=True)
        else:
            raise ValueError(f"Unsupported pre-trained model: {model_name}")

    def build_custom_model(self, config_path):
        """사용자 정의 모델 구성"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        # 여기에서 사용자 정의 모델을 구성하는 로직을 구현합니다.
        # 예: config 파일에 기반하여 동적으로 모델 구성
        return None  # 사용자 정의 모델 반환

    def select_model(self, strategy, model_name=None, config_path=None):
        """모델 선택 로직"""
        if strategy == 'baseline':
            # 간단한 baseline 모델을 선택합니다.
            return self.models[model_name](pretrained=False)
        elif strategy == 'transfer_learning':
            # 사전 학습된 모델을 로드합니다.
            return self.load_pretrained_model(model_name)
        elif strategy == 'custom':
            # 사용자 정의 모델을 구성합니다.
            return self.build_custom_model(config_path)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
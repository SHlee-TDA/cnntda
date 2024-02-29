"""Main script for model training.
1. Train a model
2. Save the history of training
3. Save the model and check point"""



import argparse
import json
import torch

from trainer import Trainer
from models import ModelSelector
from utils.tools import seed_all, print_env

if __name__ == "__main__":
    seed_all(42)
    print_env()
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file')
    args = parser.parse_args()
    
    # 1. 설정을 읽어옵니다.
    with open(args.config, 'r') as file:
        config = json.load(file)
        
    # # 2. 모델을 초기화합니다.
    model = ModelSelector(config).build()

    # 3. Trainer를 초기화하고 학습을 시작합니다.

    trainer = Trainer(model, config)
    trainer.train()
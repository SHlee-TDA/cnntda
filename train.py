"""Main script for model training.
1. Train a model
2. Save the history of training
3. Save the model and check point"""



import argparse
import json
import torch

from trainer import Trainer
from models.image_models.otrain import Otrain
from models.topology_models.tdanet import TDANet
from models.classifiers.mlp import MLP
from models.cnntda_net import CNNTDANet
from models.base_cnn import CNN
from utils.tools import seed_all, print_env, compute_output_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file')
    args = parser.parse_args()
    
    # 1. 설정을 읽어옵니다.
    with open(args.config, 'r') as file:
        config = json.load(file)
        
    # # 2. 모델을 초기화합니다.
    # image_model = Otrain(in_channels=3)
    # image_output_dim = compute_output_dim(image_model, (3, 51, 51))
    # topology_model = TDANet(in_channels=1)
    # topology_output_dim = compute_output_dim(topology_model, (1, 100))
    # classifier = MLP(input_dim= image_output_dim + topology_output_dim, num_classes=7)

    # model = CNNTDANet(
    #     image_model=image_model,
    #     topology_model=topology_model,
    #     classifier=classifier
    # )
    model = CNN()


    # 3. Trainer를 초기화하고 학습을 시작합니다.
    seed_all(42)
    print_env()
    trainer = Trainer(model, config)
    trainer.train()
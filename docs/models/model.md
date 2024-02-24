# Model


## Configuration
Example
```json
{
    "dataset_options": {
        "dataset": "HAM10000",
        "root": "data/HAM10000",
        "task": "classification",
        "transforms": [
            {
                "name": "Resize",
                "params": {
                    "size": [51, 51]
                }
            },
            {
                "name": "ToTensor",
                "params": {}
            }
        ],
        "val_size": 0.1,
        "batch_size": 32,
        "vectorization": "betti_curves"
    },
    "model_options": {
        "strategy": "baseline",
        

    },
    "training_options": {
        "n_epochs": 1000,
        "loss_function": "CrossEntropyLoss",
        "optimizer": {
            "name": "Adam",
            "params": {
                "lr": 0.0001,
                "weight_decay": 0.01
            }
        },
        "scheduler": null,
        "early_stopping": 20
    },
    "experiment_options": {
        "title": "CNNTDANet Classification",
        "ckpt_dir": "experiments/test",
        "wandb": {
            "project": "CNNTDANet Research",
            "name": "231023_classification"
        }
    }
}
```

 - `dataset_options`
    - `dataset` : Select training dataset 
    - `root` : 
    - ``

- `model_options`
   - `strategy` : `baseline`, `pretrained`, `transfer`, `custom`
        - 모델을 구성하는 전략을 선택합니다. 
            - `baseline` : 사전 정의된 baseline model을 불러옵니다.
            - `pretrained` : 추가학습을 원할 경우 사전 학습된 네트워크 파라미터를 불러와 학습합니다.
            - `transfer` : 사전학습된 ResNet 등을 전이학습을 통해 사용할 수 있습니다.
            - `custom` : config의 설정에 따라 모델을 동적으로 구축하고 학습합니다. 아키텍쳐 리서치를 위해 사용할 수 있습니다.
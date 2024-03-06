"""Main script to train the CNNTDA model (or a baseline model).

Author: Seong-Heon Lee (POSTECH MINDS)
Date: 2024-03-06
E-mail: shlee0125@postech.ac.kr
License: MIT 

This script is intended to be run from the command line interface (CLI).
Usage example:
`python3 train.py --config configs/config.json`
The `--config` argument specifies the path to the training configuration file.
Users should write their desired training settings in `json` format and provide it as an argument, which will guide the training process accordingly.
Please make sure to familiarize yourself with the guidance on the config (`docs/configs/config.md`) before using this script.

The script operates according to the following procedures:
1. Fix random seeds: To ensure experiment reproducibility, all random seeds used in this program are fixed.
2. Print execution environment: Information about the execution environment, including CPU, GPU, and memory details, is printed.
3. Parsing configuration: The configuration file specified in the CLI is read and the training settings are passed on to the subsequent functions.
4. Model build: Based on the provided configuration, a model is constructed and initialized.
5. Trainer setting: The model training process is designed and initialized according to the received configuration.
6. Training: The training is executed. Throughout the training process, various metrics about the model can be observed in real-time through `wandb`*.

* About using `wandb`:
This script utilizes `wandb` (Weights & Biases) for monitoring the training process and logging results.
`wandb` is a platform for visualizing and managing machine learning experiments.
Upon first execution of the script, a `wandb` API key will be requested.
Please prepare your `wandb` account and API key in advance by signing up at https://wandb.ai.
You can find the API key in your `wandb` site's user settings, which will be needed to authenticate with `wandb` when you run the script.
"""
import argparse
import json
import sys

from   models import ModelSelector
from   trainer import Trainer
from   utils.tools import print_env, seed_all

if __name__ == "__main__":
    # Fix all random seeds for reproducibility.
    seed_all(42)
    
    # Print information about the execution environment.
    print_env()
    
    # Parse arguments from the CLI.
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file')
    args = parser.parse_args()
    
    # Read and parse the configuration file.
    try:
        with open(args.config, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: The configuration file {args.config} was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The configuration file {args.config} is not in a valid JSON format.")
        sys.exit(1)
        
    # Initialize the model based on the configuration.
    model = ModelSelector(config).build()

    # Initialize the Trainer and start the training process.
    trainer = Trainer(model, config)
    trainer.train()
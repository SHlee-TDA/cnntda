"""Main script for Inference step


"""
import argparse
import json
import os
import sys

import torch

from   models import ModelSelector
from   trainer import Trainer
from   utils.tools import print_env

if __name__ == "__main__":

    # Print information about the execution environment.
    print_env()
    
    # Parse arguments from the CLI.
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint filename for loading model parameters')
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
    ckpt_dir = config['experiment_options'].get('ckpt_dir')
    ckpt = os.path.join(ckpt_dir, args.ckpt)
    model.load_state_dict(torch.load(ckpt)['model_state_dict'])
    # Initialize the Trainer and start the inference process.
    trainer = Trainer(model, config)
    trainer.inference_mode = True
    loss, metrics, importances = trainer.valid()

    # Display metrics
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
   
    # Update config file
    config['test_metrics'] = metrics
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Test result updated and saved to {args.config}")

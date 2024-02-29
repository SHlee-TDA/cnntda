import datetime
import math
import platform
import random
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import json
import torch
import pandas as pd
import numpy as np
import psutil
import pynvml


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
    
def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Fix all random seeds as {seed}')

def print_env():
    """
    현재 사용중인 실험환경에 대한 기본정보를 프린트합니다.
    다음의 정보를 프린트합니다.
    - 실험날짜
    - Python version
    - Pytorch version
    - OS
    - CPU spec
    - RAM spec
    - GPU spec
    """
    print('========== System Information ==========')
    # 오늘의 날짜
    current_date = datetime.date.today()
    print(f'DATE : {current_date}')

    # Python 버전
    python_version = platform.python_version()
    print(f'Pyton Version : {python_version}')

    # Pytorch 버전
    # 이 환경에 PyTorch가 설치되어 있는지 확인합니다.
    try:
        import torch
        pytorch_version = torch.__version__

    except ImportError:
        pytorch_version = "PyTorch not installed"

    print(f'PyTorch Version : {pytorch_version}')
    # 현재 작업환경의 os
    os_info = platform.system() + " " + platform.release()
    print(f'OS : {os_info}')

    # 현재 작업환경의 CPU 스펙
    cpu_info = platform.processor()
    print(f'CPU spec : {cpu_info}')

    # 현재 작업환경의 Memory 스펙
    mem_info = psutil.virtual_memory().total
    print(f'RAM spec : {mem_info / (1024**3):.2f} GB')

    # 현재 작업환경의 GPU 스펙
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    print("="*30)
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()

        print(f"Device {i}:")
        print(f"Name: {name}")
        print(f"Total Memory: {memory_info.total / 1024**2} MB")
        print(f"Driver Version: {driver_version}")
        print("="*30)

    pynvml.nvmlShutdown()


def compute_output_dim(model, input_shape=(3, 51, 51)):
    """Compute the number of output neurons of convolutional layers
    """
    # Create a dummy input with the given shape
    dummy_input = torch.randn(1, *input_shape)
    # Forward pass the dummy input through the model
    dummy_output = model(dummy_input)
    # Return the output shape
    return dummy_output.shape[-1]
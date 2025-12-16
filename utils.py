import numpy as np
import random
import torch
import os


def generate_wavetable(length, f):
    wavetable = np.zeros((length,), dtype=np.float32)
    for i in range(length):
        wavetable[i] = f(2 * np.pi * i / length)
    return wavetable


def sawtooth_waveform(x):
    """Sawtooth with period 2 pi."""
    return (x + np.pi) / np.pi % 2 - 1


def square_waveform(x):
    """Square waveform with period 2 pi."""
    return np.sign(np.sin(x))

def set_seed(seed: int = 42) -> None:
    """
    Sets the global seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    
    # For deterministic behavior on CuDNN backend (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set as {seed}")
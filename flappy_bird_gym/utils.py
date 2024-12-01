# flappy_bird_gym/utils.py

import numpy as np
import random
import torch

def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
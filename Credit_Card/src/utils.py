import logging
import random
import numpy as np
import torch

def setup_logging() -> logging.Logger:
    """Utility function to set up the logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger

def set_seed(seed):
    """
    Utility function to set the seed for reproducibility.
    
    Args:
        - seed (int): The seed to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


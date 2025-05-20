import torch
import os
from datasets import load_from_disk
from models import get_tokenizer
import hydra


def get_data(cfg):
    """
    Loads the dataset from the given path.
    """
    if cfg.data.type == "hf":
        raise NotImplementedError("HF datasets are not supported yet.")
    elif cfg.data.type == "disk":
        dataset = load_from_disk(cfg.data.path)
    else:
        raise ValueError(f"Unknown data type: {cfg.data.type}")
    
    return dataset







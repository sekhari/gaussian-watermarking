import torch
import os

import transformers
import tqdm

from omegaconf import OmegaConf




def get_model(cfg):
    """
    Download pretrained model from Hugging Face model hub.
    """

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.name)
    return model


def get_tokenizer(cfg):
        """
        Downlaods tokenizer from Huggingface
        """
        tokenizer_name = cfg.tokenizer.name or cfg.model.name
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer



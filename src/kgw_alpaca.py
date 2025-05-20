import datasets

import os
import json

import time
import torch
import hydra
from omegaconf import OmegaConf
from utils import save_amulet_metadata, hf_login


import transformers
from kgw_generate import _tokenize_prompts, get_kgw_config, get_generation_config
from generate_text import get_model_type





def _parse_alpaca_batch(instructions, outputs, prompt_batch, tokenizer, model, generation_config, watermarking_config, device):
    """
    Given a batch of prompts, generates using HF and then parses the outputs into a list of dictionaries appended to the outputs list.
    """
    tokenized_prompts = _tokenize_prompts(prompt_batch, tokenizer, device)
    model_outputs = model.generate(**tokenized_prompts, generation_config=generation_config, watermarking_config=watermarking_config)
    model_output_strings = tokenizer.batch_decode(model_outputs)
    for i in range(len(prompt_batch)):
        instructions.append(prompt_batch[i])
        outputs.append(model_output_strings[i])

    
        



def get_alpaca_eval_responses(model, tokenizer, eval_set, cfg):
    
    
    watermarking_config = get_kgw_config(cfg)
    generation_config = get_generation_config(cfg)
    device = cfg.device
    batch_size = cfg.other_gen.batch_size
    prompts = eval_set['instruction']

    instructions, outputs = [], []
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i + batch_size]
        _parse_alpaca_batch(instructions, outputs, prompt_batch, tokenizer, model, generation_config, watermarking_config, device)



    model_names = [cfg.model.name] * len(prompts)
    datumsets = ['alpaca_eval'] * len(prompts)

    generations_dict = {
        'instruction': instructions,
        'output': outputs,
        'generator': model_names,
        'dataset': datumsets
    }
    generations = datasets.Dataset.from_dict(generations_dict)
    return generations




@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):


    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    if cfg.amlt:
        save_amulet_metadata(cfg)
        cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')
        output_dir = cfg.master_parent
    else:
        model_type = get_model_type(cfg.model.name)
        output_dir = os.path.join(cfg.master_parent, 'data/alpaca_kgw', model_type, f"{cfg.other_gen.bias}")
        os.makedirs(output_dir, exist_ok=True)


    hf_login(cfg)


    data_loading_start = time.time()
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    if cfg.other_gen.alpaca_max_samples is not None:
        eval_set = eval_set.take(cfg.other_gen.alpaca_max_samples)
    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")



    model_loading_start = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model.name).to(cfg.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")




    eval_start = time.time()
    alpaca_generations = get_alpaca_eval_responses(model, tokenizer, eval_set, cfg)
    eval_end = time.time()
    print(f"\nEvaluation took {eval_end - eval_start:.0f} seconds\n")




    out_path = os.path.join(output_dir, 'watermarked')
    print(f"Saving watermarked responses to {out_path}")
    alpaca_generations.save_to_disk(out_path)


    print("Example response:")
    index = 0
    print(f"({index}): {alpaca_generations[index]['output']}")











if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"Master time: {master_end - master_start:.0f} seconds")




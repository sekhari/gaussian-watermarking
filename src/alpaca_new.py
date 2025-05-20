import datasets

import os
import json

import time
import torch
import hydra
from omegaconf import OmegaConf
from utils import save_amulet_metadata, hf_login, override_watermark_params, get_watermark_path


import transformers
from generate_text import get_model_type, get_watermark_param_names, get_sampling_params
from hf_core import VanillaLMWatermarker, LowRankLMWatermarker, LaserizedLowRankLMWatermarker
from vllm import LLM, SamplingParams


def _parse_alpaca_output(outputs, model_name):
    instructions, responses = [], []

    for output in outputs:
        instructions.append(output.prompt)
        responses.append(output.outputs[0].text)

    model_names = [model_name] * len(instructions)
    datumsets = ['alpaca_eval'] * len(instructions)
    return instructions, responses, model_names, datumsets

def get_alpaca_eval_responses(model, eval_set, cfg, max_tries=5):
    sampling_params = get_sampling_params(cfg)
    prompts = eval_set['instruction']
    tries = 0
    completed =  False
    while not completed and tries < max_tries:
        try:
            outputs = model.generate(prompts, sampling_params)
            completed = True
        except Exception as e:
            print(f"Error generating responses, trying again.  Error: {e}")
            tries += 1
            sleep_time = 5
            print(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)


    instructions, responses, model_names, datumsets = _parse_alpaca_output(outputs, cfg.model.name)

    generations_dict = {
        'instruction': instructions,
        'output': responses,
        'generator': model_names,
        'dataset': datumsets
    }
    generations = datasets.Dataset.from_dict(generations_dict)
    return generations



def get_watermark_path(watermark_overrides, seed, parent='../models'):
    """
    Returns the path to the watermark model.  Assumes rank is none, in which case the VanillaWatermarker is assumed.  If rank is not none, the path to a LowRankWatermarker is returned
    """
    watermark_overrides = watermark_overrides.replace('/', '-')
    return os.path.join(parent, f'{watermark_overrides}_seed_{seed}')


@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):
    print(cfg.model.watermark_overrides)
    watermark_overrides = cfg.model.watermark_overrides
    if len(watermark_overrides.split('@')) == 1:
        do_watermark = False
        print("\nNo watermarking\n")
    else:
        do_watermark = True
        print("\nWatermarking\n")

        override_watermark_params(cfg)
        

    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)

    model_type = get_model_type(cfg.model.name)
    if cfg.amlt:
        save_amulet_metadata(cfg)
        cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')
        output_dir = cfg.master_parent
    
    else:
        fname_model_type = model_type.replace('/', '-')
        fname_overrides = cfg.model.watermark_overrides.replace('/', '-') 
        if do_watermark:
            output_dir = os.path.join(cfg.master_parent, 'data/alpaca_gaussmark', fname_model_type, fname_overrides)
        else:
            output_dir = os.path.join(cfg.master_parent, 'data/alpaca_gaussmark', fname_model_type, f"Base_{cfg.seed}")

        os.makedirs(output_dir, exist_ok=True)





    # if os.path.exists(os.path.join(output_dir, 'responses.json')):
    #     print(f"Responses already exist at {output_dir}, skipping")
    #     return


    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)

    hf_login(cfg)

    model_loading_start = time.time()
    if cfg.model.watermark_variance == 0 or not do_watermark:
        model = LLM(cfg.model.name, tokenizer=cfg.tokenizer.name, gpu_memory_utilization=cfg.sampling.gpu_memory_utilization, max_num_seqs=cfg.sampling.max_num_seqs)
    else:
        # model_path = get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop)
        model_path = get_watermark_path(cfg.model.watermark_overrides, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'))
        if os.path.exists(model_path) and cfg.model.load_prewatermarked:
            print(f"\nLoading model from {model_path}\n")
            model = VanillaLMWatermarker.load_pretrained(model_path, keep_base_model=False, keep_watermarked_model=True, gpu_memory_utilization=cfg.sampling.gpu_memory_utilization, max_num_seqs=cfg.sampling.max_num_seqs) # Only need watermarked model for 

        else:
            print(f"\nWatermarking model {cfg.model.name} with seed {cfg.model.seed}\n")


            

            if model_type is not None: # get watermark param names from config
                watermark_param_names = get_watermark_param_names(cfg)
            else:
                watermark_param_names = OmegaConf.to_object(cfg.model.watermark_param_names)
            print(f"\nWatermarking parameters: {watermark_param_names}\n")

            if cfg.model.rank_to_drop is None or cfg.model.rank_to_drop == 0:
                model = VanillaLMWatermarker(
                    cfg.model.name,
                    watermark_param_names=watermark_param_names,
                    path = model_path,
                    loss = cfg.model.watermark_loss,
                    variance=cfg.model.watermark_variance,
                    seed = cfg.seed,
                    tokenizer_name=cfg.tokenizer.name,
                    keep_base_model=False,
                    keep_watermarked_model=True,
                    gpu_memory_utilization=cfg.sampling.gpu_memory_utilization
                )
            elif cfg.model.laserize:
                model = LaserizedLowRankLMWatermarker(
                    cfg.model.name,
                    watermark_param_names=watermark_param_names,
                    path = model_path,
                    loss = cfg.model.watermark_loss,
                    variance=cfg.model.watermark_variance,
                    seed = cfg.seed,
                    tokenizer_name=cfg.tokenizer.name,
                    keep_base_model=False,
                    keep_watermarked_model=True,
                    rank_to_drop=cfg.model.rank_to_drop,
                    gpu_memory_utilization=cfg.sampling.gpu_memory_utilization
                )

            else:
                    model = LowRankLMWatermarker(
                    cfg.model.name,
                    watermark_param_names=watermark_param_names,
                    path = model_path,
                    loss = cfg.model.watermark_loss,
                    variance=cfg.model.watermark_variance,
                    seed = cfg.seed,
                    tokenizer_name=cfg.tokenizer.name,
                    keep_base_model=False,
                    keep_watermarked_model=True,
                    rank_to_drop=cfg.model.rank_to_drop,
                    gpu_memory_utilization=cfg.sampling.gpu_memory_utilization
                )

    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")


    data_loading_start = time.time()
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    if cfg.data.max_samples is not None:
        try:
            eval_set = eval_set.take(cfg.data.max_samples)
        except Exception as e:
            print(f"Could not take {cfg.data.max_samples} samples from eval set, using full eval set instead")
            print(e)
            

    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")

    eval_start = time.time()
    alpaca_generations = get_alpaca_eval_responses(model, eval_set, cfg)
    eval_end = time.time()
    print(f"\nEvaluation took {eval_end - eval_start:.0f} seconds\n")




    out_path = os.path.join(output_dir, 'watermarked')
    print(f"Saving watermarked responses to {out_path}")
    alpaca_generations.save_to_disk(out_path)


    # print("Example response:")
    # index = 0
    # print(f"({index}): {alpaca_generations[index]['output']}")











if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"\n\nMaster time: {master_end - master_start:.0f} seconds\n\n")


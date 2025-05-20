import torch
from vllm import LLM, SamplingParams
import datasets
import numpy as np

import os
import json

import time

import hydra
from omegaconf import OmegaConf

from utils import save_amulet_metadata, get_watermark_path, hf_login, save_config, override_watermark_params
from hf_core import VanillaLMWatermarker, LowRankLMWatermarker, LaserizedLowRankLMWatermarker

from generate_text import get_model_type, get_watermark_param_names, get_data




def get_cumulative_logprobs(response):
    prompt_token_ids = response.prompt_token_ids
    cum_logprobs = 0.0
    for i, prompt_logprob in enumerate(response.prompt_logprobs):
        if i == 0:
            continue
        log_prob = prompt_logprob[prompt_token_ids[i]].logprob
        cum_logprobs += log_prob
    return cum_logprobs



def get_perplexities(llm, prompts):
    """
    Given a vllm llm, returns the cumulative log probabilities under the reference model of the answers
    """

    log_probs_config = SamplingParams(
        n=1,
        max_tokens=1,
        prompt_logprobs=0,
    )

    responses = llm.generate(prompts, log_probs_config)
    outputs = []
    for response in responses:
        if response is None:
            continue
        if response.prompt_logprobs is None:
            continue
        prompt = response.prompt
        prompt_len = len(response.prompt_token_ids)
        prompt_cumlogprob = get_cumulative_logprobs(response)
        per_token_logprob = prompt_cumlogprob / prompt_len
        perplexity = float(np.exp(-per_token_logprob))
        outputs.append({
            'prompt': prompt,
            'perplexity': perplexity,
            'length': prompt_len,
            'logprob': prompt_cumlogprob
        })
    
    return outputs


def get_perplexity_statistics(outputs):

    logprobs, lengths, perplexities = [], [], []
    for output in outputs:
        logprobs.append(output['logprob'])
        lengths.append(output['length'])
        perplexities.append(output['perplexity'])
    
    return {
        'avg_logprob': np.mean(logprobs),
        'std_logprob': np.std(logprobs),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'avg_perplexity': np.mean(perplexities),
        'std_perplexity': np.std(perplexities)
    }





@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):
    override_watermark_params(cfg)
    
    master_start = time.time()

    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    if cfg.amlt:
        save_amulet_metadata(cfg)
        cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')

    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)

    hf_login(cfg)




    model_loading_start = time.time()
    if cfg.model.watermark_variance == 0:
        model = LLM(cfg.model.name, tokenizer=cfg.tokenizer.name, gpu_memory_utilization=cfg.sampling.gpu_memory_utilization, max_num_seqs=cfg.sampling.max_num_seqs)
    else:
        model_path = get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop)
    
        print(f"\nWatermarking model {cfg.model.name} with seed {cfg.model.seed}\n")


        model_type = get_model_type(cfg.model.name)

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
                gpu_memory_utilization=cfg.sampling.gpu_memory_utilization,
                max_num_seqs=cfg.sampling.max_num_seqs
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
                gpu_memory_utilization=cfg.sampling.gpu_memory_utilization,
                max_num_seqs=cfg.sampling.max_num_seqs
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
                gpu_memory_utilization=cfg.sampling.gpu_memory_utilization,
                max_num_seqs=cfg.sampling.max_num_seqs
            )

    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")


    data_loading_start = time.time()
    prompts = get_data(cfg)
    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")


    perplexity_start = time.time()
    perplexities = get_perplexities(model, prompts)
    perplexity_end = time.time()
    print(f"\nPerplexity calculation took {perplexity_end - perplexity_start:.0f} seconds\n")



    statistics = get_perplexity_statistics(perplexities)
    print(f"Average Perplexity: {statistics['avg_perplexity']:.4f} +/- {statistics['std_perplexity']:.4f}")
    print(f"Average Log Probability: {statistics['avg_logprob']:.4f} +/- {statistics['std_logprob']:.4f}")
    print(f"Average Length: {statistics['avg_length']:.4f} +/- {statistics['std_length']:.4f}\n")


    output_path = os.path.join(cfg.master_parent, 'perplexities.json')
    print(f"\nSaving perplexities to {output_path}\n")
    with open(output_path, 'w') as f:
        json.dump(perplexities, f)
    



    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")


if __name__ == "__main__":
    main()
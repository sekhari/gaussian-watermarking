import torch
from vllm import LLM, SamplingParams
import transformers
import datasets

import os
import json
import time

import hydra
from omegaconf import OmegaConf

from utils import hf_login, save_amulet_metadata, override_watermark_params, get_watermark_path
from generate_text import get_model_type, get_watermark_param_names

from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment


from hf_core import VanillaLMWatermarker, LowRankLMWatermarker, LaserizedLowRankLMWatermarker
import gc





# def get_model_type_capital(cfg):

#     model_name = cfg.model.name

#     if 'mistral' in model_name:
#         model_type = 'Mistral'
#     elif 'Phi' in model_name or 'phi' in model_name:
#         model_type = 'Phi-3'
#     elif 'Llama-3' in model_name:# Llama-3
#         model_type = 'Llama-3'
#     else: # Llama-2
#         model_type = 'Llama-2'
    
#     return model_type



def remove_llm(llm):

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("LLM removed")


def get_sampling_params(cfg):
    return SamplingParams(
        n = 1,
        best_of=1,
        top_p = cfg.sampling.top_p,
        seed = cfg.sampling.seed + 111,
        max_tokens=cfg.sampling.max_tokens,
        temperature=cfg.sampling.temperature,
        logprobs=0, # Return logprobs
        top_k=cfg.sampling.top_k,
    )



def get_alpaca_eval_responses(cfg, model_path, eval_set, base_model=False):
    """
    Given a model path and eval_set, returns the generations and prompts
    """
    ## Getting config
    overrides = cfg.model.watermark_overrides
    model_type = get_model_type(cfg.model.name)
    generator = model_type
    if base_model:
        generator += '_Base'
        # path = os.path.join(model_path, 'base-model')
        path = model_path
    else:
        generator += f'_{overrides}'
        path = os.path.join(model_path, 'watermarked-model')

    print(f"Loading model from {path}")
    ## Loading model and generating responses)
    llm = LLM(path, tokenizer=cfg.tokenizer.name, gpu_memory_utilization=cfg.lm_eval.gpu_memory_utilization)
    samplng_params = get_sampling_params(cfg.lm_eval)
    print(f"Sampling params: {samplng_params}")
    responses = llm.generate(eval_set['instruction'], sampling_params=samplng_params)
    
    remove_llm(llm)

    ## TODO construct new datasets.Dataset and save that instead
    instruction, output, model_names, datum = [], [], [], []
    for i in range(len(responses)):
        response = responses[i]
        instruction.append(response.prompt)
        output.append(responses[i].outputs[0].text)
        model_names.append(generator)
        datum.append(eval_set[i]['dataset'])
    

    generations_dict = {
        'instruction': instruction,
        'output': output,
        'generator': model_names,
        'dataset': datum
    }

    generations = datasets.Dataset.from_dict(generations_dict)

    # for i, example in enumerate(eval_set):

    #     example['output'] = responses[i].outputs[0].text
    #     example['generator'] = generator
    
    return generations






@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):

    if cfg.model.watermark_overrides is None:
        cfg.lm_eval.unwatermarked = True
    elif cfg.model.watermark_overrides.endswith('@___@'):
        cfg.lm_eval.unwatermarked = True
        cfg.model.name = cfg.model.watermark_overrides[:-5]
    else:
        override_watermark_params(cfg)
    
    master_start = time.time()

    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    if cfg.amlt:
        cfg.lm_eval.model_path = None ## TODO: Load from storage instead?
        save_amulet_metadata(cfg)
        cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')
        output_dir = cfg.master_parent
    else:
        model_type = get_model_type(cfg.model.name)
        output_dir = os.path.join(cfg.master_parent, 'data/alpaca', model_type)
        os.makedirs(output_dir, exist_ok=True)

    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)

    hf_login(cfg)

    if cfg.lm_eval.unwatermarked:
        print(f"Unwatermarked model: {cfg.model.name}")
    # if not cfg.lm_eval.unwatermarked:
    else:
        print(f"Watermark overrides: {cfg.model.watermark_overrides}")
        if cfg.lm_eval.model_path is None:
            model_path = os.path.join(get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop))
            print(f"Model path: {model_path}")
        else: ## For local tests
            model_path = os.path.join(cfg.master_parent, cfg.lm_eval.model_path)

        if not os.path.exists(model_path): # Watermarks model if it doesn't exist and we are not intentionally evaluating un-watermarked model
            print(f"Model not found at {model_path}.  Watermarking model...")

            model_type = get_model_type(cfg.model.name)
            # if model_type is not None: # get watermark param names from config
            watermark_param_names = get_watermark_param_names(cfg)
            # else:
            #     watermark_param_names = OmegaConf.to_object(cfg.model.watermark_param_names)
            # print(f"\nWatermarking parameters: {watermark_param_names}\n")
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
                    keep_watermarked_model=True
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
                    rank_to_drop=cfg.model.rank_to_drop
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
                    rank_to_drop=cfg.model.rank_to_drop
                )
            
            del model ## To avoid memory leak
            gc.collect()
            torch.cuda.empty_cache()

        else:
            print(f"Model found at {model_path}")



    data_loading_start = time.time()
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    if cfg.lm_eval.max_samples is not None:
        eval_set = eval_set.take(cfg.lm_eval.max_samples)
    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")

    eval_start = time.time()
    if cfg.lm_eval.unwatermarked:
        model_path = cfg.model.name
        base_model_responses = get_alpaca_eval_responses(cfg, model_path, eval_set, base_model=True)
        out_path = os.path.join(output_dir, 'base_model')
        print(f"Saving base model responses to {out_path}")
        base_model_responses.save_to_disk(out_path)
        responses = base_model_responses
    else:   
        watermarked_responses = get_alpaca_eval_responses(cfg, model_path, eval_set, base_model=False)
        out_path = os.path.join(output_dir, 'watermarked')
        print(f"Saving watermarked responses to {out_path}")
        watermarked_responses.save_to_disk(out_path)
        responses = watermarked_responses

    eval_end = time.time()
    print(f"\nEvaluation took {eval_end - eval_start:.0f} seconds\n")
    

    print("Example response:")
    index = 0
    print(f"({index}): {responses[index]['output']}")


    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")    





if __name__ == "__main__":
    main()








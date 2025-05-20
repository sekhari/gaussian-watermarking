import torch
import json
import os
import time


import hydra 
from omegaconf import OmegaConf
from utils import save_amulet_metadata, get_watermark_path, copy_from_blob, hf_login, save_config, override_watermark_params
import shutil


from tqdm import tqdm
from hf_core import VanillaLMWatermarker, LowRankLMWatermarker, LaserizedLowRankLMWatermarker






def load_text(cfg):
    """
    Gets watermarked text from data saved to disk
    """
    responses_path = os.path.join(cfg.master_parent, 'generations.json')
    if not cfg.path_to_generations is None:
        copy_from_blob(cfg, cfg.path_to_generations)
        response_path = os.path.join(cfg.master_parent, cfg.path_to_generations)

    else:
         assert os.path.exists(responses_path), f"Generations not found at {responses_path}"


    with open(responses_path, 'r') as f:
        generations = json.load(f)
    return generations



def add_pvalues(generations, watermarker, verbose=True, block_size=None, num_samples=None):
    """
    Given list of generations, adds p-values to each generation according to watermarker
    """
    if type(block_size) == int: # If block_size is an int, convert to None if 0 or less for non-robust detection
        if block_size < 1:
            block_size = None
            num_samples = None


    generations_with_pvalues = []
    if verbose:
        bar = tqdm(enumerate(generations))
    else:
        bar = enumerate(generations)

    if block_size is not None:
        get_scores_and_pvalues = lambda prompt,output: watermarker.get_robust_score_and_pvalue(prompt, output, block_size=block_size, num_samples=num_samples)
    else:
        get_scores_and_pvalues = lambda prompt,output: watermarker.get_score_and_pvalue_cumnormal(prompt, output)

    for i, generation in bar:
        prompt, output = generation['prompt_token_ids'], generation['response_token_ids']
        # score, pvalue, var_under_null, log_prob = watermarker.get_score_and_pvalue_cumnormal(prompt, output)
        score, pvalue, var_under_null, log_prob = get_scores_and_pvalues(prompt, output)

        generation['pvalue'] = pvalue
        generation['score'] = score
        generation['var_under_null'] = var_under_null
        generation['base_model_logprobs'] = log_prob
        if generation['pvalue'] < 1.0:
            generation['passed'] = True
        else:
            generation['passed'] = False
            # print(f"Failed p-value test for generation {i}. Generation: {generation['response']}")



        generations_with_pvalues.append(generation)
    return generations_with_pvalues






@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):
    override_watermark_params(cfg)
    if cfg.model.watermark_variance is None or cfg.model.watermark_variance == 0:
        print("No watermark variance specified, skipping watermark detection")
        return

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
    model_path = get_watermark_path(cfg.model.name, cfg.model.seed, parent=os.path.join(cfg.master_parent, 'models'), rank=cfg.model.rank_to_drop)
    
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    print(f"Loading model from {model_path}")
    if cfg.model.rank_to_drop is None or cfg.model.rank_to_drop == 0:
        model = VanillaLMWatermarker.load_pretrained(model_path, keep_base_model=True, keep_watermarked_model=False) # Only need base model for watermark detection
    elif cfg.model.laserize:
        model = LaserizedLowRankLMWatermarker.load_pretrained(model_path, keep_base_model=True, keep_watermarked_model=False)
    else:
        model = LowRankLMWatermarker.load_pretrained(model_path, keep_base_model=True, keep_watermarked_model=False)
    
    model.to(cfg.device)
    model_loading_end = time.time()
    print(f"\nModel loading took {model_loading_end - model_loading_start:.0f} seconds\n")




    data_loading_start = time.time()
    generations = load_text(cfg)
    data_loading_end = time.time()
    print(f"\nData loading took {data_loading_end - data_loading_start:.0f} seconds\n")


    watermarking_start = time.time()
    generations_with_pvalues = add_pvalues(generations, model, verbose=cfg.watermarking.verbose)
    watermarking_end = time.time()
    print(f"\nWatermark detection took {watermarking_end - watermarking_start:.0f} seconds\n")

    temp_path = os.path.join(cfg.master_parent, 'generations_with_pvalues.json')
    with open(temp_path, 'w') as f:
        json.dump(generations_with_pvalues, f)
    


    pvalues = []
    log_probs = []
    base_model_logprobs = []
    for generation in generations_with_pvalues:
        pvalues.append(generation['pvalue'])
        log_probs.append(generation['logprobs'])
        base_model_logprobs.append(generation['base_model_logprobs'])
    
    median_pval = torch.tensor(pvalues).median().item()
    median_log_prob = torch.tensor(log_probs).median().item()
    median_base_model_log_prob = torch.tensor(base_model_logprobs).median().item()

    print(f"Median p-value: {median_pval:.3f}")
    print(f"Median log-prob: {median_log_prob:.3f}")
    print(f"Median base model log-prob: {median_base_model_log_prob:.3f}")

    ## Remove the old generations file and replace with the new one
    os.remove(os.path.join(cfg.master_parent, 'generations.json'))
    shutil.move(temp_path, os.path.join(cfg.master_parent, 'generations.json'))

    # save_config(cfg)

    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")    



if __name__ == "__main__":
    main()
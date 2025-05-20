import os
from hydra.utils import to_absolute_path as abspath
from omegaconf import OmegaConf
import hydra
import torch
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers


from huggingface_hub import login

def copy_from_blob(cfg, path):
    if os.path.exists(abspath(path)):
        return False

    copy_cmd = f'mkdir -p {abspath(path)} && cp -r {abspath(os.path.join(cfg.fs.blob_root, "uploads", path))} {os.path.dirname(abspath(path))}'
    print('Copying from blob:', copy_cmd)
    os.system(copy_cmd)
    return True


def save_config(cfg):
    """
    Saves config to json file.  If `config.json` already exists, then updates the config as needed.
    """


    config = OmegaConf.to_object(cfg)
    config_path = os.path.join(cfg.master_parent, 'config.json')
    if os.path.exists(config_path):
        print("\nUpdating existing config\n")
        with open(config_path, 'r') as f:
            old_config = json.load(f)
        if old_config is not None:
            config = old_config.update(config)
            


    print(f"\nSaving config to {config_path}\n")
    with open(config_path, 'w') as f:
        json.dump(config, f)




def save_amulet_metadata(cfg):
    if os.environ.get('SUBMISSION_SOURCE') != 'AMLT':
        return


    cfg.meta = OmegaConf.create({})
    cfg.meta.original_dir = hydra.utils.get_original_cwd()
    cfg.meta.run_dir = os.getcwd()
    cfg.meta.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    cfg.meta.amulet = OmegaConf.create({})
    cfg.meta.amulet.project_name = os.environ.get('AMLT_PROJECT_NAME')
    cfg.meta.amulet.experiment_name = os.environ.get('AMLT_EXPERIMENT_NAME')
    cfg.meta.amulet.job_name = os.environ.get('AMLT_JOB_NAME')
    cfg.meta.amulet.output_dir = os.environ.get('AMLT_OUTPUT_DIR')
    
    ## Make sure everything saved does not get deleted after job finishes
    cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')



def get_watermark_path(model_name, seed, parent='../models', rank=None):
    """
    Returns the path to the watermark model.  Assumes rank is none, in which case the VanillaWatermarker is assumed.  If rank is not none, the path to a LowRankWatermarker is returned
    """
    model_name = model_name.replace('/', '-')
    if rank is not None:
        return os.path.join(parent, f'{model_name}_seed_{seed}_rank_{rank}')
    else:
        return os.path.join(parent, f'{model_name}_seed_{seed}')






def get_token_level_logits(model, dataset, prompt_length=1):
    """
    Returns token level logits of a dataset according to a given model
    """
    return torch.nn.functional.log_softmax(model(dataset).logits, 2).gather(2, dataset.roll(-1).unsqueeze(2)).squeeze(2)[:,prompt_length:-1]



def get_sequence_level_logits(model, dataset, prompt_length=1):
    """
    Returns sequence level logits of a dataset according to a given model
    """
    return get_token_level_logits(model, dataset, prompt_length).sum(1)



def hf_login(cfg):
    """
    Loads huggingface token from file and logs in
    """

    with open(cfg.hf_token_path, 'r') as f:
        hf_token = f.read()
    
    login(token=hf_token, add_to_git_credential=False)
    del hf_token
    print('Logged in to Huggingface!')




def get_tokenizer(cfg):
    """
    Given a config, returns the appropriate tokenizer.
    """
    return transformers.AutoTokenizer.from_pretrained(cfg.model.name)











############################################################################################################
# Functions for analyzing generations
############################################################################################################

def _get_params_llama_vanilla(dirname):
    """
    Given a directory name, returns the model parameters and variance as saved in the directory name
    """
    span = re.search('lay_-?\d+', dirname).span(0)

    var_span = re.search('var_\de-?\d+', dirname).span(0)
    variance = float(dirname[var_span[0]+4:var_span[1]])

    param_info = dirname[span[0]+4:var_span[0]-1]
    if len(param_info) == 0:
        return '-1_mlp_weight', variance


    return param_info, variance



def _get_params_llama(dirname):
    """
    Given a directory name, returns the model parameters and variance as saved in the directory name
    """
    span = re.search('lay_-?\d+', dirname).span(0)

    var_span = re.search('var_\de-?\d+', dirname).span(0)
    variance = float(dirname[var_span[0]+4:var_span[1]])

    param_info = dirname[span[0]+4:var_span[0]-1]
    if len(param_info) == 0:
        param_info = '-1_mlp_weight'
        if '_ran_' in dirname:
            span = re.search('ran_\d+', dirname).span(0)
            param_info += '_r' + dirname[span[0]:span[1]]
        else:
            param_info += '_r0'
        return param_info, variance

    if '_ran_' in dirname:
        span = re.search('ran_\d+', dirname).span(0)
        rank = int(dirname[span[0]+4:span[1]])
        param_info += '_r' + dirname[span[0]+4:span[1]]
    else:
        param_info += '_r0'
        rank = 0

    if not '_ran_' in param_info: ## Annoying hack to fix the fact that I didn't save the rank in the directory name

        addition = ['ran', str(rank)]
        param_info = '_'.join(param_info.split('_')[:-1] + addition + [param_info.split('_')[-1]])


    return param_info, variance



def get_generations(parent, _get_params=_get_params_llama_vanilla):
    """
    Given a path to the parent directory of saved amulet generations, returns a dictionary of generations, where the key is the model parameters and the value is the generations list
    """
    generations = {}

    for dirname in os.listdir(parent):
        key = (*_get_params(dirname),)
        gen_path = os.path.join(parent, dirname, 'generations.json')
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                generations[key] = json.load(f)
        else:
            print(f"No generations found for {key}")

    return generations





def get_pvalues(gens):
    """
    Returns the p-value for each generation
    """
    pvals = []
    for gen in gens:
        pvals.append(gen['pvalue'])
    return pvals


def get_scores(gens):
    """
    Returns the test statistic for each generation
    """
    scores = []
    for gen in gens:
        scores.append(gen['score'])
    return scores

def get_logprobs(gens):
    """
    Returns the logprobs and base model logprobs for each generation
    """
    logprobs, base_model_logprobs = [], []
    for gen in gens:
        logprobs.append(gen['logprobs'])
        base_model_logprobs.append(gen['base_model_logprobs'])
    return logprobs, base_model_logprobs

def get_variances(gens):
    """
    Gets the variance under the null for each generation
    """
    variances = []
    for gen in gens:
        variances.append(gen['var_under_null'])
    return variances

def get_lengths(gens):
    """
    Returns the length of each generation
    """
    lengths = []
    for gen in gens:
        length = len(gen['response_token_ids'])
        lengths.append(length)
    return lengths



def get_all(gens, min_pval=1e-10):
    """
    Returns a pandas dataframe with data from the generations for plotting and further analysis
    """
    pvals = get_pvalues(gens)
    scores = get_scores(gens)
    logprobs, base_model_logprobs = get_logprobs(gens)
    variances = get_variances(gens)
    lengths = get_lengths(gens)

    data = pd.DataFrame({'pvalue': pvals, 'score': scores, 'logprob': logprobs, 'base_model_logprob': base_model_logprobs, 'variance': variances, 'length': lengths})
    data['pvalue_clipped'] = np.clip(data['pvalue'], min_pval, 1)
    data['log_pvalue'] = -np.log(data['pvalue_clipped']) #.replace(-np.inf, -np.log(min_pval))
    data.drop(columns=['pvalue_clipped'], inplace=True)
    data['token_logprob'] = data['logprob'] / data['length']
    data['base_model_token_logprob'] = data['base_model_logprob'] / data['length']
    data['logprob_diff_per_token'] = data['token_logprob'] - data['base_model_token_logprob']
    data['normalized_score'] = data['score'] / np.sqrt(data['variance'])


    return data



def get_empirical_cdf(pvals, grid_size=500):
    """
    Returns an empirical ROC curve for p-values
    """
    grid = np.linspace(0, 1, grid_size)
    cdf = np.zeros(grid_size)
    for i, val in enumerate(grid):
        cdf[i] = np.mean(pvals < val)
    return grid, cdf





def _process_evalharness_results(results_dict):
    """
    Parses necessary information from the evalharness results
    """
    output = {}
    results = results_dict['results']
    for key, val in results.items():
        result = {}
        result['task'] = key
        if key == 'gsm8k':
            
            result['exact_match,flexible-extract'] = val['exact_match,flexible-extract']
            result['exact_match,flexible-extract_stderr'] = val['exact_match_stderr,flexible-extract']
        elif key == 'gsm8k_cot_self_consistency':
            result['exact_match,flexible-extract'] = val['exact_match,maj@64']
            result['exact_match,flexible-extract_stderr'] = val['exact_match_stderr,maj@64']
            # result['exact_match,flexible-extract'] = val['exact_match,maj@8']
            # result['exact_match,flexible-extract_stderr'] = val['exact_match_stderr,maj@8']

            # result['exact_match,flexible-extract'] = val['exact_match,score-first']
            # result['exact_match,flexible-extract_stderr'] = val['exact_match_stderr,score-first']

        else:
            for key in val.keys():
                if key == 'alias':
                    continue
                elif ',none' in key:
                    new_key = key.replace(',none', '')
                    result[new_key] = val[key]
        
        
        output[result['task']] = result
    return output



def get_metrics(parent, _get_params=_get_params_llama_vanilla):
    """
    Returns eval harness metrics for all experiments in a directory
    """
    metrics = {}
    for dirname in os.listdir(parent):
        key = (*_get_params(dirname),)
        if parent == 'llama-watermark': # Kludge for earlier file handling
            metrics_path = os.path.join(parent, dirname, 'results.json')
        else:
            metrics_path = os.path.join(parent, dirname, 'data/eval_results/results.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                results = json.load(f)
            metrics[key] = _process_evalharness_results(results)
        else:
            print(f"No metrics found for {key}")

        
    return metrics


def plot_superglue_task(results, task, param, include_gsm8k=False):
    """
    Plots the superglue task results
    """
    SUPERGLUE_TASKS = [
        'boolq',
        'cb',
        'copa',
        'multirc',
        'record',
        'rte',
        'wic',
        'wsc',
    ]
    if include_gsm8k:
        SUPERGLUE_TASKS += ['gsm8k', 'gsm8k_cot_self_consistency'] # GSM8K is not a superglue task but included for math reasoning

    assert task in SUPERGLUE_TASKS, f"Task {task} not in superglue tasks"

    if task == 'record':
        metric = 'em'
    elif task in {'gsm8k', 'gsm8k_cot_self_consistency'}:
        metric = 'exact_match,flexible-extract'
    else:
        metric = 'acc'

    vars = []
    scores = []
    errors = []

    llama_score = 0.0
    llama_stderr = 0.0

    ## Sort by key
    results = {key: results[key] for key in sorted(results.keys())}

    for key, val in results.items():
        if key[0] == 0 and key[1] == 0:
            label = 'No Watermark'
        else:
            label = f'Var {key[1]}, r {key[0]}'
        vars.append(label)

        score = val[task][metric]
        error = val[task][metric + '_stderr']

        if key[0] == 0 and key[1] == 0:
            llama_score = score
            llama_stderr = error


        scores.append(score)
        errors.append(error)


    fig, ax = plt.subplots()

    ax.bar(vars, scores, yerr=errors, capsize=5, width=0.5, alpha=0.5)
    if metric == 'acc':
        ylabel = 'Accuracy'
    elif metric == 'em':
        ylabel = 'Exact Match'
    else:
        ylabel = metric


    ax.axhline(llama_score, color='red', label='Llama', linestyle='--', linewidth=1)
    ax.axhline(llama_score + llama_stderr, color='red', linestyle='--', linewidth=1, alpha=0.1)
    ax.axhline(llama_score - llama_stderr, color='red', linestyle='--', linewidth=1, alpha=0.1)
    # ax.fill_between([*ax.get_xlim()], llama_score - llama_stderr, llama_score + llama_stderr, color='red', alpha=0.1)
        
    ax.set_ylabel(ylabel)
    ax.set_title(f'SuperGLUE {task} ({param})')
    fig.autofmt_xdate()
    return fig, ax


def get_superglue_results(results_dict):
    """
    Given a dictionary of results with key being the task and value being the resulting value, returns the average superglue score
    """
    superglue_scores = []
    std_errors = []
    keys = []
    has_gsm8k = False
    for key, val in results_dict.items():
        keys.append(key)
        if key == 'record':
            superglue_scores.append(val['em'])
            std_errors.append(val['em_stderr'])
        elif key in {'gsm8k', 'gsm8k_cot_self_consistency'}:
            has_gsm8k = True
            # superglue_scores.append(val['exact_match,flexible-extract'])
            # std_errors.append(val['exact_match,flexible-extract_stderr'])
            gsm8k_score = val['exact_match,flexible-extract']
            gsm8k_stderr = val['exact_match,flexible-extract_stderr']
        elif key == 'gsm8k_cot_llama':
            continue
        else:
            superglue_scores.append(val['acc'])
            std_errors.append(val['acc_stderr'])
    if has_gsm8k:
        return keys, superglue_scores, std_errors, (gsm8k_score, gsm8k_stderr)
    else:
        return keys, superglue_scores, std_errors, (None, None)


def print_random_examples(generations, param, var=1e-5, num_examples=5, idxs=None):
    gen = generations[(param, var)]
    prompts, responses = [], []
    for example in gen:
        prompts.append(example['prompt'])
        responses.append(example['response'])

    if idxs is not None:
        assert num_examples <= len(idxs), "Number of examples must be less than or equal to the number of indices"
    else:
        idxs = np.random.choice(len(gen), num_examples, replace=False)

    for i in range(num_examples):
        idx = idxs[i]
        print(f"({idx})\t" + prompts[idx] + responses[idx] + '\n')



def get_auc(grid, cdf):
    """
    Given a grid and cdf, returns the area under the curve
    """
    return np.trapz(cdf, grid)




def is_good_model(gens, metric_dict, unwatermarked_metrics, auc_threshold=0.9, tolerance=0.0, min_prob=1e-20, grid_size=500, include_gsm8k=True):
    """
    Returns whether the model is good for watermarking.  If metric_criterion is 'all', then all metrics must pass the threshold.
    """
    ## Checking if metrics are good
    scores = {}
    std_errors = {}

    benchmark_scores = {}
    benchmark_std_errors = {}
    for key, val in metric_dict.items():

        if key == 'record':
            scores[key] = val['em']
            std_errors[key] = val['em_stderr']
            benchmark_scores[key] = unwatermarked_metrics[key]['em']
            benchmark_std_errors[key] = unwatermarked_metrics[key]['em_stderr']

        elif key in {'gsm8k', 'gsm8k_cot_self_consistency'}:
            if include_gsm8k:
                scores[key] = val['exact_match,flexible-extract']
                std_errors[key] = val['exact_match,flexible-extract_stderr']
                benchmark_scores[key] = unwatermarked_metrics[key]['exact_match,flexible-extract']
                benchmark_std_errors[key] = unwatermarked_metrics[key]['exact_match,flexible-extract_stderr']
            else:
                continue
        else:
            scores[key] = val['acc']
            std_errors[key] = val['acc_stderr']

            benchmark_scores[key] = unwatermarked_metrics[key]['acc']
            benchmark_std_errors[key] = unwatermarked_metrics[key]['acc_stderr']
            
  
    has_good_metrics = True
    for key in scores.keys():
        
        upper_confidence_interval = scores[key] + std_errors[key]

        threshold = benchmark_scores[key] - benchmark_std_errors[key] - tolerance
        if upper_confidence_interval < threshold:
            has_good_metrics = False
            break

    
    ## Checking if pvalues are good
    data = get_all(gens, min_prob) 
    grid, cdf = get_empirical_cdf(data['pvalue'], grid_size=grid_size)
    auc = get_auc(grid, cdf)
    if auc < auc_threshold:
        return False, auc, has_good_metrics
    else:
        return True, auc, has_good_metrics


def get_good_models(parents, auc_threshold=0.9, tolerance=0.0, min_prob=1e-20, grid_size=500, _get_params=_get_params_llama, include_gsm8k=True):
    """
    Returns a dictionary of models that pass the criteria
    """
    good_models = {}
    for parent in parents:
        generations = get_generations(parent, _get_params=_get_params)
        metrics = get_metrics(parent, _get_params=_get_params)
        

        unwatermarked_parent = './amlt/09-04-eval-models'
        if 'mistral' in parent:
            model = 'Mistral'
            unwatermarked_path = os.path.join(unwatermarked_parent, 'grid_09-04-eval-models_mod_mistralai_Mistral-7B-v0.3')
        elif 'Phi' in parent or 'phi' in parent:
            model = 'Phi-3'
            # unwatermarked_path = os.path.join(unwatermarked_parent, 'grid_09-04-eval-models_mod_microsoft_Phi-3-mini-4k-instruct')
            unwatermarked_path = './amlt/09-13-eval-phi.yaml/grid_09-13-eval-phi.yaml_mod_microsoft_Phi-3-mini-4k-instruct'
        # elif 'Llama-3' in parent:
        elif '09-04-watermark-llama' in parent: # Llama-3
            model = 'Llama-3'
            unwatermarked_path = os.path.join(unwatermarked_parent, 'grid_09-04-eval-models_mod_meta-llama_Meta-Llama-3.1-8B')
        else: # Llama-2
            model = 'Llama-2'
            unwatermarked_path = os.path.join(unwatermarked_parent, 'grid_09-04-eval-models_mod_meta-llama_Llama-2-7b-hf')

        with open(os.path.join(unwatermarked_path, 'data/eval_results/results.json')) as f:
            unwatermarked_metrics = json.load(f)

        unwatermarked_metrics = _process_evalharness_results(unwatermarked_metrics)


        basename = os.path.basename(parent)
        for key, value in generations.items():
            
            id = (model, *key)

            try:
                good_pvalues, auc, has_good_metrics = is_good_model(value, metrics[key], unwatermarked_metrics, auc_threshold=auc_threshold, tolerance=tolerance, min_prob=min_prob, grid_size=grid_size, include_gsm8k=include_gsm8k)
                if good_pvalues and has_good_metrics:
                    good_models[id] = auc
            except:
                print(f"{id} failed.")
    
    return good_models




def get_watermark_overrides(key, cfg):
    """
    Quickly generates overrides for watermarking.  Useful if we want to study some small set of pre-selected models
    """
    key_list = key.split('@___@')
    model_name, rank, watermark_param, var = key_list
    

    cfg.model.name = model_name
    cfg.model.rank_to_drop = int(rank)
    cfg.model.watermark_param_names = [watermark_param]
    cfg.model.watermark_variance = float(var)




    
def override_watermark_params(cfg):
    if cfg.model.watermark_overrides is not None:
        print("Overriding watermark parameters")
        get_watermark_overrides(cfg.model.watermark_overrides, cfg)

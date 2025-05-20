import matplotlib.pyplot as plt
import os
import numpy as np
import json
import re
import matplotlib
from src.utils import get_auc, get_empirical_cdf, get_all, _process_evalharness_results, get_superglue_results
import pandas as pd
import seaborn as sns




plt.rcParams.update({
    # Fonts (increased further for visibility)
    'font.size': 28,                      # Default font size
    'font.family': 'serif',               # Set font family to 'serif'
    'font.serif': ['Times New Roman', 'DejaVu Math TeX Gyre', 'DejaVu Serif'],  # Serif fonts
    'axes.titlesize': 30,                 # Title font size (even larger)
    'axes.labelsize': 30,                 # Axis label font size
    'legend.fontsize': 26,                # Legend font size
    'xtick.labelsize': 24,                # X-tick label font size
    'ytick.labelsize': 24,                # Y-tick label font size
    
    # Figure size and DPI
    'figure.figsize': [10, 8],            # Same figure size, but adjust as needed
    'figure.dpi': 100,                    # Higher resolution for clarity
    
    # Line properties
    'lines.linewidth': 4,                 # Default line thickness
    'lines.markersize': 15,               # Marker size
    
    # Axes
    'axes.grid': True,                    # Grid on
    'grid.color': 'gray',                 # Grid color
    'grid.linestyle': '-',                # Solid grid line style
    'grid.linewidth': 0.75,               # Grid line width
    'axes.edgecolor': 'black',            # Axes border color
    'axes.linewidth': 1.5,                # Thicker axes border
    'axes.titleweight': 'bold',           # Bold titles
    
    # Set default x-axis to start at 0
    'axes.autolimit_mode': 'round_numbers',  # Round axis limits
    'axes.xmargin': 0.0,                    # Remove extra margin on x-axis
    'axes.ymargin': 0.05,                 # Slight y-axis margin
    
    # Ticks
    'xtick.major.size': 7,                # Major tick size
    'xtick.major.width': 1.5,             # Major tick width
    'ytick.major.size': 7,                # Major tick size
    'ytick.major.width': 1.5,             # Major tick width
    
    # Legend
    'legend.frameon': True,               # Frame around legend
    'legend.framealpha': 0.9,             # Slightly transparent frame
    'legend.fancybox': True,              # Rounded corners
    
    # Save figure
    'savefig.dpi': 300,                   # High resolution for saving
    'savefig.format': 'png',              # Save format
    'savefig.bbox': 'tight',              # Tight layout for saving
})
default_cycler = plt.rcParams['axes.prop_cycle']
default_cycler = [c['color'] for c in default_cycler]




TASKS = {
        'boolq':'acc',
        'cb':'acc',
        'copa':'acc',
        'multirc':'acc',
        'record': 'f1',
        'rte':'acc',
        'wic':'acc',
        'wsc':'acc',
        'gsm8k_cot_self_consistency': 'exact_match,flexible-extract'
    }




def recursively_update_dict(old_dict, updates):
    for key in updates.keys():
        if key in old_dict.keys():
            if isinstance(old_dict[key], dict):
                recursively_update_dict(old_dict[key], updates[key])
            else:
                old_dict[key] = updates[key]
        else:
            old_dict[key] = updates[key]
            
def _get_rank(param_info):
    """
    Given a key, returns the rank of the model that was dropped
    """
    return int(param_info.split('_r')[-1])
    

def _get_layer(param_info):
    """
    Given a key, returns the layer of the model that was dropped
    """
    return int(param_info.split('_')[0])


def _get_mlp_type(param_info):

    layer = _get_layer(param_info)
    if layer == -1:
        return 'mlp_weight'
    else:
        mlp_type = '_'.join(param_info.split('_')[2:4])
        return mlp_type
def _get_seed(dirname):
    found = re.search('_see_(\d+)', dirname)
    if found:
        return int(found.group(1))
    else:
        return None

def check_match(key, param_data):
    """
    Given a key, check if it matches the desired data to display
    """
    param_info = key[0]
    matches = True
    if 'rank' in param_data.keys():
        if _get_rank(param_info) != param_data['rank']:
            matches = False
    layer = _get_layer(param_info)
    if 'layer' in param_data.keys():
        if param_data['layer'] != layer:
            matches = False
    if 'mlp_type' in param_data.keys() and layer != -1:
        mlp_type = '_'.join(param_info.split('_')[2:4])
        if param_data['mlp_type'] != mlp_type:
            matches = False


    return matches
    
def get_title(param_data):
    """
    Given the parameters, return the title for the plot
    """
    title = ''
    if 'layer' in param_data.keys():
        title += str(param_data['layer'])
    if 'mlp_type' in param_data.keys():
        title += '_' + param_data['mlp_type']
    if 'rank' in param_data.keys():
        title += '_r' + str(param_data['rank'])
    
    if 'gate_up_proj' in title:
        title = title.replace('gate_up_proj', 'mlp_gate_up_proj')
    
    return title



################## Getting data #########################


def get_params(dir):
    model_path = os.path.join(dir, 'models')
    param_path = os.path.join(model_path, os.listdir(model_path)[-1], 'params.json')
    with open(param_path) as f:
        config = json.load(f)


    model_name = config['tokenizer_name']
    param_names = config['watermark_param_names'][0]
    variance = config['variance']
    if 'rank_to_drop' in config.keys():
        rank_to_drop = config['rank_to_drop']
    else:
        rank_to_drop = 0

    if '_ran_' in dir:

        found = re.search(r'_ran_(\d+)', dir)
        if found:
            rank_to_drop = int(found.group(1))


    key = (param_names, variance, rank_to_drop)
    
    return key, model_name

def get_generations(parent):
    """
    Given a path to the parent directory of saved amulet generations, returns a dictionary of generations, where the key is the model parameters and the value is the generations list
    """
    generations = {}

    for dirname in os.listdir(parent):
        dir = os.path.join(parent, dirname)
        try:
            key, model_name = get_params(dir)
            if model_name not in generations.keys():
                generations[model_name] = {}
            with open(os.path.join(dir, 'generations.json')) as f:
                generations[model_name][key] = json.load(f)

        except Exception as e:
            print(e)
            print(f"Error processing {dir}")
        # gen_path = os.path.join(parent, dirname, 'generations.json')
        # if os.path.exists(gen_path):
        #     with open(gen_path) as f:
        #         generations[key] = json.load(f)
        # else:
        #     print(f"No generations found for {key}")

    return generations

def get_unwatermarked_metrics(model):
    """
    Get the unwatermarked metrics for a model
    """
    unwatermarked_parent = './amlt/09-18-eval-models/'
    if 'mistral' in model:
        base_model = 'Mistral'
        unwatermarked_path = os.path.join(unwatermarked_parent, 'grid_09-18-eval-models_mod_mistralai_Mistral-7B-v0.3')

    elif 'Phi' in model or 'phi' in model:
        base_model = 'Phi-3'
        unwatermarked_path = os.path.join(unwatermarked_parent, 'grid_09-18-eval-models_mod_microsoft_Phi-3-mini-4k-instruct')
    # elif 'Llama-3' in parent:
    elif 'llama' in model and '3' in model: # Llama-3
        base_model = 'Llama-3'
        # unwatermarked_path = 'amlt/09-19-eval-llama/grid_09-19-eval-llama_mod_meta-llama_Meta-Llama-3.1-8B'
        unwatermarked_path = 'amlt/10-09-eval-llama/grid_10-09-eval-llama_mod_meta-llama_Meta-Llama-3.1-8B'
    else:
        raise ValueError(f"Model {model} not recognized")
    
    with open(os.path.join(unwatermarked_path, 'data/eval_results/results.json')) as f:
        unwatermarked_metrics = json.load(f)

    return _process_evalharness_results(unwatermarked_metrics), base_model


def get_metrics(parent,  key_is_path=False, include_unwatermarked=False):
    """
    Returns eval harness metrics for all experiments in a directory
    """
    metrics = {}
    for dirname in os.listdir(parent):
        try:
            dir = os.path.join(parent, dirname)
            key, model_name = get_params(dir)
            metrics_path = os.path.join(parent, dirname, 'data/eval_results/results.json')
            with open(metrics_path) as f:
                results = json.load(f)
            
            if model_name not in metrics.keys():
                metrics[model_name] = {}
            
            metrics[model_name][key] = _process_evalharness_results(results)
            if key_is_path:
                metrics[model_name][key]['path'] = dir
        except Exception as e:

            print(f"Error processing {dir}: {e}")
    if include_unwatermarked:
        for model in metrics.keys():
            unwatermarked_metrics, _ = get_unwatermarked_metrics(model)
            metrics[model]['unwatermarked'] = unwatermarked_metrics

    return metrics



################## OLD PLOTTING FUNCTIONS #########################

def get_layer_mlp_type(param_name):
    splits = param_name.split('.')
    return splits[2], splits[4]

def is_match(param_info, param):
    
    keys = set(param_info.keys())
    param_names, variance, rank_to_drop = param
    layer, mlp_type = get_layer_mlp_type(param_names)

    param_data ={
        'layer': int(layer),
        'mlp_type': mlp_type,
        'rank': rank_to_drop,
        'variance': variance
    }
    matches = True
    for key in keys:
        if param_info[key] != param_data[key]:
            matches = False
            break
    return matches



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
        if key[1] == 0 and key[2] == 0:
            label = 'No Watermark'
        else:
            label = f'Var {key[1]}, r {key[2]}'
        vars.append(label)

        score = val[task][metric]
        error = val[task][metric + '_stderr']

        if key[1] == 0 and key[2] == 0:
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



################## Getting good models #########################

def get_watermark_overrides(good_models_key):

    model, key = good_models_key
    param_name, var, rank = key
    layer, mlp_type = get_layer_mlp_type(param_name)

    param_name = f'{layer}@@@{mlp_type}@@@weight'
    return '@___@'.join([model, str(rank), param_name, str(var)])


def is_good_model(gens, metric_dict, unwatermarked_metrics, fraction_passed_threshold=0.75, sigfig_threshold=0.05, tolerance=0.0, min_prob=1e-20, grid_size=500, include_gsm8k=True):
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
        elif key == 'gsm8k_cot_llama':
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
            failure = threshold - upper_confidence_interval + tolerance
            print(f"Model failed {key} test by {failure:.4f}")
            has_good_metrics = False
            break

    
    ## Checking if pvalues are good
    data = get_all(gens, min_prob) 
    frac_passed = len(data[data['pvalue'] < sigfig_threshold]) / len(data)
    if frac_passed < fraction_passed_threshold:
        return False, frac_passed, has_good_metrics
    else:
        return True, frac_passed, has_good_metrics
    

def get_good_models(parents, fraction_passed_threshold=0.75, sigfig_threshold=0.05, tolerance=0.0, min_prob=1e-20, grid_size=500, include_gsm8k=True, key_is_path=False, verbose=False):
    """
    Returns a dictionary of models that pass the criteria
    """
    good_models = {}
    for parent in parents:
        generations = get_generations(parent)
        metrics = get_metrics(parent, key_is_path=key_is_path)
        
        for model in generations.keys():
            

            
            unwatermarked_metrics, base_model = get_unwatermarked_metrics(model)



            for key, value in generations[model].items():
                try:
                    if key_is_path:
                        path = metrics[model][key]['path']
                        del metrics[model][key]['path']

                    
                    good_pvalues, frac_passed, has_good_metrics = is_good_model(value, metrics[model][key], unwatermarked_metrics, fraction_passed_threshold=0.75, sigfig_threshold=0.05, tolerance=tolerance, min_prob=min_prob, grid_size=grid_size, include_gsm8k=include_gsm8k)
                    
                    if good_pvalues and has_good_metrics:
                        if not key_is_path:
                            good_models[(model, key)] = frac_passed
                        else:
                            good_models[(model, key)] = [frac_passed, path]
                except Exception as e:
                    if verbose:
                        print(f"{key} failed with {e}")
    
    return good_models






def print_good_models(parent, good_models):
    """
    Given parent, prints the good models in csv form
    """
    good_models = dict(sorted(good_models.items(), key=lambda x: x[1], reverse=True))
    for key, (auc, path) in good_models.items():
        override = get_watermark_overrides(key)
        exp_name = os.path.basename(parent)
        if 'laser' in path:
                laser = True
        else:
                laser = False
        print(f"{override},{exp_name},{os.path.basename(path)},{auc},{laser}")



################## PLOTTING FUNCTIONS #########################


def get_quantiles(datum, quantiles=[0.25, 0.5, 0.75]):
    """
    Given a datum, returns the quantiles
    """
    pvals = datum['pvalue']

    outs = (pvals.quantile(q) for q in quantiles)
    return outs


def plot_by_layer_var(data, variance,title, quantiles=[0.25, 0.5, 0.75], use_log=False):
    """
    Plots the pvalues by layer
    """
    medians = {}
    ubs = {}
    lbs = {}
    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if var != variance:
            continue
        
        if mlp_type not in medians.keys():
            medians[mlp_type] = {}
            ubs[mlp_type] = {}
            lbs[mlp_type] = {}
        lb, med, ub = get_quantiles(datum, quantiles)
        if use_log:
            med = np.log10(max(med, 1e-15))
            ub = np.log10(max(ub, 1e-15))
            lb = np.log10(max(lb, 1e-15))
        
        medians[mlp_type][layer] = med
        ubs[mlp_type][layer] = ub
        lbs[mlp_type][layer] = lb
    
    for mlp_type in medians.keys():
        medians[mlp_type] = {k: v for k, v in sorted(medians[mlp_type].items(), key=lambda x: x[0])}
        ubs[mlp_type] = {k: v for k, v in sorted(ubs[mlp_type].items(), key=lambda x: x[0])}
        lbs[mlp_type] = {k: v for k, v in sorted(lbs[mlp_type].items(), key=lambda x: x[0])}
        
    

    fig, ax = plt.subplots()
    for mlp_type in medians.keys():
        ax.plot(list(medians[mlp_type].keys()), list(medians[mlp_type].values()), label=mlp_type, marker='o')
        ax.fill_between(list(medians[mlp_type].keys()), list(lbs[mlp_type].values()), list(ubs[mlp_type].values()), alpha=0.2)



    ax.set_title(title)
    ax.set_xlabel('Layer')
    if use_log:
        ax.set_ylabel('Log10 Median P-value')
    else:
        ax.set_ylabel('Median P-value')

    ax.legend()
    return fig, ax


def plot_by_layer_mlptype(data, fixed_mlp_type, title, quantiles=[0.25, 0.5, 0.75], use_log=False):
    """
    Plots the pvalues by layer with fixed mlp_type
    """
    medians = {}
    ubs = {}
    lbs = {}
    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        
        if var not in medians.keys():
            medians[var] = {}
            ubs[var] = {}
            lbs[var] = {}
        lb, med, ub = get_quantiles(datum, quantiles)
        if use_log:
            med = np.log10(max(med, 1e-15))
            ub = np.log10(max(ub, 1e-15))
            lb = np.log10(max(lb, 1e-15))
        
        medians[var][layer] = med
        ubs[var][layer] = ub
        lbs[var][layer] = lb

    
    for var in medians.keys():
        medians[var] = {k: v for k, v in sorted(medians[var].items(), key=lambda x: x[0])}
        ubs[var] = {k: v for k, v in sorted(ubs[var].items(), key=lambda x: x[0])}
        lbs[var] = {k: v for k, v in sorted(lbs[var].items(), key=lambda x: x[0])}
        
    
    medians = {k: v for k, v in sorted(medians.items(), key=lambda x: x[0])}
    ubs = {k: v for k, v in sorted(ubs.items(), key=lambda x: x[0])}
    lbs = {k: v for k, v in sorted(lbs.items(), key=lambda x: x[0])}



    fig, ax = plt.subplots()
    for var in medians.keys():
        ax.plot(list(medians[var].keys()), list(medians[var].values()), label=var, marker='o')
        ax.fill_between(list(medians[var].keys()), list(lbs[var].values()), list(ubs[var].values()), alpha=0.2)



    ax.set_title(title)
    ax.set_xlabel('Layer')
    if use_log:
        ax.set_ylabel('Log10 Median P-value')
    else:
        ax.set_ylabel('Median P-value')

    ax.legend()
    return fig, ax


def plot_by_variance_layer(data, fixed_layer, title, quantiles=[0.25, 0.5, 0.75], use_log=False):
    """
    Plots the pvalues by layer with fixed mlp_type
    """
    medians = {}
    ubs = {}
    lbs = {}
    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if layer != fixed_layer:
            continue
        
        if mlp_type not in medians.keys():
            medians[mlp_type] = {}
            ubs[mlp_type] = {}
            lbs[mlp_type] = {}
        lb, med, ub = get_quantiles(datum, quantiles)
        if use_log:
            med = np.log10(max(med, 1e-15))
            ub = np.log10(max(ub, 1e-15))
            lb = np.log10(max(lb, 1e-15))
        
        medians[mlp_type][var] = med
        ubs[mlp_type][var] = ub
        lbs[mlp_type][var] = lb

    
    for mlp_type in medians.keys():
        medians[mlp_type] = {k: v for k, v in sorted(medians[mlp_type].items(), key=lambda x: x[0])}
        ubs[mlp_type] = {k: v for k, v in sorted(ubs[mlp_type].items(), key=lambda x: x[0])}
        lbs[mlp_type] = {k: v for k, v in sorted(lbs[mlp_type].items(), key=lambda x: x[0])}
        
    
    medians = {k: v for k, v in sorted(medians.items(), key=lambda x: x[0])}
    ubs = {k: v for k, v in sorted(ubs.items(), key=lambda x: x[0])}
    lbs = {k: v for k, v in sorted(lbs.items(), key=lambda x: x[0])}



    fig, ax = plt.subplots()
    for mlp_type in medians.keys():
        ax.plot(list(medians[mlp_type].keys()), list(medians[mlp_type].values()), label=mlp_type, marker='o')
        ax.fill_between(list(medians[mlp_type].keys()), list(lbs[mlp_type].values()), list(ubs[mlp_type].values()), alpha=0.2)



    ax.set_title(title)
    ax.set_xlabel('Variance')
    if use_log:
        ax.set_ylabel('Log10 Median P-value')
    else:
        ax.set_ylabel('Median P-value')

    ax.legend()
    return fig, ax

def plot_by_variance_mlptype(data, fixed_mlp_type, title, quantiles=[0.25, 0.5, 0.75], use_log=False):
    """
    Plots the pvalues by layer with fixed mlp_type
    """
    medians = {}
    ubs = {}
    lbs = {}
    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        
        if layer not in medians.keys():
            medians[layer] = {}
            ubs[layer] = {}
            lbs[layer] = {}
        lb, med, ub = get_quantiles(datum, quantiles)
        if use_log:
            med = np.log10(max(med, 1e-15))
            ub = np.log10(max(ub, 1e-15))
            lb = np.log10(max(lb, 1e-15))
        
        medians[layer][var] = med
        ubs[layer][var] = ub
        lbs[layer][var] = lb

    
    for layer in medians.keys():
        medians[layer] = {k: v for k, v in sorted(medians[layer].items(), key=lambda x: x[0])}
        ubs[layer] = {k: v for k, v in sorted(ubs[layer].items(), key=lambda x: x[0])}
        lbs[layer] = {k: v for k, v in sorted(lbs[layer].items(), key=lambda x: x[0])}
        
    
    medians = {k: v for k, v in sorted(medians.items(), key=lambda x: x[0])}
    ubs = {k: v for k, v in sorted(ubs.items(), key=lambda x: x[0])}
    lbs = {k: v for k, v in sorted(lbs.items(), key=lambda x: x[0])}



    fig, ax = plt.subplots()
    for layer in medians.keys():
        ax.plot(list(medians[layer].keys()), list(medians[layer].values()), label=layer, marker='o')
        ax.fill_between(list(medians[layer].keys()), list(lbs[layer].values()), list(ubs[layer].values()), alpha=0.2)



    ax.set_title(title)
    ax.set_xlabel('Variance')
    if use_log:
        ax.set_ylabel('Log10 Median P-value')
    else:
        ax.set_ylabel('Median P-value')

    ax.legend()
    return fig, ax



def plot_by_rank_mlptype(data,fixed_mlp_type, fixed_layer, title, quantiles=[0.25, 0.5, 0.75], use_log=False):
    """
    Plots the pvalues by layer with fixed mlp_type
    """
    medians = {}
    ubs = {}
    lbs = {}
    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        rank = key[2]
        if layer != fixed_layer:
            continue
        if mlp_type != fixed_mlp_type:
            continue
        
        if rank not in medians.keys():
            medians[rank] = {}
            ubs[rank] = {}
            lbs[rank] = {}
        lb, med, ub = get_quantiles(datum, quantiles)
        if use_log:
            med = np.log10(max(med, 1e-15))
            ub = np.log10(max(ub, 1e-15))
            lb = np.log10(max(lb, 1e-15))
        
        medians[rank][var] = med
        ubs[rank][var] = ub
        lbs[rank][var] = lb

    
    for rank in medians.keys():
        medians[rank] = {k: v for k, v in sorted(medians[rank].items(), key=lambda x: x[0])}
        ubs[rank] = {k: v for k, v in sorted(ubs[rank].items(), key=lambda x: x[0])}
        lbs[rank] = {k: v for k, v in sorted(lbs[rank].items(), key=lambda x: x[0])}
        
    
    medians = {k: v for k, v in sorted(medians.items(), key=lambda x: x[0])}
    ubs = {k: v for k, v in sorted(ubs.items(), key=lambda x: x[0])}
    lbs = {k: v for k, v in sorted(lbs.items(), key=lambda x: x[0])}



    fig, ax = plt.subplots()
    for rank in medians.keys():
        ax.plot(list(medians[rank].keys()), list(medians[rank].values()), label=rank, marker='o')
        ax.fill_between(list(medians[rank].keys()), list(lbs[rank].values()), list(ubs[rank].values()), alpha=0.2)



    ax.set_title(title)
    ax.set_xlabel('Variance')
    if use_log:
        ax.set_ylabel('Log10 Median P-value')
    else:
        ax.set_ylabel('Median P-value')
    ax.set_xscale('log')
    ax.legend()
    return fig, ax






def plot_metric_by_variance_mlp_type(metrics,unwatermarked_metrics, task, fixed_mlp_type, title):

    scores = {}
    stderrs = {}
    

    unwatermarked_score = unwatermarked_metrics[task][TASKS[task]]
    unwatermarked_stderr = unwatermarked_metrics[task][TASKS[task] + '_stderr']

    for key, val in metrics.items():

        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        if layer not in scores.keys():
            scores[layer] = {}
            stderrs[layer] = {}

        scores[layer][var] = val[task][TASKS[task]]
        stderrs[layer][var] = val[task][TASKS[task] + '_stderr']


    for layer in scores.keys():
        scores[layer] = {k: v for k, v in sorted(scores[layer].items(), key=lambda x: x[0])}
        stderrs[layer] = {k: v for k, v in sorted(stderrs[layer].items(), key=lambda x: x[0])}
    
    scores = {k: v for k, v in sorted(scores.items(), key=lambda x: x[0])}
    stderrs = {k: v for k, v in sorted(stderrs.items(), key=lambda x: x[0])}

    fig, ax = plt.subplots()
    for layer in scores.keys():
        xs = np.array(list(scores[layer].keys()))
        means = np.array(list(scores[layer].values()))
        std = np.array(list(stderrs[layer].values()))
        ax.plot(xs, means, label=layer, marker='o')
        ax.fill_between(xs, means - std, means + std, alpha=0.2)


    ax.axhline(unwatermarked_score, color='black', label='Llama', linestyle='--')
    ax.axhline(unwatermarked_score + unwatermarked_stderr, color='gray', linestyle='--',)
    ax.axhline(unwatermarked_score - unwatermarked_stderr, color='gray', linestyle='--', )
    ax.set_title(title)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Accuracy')
    # ax.legend()
    ax.set_xscale('log')
    return fig, ax



def plot_metric_by_variance_rank(metrics, unwatermarked_metrics, task, fixed_mlp_type, fixed_layer, title):

    scores = {}
    stderrs = {}
    

    unwatermarked_score = unwatermarked_metrics[task][TASKS[task]]
    unwatermarked_stderr = unwatermarked_metrics[task][TASKS[task] + '_stderr']

    for key, val in metrics.items():

        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        rank = key[2]
        if mlp_type != fixed_mlp_type:
            continue
        if layer != fixed_layer:
            continue

        if rank not in scores.keys():
            scores[rank] = {}
            stderrs[rank] = {}

        scores[rank][var] = val[task][TASKS[task]]
        stderrs[rank][var] = val[task][TASKS[task] + '_stderr']


    for rank in scores.keys():
        scores[rank] = {k: v for k, v in sorted(scores[rank].items(), key=lambda x: x[0])}
        stderrs[rank] = {k: v for k, v in sorted(stderrs[rank].items(), key=lambda x: x[0])}
    
    scores = {k: v for k, v in sorted(scores.items(), key=lambda x: x[0])}
    stderrs = {k: v for k, v in sorted(stderrs.items(), key=lambda x: x[0])}

    fig, ax = plt.subplots()
    for rank in scores.keys():
        xs = np.array(list(scores[rank].keys()))
        means = np.array(list(scores[rank].values()))
        std = np.array(list(stderrs[rank].values()))
        ax.plot(xs, means, label=rank, marker='o')
        ax.fill_between(xs, means - std, means + std, alpha=0.2)


    ax.axhline(unwatermarked_score, color='black', label='Llama', linestyle='--')
    ax.axhline(unwatermarked_score + unwatermarked_stderr, color='gray', linestyle='--',)
    ax.axhline(unwatermarked_score - unwatermarked_stderr, color='gray', linestyle='--', )
    ax.set_title(title)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_xscale('log')
    return fig, ax




def get_average_superglue_scores(metrics_datum, unwatermarked_datum, verbose=False):
    """
    Get average superglue score and standard error, as normalized by the unwatermarked score for each task
    """
    SUPERGLUE_TASKS = {
        'boolq':'acc',
        'cb':'acc',
        'copa':'acc',
        'multirc':'acc',
        'record': 'f1',
        'rte':'acc',
        'wic':'acc',
        'wsc':'acc',
    }

    normalized_scores, normalized_stderr = {}, {}
    for task, val in metrics_datum.items():

        if task in SUPERGLUE_TASKS.keys():
            raw_score = val[SUPERGLUE_TASKS[task]]
            raw_stderr = val[SUPERGLUE_TASKS[task] + '_stderr']

            unwatermarked_score = unwatermarked_datum[task][SUPERGLUE_TASKS[task]]

            normalized_scores[task] = raw_score / unwatermarked_score
            normalized_stderr[task] = raw_stderr / unwatermarked_score
    
    average_normalized_score = np.mean(list(normalized_scores.values()))
    average_normalized_stderr = np.linalg.norm(list(normalized_stderr.values())) / np.sqrt(len(normalized_scores))
    if verbose:
        for task, score in normalized_scores.items():
            print(f"{task}: {score:.4f} +/- {normalized_stderr[task]:.4f}")
    return average_normalized_score, average_normalized_stderr



def plot_rank_by_variance(data, fixed_layer, fixed_mlp_type, title, use_log=False, quantiles=[0.25, 0.5, 0.75]):
    
    lbs, medians, ubs = {}, {}, {}

    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        rank = key[2]
        if layer == fixed_layer and mlp_type == fixed_mlp_type:
            
            if rank not in medians.keys():
                medians[rank] = {}
                lbs[rank] = {}
                ubs[rank] = {}
            
            lb, med, ub = get_quantiles(datum, quantiles)
            lbs[rank][var] = lb
            medians[rank][var] = med
            ubs[rank][var] = ub
        
    for rank in medians.keys():
        medians[rank] = {k: v for k, v in sorted(medians[rank].items(), key=lambda item: item[0])}
        lbs[rank] = {k: v for k, v in sorted(lbs[rank].items(), key=lambda item: item[0])}
        ubs[rank] = {k: v for k, v in sorted(ubs[rank].items(), key=lambda item: item[0])}
    
    medians = {k: v for k, v in sorted(medians.items(), key=lambda item: item[0])}
    lbs = {k: v for k, v in sorted(lbs.items(), key=lambda item: item[0])}
    ubs = {k: v for k, v in sorted(ubs.items(), key=lambda item: item[0])}


    fig, ax = plt.subplots()
    for rank, datum in medians.items():
        ax.plot(list(datum.keys()), list(datum.values()), label=f"Rank {rank}", marker='o')
        ax.fill_between(list(datum.keys()), list(lbs[rank].values()), list(ubs[rank].values()), alpha=0.2)
    
    ax.set_title(title)
    ax.set_xlabel('Variance')
    ax.set_ylabel('P-value')
    ax.legend()
    ax.set_xscale('log')
    return fig, ax


def plot_task_rank_by_variance(metrics, unwatermarked_metrics, task, fixed_layer, fixed_mlp_type, title):
    """
    Plots the task metrics against variance, with each curve representing a different rank of the rand-reduced watermark
    """
    means, stderrs = {}, {}

    for key, datum in metrics.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        rank = key[2]
        if layer == fixed_layer and mlp_type == fixed_mlp_type:
            
            if rank not in means.keys():
                means[rank] = {}
                stderrs[rank] = {}

            
            means[rank][var] = datum[task][TASKS[task]]
            stderrs[rank][var] = datum[task][TASKS[task] + '_stderr']


            
        
    for rank in means.keys():
        means[rank] = {k: v for k, v in sorted(means[rank].items(), key=lambda item: item[0])}
        stderrs[rank] = {k: v for k, v in sorted(stderrs[rank].items(), key=lambda item: item[0])}

    
    means = {k: v for k, v in sorted(means.items(), key=lambda item: item[0])}
    stderrs = {k: v for k, v in sorted(stderrs.items(), key=lambda item: item[0])}


    unwat_mean = unwatermarked_metrics[task][TASKS[task]]
    unwat_std = unwatermarked_metrics[task][TASKS[task] + '_stderr']

    fig, ax = plt.subplots()
    for rank, datum in means.items():
        xs = list(datum.keys())
        ys = np.array(list(datum.values()))

        yerrs = np.array(list(stderrs[rank].values()))

        lbs = ys - yerrs
        ubs = ys + yerrs

        ax.plot(xs, ys, label=f"Rank {rank}", marker='o')
        ax.fill_between(xs, lbs, ubs, alpha=0.2)
    
    ax.axhline(unwat_mean, color='black', linestyle='--', label='Unwatermarked')
    ax.fill_between(xs, unwat_mean - unwat_std, unwat_mean + unwat_std, alpha=0.2, color='black')

    ax.set_title(title)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_xscale('log')
    return fig, ax








################## PLOTTING FUNCTIONS FOR COLORED TABLES ####################



def get_pval_colored_tables(data, fixed_mlp_type):
    """
    Plots the colored tables of pvalues
    """
    medians = {}

    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        
        if layer not in medians.keys():
            medians[layer] = {}
            
        _, med = get_quantiles(datum, [0.25, 0.5])
        medians[layer][var] = med

    
    for layer in medians.keys():
        medians[layer] = {k: v for k, v in sorted(medians[layer].items(), key=lambda x: x[0])}

    medians = {k: v for k, v in sorted(medians.items(), key=lambda x: x[0], reverse=True)}



    array = np.zeros((len(medians.keys()), len(medians[next(iter(medians.keys()))].keys())))
    for i, layer in enumerate(medians.keys()):
        for j, var in enumerate(medians[layer].keys()):
            array[i, j] = medians[layer][var]
    

    fig, ax = plt.subplots()

    yticklabels = list(medians.keys())
    yticklabels = [f'Layer {y}' for y in yticklabels]
    xticklabels = list(medians[next(iter(medians.keys()))].keys())
    xticklabels = [f'{x:.0e}' for x in xticklabels]
    sns.heatmap(array, ax=ax, cmap='viridis_r', annot=True, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    # ax.tick_params(axis=u'both', which=u'both',length=0)
    # ax.set_xlabel('Var', x=-0.1)
    ax.text(-0.17,1.02, 'Var',transform=ax.transAxes, ha='left', va='bottom' )
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=False, labeltop=True, labelbottom=False, bottom=False)
    ax.tick_params(axis='y', left=False, labelleft=True, labelright=False, right=False, labelrotation=0)
    ax.grid(False)
    

    return fig, ax


def get_pval_colored_tables_rank(data, fixed_mlp_type, fixed_layer):
    """
    Plots the colored tables of pvalues
    """
    medians = {}

    for key, datum in data.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        rank = key[2]
        if mlp_type != fixed_mlp_type:
            continue
        if layer != fixed_layer:
            continue
        
        if rank not in medians.keys():
            medians[rank] = {}
            
        _, med = get_quantiles(datum, [0.25, 0.5])
        medians[rank][var] = med

    
    for rank in medians.keys():
        medians[rank] = {k: v for k, v in sorted(medians[rank].items(), key=lambda x: x[0])}

    medians = {k: v for k, v in sorted(medians.items(), key=lambda x: x[0], reverse=True)}



    array = np.zeros((len(medians.keys()), len(medians[next(iter(medians.keys()))].keys())))
    for i, rank in enumerate(medians.keys()):
        for j, var in enumerate(medians[rank].keys()):
            array[i, j] = medians[rank][var]
    

    fig, ax = plt.subplots()

    yticklabels = list(medians.keys())
    yticklabels = [f'Rank {y}' for y in yticklabels]
    xticklabels = list(medians[next(iter(medians.keys()))].keys())
    xticklabels = [f'{x:.0e}' for x in xticklabels]
    sns.heatmap(array, ax=ax, cmap='viridis_r', annot=True, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    # ax.tick_params(axis=u'both', which=u'both',length=0)
    # ax.set_xlabel('Var', x=-0.1)
    ax.text(-0.17,1.02, 'Var',transform=ax.transAxes, ha='left', va='bottom' )
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=False, labeltop=True, labelbottom=False, bottom=False)
    ax.tick_params(axis='y', left=False, labelleft=True, labelright=False, right=False, labelrotation=0)
    ax.grid(False)
    

    return fig, ax


def get_avg_superglue_colored_tables(metrics, unwatermarked_metrics, fixed_mlp_type):
    """
    Plots the colored tables of pvalues
    """
    mean_scores = {}

    for key, metric in metrics.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        
        if layer not in mean_scores.keys():
            mean_scores[layer] = {}
            
        mean_score, _ = get_average_superglue_scores(metric, unwatermarked_metrics)
        mean_scores[layer][var] = mean_score

    
    for layer in mean_scores.keys():
        mean_scores[layer] = {k: v for k, v in sorted(mean_scores[layer].items(), key=lambda x: x[0])}

    mean_scores = {k: v for k, v in sorted(mean_scores.items(), key=lambda x: x[0], reverse=True)}



    array = np.zeros((len(mean_scores.keys()), len(mean_scores[next(iter(mean_scores.keys()))].keys())))
    for i, layer in enumerate(mean_scores.keys()):
        for j, var in enumerate(mean_scores[layer].keys()):
            array[i, j] = mean_scores[layer][var]
    

    fig, ax = plt.subplots()

    yticklabels = list(mean_scores.keys())
    yticklabels = [f'Layer {y}' for y in yticklabels]
    xticklabels = list(mean_scores[next(iter(mean_scores.keys()))].keys())
    xticklabels = [f'{x:.0e}' for x in xticklabels]
    sns.heatmap(array, ax=ax, cmap='viridis', annot=True, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    # ax.tick_params(axis=u'both', which=u'both',length=0)
    # ax.set_xlabel('Var', x=-0.1)
    ax.text(-0.17,1.02, 'Var',transform=ax.transAxes, ha='left', va='bottom' )
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=False, labeltop=True, labelbottom=False, bottom=False)
    ax.tick_params(axis='y', left=False, labelleft=True, labelright=False, right=False, labelrotation=0)
    ax.grid(False)
    

    return fig, ax



def get_superglue_colored_tables(metrics, unwatermarked_metrics, task, fixed_mlp_type, normalize=False):
    """
    Plots the colored tables of pvalues.  If normalize is True, then the scores are normalized by the unwatermarked score.  Otherwise, the raw score is used and the color is determined by whether or not the score is greater than the unwatermarked score.
    """
    mean_scores = {}
    thresholds = {}
    stderrs = {}
    threshold_stderrs = {}
    for key, metric in metrics.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        
        if layer not in mean_scores.keys():
            mean_scores[layer] = {}
            thresholds[layer] = {}
            stderrs[layer] = {}
            threshold_stderrs[layer] = {}
            
        
        raw_mean = metric[task][TASKS[task]]
        threshold = unwatermarked_metrics[task][TASKS[task]] - unwatermarked_metrics[task][TASKS[task] + '_stderr']
        stderr = metric[task][TASKS[task] + '_stderr']
        threshold_stderr = unwatermarked_metrics[task][TASKS[task] + '_stderr']

        if normalize:
            mean = raw_mean / unwatermarked_metrics[task][TASKS[task]]
        else:
            mean = raw_mean

        mean_scores[layer][var] = mean
        thresholds[layer][var] = threshold
        stderrs[layer][var] = stderr
        threshold_stderrs[layer][var] = threshold_stderr

    
    for layer in mean_scores.keys():
        mean_scores[layer] = {k: v for k, v in sorted(mean_scores[layer].items(), key=lambda x: x[0])}
        thresholds[layer] = {k: v for k, v in sorted(thresholds[layer].items(), key=lambda x: x[0])}
        stderrs[layer] = {k: v for k, v in sorted(stderrs[layer].items(), key=lambda x: x[0])}
        threshold_stderrs[layer] = {k: v for k, v in sorted(threshold_stderrs[layer].items(), key=lambda x: x[0])}

    mean_scores = {k: v for k, v in sorted(mean_scores.items(), key=lambda x: x[0], reverse=True)}
    thresholds = {k: v for k, v in sorted(thresholds.items(), key=lambda x: x[0], reverse=True)}
    stderrs = {k: v for k, v in sorted(stderrs.items(), key=lambda x: x[0], reverse=True)}
    threshold_stderrs = {k: v for k, v in sorted(threshold_stderrs.items(), key=lambda x: x[0], reverse=True)}



    # array = np.zeros((len(mean_scores.keys()), len(mean_scores[next(iter(mean_scores.keys()))].keys())))
    array = np.zeros((len(mean_scores.keys()), 5))
    thresholds_array = np.zeros((len(mean_scores.keys()), 5))
    stderrs_array = np.zeros((len(mean_scores.keys()), 5))
    threshold_stderr_array = np.zeros((len(mean_scores.keys()), 5))

    for i, layer in enumerate(mean_scores.keys()):
        for j, var in enumerate(mean_scores[layer].keys()):
            array[i, j] = mean_scores[layer][var]
            thresholds_array[i, j] = thresholds[layer][var]
            stderrs_array[i, j] = stderrs[layer][var]
            threshold_stderr_array[i, j] = threshold_stderrs[layer][var]
    
    passed = np.zeros_like(array)
    passed[array + stderrs_array > thresholds_array - threshold_stderr_array] = 1
    

    fig, ax = plt.subplots()

    yticklabels = list(mean_scores.keys())
    yticklabels = [f'Layer {y}' for y in yticklabels]
    xticklabels = list(mean_scores[next(iter(mean_scores.keys()))].keys())
    xticklabels = [f'{x:.0e}' for x in xticklabels]
    if normalize:
        sns.heatmap(array, ax=ax, cmap='viridis', annot=True, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)

    else:
        cmap = plt.get_cmap('viridis')
        min_color = cmap(0.0)
        max_color = cmap(1.0)
        colors = (min_color, max_color)
        sns.heatmap(passed, ax=ax, cmap=matplotlib.colors.ListedColormap(colors), annot=array, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    
    # ax.tick_params(axis=u'both', which=u'both',length=0)
    # ax.set_xlabel('Var', x=-0.1)
    ax.text(-0.17,1.02, 'Var',transform=ax.transAxes, ha='left', va='bottom' )
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=False, labeltop=True, labelbottom=False, bottom=False)
    ax.tick_params(axis='y', left=False, labelleft=True, labelright=False, right=False, labelrotation=0)
    ax.grid(False)
    

    return fig, ax

def get_gsm8k_colored_tables(metrics, unwatermarked_metrics, fixed_mlp_type, normalize=False):
    """
    Plots the colored tables of GSM8K scores. If normalize is True, then the scores are normalized by the unwatermarked score.  Otherwise, the raw score is used and the color is determined by whether or not the score is greater than the unwatermarked score.
    """
    mean_scores = {}
    thresholds = {}
    stderrs = {}
    threshold_stderrs = {}

    for key, metric in metrics.items():
        layer, mlp_type = get_layer_mlp_type(key[0])
        layer = int(layer)
        var = key[1]
        if mlp_type != fixed_mlp_type:
            continue
        
        if layer not in mean_scores.keys():
            mean_scores[layer] = {}
            thresholds[layer] = {}
            stderrs[layer] = {}
            threshold_stderrs[layer] = {}
        
        raw_mean = metric['gsm8k_cot_self_consistency']['exact_match,flexible-extract']
        if normalize:
            mean = raw_mean / unwatermarked_metrics['gsm8k_cot_self_consistency']['exact_match,flexible-extract']
        else:
            mean = raw_mean

        mean_scores[layer][var] = mean
        thresholds[layer][var] = unwatermarked_metrics['gsm8k_cot_self_consistency']['exact_match,flexible-extract']
        stderrs[layer][var] = metric['gsm8k_cot_self_consistency']['exact_match,flexible-extract_stderr']
        threshold_stderrs[layer][var] = unwatermarked_metrics['gsm8k_cot_self_consistency']['exact_match,flexible-extract_stderr']


    
    for layer in mean_scores.keys():
        mean_scores[layer] = {k: v for k, v in sorted(mean_scores[layer].items(), key=lambda x: x[0])}
        thresholds[layer] = {k: v for k, v in sorted(thresholds[layer].items(), key=lambda x: x[0])}
        stderrs[layer] = {k: v for k, v in sorted(stderrs[layer].items(), key=lambda x: x[0])}
        threshold_stderrs[layer] = {k: v for k, v in sorted(threshold_stderrs[layer].items(), key=lambda x: x[0])}


    mean_scores = {k: v for k, v in sorted(mean_scores.items(), key=lambda x: x[0], reverse=True)}
    thresholds = {k: v for k, v in sorted(thresholds.items(), key=lambda x: x[0], reverse=True)}
    stderrs = {k: v for k, v in sorted(stderrs.items(), key=lambda x: x[0], reverse=True)}
    threshold_stderrs = {k: v for k, v in sorted(threshold_stderrs.items(), key=lambda x: x[0], reverse=True)}



    # array = np.zeros((len(mean_scores.keys()), len(mean_scores[next(iter(mean_scores.keys()))].keys())))
    array = np.zeros((len(mean_scores.keys()),  5))
    thresholds_array = np.zeros((len(mean_scores.keys()), 5))
    stderrs_array = np.zeros((len(mean_scores.keys()), 5))
    threshold_stderr_array = np.zeros((len(mean_scores.keys()), 5))


    for i, layer in enumerate(mean_scores.keys()):
        for j, var in enumerate(mean_scores[layer].keys()):
            array[i, j] = mean_scores[layer][var]
            thresholds_array[i, j] = thresholds[layer][var]
            stderrs_array[i, j] = stderrs[layer][var]
            threshold_stderr_array[i, j] = threshold_stderrs[layer][var]
    

    passed = np.zeros_like(array)
    passed[array + stderrs_array > thresholds_array - threshold_stderr_array] = 1

    fig, ax = plt.subplots()

    yticklabels = list(mean_scores.keys())
    yticklabels = [f'Layer {y}' for y in yticklabels]
    xticklabels = list(mean_scores[next(iter(mean_scores.keys()))].keys())
    xticklabels = [f'{x:.0e}' for x in xticklabels]


    if normalize:
        sns.heatmap(array, ax=ax, cmap='viridis', annot=True, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)

    else:
        cmap = plt.get_cmap('viridis')
        min_color = cmap(0.0)
        max_color = cmap(1.0)
        colors = (min_color, max_color)
        sns.heatmap(passed, ax=ax, cmap=matplotlib.colors.ListedColormap(colors), annot=array, fmt='.4f', cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)


    # ax.tick_params(axis=u'both', which=u'both',length=0)
    # ax.set_xlabel('Var', x=-0.1)
    ax.text(-0.17,1.02, 'Var',transform=ax.transAxes, ha='left', va='bottom' )
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', top=False, labeltop=True, labelbottom=False, bottom=False)
    ax.tick_params(axis='y', left=False, labelleft=True, labelright=False, right=False, labelrotation=0)
    ax.grid(False)
    

    return fig, ax


#################### Winrate Calculations ####################

def get_winrate(parent):
    found_responses = False
    found_flipped = False
    for root, dirs, files in os.walk(parent):
        if found_responses and found_flipped:
            break

        for file in files:

            if 'responses.json' in file:
                with open(os.path.join(root, file), 'r') as f:
                    responses = json.load(f)
                
                found_responses = True
            elif 'flipped.json' in file:
                with open(os.path.join(root, file), 'r') as f:
                    flipped = json.load(f)
                found_flipped = True
    

    watermarked_wins = 0
    base_wins = 0
    for i in range(len(responses)):
        response = responses[i]
        if flipped[i]:
            if response  == 'm':
                watermarked_wins += 1

            elif response  == 'M':
                base_wins += 1
        else:
            if response  == 'm':
                base_wins += 1
            elif response == 'M':
                watermarked_wins += 1

    return watermarked_wins, base_wins



def get_baseline_winrates():

    llama_baseline, mistral_baseline, phi_baseline = {}, {}, {}

    wat_wins, base_wins = get_winrate(os.path.join('alpaca_win_rate', '_home_blockadam_gaussian-watermarking_amlt_09-19-alpaca-generate-baseline_grid_09-19-alpaca-generate-baseline_see_1339_wat_meta-llama_Meta-Llama-3.1-8B_____'))
    llama_baseline[1339] = wat_wins / (wat_wins + base_wins)
    wat_wins, base_wins = get_winrate(os.path.join('alpaca_win_rate', '_home_blockadam_gaussian-watermarking_amlt_09-19-alpaca-generate-baseline_grid_09-19-alpaca-generate-baseline_see_133339_wat_meta-llama_Meta-Llama-3.1-8B_____'))
    llama_baseline[133339] = wat_wins / (wat_wins + base_wins)


    wat_wins, base_wins = get_winrate(os.path.join('alpaca_win_rate', '_home_blockadam_gaussian-watermarking_amlt_09-19-alpaca-generate-baseline_grid_09-19-alpaca-generate-baseline_see_1339_wat_mistralai_Mistral-7B-v0.3_____'))
    mistral_baseline[1339] = wat_wins / (wat_wins + base_wins)
    wat_wins, base_wins = get_winrate(os.path.join('alpaca_win_rate', '_home_blockadam_gaussian-watermarking_amlt_09-19-alpaca-generate-baseline_grid_09-19-alpaca-generate-baseline_see_133339_wat_mistralai_Mistral-7B-v0.3_____'))
    mistral_baseline[133339] = wat_wins / (wat_wins + base_wins)

    wat_wins, base_wins = get_winrate(os.path.join('alpaca_win_rate', '_home_blockadam_gaussian-watermarking_amlt_09-19-alpaca-generate-baseline_grid_09-19-alpaca-generate-baseline_see_1339_wat_microsoft_Phi-3-mini-4k-instruct_____'))
    phi_baseline[1339] = wat_wins / (wat_wins + base_wins)
    wat_wins, base_wins = get_winrate(os.path.join('alpaca_win_rate', '_home_blockadam_gaussian-watermarking_amlt_09-19-alpaca-generate-baseline_grid_09-19-alpaca-generate-baseline_see_133339_wat_microsoft_Phi-3-mini-4k-instruct_____'))
    phi_baseline[133339] = wat_wins / (wat_wins + base_wins)


    llama_mean, llama_std = np.mean(list(llama_baseline.values())), np.std(list(llama_baseline.values()))
    mistral_mean, mistral_std = np.mean(list(mistral_baseline.values())), np.std(list(mistral_baseline.values()))
    phi_mean, phi_std = np.mean(list(phi_baseline.values())), np.std(list(phi_baseline.values()))

    return llama_mean, llama_std, mistral_mean, mistral_std, phi_mean, phi_std


def _get_win_rate_path(row):
    """
    Helper for getting path to win rates for a given row of the good_models dataframe
    """
    parent = '/home/blockadam/gaussian-watermarking/alpaca_win_rate'
    dir = '_home_blockadam_gaussian-watermarking_amlt_' + row['exp_name'] + '_' + row['run_name']
    for root, dirs, files in os.walk(os.path.join(parent, dir)):
        if 'wins.json' in files:
            return os.path.join(root, 'wins.json')
    
    ## Older data has different directory structure
    dir = '._amlt_' + row['exp_name'] + '_' + row['run_name']
    for root, dirs, files in os.walk(os.path.join(parent, dir)):
        if 'wins.json' in files:
            return os.path.join(root, 'wins.json')
    
     


def _get_model_from_row(row):
    if 'llama' in row['watermark_overrides'] or 'Llama' in row['watermark_overrides']:
        return 'Llama'
    elif 'mistral' in row['watermark_overrides'] or 'Mistral' in row['watermark_overrides']:
        return 'Mistral'
    elif 'phi' in row['watermark_overrides'] or 'Phi' in row['watermark_overrides']:
        return 'Phi'
    else:
        raise ValueError(f"Could not determine model from {row['watermark_overrides']}")


def add_win_rates(good_models, verbose=False):
    """ 
    Given a dataframe of good models, adds the win rate to the dataframe
    """
    llama_mean, llama_std, mistral_mean, mistral_std, phi_mean, phi_std = get_baseline_winrates()

    for i in range(good_models.shape[0]):
        try:
            win_rate_path = _get_win_rate_path(good_models.loc[i])
            with open(win_rate_path, 'r') as f:
                wins = json.load(f)
            win_rate = wins['watermarked_wins'] / wins['total']
            good_models.loc[i, 'win_rate'] = win_rate

            model = _get_model_from_row(good_models.loc[i])
            if model == 'Llama':
                good_models.loc[i, 'baseline_mean'] = llama_mean
                good_models.loc[i, 'baseline_std'] = llama_std
            elif model == 'Mistral':
                good_models.loc[i, 'baseline_mean'] = mistral_mean
                good_models.loc[i, 'baseline_std'] = mistral_std
            elif model == 'Phi':
                good_models.loc[i, 'baseline_mean'] = phi_mean
                good_models.loc[i, 'baseline_std'] = phi_std

        except:
            if verbose:
                print(f"Error processing {good_models.loc[i]}")

    return good_models



#################### Detection Time Calculations ####################


def get_generation_detection_times(lines):
    
    found_gen_time = False
    found_detection_time = False
    for line in lines:

        if found_gen_time and found_detection_time:
            break
        if "Generation took " in line:
            gen_time = float(line.split(' ')[2])
            gen_time = gen_time / 1000
            found_gen_time = True
        
        elif 'Watermark detection took ' in line:
            detection_time = float(line.split(' ')[3])
            detection_time = detection_time / 1000
            found_detection_time = True
        else:
            continue
    
    return gen_time, detection_time


def get_detection_times(parent):
    detection_times = []
    generation_times = []

    for child in os.listdir(parent):
        if 'grid' in child:
            try:
                with open(os.path.join(parent, child, 'stdout.txt')) as f:
                    lines = f.readlines()
                gen_time, detection_time = get_generation_detection_times(lines)
                detection_times.append(detection_time)
                generation_times.append(gen_time)
            except Exception as e:
                print(f"Error with {child}")
                print(e)
                continue

    return generation_times, detection_times


def _get_numtokens_times(parent):

    detection_times = {}
    generation_times = {}
    for child in os.listdir(parent):
        if 'grid' in child:
            try:
                
                numtokens_pattern = 'max_(\d+)'
                numtokens = int(re.search(numtokens_pattern, child).group(1))

                model_name = '/'.join(child.split('_')[5:7])

                with open(os.path.join(parent, child, 'stdout.txt')) as f:
                    lines = f.readlines()
                gen_time, detection_time = get_generation_detection_times(lines)
                if model_name not in detection_times.keys():
                    detection_times[model_name] = {}
                    generation_times[model_name] = {}
                
                if numtokens not in detection_times[model_name].keys():
                    detection_times[model_name][numtokens] = []
                    generation_times[model_name][numtokens] = []

                detection_times[model_name][numtokens].append(detection_time)
                generation_times[model_name][numtokens].append(gen_time)

            except Exception as e:
                print(f"Error with {child}")
                print(e)
                continue
        
    return generation_times, detection_times

def get_num_tokens_times(parents):

    all_detection_times = {}
    all_generation_times = {}
    for parent in parents:
        
        generation_times, detection_times = _get_numtokens_times(parent)
        recursively_update_dict(all_generation_times, generation_times)
        recursively_update_dict(all_detection_times, detection_times)
    
    for model, times_dict in all_detection_times.items():
        times_dict = {k:v for k, v in sorted(times_dict.items(), key=lambda x: x[0])}
        all_detection_times[model] = times_dict
    
    for model, times_dict in all_generation_times.items():
        times_dict = {k:v for k, v in sorted(times_dict.items(), key=lambda x: x[0])}
        all_generation_times[model] = times_dict

    return all_generation_times, all_detection_times

def _get_numtokens_times_seeds(parent):

    detection_times = {}
    generation_times = {}
    for child in os.listdir(parent):
        if 'grid' in child:
            try:
                
                numtokens_pattern = 'max_(\d+)'
                numtokens = int(re.search(numtokens_pattern, child).group(1))


                raw_model_name = child.split('_wat_')[1].split('_____')[0]
                model_name = raw_model_name.replace('_', '/')

                with open(os.path.join(parent, child, 'stdout.txt')) as f:
                    lines = f.readlines()
                gen_time, detection_time = get_generation_detection_times(lines)
                if model_name not in detection_times.keys():
                    detection_times[model_name] = {}
                    generation_times[model_name] = {}
                
                if numtokens not in detection_times[model_name].keys():
                    detection_times[model_name][numtokens] = []
                    generation_times[model_name][numtokens] = []

                if numtokens not in detection_times[model_name].keys():
                    detection_times[model_name][numtokens] = []
                    generation_times[model_name][numtokens] = []


                detection_times[model_name][numtokens].append(detection_time)
                generation_times[model_name][numtokens].append(gen_time)

            except Exception as e:
                print(f"Error with {child}")
                print(e)
                continue
        
    return generation_times, detection_times

def get_num_tokens_times_seeds(parents):

    all_detection_times = {}
    all_generation_times = {}
    for parent in parents:
        
        generation_times, detection_times = _get_numtokens_times_seeds(parent)
        recursively_update_dict(all_generation_times, generation_times)
        recursively_update_dict(all_detection_times, detection_times)
    
    for model, times_dict in all_detection_times.items():
        times_dict = {k:v for k, v in sorted(times_dict.items(), key=lambda x: x[0])}
        all_detection_times[model] = times_dict
    
    for model, times_dict in all_generation_times.items():
        times_dict = {k:v for k, v in sorted(times_dict.items(), key=lambda x: x[0])}
        all_generation_times[model] = times_dict

    return all_generation_times, all_detection_times


def get_numtokens_times_lbs_ubs_means(numtokens_times):
    
    means, stds = {}, {}
    for numtokens, vals in numtokens_times.items():
        means[numtokens] = np.mean(vals)
        stds[numtokens] = np.std(vals)
    
    means = {k: v for k, v in sorted(means.items(), key=lambda item: int(item[0]))}
    stds = {k: v for k, v in sorted(stds.items(), key=lambda item: int(item[0]))}
    numtokens = list(means.keys())
    means = list(means.values())
    stds = list(stds.values())
    ubs = [means[i] + stds[i] for i in range(len(means))]
    lbs = [means[i] - stds[i] for i in range(len(means))]
    return numtokens, lbs, ubs, means

##################### Number of Tokens Calculations ####################


def _get_numtokens_generations(parent):
    """
    Given a path to the parent directory of saved amulet generations, returns a dictionary of generations, where the key is the model parameters and the value is the generations list
    """
    generations = {}
    data = {}
    for dirname in os.listdir(parent):
        dir = os.path.join(parent, dirname)
        found = re.search('_max_(\d+)', dir)
        num_tokens = int(found.group(1))
        try:
            key, model_name = get_params(dir)
            if model_name not in generations.keys():
                generations[model_name] = {}
                data[model_name] = {}
            
            if key not in generations[model_name].keys():
                generations[model_name][key] = {}
                data[model_name][key] = {}

            seed = _get_seed(dirname)


            with open(os.path.join(dir, 'generations.json')) as f:

                temp_gens = json.load(f)
            
            
            temp_data = get_all(temp_gens, 1e-20)


            if seed is not None:
                if num_tokens not in generations[model_name][key].keys():
                    generations[model_name][key][num_tokens] = {}
                    data[model_name][key][num_tokens] = {}


                generations[model_name][key][num_tokens][seed] = temp_gens
                data[model_name][key][num_tokens][seed] = temp_data
                
            else:
                generations[model_name][key][num_tokens] = temp_gens
                data[model_name][key][num_tokens] = temp_data

        except Exception as e:
            print(e)
            print(f"Error processing {dir}")
        # gen_path = os.path.join(parent, dirname, 'generations.json')
        # if os.path.exists(gen_path):
        #     with open(gen_path) as f:
        #         generations[key] = json.load(f)
        # else:
        #     print(f"No generations found for {key}")

    return generations, data

def get_num_tokens_generations(parents):

    generations = {}
    data = {}
    for parent in parents:
        gens, datum = _get_numtokens_generations(parent)
        recursively_update_dict(generations, gens)
        recursively_update_dict(data, datum)

    return generations, data



def get_numtokens_numpassed(data_dict, sig_pval=0.05, has_seeds=True):

    num_passed, num_tokens = [], []
    data = {key: data_dict[key] for key in sorted(data_dict.keys())}
        
    
    for key, data in data.items():

        num_tokens.append(key)
        if has_seeds:
            temp_num_passed = []

            for seed_data in data.values():
                temp_num_passed.append(np.mean(seed_data['pvalue'] < sig_pval))
            
            temp_num_passed = ( np.min(temp_num_passed), np.mean(temp_num_passed), np.max(temp_num_passed))

            num_passed.append(temp_num_passed)
        else:
            num_passed.append(data[data['pvalue'] < sig_pval].shape[0] / data.shape[0])

    return num_tokens, num_passed

def get_nunmtokens_medians(data_dict, quantiles=[0.25, 0.75], has_seeds=True):
        
        medians, num_tokens = [], []
        additional_quantiles = []
        data = {key: data_dict[key] for key in sorted(data_dict.keys())}
        for key, data in data.items():

            num_tokens.append(key)

            if has_seeds:
                temp_medians = []
                for seed_data in data.values():
                    temp_medians.append(seed_data['pvalue'].median())
                medians.append((np.min(temp_medians), np.mean(temp_medians), np.max(temp_medians)))

            else:
                medians.append(data['pvalue'].median())
                if len(quantiles) > 0:
                    quantile = {quantile: data['pvalue'].quantile(quantile) for quantile in quantiles}
                    additional_quantiles.append(quantile)
    
        return num_tokens, medians, additional_quantiles

def get_avg_squared_norms(data_dict, added_var, has_seeds=True):
    avg_squared_norms, num_tokens, stdderrs = [], [], []

    data = {key: data_dict[key] for key in sorted(data_dict.keys())}
    for key, data in data.items():
        num_tokens.append(key)

        if has_seeds:
            avg_squared_norm = []
            for seed_data in data.values():
                avg_squared_norm.append(seed_data['variance'].mean() / added_var)
            avg_squared_norm = (np.min(avg_squared_norm), np.mean(avg_squared_norm), np.max(avg_squared_norm))
            avg_squared_norms.append(avg_squared_norm)
            
            stdderrs.append(np.std(avg_squared_norm) / added_var)
        else:
            avg_squared_norm = data['variance'].mean() / added_var
            avg_squared_norms.append(avg_squared_norm)
            stdderrs.append(data['variance'].std() / added_var)

    return num_tokens, avg_squared_norms, stdderrs


def get_median_pval_by_norm(data_df, grid_size=500, quantiles = [0.25, 0.75]):
    
    grid = np.linspace(0, data_df['squared_norm'].max(), grid_size)
    medians, lbs, ubs = [], [], []

    for i in range(grid_size - 1):
        mask =  (data_df['squared_norm'] < grid[i + 1])
        medians.append(data_df[mask]['pvalue'].median())
        lbs.append(data_df[mask]['pvalue'].quantile(quantiles[0]))
        ubs.append(data_df[mask]['pvalue'].quantile(quantiles[1]))
    
    return grid[:-1], medians, lbs, ubs




#################### Corruption Ablations ######################

def build_get_row_lookup(good_models):
    
    row_lookup = {}
    master_parent = '/home/blockadam/gaussian-watermarking/amlt'
    for i in range(good_models.shape[0]):
        try:
            row = good_models.loc[i]

            dir = os.path.join(master_parent, row['exp_name'], row['run_name'])
            key, model_name = get_params(dir)
            row_lookup[(key, model_name)] = i
        except Exception as e:
            print(f"Error processing row {i}: {e}")
    
    return row_lookup


good_models_df = pd.read_csv('good_models.csv')
row_lookup = build_get_row_lookup(good_models_df)


def get_corrupt_pvalues(corrupt_gen, has_seeds=True):

    if has_seeds:

        pvals = {}
        for seed, seed_data in corrupt_gen.items():
            
            seed_pvals = [gen['pvalue'] for gen in seed_data]
            seed_pvals = pd.DataFrame({'pvalue':seed_pvals})
            pvals[seed] = seed_pvals
        
        return pvals

    else:
        pvals = [gen['pvalue'] for gen in corrupt_gen]
        return pd.DataFrame({'pvalue':pvals})
    


def get_path_to_uncorrupted(watermark_override, laserized, good_models, parent='/home/blockadam/gaussian-watermarking/amlt'):
    """
    Given a watermark override, returns the path to the model
    """

    row = row_lookup[watermark_override]
    row = good_models_df.iloc[row]
    if row.shape[0] == 0:
        raise ValueError(f"Could not find model with watermark override {watermark_override} and laserized {laserized}")


    exp_name = row['exp_name']
    run_name = row['run_name']
    return os.path.join(parent, exp_name, run_name)


def get_watermark_overrides_path(path):
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = json.load(f)
    return config['model']['watermark_overrides']

def get_corruption_type(path):
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    corruption_type = config['corruption_robust']['corruptions'][0]['alias']
    
    # print(config['corruption_robust']['corruptions'][0])
    if corruption_type == 'roundtrip_translation':
        num_corrupted_tokens = None
    else:
        num_corrupted_tokens = float(config['corruption_robust']['corruptions'][0]['token_frac'])
        location = config['corruption_robust']['corruptions'][0]['kwargs']['location']
        temps = corruption_type.split('_')
        corruption_type = '_'.join([temps[0], location, temps[2]])
    
    return corruption_type, num_corrupted_tokens


def _get_corruptions(parent, good_models):
    """
    Returns corruption metrics for all experiments in a directory
    """
    if 'laser' in parent:
        laserized = True
    else:
        laserized = False
    
    generations = {}
    for run_name in os.listdir(parent):

        path = os.path.join(parent, run_name)
        try:
            key, model_name = get_params(path)
            key = (key, model_name)
            if key not in generations.keys():
                generations[key] = {}
                # path_to_uncorrupted = get_path_to_uncorrupted(watermark_overrides, laserized, good_models)
                path_to_uncorrupted = get_path_to_uncorrupted(key, laserized, good_models)
                with open(os.path.join(path_to_uncorrupted, 'generations.json'), 'r') as f:
                    generations[key]['uncorrupted'] = json.load(f)
            
            
            with open(os.path.join(path, 'generations.json'), 'r') as f:
                gens = json.load(f)

            corruption_type, num_corrupted_tokens = get_corruption_type(path)
            if corruption_type == 'roundtrip_translation':
                generations[key][corruption_type] = gens
            elif corruption_type not in generations[key].keys():
                generations[key][corruption_type] = {
                    num_corrupted_tokens: gens
                }
            else:
                generations[key][corruption_type][num_corrupted_tokens] = gens
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return generations


def get_corruptions(parents, good_models):

    generations = {}
    for parent in parents:
        recursively_update_dict(generations, _get_corruptions(parent, good_models))
    
    return generations





def _get_roundtrip_translation(parent):

    generations = {}
    for run_name in os.listdir(parent):

        path = os.path.join(parent, run_name)
        try:
            key, model_name = get_params(path)
            key = (key, model_name)
            if key not in generations.keys():
                generations[key] = {}
                # path_to_uncorrupted = get_path_to_uncorrupted(watermark_overrides, laserized, good_models)
                path_to_uncorrupted = get_path_to_uncorrupted(key, False, good_models_df)
                with open(os.path.join(path_to_uncorrupted, 'generations.json'), 'r') as f:
                    generations[key]['uncorrupted'] = json.load(f)
            
            
            with open(os.path.join(path, 'generations.json'), 'r') as f:
                gens = json.load(f)

            generations[key]['roundtrip_translation'] = gens
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return generations


def get_roundtrip_translations(parents):
    
        generations = {}
        for parent in parents:
            recursively_update_dict(generations, _get_roundtrip_translation(parent))
        
        return generations





def get_uncorrupted_seeds(model, param, uncorrupted_gens, num_tokens=1000):
        return uncorrupted_gens[model][param][num_tokens]


def _get_corruptions_seeds(parent, good_models, uncorrupted_gens):
    """
    Returns corruption metrics for all experiments in a directory
    """

    
    generations = {}
    for run_name in os.listdir(parent):

        path = os.path.join(parent, run_name)
        try:
            key, model_name = get_params(path)
            key = (key, model_name)
            if key not in generations.keys():
                generations[key] = {}
                generations[key]['uncorrupted'] = get_uncorrupted_seeds(key[1], key[0], uncorrupted_gens, num_tokens=1000)

                
            
            
            with open(os.path.join(path, 'generations.json'), 'r') as f:
                gens = json.load(f)

            corruption_type, num_corrupted_tokens = get_corruption_type(path)
            seed = _get_seed(run_name)
            
            if corruption_type == 'roundtrip_translation':
                generations[key][corruption_type] = gens
            else: 
                if corruption_type not in generations[key].keys():
                    generations[key][corruption_type] = {}

                if num_corrupted_tokens not in generations[key][corruption_type].keys():
                    generations[key][corruption_type][num_corrupted_tokens] = {}
                
                generations[key][corruption_type][num_corrupted_tokens][seed] = gens


        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return generations


def get_corruptions_seeds(parents, good_models, uncorrupted_gens):

    generations = {}
    for parent in parents:
        recursively_update_dict(generations, _get_corruptions_seeds(parent, good_models, uncorrupted_gens))
    
    return generations










def get_perplexities_params(dirname):
    params = dirname.split('_wat_')[-1]

    model_name_raw, rank_to_drop, param_name_raw, variance = params.split('_____')
    model_name = model_name_raw.replace('_', '/')
    variance = float(variance)
    rank_to_drop = int(rank_to_drop)
    layer_idx, proj_type, param_type = param_name_raw.split('___')
    param_names = f'model.layers.{layer_idx}.mlp.{proj_type}.{param_type}'

    key = (param_names, variance, rank_to_drop)
    
    return key, model_name

def _get_perplexities_data(data):
    perplexities, lengths, logprobs = [], [], []
    for response in data:
        perplexities.append(response['perplexity'])
        lengths.append(response['length'])
        logprobs.append(response['logprob'])
    return perplexities, lengths, logprobs

def _get_perplexities(parent):
    """
    Given the parent of an experiment name, get the perplexities for the experiment
    """
    perplexities, lengths, logprobs = {}, {}, {}
    for dirname in os.listdir(parent):
        dir = os.path.join(parent, dirname)
        try:
            key, model_name = get_perplexities_params(dir)
            if model_name not in perplexities.keys():
                perplexities[model_name] = {}
                lengths[model_name] = {}
                logprobs[model_name] = {}

            if key[1] == 0.0:
                key = ('', 0.0, 0)
            
            if key not in perplexities[model_name].keys():
                perplexities[model_name][key] = {}
                lengths[model_name][key] = {}
                logprobs[model_name][key] = {}


            seed = _get_seed(dirname)
            path = os.path.join(dir, 'perplexities.json')
            with open(path, 'r') as f:
                data = json.load(f)
            
            temp_perplexities, temp_lengths, temp_logprobs = _get_perplexities_data(data)
            perplexities[model_name][key][seed] = temp_perplexities
            lengths[model_name][key][seed] = temp_lengths
            logprobs[model_name][key][seed] = temp_logprobs

        except Exception as e:
            print(e)
            print(f"Error processing {dir}")
    
    return perplexities, lengths, logprobs

def get_perplexities(parents):
    perplexities, lengths, logprobs = {}, {}, {}
    for parent in parents:
        temp_perplexities, temp_lengths, temp_logprobs = _get_perplexities(parent)
        recursively_update_dict(perplexities, temp_perplexities)
        recursively_update_dict(lengths, temp_lengths)
        recursively_update_dict(logprobs, temp_logprobs)
    
    return perplexities, lengths, logprobs


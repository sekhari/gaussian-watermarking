import os
from vllm import LLM, SamplingParams
import transformers
import datasets
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_text import _truncate_prompt, _phi3_ify, _get_llama_param_name, _get_mistral_param_name
from utils import get_sequence_level_logits
import time



def _parse_response(request_output):
    """
    Given a vllm request output, returns a parsed dictionary containing the necessary fields
    """
    output = {}

    output['prompt'] = request_output.prompt
    output['prompt_token_ids'] = request_output.prompt_token_ids
    responses = request_output.outputs # take most likely response

    texts = [response.text for response in responses]
    logprobs = [response.cumulative_logprob for response in responses]
    response_token_ids = [response.token_ids for response in responses]
    output['response'] = texts
    output['response_token_ids'] = response_token_ids
    output['logprobs'] = logprobs


    return output

def _get_gradient_norm(model, prompt, response, params, device='cuda'):
    prompt_tensor = torch.tensor(prompt).unsqueeze(0).to(device).long()
    output_tensor = torch.tensor(response).unsqueeze(0).to(device).long()
    prompt_len = prompt_tensor.shape[1]

    inputs = torch.cat([prompt_tensor, output_tensor], dim=1)
    model.zero_grad()
    cumulative_logits = get_sequence_level_logits(model, inputs, prompt_length=prompt_len)
    cumulative_logits.backward()
    return {name: param.grad.norm().item() for name, param in model.named_parameters() if name in params}

def get_gradients(model, outputs, params, device='cuda'):
    """
    Given a model and a list of outputs, returns the gradients of the model's parameters with respect to the outputs
    """
    for output in outputs:
        prompt_token_ids = output['prompt_token_ids']
        grad_norms = []
        for response_token_ids in output['response_token_ids']:
            grad_norms.append(_get_gradient_norm(model, prompt_token_ids, response_token_ids, params, device=device))
        
        output['grad_norms'] = grad_norms
    return outputs



def get_cfg(alias):
    if alias == 'Phi':

        model_name = 'microsoft/Phi-3-mini-4k-instruct' # 'meta-llama/Meta-Llama-3.1-8B' 'mistralai/Mistral-7B-v0.3' 
        params = {
            '20@@@down_proj@@@weight',
        }
    elif alias == 'Llama':
        model_name = 'meta-llama/Meta-Llama-3.1-8B'
        params = {
            '28@@@up_proj@@@weight'
        }
    elif alias == 'Mistral':
        model_name = 'mistralai/Mistral-7B-v0.3'
        params = {
            '20@@@up_proj@@@weight'
        }
    else: 
        raise ValueError(f"Unknown alias {alias}")

    seed = 133337


    ## Sampling parameters
    gpu_memory_utilization = 0.8
    temperature = 1.0
    num_responses = 150
    max_tokens = 1000

    ## Data Parameters
    max_samples = 5
    data_name = 'allenai/c4'
    subdata_name = 'realnewslike'
    split = 'train'
    prompt_field_name = 'text'
    max_prompt_length = 100


    ## Miscellaneous
    master_parent = '/home/blockadam/gaussian-watermarking'
    device = 'cuda'

    if 'Phi' in model_name:
        _get_param_name = _get_llama_param_name
    elif 'Llama' in model_name:
        _get_param_name = _get_llama_param_name
    elif 'Mistral' in model_name:
        _get_param_name = _get_mistral_param_name
    else:
        raise ValueError(f"Unknown model name {model_name}")

    param_names = []
    for param in params:
        layer_idx, proj_type, param_type = param.split('@@@')
        param_name = _get_param_name(layer_idx, proj_type, param_type)
        param_names.append(param_name)
    params = set(param_names)

    config = {
        'model_name': model_name,
        'params': params,
        'seed': seed,
        'gpu_memory_utilization': gpu_memory_utilization,
        'temperature': temperature,
        'num_responses': num_responses,
        'max_tokens': max_tokens,
        'max_samples': max_samples,
        'data_name': data_name,
        'subdata_name': subdata_name,
        'split': split,
        'prompt_field_name': prompt_field_name,
        'max_prompt_length': max_prompt_length,
        'master_parent': master_parent,
        'device': device,
        'alias': alias
    }
    return config







def main(alias):

    config = get_cfg(alias)
    model_name = config['model_name']
    params = config['params']
    seed = config['seed']
    gpu_memory_utilization = config['gpu_memory_utilization']
    temperature = config['temperature']
    num_responses = config['num_responses']
    max_tokens = config['max_tokens']
    max_samples = config['max_samples']
    data_name = config['data_name']
    subdata_name = config['subdata_name']
    split = config['split']
    prompt_field_name = config['prompt_field_name']
    max_prompt_length = config['max_prompt_length']
    master_parent = config['master_parent']
    device = config['device']

    data_start = time.time()
    data = datasets.load_dataset(data_name, subdata_name, split=split, streaming=True)
    prompts = []
    counter = 0
    for datum in iter(data):
        prompt = datum[prompt_field_name]
        prompts.append(prompt)
        counter += 1
        if counter == max_samples:
            break

    truncate = lambda prompt:_truncate_prompt(prompt, max_prompt_length)
    prompts = [truncate(prompt) for prompt in prompts]
    if 'Phi-3' in model_name:
        prompts = [_phi3_ify(prompt) for prompt in prompts]
    
    data_end = time.time()
    print(f"Data loading took {data_end - data_start:.0f} seconds")


    generation_start = time.time()
    sampling_params = SamplingParams(
        n = num_responses,
        seed = seed,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=0, # Return logprobs
        top_k=-1,
    )
    llm = LLM(model_name, gpu_memory_utilization=gpu_memory_utilization)

    responses = llm.generate(prompts, sampling_params)
    responses = [_parse_response(response) for response in responses]
    generation_end = time.time()
    print(f"Generation took {generation_end - generation_start:.0f} seconds")

    path = os.path.join(master_parent,'data', 'gradient_norms', f'{alias}_responses.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving responses to {path}")
    with open(path, 'w') as f:
        json.dump(responses, f)
    
    del llm

    gradients_start = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    for name, param in model.named_parameters():
        if name in params:
            param.requires_grad = True
        else:
            param.requires_grad = False
    

    responses = get_gradients(model, responses, params, device=device)
    gradients_end = time.time()
    print(f"Getting gradients took {gradients_end - gradients_start:.0f} seconds")

    print(f"Saving responses to {path}")
    with open(path, 'w') as f:
        json.dump(responses, f)





if __name__ == '__main__':
    master_start = time.time()

    data_parent = '/home/blockadam/gaussian-watermarking/data/gradient_norms'
    aliases = {
        'Phi', 'Llama', 'Mistral'
    }
    for alias in aliases:
        datas = os.listdir(data_parent)
        if f'{alias}_responses.json' in datas:
            print(f"Skipping {alias} as responses already exist")
            continue
        else:
            start = time.time()
            print(f"Getting gradients for {alias}")
            main(alias)
            end = time.time()
            print(f"{alias} took {end - start:.0f} seconds")

    master_end = time.time()
    print(f"Total time taken: {master_end - master_start:.0f} seconds")

    # phi_start = time.time()
    # alias = 'Phi'
    # main(alias)
    # phi_end = time.time()
    # print(f"Phi took {phi_end - phi_start:.0f} seconds")

    # llama_start = time.time()
    # alias = 'Llama'
    # main(alias)
    # llama_end = time.time()
    # print(f"Llama took {llama_end - llama_start:.0f} seconds")

    # mistral_start = time.time()
    # alias = 'Mistral'
    # main(alias)
    # mistral_end = time.time()
    # print(f"Mistral took {mistral_end - mistral_start:.0f} seconds")
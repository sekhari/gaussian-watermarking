import datasets
from openai import AzureOpenAI, AuthenticationError
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential, get_bearer_token_provider
import re
import numpy as np


import os
import json

import time
import torch
import hydra
from omegaconf import OmegaConf
from utils import save_amulet_metadata, hf_login


from kgw_generate import _tokenize_prompts, get_kgw_config, get_generation_config
from generate_text import get_model_type
from eval_alpaca_generations import process_text, TEMPLATES


COMPARATOR_SEED = 1339

def get_base_model_alpaca_generations(model_name, path_to_source='/home/blockadam/gaussian-watermarking/data/alpaca_gaussmark'):
    """
    Gets the generations for the KGW ALPACA evaluation.

    Args:
    alias (str):
        The alias of the model.
    bias (str):
        The bias of the model.
    path_to_source (str):
        The path to the source directory.
    """
    alias = get_alias(model_name)
    path = os.path.join(path_to_source, alias, f'Base_{COMPARATOR_SEED}', 'watermarked')
    return datasets.load_from_disk(path)




def get_client(api_version, model_name, model_version, instance='msrne/shared'):
    """
    Gets the client for the Azure OpenAI service.

    Args:

    api_version (str): 
        The version of the API to use.
    model_name (str):
        The name of the model.
    model_version (str):
        The version of the model.
    instance (str):
        The instance to use. Default is 'msrne/shared'.  Also can use 'gcr/shared' for the GCR instance.
    """
    assert instance in ['msrne/shared', 'gcr/shared'], 'Instance must be either `msrne/shared` or `gcr/shared`'

    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            exclude_cli_credential=True,
            # Exclude other credentials we are not interested in.
            exclude_environment_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
            # Azure ML Compute jobs that has the client id of the
            # user-assigned managed identity in it.
            # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
            # In case it is not set the ManagedIdentityCredential will
            # default to using the system-assigned managed identity, if any.
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        )
    ),scope)

    deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
    instance = 'gcr/shared' # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly) 
    endpoint = f'https://trapi.research.microsoft.com/{instance}'

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )

    return client, deployment_name

def get_response(client, deployment_name, query):
    if type(query) == str:
         messages = [
            {
                "role": "user",
                "content": query,
            }
        ]
    elif type(query) == list:
        messages = query
    else:
        raise ValueError('Query must be either a string or a list of strings')
    
    response = client.chat.completions.create(
    model=deployment_name,
    messages=messages
    ) 
    response_content = response.choices[0].message.content
    return response_content

def get_wins(responses):
    """
    Get the wins from the responses.
    """
    watermarked_wins = 0
    base_wins = 0
    for i in range(len(responses)):
        output = responses[i]
        flipped = output['flipped']
        response = output['response']
        if flipped:
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


def get_alias(model_name):
    if 'phi' in model_name.lower():
        alias = 'phi'
    elif 'llama' in model_name.lower():
        alias = 'llama'
    elif 'mistral' in model_name.lower():
        alias = 'mistral'
    else:
        raise ValueError('Model name must contain either "phi", "llama", or "mistral"')

    return alias


def get_gaussmark_alpaca_generations(alias, watermark_overrides, path_to_source='/home/blockadam/gaussian-watermarking/data/alpaca_gaussmark'):
    """
    Gets the generations for the ALPACA evaluation.

    Args:
    watermark_overrides (str):
        The watermark overrides for the model.  If base model, use Base_{seed} for the watermark overrides for one of the available seeds.
    path_to_source (str):
        The path to the source directory.
    """
    path = os.path.join(path_to_source, alias, watermark_overrides, 'watermarked')
    return datasets.load_from_disk(path)


def _get_instruction(prompt):
    
    found = re.search(r"Instruction\n\n{\n    'instruction': .*,\n", prompt)
    if found is None:
        return None
    else:
        return found.group(0)[34:-2]


def get_responses(client, deployment_name, queries, is_flipped, save_path, save_interval=10):
    responses = []
    attempted = 0
    if os.path.exists(save_path):
        print(f"Loading responses from {save_path}")
        with open(save_path, 'r') as f:
            responses = json.load(f)
        print(f"Loaded {len(responses)} responses out of {len(queries)}")
        if len(responses) == len(queries):
            print("All responses loaded")
            return responses
        else:
            all_responded_instructions = {_get_instruction(response['prompt']) for response in responses}
    else:
        all_responded_instructions = set()

    for i, query in enumerate(queries):
        prompt = query[1]['content']
        completed = False
        if i % save_interval == 0 and i > 0:
            print(f"({i}) Saving responses to {save_path}")
            with open(save_path, 'w') as f:
                json.dump(responses, f)

        if prompt in all_responded_instructions:
            completed = True

        while not completed:
            try:
                response = get_response(client, deployment_name, query)
                temp = {
                    'prompt': prompt,
                    'flipped': is_flipped[i],
                    'response': response
                }
                responses.append(temp)
                instruction = _get_instruction(prompt)
                if instruction is not None:
                    all_responded_instructions.add(instruction)
                else:
                    # print(f"Instruction not found in prompt: {prompt}")
                    print(f"Instruction not found in prompt.")

            except Exception as e:
                attempted += 1
                if e is AuthenticationError:
                    raise e
                elif e is KeyboardInterrupt:
                    raise e
                else:
                    if attempted <= 3:
                        print(e)
                        sleep_time = 2
                        print(f"Retrying in {sleep_time} second")
                        time.sleep(sleep_time)
                        continue
                    elif attempted <= 4:
                        print(e)
                        print("Failed more than 3 times.")
                        sleep_time = 60
                        print(f"Retrying in {sleep_time} seconds")
                        time.sleep(sleep_time)
                        continue
                    else:
                        print(e)
                        print("Failed more than 4 times.")
                        completed = True
                        continue
            completed = True


    return responses


@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):
    model_type = get_alias(cfg.model.watermark_overrides)

    if '@@' not in cfg.model.watermark_overrides:    ## Get base model comparisons
        watermark_overrides = cfg.model.watermark_overrides.split('@')[1]
    else:
        watermark_overrides = cfg.model.watermark_overrides

    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    if cfg.amlt:
        save_amulet_metadata(cfg)
        cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')
        output_dir = cfg.master_parent
    else:
        output_dir = os.path.join(cfg.master_parent, 'data/alpaca_gaussmark', model_type, watermark_overrides)
        os.makedirs(output_dir, exist_ok=True)



    data_start = time.time()
    base_model_gens = get_base_model_alpaca_generations(model_type)
    watermarked_gens = get_gaussmark_alpaca_generations(model_type, watermark_overrides)
    data_end = time.time()
    print(f"Data loading time: {data_end - data_start:.0f} seconds")




    prompts, is_flipped = process_text(base_model_gens, watermarked_gens, template='alpaca_eval_clf_gpt4_turbo', max_generations=cfg.other_gen.alpaca_max_samples)







    client, deployment_name = get_client(cfg.other_gen.oai_api_version, cfg.other_gen.oai_model_name, cfg.other_gen.oai_model_version, instance='msrne/shared')
    responses = get_responses(client, deployment_name, prompts, is_flipped, save_path=os.path.join(output_dir, 'responses.json'), save_interval=10)


    print(f"Saving responses to {output_dir}")
    with open(os.path.join(output_dir, f'responses.json'), 'w') as f:
        json.dump(responses, f)
    # with open(os.path.join(output_dir, f'flipped.json'), 'w') as f:
        # json.dump(is_flipped, f)
    

    
    watermarked_wins, base_wins = get_wins(responses)
    wins = watermarked_wins / (watermarked_wins + base_wins) 
    print(f"Watermarked model wins {wins:.2f} of the time")
    print(f"Saving wins to {output_dir}...")
    # with open(os.path.join(output_dir, f'wins.json'), 'w') as f:
    #     outdict = {
    #         'watermarked_wins': watermarked_wins,
    #         'base_wins': base_wins,
    #         'total': watermarked_wins + base_wins,
    #         'bias': cfg.other_gen.bias,            
    #     }
    #     json.dump(outdict, f)













if __name__ == "__main__":
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"Master time: {master_end - master_start:.0f} seconds")




import os
from openai import AzureOpenAI, AuthenticationError
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential,get_bearer_token_provider

import datasets
import hydra
from omegaconf import OmegaConf
import time
from utils import save_amulet_metadata
from tqdm import tqdm
import numpy as np  
import json
import re
from datetime import datetime

def get_alpaca_generations(parent):
    """
    Loads generations from the parent directory.  Expects the following structure:
    parent
    ├── base_model_dir
    │   └── base_model

    ├── watermarked_dir
    │   └── watermarked

    """

    base_model_gens = {}
    watermarked_gens = {}
    for dir in os.listdir(parent):

        model = dir.split('_')[4]
        params = '_'.join(dir.split('_')[5:])
        params = re.sub(r'^_+', '', params)
        params = re.sub(r'__+', '___', params)

        data_parent = os.path.join(parent, dir)
        dirname = os.listdir(data_parent)[0]



        if dirname == 'base_model':
            is_watermarked = False
        elif dirname == 'watermarked':
            is_watermarked = True
        else:
            print(f"Skipping {dir}/{dirname}")
            continue
        
        data = datasets.load_from_disk(os.path.join(data_parent, dirname))
        if is_watermarked:
            if not model in watermarked_gens.keys():
                watermarked_gens[model] = {}
            watermarked_gens[model][params] = data
        else:
            base_model_gens[model] = data
            
    return base_model_gens, watermarked_gens




def get_watermarked_alpaca_generations(watermarked_path):
    """
    Given path to an amulet output directory for a given job, returns the generations for the watermarked model.
    """
    watermarked_gens = datasets.load_from_disk(os.path.join(watermarked_path, 'watermarked'))
    return watermarked_gens



def get_base_model_alpaca_generations(model):
    """
    Given a model name, returns the generations for the base model on the alpaca dataset.
    """
    base_alpaca_parent = '/home/blockadam/gaussian-watermarking/amlt/09-16-alpaca-generate'
    if model == 'mistralai/Mistral-7B-v0.3' or model == 'mistral_mistralai/Mistral-7B-v0.3' or model == 'mistral_Base':
        job_name = 'grid_09-16-alpaca-generate_wat_mistralai_Mistral-7B-v0.3_____'
    elif model == 'microsoft/Phi-3-mini-4k-instruct' or  model == 'phi_microsoft/Phi-3-mini-4k-instruct' or model == 'phi_Base':
        job_name = 'grid_09-16-alpaca-generate_wat_microsoft_Phi-3-mini-4k-instruct_____'
    elif model == 'meta-llama/Llama-2-7b-hf' or model == 'llama_meta-llama/Llama-2-7b-hf':
        job_name = 'grid_09-16-alpaca-generate_wat_meta-llama_Llama-2-7b-hf_____'
    elif model == 'meta-llama/Meta-Llama-3.1-8B' or model == 'llama_meta-llama/Meta-Llama-3.1-8B' or model == 'llama_Base':
        job_name = 'grid_09-16-alpaca-generate_wat_meta-llama_Meta-Llama-3.1-8B_____'
    else:
        raise ValueError(f"Model {model} not found")
    
    base_model_gens = datasets.load_from_disk(os.path.join(base_alpaca_parent, job_name, 'base_model'))
    return base_model_gens



def get_seed(run_name):
    """
    given run name, returns the seed
    """
    found = re.search('_see_(\d+)', run_name)
    seed = found.group(1)
    return seed

def get_all_alpaca_gens(run_paths):
    """
    Given a list of paths to alpaca generations, returns the generations for all models.
    """
    base_model_gens = {}
    watermarked_gens = {}
    for path in run_paths:
        try:
            watermarked_datum = get_watermarked_alpaca_generations(path)
            info = watermarked_datum['generator'][0]
            model, rank, param_info, var = info.split('@___@')
            layer, mlp_type, _ = param_info.split('@@@')
            if not model in watermarked_gens.keys():
                watermarked_gens[model] = {}
                base_model_gens[model] = get_base_model_alpaca_generations(model)
            

            key = f'{rank}___{layer}___{mlp_type}___weight___{var}'
            watermarked_gens[model][key] = watermarked_datum
        except: ## Getting baseline reads of the unwatermarked model's generations iwth different seeds
            watermarked_datum = datasets.load_from_disk(os.path.join(path, 'watermarked'))
            model = watermarked_datum['generator'][0]
            run_name = os.path.basename(path)
            seed = get_seed(run_name)
            
            if not model in watermarked_gens.keys():
                watermarked_gens[model] = {}
                base_model_gens[model] = get_base_model_alpaca_generations(model)
            watermarked_gens[model][seed] = watermarked_datum

    
    return base_model_gens, watermarked_gens



def get_watermarked_overrides(watermarked_path):
    """
    Given path to an amulet output directory for a given job, returns the watermark overrides for the watermarked model.
    """
    watermarked_datum = get_watermarked_alpaca_generations(watermarked_path)
    return watermarked_datum['generator'][0]


TEMPLATES = {
    'alpaca_eval_clf_gpt4_turbo': "<|im_start|>system\nYou are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.\n<|im_end|>\n<|im_start|>user\nI require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.\n\n## Instruction\n\n{\n    'instruction': {instruction},\n}\n\n## Model Outputs\n\nHere are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.\n\n{\n    {\n        'model_identifier': 'm',\n        'output': {output_1}\n    },\n    {\n        'model_identifier': 'M',\n        'output': {output_2}\n    }\n}\n\n## Task\n\nEvaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.\n\n## Best Model Identifier\n<|im_end|>\n"
}




def _get_role(interaction):
    """
    Get the role of the interaction.
    """

    if interaction.startswith('<|im_start|>'):
        interaction = interaction[len('<|im_start|>'):]
    return interaction.split('\n')[0].strip().lower()

def _get_content(interaction):
    """
    Get the content of the interaction
    """
    return '\n'.join(interaction.split('\n')[1:]).strip()

def preprocess_template(template):
    """
    Turn alpaca eval template into chat format.
    """

    raw_interactions = template.split('<|im_end|>')

    interactions = []
    for interaction in raw_interactions:
        interaction = interaction.strip()
        role = _get_role(interaction)
        content = _get_content(interaction)
        interactions.append({
            'role': role,
            'content': content
        })
    return interactions



def _process_example(instruction, base_output, watermarked_output, template, flipped):
    """
    Gets the chat format for the example.  Template is assumed to be a list of dictionaries with 'role' and 'content' keys of length 2.  The content key is assumed to have a placeholder for the instruction, base_output, and watermarked_output.
    """

    chat = template[1]['content']
    chat = chat.replace('{instruction}', instruction)
    if flipped:
        chat = chat.replace('{output_1}', watermarked_output)
        chat = chat.replace('{output_2}', base_output)
    else:
        chat = chat.replace('{output_1}', base_output)
        chat = chat.replace('{output_2}', watermarked_output)
    
    processed_example = [
        template[0],
        {
            'role': template[1]['role'],
            'content': chat
        }
    ]
    return processed_example



def process_text(base_model_gens, watermarked_gens, template='alpaca_eval_clf_gpt4_turbo', max_generations=None):
    """
    Processes the text generations from the base model and watermarked model.  Templates are from alpaca_eval.  Available templates are:

    - alpaca_eval_clf_gpt4_turbo (https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt)
    """

    template = TEMPLATES[template]

    template = preprocess_template(template)

    prompts = []
    
    num_generations = len(base_model_gens)
    if max_generations is not None:
        num_generations = min(num_generations, max_generations)

    is_flipped = np.random.choice([True, False], num_generations, replace=True).tolist()
    for i in range(num_generations):
        instruction = base_model_gens[i]['instruction']
        base_output = base_model_gens[i]['output']
        watermarked_output = watermarked_gens[i]['output']
        flipped = is_flipped[i]
        prompts.append(_process_example(instruction, base_output, watermarked_output, template, flipped))
    
    return prompts, is_flipped


def get_client():
    RESOURCE_NAME = 'gcraoai7sw1'
    ENDPOINT = f"https://{RESOURCE_NAME}.openai.azure.com/"
    API_VERSION = "2024-02-15-preview"  # Replace with the appropriate API version

    # ChainedTokenCredential example borrowed from
    # https://github.com/technology-and-research/msr-azure/blob/main/knowledge-base/how-to/Access-Storage-Without-     Keys-in-Azure-ML.md
    # Attribution: AI4Science
    azure_credential = ChainedTokenCredential(AzureCliCredential(),
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
    # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-     authentication#compute-cluster
    # In case it is not set the ManagedIdentityCredential will
    # default to using the system-assigned managed identity, if any.
    managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
    )
    )


    token_provider = get_bearer_token_provider(azure_credential,
        "https://cognitiveservices.azure.com/.default")
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        azure_ad_token_provider=token_provider
    ) 
    return client



def get_responses(prompts):
    """
    Queries openai API to get responses to the prompts.
    """
    client = get_client()
    responses = []

    bar = tqdm(enumerate(prompts))
    for _,prompt in bar:
        
        completed = False
        while not completed:
            try:
                completion = client.chat.completions.create(
                    model= "gpt-4o-gs", # 'gpt-4-turbo', # "gpt-4o-gs", #Make sure update your model accordingly
                    messages=prompt
                )

                
                responses.append(completion.choices[0].message.content)
                completed = True

            except Exception as e:
                # raise e
                if e is AuthenticationError:
                    raise e
                elif e is KeyboardInterrupt:
                    raise e
                else:
                    print(e)
                    sleep_time = 10
                    print(f"Retrying in {sleep_time} second")
                    time.sleep(sleep_time)
                    continue


    return responses


def get_param(watermark_override):
    """
    convert watermark override into key for dictionary
    """

    # model_name, rank, layer, mlp_type, weight, var = watermark_override.split('@___@')
    # param = f'{rank}___{layer}___{mlp_type}___{weight}___{var}'
    model_name, rank, param_type, var = watermark_override.split('@___@')
    layer, mlp_type, weight = param_type.split('@@@')
    param = f'{rank}___{layer}___{mlp_type}___{weight}___{var}'
    return model_name, param



def get_model_name_baseline(path):
    """
    given the path to the baseline, returns the model name
    """
    if 'Llama-3' in path:
        return 'llama_Base'
    elif 'Llama-2' in path:
        return 'llama_Base'
    elif 'Phi' in path:
        return 'phi_Base'
    elif 'Mistral' in path:
        return 'mistral_Base'
    else:
        raise ValueError(f"Model not found in {path}")




def get_wins(responses, flipped):
    """
    Get the wins from the responses.
    """
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



def get_current_date_mm_dd():
    return datetime.now().strftime("%m-%d")





@hydra.main(config_path="../hydra_configs", config_name="hf_master", version_base=None)
def main(cfg):

    if cfg.model.watermark_overrides is None:
        cfg.model.watermark_overrides = get_watermarked_overrides(cfg.lm_eval.generations_parent)

    master_start = time.time()
    np.random.seed(cfg.seed + 10101)


    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    if cfg.amlt:
        cfg.lm_eval.model_path = None ## TODO: Load from storage instead?
        save_amulet_metadata(cfg)
        cfg.master_parent = os.environ.get('AMLT_OUTPUT_DIR')
        output_dir = cfg.master_parent
    # else:
    #     model_type = get_model_type(cfg.model.name)
    #     output_dir = os.path.join(cfg.master_parent, 'data/alpaca', model_type)
    #     os.makedirs(output_dir, exist_ok=True)

    if 'wandb' in cfg and 'key' in cfg.wandb:
            # print cfg without wandb.key, then restore
            cached_key, cfg.wandb.key = cfg.wandb.key, None
            print('Config:', cfg)
            cfg.wandb.key = cached_key
    else:
        print('Config:', cfg)




    # base_model_gens, watermarked_gens = get_alpaca_generations(os.path.join(cfg.master_parent, cfg.lm_eval.generations_parent))
    parent = '/'.join(cfg.lm_eval.generations_parent.split('/')[:-1])
    paths = [os.path.join(parent, path) for path in os.listdir(parent)]



    # base_model_gens, watermarked_gens = get_all_alpaca_gens(paths)


    try:
        model_name, param = get_param(cfg.model.watermark_overrides)
    except:
        model_name = get_model_name_baseline(cfg.lm_eval.generations_parent)
        param = get_seed(os.path.basename(cfg.lm_eval.generations_parent))
    
    # base_model_gens = base_model_gens[model_name]
    # watermarked_gens = watermarked_gens[model_name][param]

    base_model_gens = get_base_model_alpaca_generations(model_name)
    watermarked_gens = get_watermarked_alpaca_generations(cfg.lm_eval.generations_parent)

    prompts, is_flipped = process_text(base_model_gens, watermarked_gens, template='alpaca_eval_clf_gpt4_turbo', max_generations=cfg.lm_eval.max_generations)

    responses = get_responses(prompts)    

    output_dir = os.path.join(cfg.master_parent, cfg.lm_eval.outputs_parent, cfg.lm_eval.generations_parent.replace('/', '_'),  get_current_date_mm_dd(), cfg.model.watermark_overrides)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving responses to {output_dir}")
    with open(os.path.join(output_dir, f'responses.json'), 'w') as f:
        json.dump(responses, f)
    with open(os.path.join(output_dir, f'flipped.json'), 'w') as f:
        json.dump(is_flipped, f)
    
    with open(os.path.join(output_dir, f'responses.json'), 'r') as f:
        responses = json.load(f)
    
    watermarked_wins, base_wins = get_wins(responses, is_flipped)
    wins = watermarked_wins / (watermarked_wins + base_wins) 
    print(f"Watermarked model wins {wins:.2f} of the time")
    print(f"Saving wins to {output_dir}...")
    with open(os.path.join(output_dir, f'wins.json'), 'w') as f:
        outdict = {
            'param': cfg.model.watermark_overrides,
            'watermarked_wins': watermarked_wins,
            'base_wins': base_wins,
            'total': watermarked_wins + base_wins,
            'path': cfg.lm_eval.generations_parent
        }
        json.dump(outdict, f)
    






    master_end = time.time()
    print(f"\nMaster script took {master_end - master_start:.0f} seconds\n")    





if __name__ == "__main__":
    main()

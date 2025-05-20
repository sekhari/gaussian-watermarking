import os
import time
import re
import pandas as pd
from eval_alpaca_generations import get_current_date_mm_dd, get_watermarked_overrides

def get_watermark_overrrides_from_dir(dirname):
    
    found = re.search('wat_', dirname)
    overrides = dirname[found.end():]
    

    if dirname.endswith('_____'):
        is_watermarked = False
        model_name, *others = overrides.split('_____')
    else:
        is_watermarked = True
        model_name, rank, param_info, variance = overrides.split('_____')
        layer, proj_type, param_type = param_info.split('___')
    
    model_name = model_name.split('_')[1]

    if is_watermarked:
        overrides_name = '@___@'.join([model_name, rank, layer, proj_type, param_type, variance])
    else:
        overrides_name = model_name
        

    return overrides_name, is_watermarked


def get_good_model_paths(good_models_path):

    good_models = pd.read_csv(good_models_path)
    paths = []
    for ind in range(len(good_models)):
        path = os.path.join('/home/blockadam/gaussian-watermarking/amlt', good_models['exp_name'][ind], good_models['run_name'][ind])
        paths.append(path)
    return paths


def get_output_path_from_dir(dirname):

    master_parent = '/home/blockadam/gaussian-watermarking/'
    outputs_parent = 'alpaca_win_rate'

    watermark_overrides = get_watermarked_overrides(dirname)
    output_dir = os.path.join(master_parent, outputs_parent, dirname.replace('/', '_'),  get_current_date_mm_dd(), watermark_overrides)
    return output_dir



def main():

    # parent = '/home/blockadam/gaussian-watermarking/amlt/09-16-alpaca-generate'
    # parents = [
    #     '/home/blockadam/gaussian-watermarking/amlt/09-18-watermark-r64',
    #     '/home/blockadam/gaussian-watermarking/amlt/09-18-watermark-r0'
    # ]
    # paths_to_runs = []
    # for parent in parents:
    #     for dirname in os.listdir(parent):
    #         path = os.path.join(parent, dirname)
    #         paths_to_runs.append(path)


    # good_models_path = '/home/blockadam/gaussian-watermarking/good_watermarked_models.csv'
    good_models_path = '/home/blockadam/gaussian-watermarking/good_models.csv'
    paths = get_good_model_paths(good_models_path)

    

    # Getting baselines
    # parent = '/home/blockadam/gaussian-watermarking/amlt/09-19-alpaca-generate-baseline'
    # # paths = []
    # for dirname in os.listdir(parent):
    #     path = os.path.join(parent, dirname)
    #     paths.append(path)



    code_path = '/home/blockadam/gaussian-watermarking/src/eval_alpaca_generations.py'


    # overides_prefix = f' amlt=False lm_eval=alpaca model.watermark_overrides='
    overides_prefix = f' amlt=False lm_eval=alpaca lm_eval.generations_parent='

    code_command_prefix = f'python {code_path}' + overides_prefix


    for path in paths:
        try:

            output_dir = get_output_path_from_dir(path)


            if os.path.exists(output_dir):
                print(f"Skipping {path} as output directory already exists: {output_dir}")
                continue

            print(f"\nWorking on {path}\n")

            dirname_start = time.time()


            # watermark_overrides, is_watermarked_model = get_watermark_overrrides_from_dir(dirname)
            # if not is_watermarked_model:
                # print(f"Skipping base model {dirname}")
                # continue

            # command = code_command_prefix + watermark_overrides
            command = code_command_prefix + path
            print(f"Running command: {command}")
            os.system(command)
            dirname_end = time.time()
            print(f"Time taken for {path}: {dirname_end - dirname_start:.0f} seconds")
        except Exception as e:
            if e == KeyboardInterrupt:
                print(f"KeyboardInterrupt: Exiting")
                break
            else:
                print(f"Error with {path}: {e}")

    # for parent in parents:
    #     print(f"\nWorking on parent: {parent}\n")
    #     for dirname in os.listdir(parent):
    #         dirname_start = time.time()
    #         path = os.path.join(parent, dirname)
    #         print(f"\nWorking on {dirname}............")
    #         # watermark_overrides, is_watermarked_model = get_watermark_overrrides_from_dir(dirname)
    #         # if not is_watermarked_model:
    #             # print(f"Skipping base model {dirname}")
    #             # continue

    #         # command = code_command_prefix + watermark_overrides
    #         command = code_command_prefix + path
    #         print(f"Running command: {command}")
    #         os.system(command)
    #         dirname_end = time.time()
    #         print(f"Time taken for {dirname}: {dirname_end - dirname_start:.0f} seconds")
       



if __name__ == '__main__':
    master_start = time.time()
    main()
    master_end = time.time()
    print(f"Total time: {master_end - master_start:.0f} seconds")
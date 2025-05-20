#!/bin/bash







echo "Running alternative base model generations"

## echo "Running Alpaca Evaluation for phi with seed 1339"
## python src/alpaca_eval_new.py amlt=False model.watermark_overrides=phi@Base_1339

# echo "Running Alpaca Evaluation for phi with seed 1761"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=phi@Base_1761


# echo "Running Alpaca Evaluation for phi with seed 1341"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=phi@Base_1341

## echo "Running Alpaca Evaluation for llama with seed 1339"
## python src/alpaca_eval_new.py amlt=False model.watermark_overrides=llama@Base_1339


# echo "Running Alpaca Evaluation for llama with seed 1341"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=llama@Base_1341


# echo "Running Alpaca Evaluation for llama with seed 1761"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=llama@Base_1761


## echo "Running Alpaca Evaluation for mistral with seed 1339"
## python src/alpaca_eval_new.py amlt=False model.watermark_overrides=mistral@Base_1339


# echo "Running Alpaca Evaluation for mistral with seed 1341"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=mistral@Base_1341


# echo "Running Alpaca Evaluation for mistral with seed 1761"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=mistral@Base_1761



echo "Running Gaussmark Alpaca Evaluation"

# echo "Running Alpaca Evaluation with watermark override: mistralai-Mistral-7B-v0.3@___@0@___@20@@@up_proj@@@weight@___@1e-05"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=mistralai-Mistral-7B-v0.3@___@0@___@20@@@up_proj@@@weight@___@1e-05

# echo "Running Alpaca Evaluation with watermark override: mistralai-Mistral-7B-v0.3@___@1024@___@30@@@up_proj@@@weight@___@1e-05"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=mistralai-Mistral-7B-v0.3@___@1024@___@30@@@up_proj@@@weight@___@1e-05


echo "Running Alpaca Evaluation with watermark override: meta-llama-Meta-Llama-3.1-8B@___@0@___@28@@@up_proj@@@weight@___@0.0003"
python src/alpaca_eval_new.py amlt=False model.watermark_overrides=meta-llama-Meta-Llama-3.1-8B@___@0@___@28@@@up_proj@@@weight@___@0.0003

echo "Running Alpaca Evaluation with watermark override: meta-llama-Meta-Llama-3.1-8B@___@512@___@29@@@down_proj@@@weight@___@0.0001"
python src/alpaca_eval_new.py amlt=False model.watermark_overrides=meta-llama-Meta-Llama-3.1-8B@___@512@___@29@@@down_proj@@@weight@___@0.0001



# echo "Running Alpaca Evaluation with watermark override: microsoft-Phi-3-mini-4k-instruct@___@0@___@20@@@down_proj@@@weight@___@0.001"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=microsoft-Phi-3-mini-4k-instruct@___@0@___@20@@@down_proj@@@weight@___@0.001

# echo "Running Alpaca Evaluation with watermark override: microsoft-Phi-3-mini-4k-instruct@___@1024@___@31@@@gate_up_proj@@@weight@___@0.0003"
# python src/alpaca_eval_new.py amlt=False model.watermark_overrides=microsoft-Phi-3-mini-4k-instruct@___@1024@___@31@@@gate_up_proj@@@weight@___@0.0003
#!/bin/bash



echo "Running KGW Alpaca Evaluation"
echo "Llama, Bias 2"
python src/kgw_alpaca_eval.py amlt=False other_gen.bias=2 model.name=meta-llama/Meta-Llama-3.1-8B
echo "Llama, Bias 1"
python src/kgw_alpaca_eval.py amlt=False other_gen.bias=1 model.name=meta-llama/Meta-Llama-3.1-8B


echo "Mistral, Bias 2"
python src/kgw_alpaca_eval.py amlt=False other_gen.bias=2 model.name=mistralai/Mistral-7B-v0.3
echo "Mistral, Bias 1"
python src/kgw_alpaca_eval.py amlt=False other_gen.bias=1 model.name=mistralai/Mistral-7B-v0.3

echo "Phi, Bias 2"
python src/kgw_alpaca_eval.py amlt=False other_gen.bias=2 model.name=microsoft/Phi-3-mini-4k-instruct
echo "Phi, Bias 1"
python src/kgw_alpaca_eval.py amlt=False other_gen.bias=1 model.name=microsoft/Phi-3-mini-4k-instruct
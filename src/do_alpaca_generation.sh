#!/bin/bash





watermark_overrides=('mistralai/Mistral-7B-v0.3@___@0@___@20@@@up_proj@@@weight@___@1e-05' 'meta-llama/Meta-Llama-3.1-8B@___@0@___@28@@@up_proj@@@weight@___@0.0003' 'microsoft/Phi-3-mini-4k-instruct@___@0@___@20@@@down_proj@@@weight@___@0.001' 'mistralai/Mistral-7B-v0.3@___@1024@___@30@@@up_proj@@@weight@___@1e-05' 'meta-llama/Meta-Llama-3.1-8B@___@512@___@29@@@down_proj@@@weight@___@0.0001' 'microsoft/Phi-3-mini-4k-instruct@___@1024@___@31@@@gate_up_proj@@@weight@___@0.0003' 'microsoft/Phi-3-mini-4k-instruct' 'mistralai/Mistral-7B-v0.3' 'meta-llama/Meta-Llama-3.1-8B')
seed=1337

echo "Running Alpaca Evaluation"
for watermark_override in "${watermark_overrides[@]}"; do
    python src/alpaca_new.py amlt=False seed=$seed model.watermark_overrides=${watermark_override}
done



models=('mistralai/Mistral-7B-v0.3' 'meta-llama/Meta-Llama-3.1-8B' 'microsoft/Phi-3-mini-4k-instruct')
seeds=(1339 1341 1761)
echo "Running alternative base model generations"
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running $model with seed $seed"
        python src/alpaca_new.py amlt=False seed=$seed model.name=$model model.watermark_overrides=$model
    done
done
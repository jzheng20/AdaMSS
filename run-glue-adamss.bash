#!/bin/bash
#export HF_HOME="/home/jzheng2023/work/.cache"
 

 

valid_data_names=(
    "cola"
    "mrpc"
    "qnli"
    "rte"
    "stsb"
    "mrpc"
    "sst2"
)

valid_models=(
    "roberta-base" 
    "roberta-large"
)

valid_seeds=(
    0
    11111
    22222
    33333
    44444 
)

# Parse positional arguments with defaults
data_name="${1:-sst2}" 
model="${2:-roberta-large}" 
head_lr="${3:-0.0005}"
fft_lr="${4:-0.001}"
weight_decay="${5:-0.0}"
ri="${6:-1}"
seed="${7:-0}"


# Validate dataset 
if [[ ! " ${valid_data_names[@]} " =~ " ${data_name} " ]]; then
    echo "Error: '${data_name}' is not the valid dataset: ${valid_data_names[*]}"
    exit 1
fi

# Validate dataset 
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: '${model}' is not the valid model: ${valid_models[*]}"
    exit 1
fi

# Validate dataset 
if [[ ! " ${valid_seeds[@]} " =~ " ${seed} " ]]; then
    echo "Error: '${seed}' is not the valid seed: ${valid_seeds[*]}"
    exit 1
fi

sanitized_model=${model//\//_}
echo "================================================================="
echo "Running NLU_GLUE_adamss with:"
echo "Data: $data_name" 
echo "Model: $model" 
echo "head_lr: $head_lr" 
echo "fft_lr: $fft_lr" 
echo "weight_decay: $weight_decay" 
echo "r: $ri" 
echo "seed: $seed" 
echo "Check NLU_GLUE_adamss_${sanitized_model}_${data_name}_head_lr${head_lr}_fft_lr${fft_lr}_weight_decay${weight_decay}_seed${seed}_r${ri}.log for details."
echo "================================================================="


mkdir -p log


#nohup env 
CUDA_VISIBLE_DEVICES=0 python NLU_GLUE_adamss.py \
    --model_name_or_path FacebookAI/${model} \
    --task "$data_name" \
    --max_length 512 \
    --head_lr "$head_lr" \
    --fft_lr "$fft_lr" \
    --weight_decay "$weight_decay" \
    --num_epoch 100 \
    --bs 32  \
    --scale 49.0 \
    --seed "$seed" \
    --adamss_R 100\
    --adamss_K 10\
    --tt 3\
    --adamss_ri "$ri"\
    --target_KK 40 \
    --init_warmup 5 \
    --final_warmup 95 \
    --mask_interval 10 \
    --MODE_SA True
    2>&1 | tee "log/NLU_GLUE_adamss_${sanitized_model}_${data_name}_head_lr${head_lr}_fft_lr${fft_lr}_weight_decay${weight_decay}_seed${seed}_r${ri}.log"

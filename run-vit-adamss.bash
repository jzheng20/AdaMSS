
#!/bin/bash
#export HF_HOME="/home/jzheng2023/work/.cache"

 
 

valid_data_names=(
    "cars"
    "cifar10"
    "cifar100"
    "eurosat"
    "fgvc"
    "pets"
    "resisc"
)

valid_models=(
    "vit-base-patch16-224-in21k" 
    "vit-large-patch16-224-in21k"
    "vit-huge-patch14-224-in21k"
)

valid_seeds=(
    7
    77
    777
    7777
    77777 
)

 
data_name="${1:-cars}" 
model="${2:-vit-large-patch16-224-in21k}" 
head_lr="${3:-0.005}"
fft_lr="${4:-0.01}"
weight_decay="${5:-0.1}"
ri="${6:-1}"
seed="${7:-7}"


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
echo "Running exec_adamss with:"
echo "Data: $data_name" 
echo "Model: $model" 
echo "head_lr: $head_lr" 
echo "fft_lr: $fft_lr" 
echo "weight_decay: $weight_decay" 
echo "r: $r" 
echo "seed: $seed" 
echo "Check exec_adamss_${sanitized_model}_${data_name}_head_lr${head_lr}_fft_lr${fft_lr}_weight_decay${weight_decay}_seed${seed}_r${r}.log for details."
echo "================================================================="


mkdir -p log


CUDA_VISIBLE_DEVICES=0 python exec_adamss.py \
    --model-name-or-path google/${model} \
    --dataset_path  \
    --dataset-name cars \
    --target_KK 200 \
    --head_lr "$head_lr" \
    --weight_decay "$weight_decay" \
    --fft_lr "$fft_lr" \
    --num_epochs 10 \
    --mode adamss \
    --adamss_R 100\
    --adamss_K 10\
    --tt 3\
    --seed "$seed" \
    --adamss_ri "$ri"\
    --init_warmup 1 \
    --final_warmup 1000 \
    --mask_interval 100 \
    --MODE_SA False

#!/bin/bash
set -e

MODEL_PATH="path of your Qwen2.5-7B"

export NNODES=1
NODE_RANK=${RANK:-0}
export NODE_RANK
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

STAGE=sft
finetuning_type=full
OUTPUT_DIR_BASE="your save path"
EPOCHS=4.0
PRECISION="bf16"
CUTOFF_LEN=32768
ignore_observation=true
ignore_observation_token=observation

# prepare the dataset file name on your dataset_info.json about the "./AFM-WebAgent-SFT-Dataset".
DATA_DATA="dataset file name from your dataset_info.json"
TEMPALTE=qwen

SWANLAB_API_KEY=xxx
SWANLAB_PROJECT=xxx

LEARNING_RATES=("1.4e-5")
WARMUP_RATIOS=("0.1")
BATCH_SIZES=(1)  # with 2 nodes
GRADIENT_ACCUMULATIONS=(16)

mkdir -p "grid_search_results"
RESULT_FILE="grid_search_results/results_$(date +%Y%m%d_%H%M%S).txt"

START_TIME=$(date +%s)

TOTAL_EXPERIMENTS=$((${#LEARNING_RATES[@]} * ${#WARMUP_RATIOS[@]} * ${#BATCH_SIZES[@]} * ${#GRADIENT_ACCUMULATIONS[@]}))
EXPERIMENT_COUNTER=0

# 网格搜索主循环
for LR in "${LEARNING_RATES[@]}"; do
    for WR in "${WARMUP_RATIOS[@]}"; do
        for BS in "${BATCH_SIZES[@]}"; do
            for GA in "${GRADIENT_ACCUMULATIONS[@]}"; do
                EXPERIMENT_COUNTER=$((EXPERIMENT_COUNTER + 1))
                EXPERIMENT_ID="exp_${EXPERIMENT_COUNTER}_lr${LR}_wr${WR}_bs${BS}_ga${GA}"
                CURRENT_OUTPUT_DIR="${OUTPUT_DIR_BASE}/${EXPERIMENT_ID}"
                SWANLAB_EXPERIMENT_NAME="qwen2.5_32b_${EXPERIMENT_ID}"
                
                EXPERIMENT_START_TIME=$(date +%s)
                
                llama_factory_status=0
                llama_factory_output=$(llamafactory-cli train \
                 --deepspeed examples/deepspeed/ds_z3_config.json \
                  --model_name_or_path "$MODEL_PATH" \
                  --trust_remote_code \
                  --stage $STAGE \
                  --do_train \
                  --finetuning_type $finetuning_type \
                  --dataset $DATA_DATA \
                  --template $TEMPALTE \
                  --cutoff_len $CUTOFF_LEN \
                  --output_dir "$CURRENT_OUTPUT_DIR" \
                  --per_device_train_batch_size "$BS" \
                  --gradient_accumulation_steps "$GA" \
                  --learning_rate "$LR" \
                  --warmup_ratio "$WR" \
                  --num_train_epochs "$EPOCHS" \
                  --${PRECISION} \
                  --save_strategy epoch \
                  --save_only_model true \
                  --report_to swanlab \
                  --logging_steps 10 \
                  --use_swanlab \
                  --swanlab_api_key $SWANLAB_API_KEY \
                  --swanlab_project $SWANLAB_PROJECT \
                  --ignore_observation_token $ignore_observation_token \
                  --ignore_observation $ignore_observation 2>&1) || llama_factory_status=$?
                
                EXPERIMENT_END_TIME=$(date +%s)
                EXPERIMENT_DURATION=$((EXPERIMENT_END_TIME - EXPERIMENT_START_TIME))
                
                if [ $llama_factory_status -eq 0 ]; then
                    
                    LOSS=$(echo "$llama_factory_output" | grep -oP 'loss: \K[\d.]+' | tail -1)
                    PERPLEXITY=$(echo "$llama_factory_output" | grep -oP 'perplexity: \K[\d.]+' | tail -1)
                    
                    echo "$LR,$WR,$BS,$GA,$LOSS,$PERPLEXITY,$EXPERIMENT_DURATION" >> grid_search_results/results_summary.csv
                else
                    echo "$llama_factory_output" | tee -a $RESULT_FILE
                fi
                
                ELAPSED_TIME=$((EXPERIMENT_END_TIME - START_TIME))
                AVG_TIME_PER_EXP=$((ELAPSED_TIME / EXPERIMENT_COUNTER))
                REMAINING_EXPS=$((TOTAL_EXPERIMENTS - EXPERIMENT_COUNTER))
                ESTIMATED_REMAINING=$((AVG_TIME_PER_EXP * REMAINING_EXPS))
                
            done
        done
    done
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
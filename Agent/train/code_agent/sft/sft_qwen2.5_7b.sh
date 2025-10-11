#!/bin/bash
set -e

MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"

export NNODES=1 # Nodes number for training
NODE_RANK=${RANK:-0}
export NODE_RANK
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
STAGE=sft
finetuning_type=full
OUTPUT_DIR="saves/AFM-CodeAgent-7B-sft"
LEARNING_RATE="3e-5"
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
EPOCHS=2.0
PRECISION="bf16"
CUTOFF_LEN=32768
ignore_observation=true
ignore_observation_token=observation

# datasets key of the `LLaMA-Factory/data/dataset_info.json`
DATA_DATA=code_agent_sft
TEMPALTE=qwen

# Swanlab
SWANLAB_API_KEY=xxx
SWANLAB_PROJECT=xxx

# train
echo "Training start: $MODEL_PATH -> $OUTPUT_DIR"
llamafactory-cli train \
  --deepspeed "./examples/deepspeed/ds_z3_config.json" \
  --model_name_or_path "$MODEL_PATH" \
  --trust_remote_code \
  --stage $STAGE \
  --do_train \
  --finetuning_type $finetuning_type \
  --dataset $DATA_DATA \
  --template $TEMPALTE \
  --cutoff_len $CUTOFF_LEN \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --${PRECISION} \
  --save_only_model true \
  --report_to swanlab \
  --use_swanlab \
  --swanlab_api_key $SWANLAB_API_KEY \
  --swanlab_project $SWANLAB_PROJECT \
  --ignore_observation_token $ignore_observation_token \
  --ignore_observation $ignore_observation

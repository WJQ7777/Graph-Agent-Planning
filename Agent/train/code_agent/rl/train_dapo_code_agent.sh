set -x

ulimit -n 65535
# =====================================================================================================================
#                                      Param
# =====================================================================================================================
ACTOR_LR=1e-6
TRAIN_BS=256
PPO_MINI_BS=32
GEN_BS=256
EPOCHS=100
STEPS=2000
N=8
PPO_MICRO_BSZ_PER_GPU=2
LOG_PROB_MICRO_BSZ_PER_GPU=16
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
# context window
max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 16))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# performance related param
SP_SIZE=4
GEN_TP=4
use_dynamic_bsz=False
# =====================================================================================================================
#                                      Env
# =====================================================================================================================
# NOTE: We recommend to use wandb as log backend. Export your own wandb project and key to use it. Remember to turn on wandb_mode if you sync online.
export WANDB_MODE="offline"
CURRENT_DIR=$(pwd)
export NNODES=8 # "your GPU group number"
export PROJECT_NAME="agent_foundation_models"
SAVE_MODEL_FOLDER="${CURRENT_DIR}/experiments"  # your save model folder
export EXPERIMENT_NAME="DAPO-QWEN32B-CodeAgent"
export BASE_MODEL="${CURRENT_DIR}/AFM/models/web_agent/AFM-CodeAgent-32B-sft"   # your train model path
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
TRAIN_DATASETS="${CURRENT_DIR}/AFM/data/web_agent/AFM-CodeAgent-RL-Dataset/CodeAgentRLDataset.parquet"   # your train dataset
VAL_DATASETS="your val datasets"
# =====================================================================================================================
#                                      Tool
# =====================================================================================================================
# code tool
CODE_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/code_tool_config/code_executor.yaml"
# search tools
SEARCH_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/search_tool_config/training_servers_config.yaml"
# afm tools
AFM_CONFIG="${CURRENT_DIR}/verl/verl/tools/config/afm_tool_config/afm_tool_config.yaml" 
# =====================================================================================================================
#                                      Train
# =====================================================================================================================
cd verl
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=true \
    data.train_files=[\"${TRAIN_DATASETS}\"] \
    data.val_files=[\"${VAL_DATASETS}\"] \
    data.train_batch_size="${TRAIN_BS}" \
    data.gen_batch_size="${GEN_BS}" \
    data.val_batch_size=4096 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.shuffle=true \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']" \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.rollout.max_model_len=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BSZ_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.logger=['wandb','tensorboard'] \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs="${EPOCHS}" \
    trainer.total_training_steps="${STEPS}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${SAVE_MODEL_FOLDER}/${EXPERIMENT_NAME}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_turns=8 \
    +actor_rollout_ref.rollout.multi_turn.format=qwen \
    actor_rollout_ref.rollout.multi_turn.use_xml_tool_parser=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CODE_CONFIG" \
    reward_model.reward_manager="afm" \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
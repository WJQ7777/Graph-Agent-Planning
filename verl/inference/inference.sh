#!/usr/bin/env bash
set -x

ulimit -n 65535
# =====================================================================================================================
#                                      Param
# =====================================================================================================================
# context window
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 30))
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
# tool call
max_turns=10
# performance related param
TP=2
# =====================================================================================================================
#                                      Env
# =====================================================================================================================
NNODES=1
MODEL_PATH="/home/jiaqi/Agent_Foundation_Models/AFM/models/AFM-MHQA-Agent-3B-sft"
OUTPUT_PATH="/home/jiaqi/Agent_Foundation_Models/experiments/inference"
DATASETS="/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/test_benchmarks/nq_full_first_10.parquet"
export VLLM_ATTENTION_BACKEND=XFORMERS
export EXPERIMENT_NAME="afm_inference"
# =====================================================================================================================
#                                      Tool
# =====================================================================================================================
# code tool
CODE_CONFIG="./verl/tools/config/code_tool_config/code_executor.yaml"
# search tools
SEARCH_CONFIG="./verl/tools/config/search_tool_config/training_servers_config.yaml"
# wiki tools
WIKI_SEARCH="./verl/tools/config/search_tool_config/wiki_rag_config.yaml"
# afm tools
AFM_CONFIG="./verl/tools/config/afm_tool_config/afm_tool_config.yaml" 
# =====================================================================================================================
#                                      Inference
# =====================================================================================================================
cd verl
python3 -m verl.trainer.main_generation \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=2 \
    data.path=[\"${DATASETS}\"] \
    data.prompt_key=prompt \
    data.batch_size=512 \
    data.n_samples=1 \
    data.output_path=${OUTPUT_PATH} \
    data.return_raw_chat=true \
    data.experiment_name=${EXPERIMENT_NAME} \
    data.filter_overlong_prompts=false \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    model.path=${MODEL_PATH} \
    rollout.name=sglang_async \
    rollout.temperature=1.0 \
    rollout.top_p=0.95 \
    rollout.prompt_length=${max_prompt_length} \
    rollout.response_length=${max_response_length} \
    rollout.tensor_model_parallel_size=${TP} \
    rollout.gpu_memory_utilization=0.6 \
    rollout.max_num_batched_tokens=65536 \
    rollout.enforce_eager=False \
    rollout.free_cache_engine=False \
    rollout.max_model_len=${infer_ppo_max_token_len} \
    rollout.multi_turn.enable=true \
    rollout.multi_turn.max_turns=${max_turns} \
    rollout.multi_turn.use_xml_tool_parser=true \
    rollout.multi_turn.tool_config_path="$WIKI_SEARCH" \
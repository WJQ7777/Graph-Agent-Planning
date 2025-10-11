# Parallel-Agent-Workflow-Planning
This is the repository for our paper "Parallel Agent Workflow Planning with Reinforcement Learning". We propose a new paradigm for LLM reasoning that enables parallel workflow planning and end-to-end RL trianing within a single model. This Work was done during TensorOpera AI internship.

## Training
### Supervised Fine-tuning
#### 1. Env Setup
```bash
conda create -n llama_factory python=3.10
conda activate llama_factory
pip install deepspeed
pip install swanlab
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
```

#### 2. Prepare SFT Dataset
Download SFT Dataset for Web/MHQA/Code Agent:
```py 
python ./Agent/data/web_agent/download.py 
python ./Agent/data/mhqa_agent/download.py 
python ./Agent/data/code_agent/download.py 
```

Add the downloaded dataset filepath to `LLaMA-Factory/data/dataset_info.json`, for example:
```json
"code_agent_sft": {
  "file_name": "path/to/downloaded/WebAgent-SFT-Dataset/WebAgentSFTDataset.json"
}
```

#### 3. Start training with default parameters
The training scripts are list in `./train`. 
Example of sft for code agent:
```bash
bash ./Agent/train/code_agent/sft/sft_qwen2.5_3b.sh
```

Note `DATA_DATA` in the training bash script should be the key in `LLaMA-Factory/data/dataset_info.json`, like `web_agent_sft`, `mhqa_agent_sft`, `code_agent_sft`.

Logs output to output_dir/training.log. We use [SwanLab](https://swanlab.cn/) for visualization (requires setup):
```bash
--swanlab_api_key xxx  # Your SWANLAB_API_KEY
--swanlab_project xxx  # Your SWANLAB_PROJECT
```

Key Configurable Parameters
```
ignore_observation=true # Whether to mask content within special tokens
ignore_observation_token=observation # Specify special token
```
**Note: Check if special tokens are properly masked and data length is appropriate after starting.**

### Reinforement Learning
#### 1. Env Setup
```bash
# Create virtual environment. 
conda create -n afm python=3.10.14 -y
conda activate afm

# Phase 1
pip install symeval@git+https://github.com/tongyx361/symeval.git@54c1a844ea4a6db486c5af8b5b4d2f383224a83b
pip install latex2sympy2==1.9.1
pip install --force-reinstall antlr4-python3-runtime==4.9.3

# Phase 2
cd verl
pip install -r requirements.txt

# Phase 3
pip install --force-reinstall protobuf==5.29.5
pip install --force-reinstall --no-deps grpcio-status==1.71.0 selenium==4.33.0

# Phase 4
cd ..
git clone https://github.com/NVIDIA/apex.git  
cd apex
python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

# Phase 5
cd verl
pip install -r requirements_sglang.txt
cd ..
```


#### 2. Tool usage
##### 2.1 Search Servers
We have developed two server-side components to support web interactions:
- A web search server
- A page crawling server

For detailed deployment instructions, please refer to `Agent/tool_servers/tool_server_readme.md`.


##### 2.2 Code Server(s)
Our Python executor leverages the powerful local isolation sandbox capabilities provided by [nsjail](https://github.com/google/nsjail). We greatly appreciate the nsjail project for enabling secure code execution.

To use this feature during training, you need to:

1. Clone and build nsjail
    ```bash
    git clone https://github.com/google/nsjail.git
    cd nsjail
    make
    ```
2. Add the absolute path to the nsjail_path in code tool configuration file `verl/verl/tools/config/code_tool_config/code_executor.yaml`:
   ```yaml
   nsjail_path: /abs_path/to/your/nsjail/nsjail
   ```


#### 4. Configuration
1. Edit the `environment.sh` file and fill in your API keys and other required credentials
2. Apply the environment settings:
```bash
source environment.sh
```

#### 5. Dataset Processing
The `./Agent/data/README.md` contains scripts and instructions for processing search agent model related data.

For code agent model, the validation datasets are already provided in the `./Agent/data/code_agent/code_math_benchmarks` folder, with corresponding processing instructions available in `./Agent/data/code_agent/code_math_benchmarks/README.md`.

The final web_agent and mhqa_agent dataset format is shown below and stored in .parquet: 
```python
{
    "data_source": data_source,
    "prompt": [
        {"role": "user", "content": sys_prompt + question}
    ],
    "reward_model": {
        "ground_truth": {"target": answer}
    },
    "extra_info": {
        "need_tools_kwargs": True,
        "question": question,
        "answer": answer,
        "tools_kwargs": tools_kwargs
    }
}
```


#### 6. Training
To start a training run:

1. All Agentic-RL script examples are listed:
    -  Web Agent: `./Agent/train/web_agent/rl/train_dapo_web_agent.sh`
    -  Code Agent: `./Agent/train/code_agent/rl/train_dapo_code_agent.sh`
    -  MHQA Agent: `./Agent/train/mhqa_agent/rl/train_ppo_mhqa_agent.sh`
2. Edit the corresponding script to specify your downloaded dataset and model
3. Make sure you have already fill in the `environment.sh` and source
4. All tool configs are listed and have been specified in training scripts: 
    - web_search and crawl_page: `verl/verl/tools/config/search_tool_config/training_servers_config.yaml`
    - code_executor: `verl/verl/tools/config/code_tool_config/code_executor.yaml`
    - wiki_search: `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml`
    - all_tools: `verl/verl/tools/config/afm_tool_config/afm_tool_config.yaml`
5. Execute the training script like:
```bash
bash ./AFM/train/web_agent/rl/train_dapo_web_agent.sh
```


## Evaluation
### Multi Hop QA (MHQA) Evaluation
1. To evaluate MHQA datasets, you should first download the AFM-MHQA-Agent-7B-rl model and test datasets
2. Transform the test dataset to parquet format.
```bash
cd ./Agent/data/mhqa_agent
bash ./prepare.sh
```
3. Then fill the corresponding dataset and model in scripts below and run
  ```bash
  bash evaluation/inference_mhqa.sh
  ```


### Web Agent Evaluation
1. To evaluate web agent, you should first download the AFM-WebAgent-32B-RL checkpoint (or your own) and test dataset.
2. Set environment variable `source environment.sh`.
3. Set `model_path` in the `run_qwen.sh` script, and serve the model with the following command `./AFM/evaluation/web_agent/run_qwen.sh`. After several minutes, the script will output like `URL Endpoint: http://10.77.225.92:10000/v1`.
4. Choose from available test sets in `./Agent/data/web_agent/test_benchmarks`: gaia, hle, webwalker, browsercomp.
5. Finally, set `URL` in `inference_web_agent.py` according to step3, and execute the python script to start webagent inference and evaluation.

```bash
python ./Agent/evaluation/web_agent/inference_web_agent.py \
    --infile  ./AFM/data/web_agent/test_benchmarks/gaia_dev_103.json \
    --outfile ./Agent/evaluation/web_agent/results/webagent_out.jsonl
```


### Code Agent Evaluation
1. All math and code related evaluation datasets are stored in the `./Agent/data/code_agent/code_math_benchmarks` folder. 
2. Please fill in the downloaded code agent model AFM-CodeAgent-32B-rl and validation datasets in `./Agent/evaluation/code_agent/eval_code_agent.sh`
3. Make sure you have build nsjail code sandbox and fill in the corresponding config. Then run

```bash
bash ./Agent/evaluation/code_agent/eval_code_agent.sh
```

In addition, if you want to evaluate livecodebench datasets, please use the scripts `./Agent/data/code_agent/livecodebench_testcases/download_and_process.py` to generate corresponding testcases. We would like to thank the [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1) for their open-source evaluation code. Our evaluation implementation for math and code training sets was inspired by and adapted from their work.


# Acknowledgement
We would like to express our sincere gratitude to the original authors and contributors of LLaMA-Factory and verl, an excellent open-source project that provided a solid foundation for our work. Our implementation has been adapted from the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [verl-agent](https://github.com/langfengQ/verl-agent) and [AFM](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models). 
For the VeRL framework, we have enhanced it with parallel tool calling in sgalng rollout phase for RL training, along with specified reward design, and related supporting features.
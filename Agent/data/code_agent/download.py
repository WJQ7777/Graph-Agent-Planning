from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds_sft = load_dataset("PersonalAILab/AFM-CodeAgent-SFT-Dataset")
ds_sft.save_to_disk("./AFM-CodeAgent-SFT-Dataset")

ds_rl = load_dataset("PersonalAILab/AFM-CodeAgent-RL-Dataset")
ds_rl.save_to_disk("./AFM-CodeAgent-RL-Dataset")
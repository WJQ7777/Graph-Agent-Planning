from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds_sft = load_dataset("PersonalAILab/AFM-MHQA-Agent-SFT-Dataset")
ds_sft.save_to_disk("./AFM-MHQA-Agent-SFT-Dataset")

ds_rl = load_dataset("PersonalAILab/AFM-MHQA-RL-Dataset")
ds_rl.save_to_disk("./AFM-MHQA-Agent-RL-Dataset")
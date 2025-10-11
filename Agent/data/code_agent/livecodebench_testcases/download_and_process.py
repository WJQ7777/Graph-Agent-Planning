"""
Generate single pkl file of livecodebench testcases for dynamic loading during training.
You can specify release_version, start_date and end_date to generate lcb v5 or lcb v6, etc. 
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks import code_generation

dataset = load_code_generation_dataset()
output_dir = "data/livecodebench_testcases"
save_dataset_as_individual_pkl(dataset, output_dir)



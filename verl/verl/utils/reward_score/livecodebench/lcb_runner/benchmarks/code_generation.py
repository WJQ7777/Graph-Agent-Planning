import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk
import os


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset

def load_code_generation_dataset_from_disk(
    dataset_path: str, 
    start_date=None, 
    end_date=None
) -> list[CodeGenerationProblem]:
    """Load LiveCodeBench dataset from local disk"""
    # Load local dataset using load_from_disk
    dataset = load_from_disk(dataset_path)
    
    # Assume dataset contains 'test' split, modify accordingly if using other splits
    if 'test' in dataset:
        dataset = dataset['test']
    else:
        # If no 'test' split, use the first available split
        dataset = next(iter(dataset.values()))
    
    # Convert to list of CodeGenerationProblem objects
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    
    # Date filtering logic remains the same
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems from {dataset_path}")
    return dataset

def save_dataset_as_individual_pkl(
    dataset: list[CodeGenerationProblem],
    output_dir: str
) -> None:
    """
    Save dataset as individual pkl files by question_id
    
    Args:
        dataset: List of loaded CodeGenerationProblem objects
        output_dir: Target directory to save pkl files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track failed entries
    failed_entries = []
    
    # Iterate through dataset and save each problem
    for problem in dataset:
        try:
            # Use question_id as filename
            file_name = f"{problem.question_id}.pkl"
            file_path = os.path.join(output_dir, file_name)
            
            # Save as pkl file
            with open(file_path, 'wb') as f:
                pickle.dump(problem, f)
                
        except Exception as e:
            # Record failed problem ID and error message
            failed_entries.append((problem.question_id, str(e)))
            print(f"Save failed - ID: {problem.question_id}, Error: {str(e)}")
    
    # Output save result statistics
    total = len(dataset)
    saved = total - len(failed_entries)
    print(f"Save completed! Processed {total} items in total:")
    print(f"Successfully saved: {saved} items")
    if failed_entries:
        print(f"Failed to save: {len(failed_entries)} items")
        print(f"Failed ID examples: {[entry[0] for entry in failed_entries[:3]]}")
        if len(failed_entries) > 3:
            print(f"  ... {len(failed_entries) - 3} more items not listed")

if __name__ == "__main__":
    # dataset = load_code_generation_dataset()
    # Usage example: Replace with your local dataset path
    # Local dataset path
    local_dataset_path = "lcb_lite/lcb_lite_v1_v3"
    
    # Target save directory
    output_dir = "dataprocess/livecodebench_v1_to_v3_single_data"
    
    # Load dataset
    dataset = load_code_generation_dataset_from_disk(local_dataset_path)
    
    # Save as individual pkl files
    save_dataset_as_individual_pkl(dataset, output_dir)

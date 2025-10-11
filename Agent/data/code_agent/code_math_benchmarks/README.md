# AFM Code Agent Benchmark Dataset  
Here contains a processed benchmark dataset **tailored for AFM Code Agent**. Compared to the original datasets, key modifications have been made to optimize its compatibility and effectiveness for the target framework.  


## Key Modifications  
The dataset has undergone the following critical adjustments to enhance usability:  

1. **Query Enhancement**  
   Added task-specific prompt prefixes to clarify task objectives:  
   - `Mathematical problem-solving Prompt` for math-related queries  
   - `Code generation Prompt` for code-related queries  

2. **Query Requirement Supplement**  
   For code benchmarks, explicit requirements for handling `entrypoint` or `stdin` have been added to streamline code parsing and execution workflows.  

3. **Test Case Adaptation**  
   Original test cases in code benchmarks have been restructured to align with the evaluation logic and technical specifications of AFM CodeAgent.  


## Data Sources  
This benchmark dataset is built upon the following open-source resources:  

1. [livecodebench v4--v5](https://huggingface.co/datasets/livecodebench/code_generation_lite)  
2. [codecontests](https://huggingface.co/datasets/deepmind/code_contests)  
3. [aime24](https://huggingface.co/datasets/math-ai/aime24)  
4. [aime25](https://huggingface.co/datasets/math-ai/aime25)  
5. [math500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)  
6. [amc23](https://huggingface.co/datasets/zwhe99/amc23)  
7. [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench)  
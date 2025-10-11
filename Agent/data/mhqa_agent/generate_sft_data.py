import json
import requests
import time
from typing import Dict, List, Any
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class OutputConverter:
    def __init__(self, api_url: str, api_key: str, model: str = "gpt-4o"):
        """
        初始化转换器
        
        Args:
            api_url: 第三方API的URL
            api_key: API密钥
            model: 使用的模型名称
        """
        self.api_url = "https://api.tensoropera.ai/v1/chat/completions"
        self.api_key = "sk-to2025llm"
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def create_conversion_prompt(self, original_output: str) -> str:
        """创建转换prompt"""
        prompt = f"""You are tasked with converting a sequential reasoning trajectory into a parallel execution format. The original trajectory follows a strict sequential pattern where each step waits for the previous one to complete. Your job is to identify dependencies and enable parallel execution where possible.

**Original Sequential Output:**
{original_output}

**Conversion Requirements:**
You can respond to questions using the following 7 functions: think, plan, wiki_search, observation, reflection and answer.
Function Descriptions:
1. think: Provide reasoning, justification, and synthesis of information before using other functions. Begin with <think> and end with </think>.
2. plan: Break down the question into sub-tasks with explicit dependencies. Format each task as:
   - Task ID: unique identifier (T1, T2, etc.)
   - Description: what to search/investigate
   - Dependencies: which tasks must complete first (use "none" if independent)
   Begin with <plan> and end with </plan>.
3. wiki_search: Execute search queries. For parallel searches, separate multiple queries with |. For single search, use one query. Begin with <wiki_search> and end with </wiki_search>.
4. observation: Results from search functions. Begin with <observation> and end with </observation>.
5. reflection: Evaluate progress and suggest plan modifications if needed. Begin with <reflection> and end with </reflection>.
6. answer: Final confident answer. Begin with <answer> and end with </answer>.

**Example Pattern for converted reasoning trajectory:**
```
Question: What occupation was shared by both John Frankenheimer and Tiffanie DeBartolo?

<think>This question requires finding information about two different people and identifying their shared occupation. I need to search for each person's career information and then compare them to find commonalities.</think>

<plan>
T1: Search for John Frankenheimer's occupations and career
- Dependencies: none
T2: Search for Tiffanie DeBartolo's occupations and career
- Dependencies: none
T3: Compare their occupations to identify shared ones
- Dependencies: T1, T2
</plan>

<think>Tasks T1 and T2 are independent and can be executed in parallel using the | separator in wiki_search. After getting both results, I'll analyze them to find the shared occupation.</think>

<wiki_search>John Frankenheimer occupation career director|Tiffanie DeBartolo occupation career director novelist</wiki_search>

<observation>
Doc 1 - John Frankenheimer:
John Frankenheimer (1930-2002) was an American film and television director. He was known for directing films such as "The Manchurian Candidate" (1962), "Seven Days in May" (1964), and "The French Connection II" (1975). He was primarily recognized as a director in both film and television industries.

Doc 2 - Tiffanie DeBartolo:
Tiffanie DeBartolo is an American novelist and film director. She is the author of novels including "God-Shaped Hole" and "How to Kill a Rock Star." She also directed the independent film "Dream for an Insomniac" (1996). She works as both a writer and director.
</observation>

<think>Now I can synthesize the results from both parallel searches:
- John Frankenheimer: Film and television director
- Tiffanie DeBartolo: Novelist and film director

The shared occupation between them is "director" - both have worked as film directors. John Frankenheimer was primarily known as a director, while Tiffanie DeBartolo works as both a novelist and director.</think>

<answer>director</answer>
```

**Important Constraints:**
1. **External Information Only**: You must rely solely on search tools to obtain all external information. Do not use internal knowledge or pre-existing facts in your reasoning.
2. **Keep Tasks Simple**: Do not over-decompose tasks. Most tasks in the dataset are relatively simple search operations that don't require excessive breakdown into sub-tasks.
3. **Maintain Search Dependency**: Ensure that the parallel execution still respects the logical flow where search results inform subsequent reasoning steps.
4. **Preserve Observation Content**: Do NOT modify the content within <observation> tags. These contain actual search results that must remain unchanged. You can even use the observation content to reverse-engineer and adjust your preceding search queries and reasoning to ensure consistency.
5. **Preserve Final Answer**: Do NOT modify the content within <answer> tags. Your task is only to optimize the reasoning and tool execution path, not to change the final answer or observation results.

**Now Please Convert the given sequential output to new parallel execution while maintaining correctness:**"""
        
        return prompt
    
    def call_api(self, prompt: str, max_retries: int = 5) -> str:
        """调用第三方API"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 1.0,
            "max_tokens": 8192
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"API调用失败，状态码: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(3 ** attempt)  # 指数退避
                        continue
                    else:
                        raise Exception(f"API调用失败: {response.status_code}")
                        
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 ** attempt)
                    continue
                else:
                    raise e
    
    def convert_single_output(self, original_output: str) -> str:
        """转换单个output"""
        try:
            prompt = self.create_conversion_prompt(original_output)
            converted_output = self.call_api(prompt)
            return converted_output.strip()
        except Exception as e:
            print(f"转换失败: {e}")
            return original_output  # 返回原始内容作为fallback
    
    def validate_conversion(self, original: str, converted: str) -> bool:
        """验证转换结果的基本有效性"""
        # 检查必要的标签是否存在
        required_tags = ['<think>', '<plan>', '<wiki_search>', '<answer>']
        for tag in required_tags:
            if tag not in converted:
                print(f"警告: 转换结果缺少必要标签 {tag}")
                return False
        
        
        return True

def process_dataset(
    input_file: str, 
    output_file: str, 
    api_url: str, 
    api_key: str,
    batch_size: int = 10,
    max_workers: int = 3
):
    """
    处理整个数据集
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径  
        api_url: API URL
        api_key: API密钥
        batch_size: 批处理大小
        max_workers: 最大并发工作线程数
    """
    
    # 读取原始数据
    print(f"读取数据集: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集大小: {len(data)} 条记录")
    
    # 初始化转换器
    converter = OutputConverter(api_url, api_key)
    
    # 处理数据
    processed_data = []
    failed_count = 0
    
    def process_item(item):
        """处理单个数据项"""
        try:
            original_output = item.get('output', '')
            if not original_output:
                return item
            
            print(f"正在转换第 {item.get('id', 'unknown')} 条记录...")
            converted_output = converter.convert_single_output(original_output)
            
            # 验证转换结果
            if converter.validate_conversion(original_output, converted_output):
                # 创建新的数据项
                new_item = item.copy()
                new_item['output'] = converted_output
                new_item['original_output'] = original_output  # 保留原始输出用于对比
                new_item['conversion_status'] = 'success'
                return new_item
            else:
                print(f"转换验证失败，保留原始输出")
                new_item = item.copy()
                new_item['conversion_status'] = 'validation_failed'
                return new_item
                
        except Exception as e:
            print(f"处理失败: {e}")
            new_item = item.copy()
            new_item['conversion_status'] = 'failed'
            new_item['error'] = str(e)
            return new_item
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 分批处理以避免过多并发请求
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            print(f"\n处理批次 {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
            
            # 提交批次任务
            future_to_item = {executor.submit(process_item, item): item for item in batch}
            
            # 收集结果
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    processed_data.append(result)
                    
                    if result.get('conversion_status') != 'success':
                        failed_count += 1
                        
                except Exception as e:
                    print(f"批次处理异常: {e}")
                    original_item = future_to_item[future]
                    original_item['conversion_status'] = 'exception'
                    original_item['error'] = str(e)
                    processed_data.append(original_item)
                    failed_count += 1
            
            # 批次间稍作休息，避免API限流
            time.sleep(2)
    
    # 保存结果
    print(f"\n保存处理结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    success_count = len(processed_data) - failed_count
    print(f"\n处理完成!")
    print(f"总记录数: {len(processed_data)}")
    print(f"成功转换: {success_count}")
    print(f"转换失败: {failed_count}")
    print(f"成功率: {success_count/len(processed_data)*100:.2f}%")

def sample_conversion_test(input_file: str, api_url: str, api_key: str, sample_size: int = 10):
    """测试转换效果，处理少量样本"""
    print("开始样本测试...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converter = OutputConverter(api_url, api_key)
    
    for i in range(min(sample_size, len(data))):
        item = data[i]
        original_output = item.get('output', '')
        
        print(f"\n=== 样本 {i+1} ===")
        # print("原始输出:")
        # print(original_output)
        
        try:
            converted_output = converter.convert_single_output(original_output)
            print("\n转换后输出:")
            print(converted_output)
            
            # 验证
            is_valid = converter.validate_conversion(original_output, converted_output)
            print(f"\n验证结果: {'通过' if is_valid else '失败'}")
            
        except Exception as e:
            print(f"转换失败: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    # 配置参数
    API_URL = "https://api.tensoropera.ai/v1"  # 替换为你的第三方API URL
    API_KEY = "sk-to2025llm"  # 替换为你的API密钥
    
    INPUT_FILE = "/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-Agent-SFT-Dataset/hotpotQA_nq_level1_level2_merge_first_second_filtered_multiprediction.json"
    OUTPUT_FILE = "/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-Agent-SFT-Dataset/hotpotQA_parallel.json"
    
    # 可以从环境变量读取配置
    API_URL = os.getenv("API_URL", API_URL)
    API_KEY = os.getenv("API_KEY", API_KEY)
    
    if API_KEY == "your-api-key-here":
        print("请设置正确的API_KEY")
        exit(1)
    
    # 选择运行模式
    mode = input("选择运行模式:\n1. 样本测试 (处理3个样本)\n2. 完整处理\n请输入 1 或 2: ").strip()
    
    if mode == "1":
        # 样本测试
        sample_conversion_test(INPUT_FILE, API_URL, API_KEY, sample_size=10)
    elif mode == "2":
        # 完整处理
        process_dataset(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            api_url=API_URL,
            api_key=API_KEY,
            batch_size=1,  # 可以根据API限制调整
            max_workers=1  # 可以根据API限制调整
        )
    else:
        print("无效选择")

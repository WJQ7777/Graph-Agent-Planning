import pandas as pd
import numpy as np
import json
import re

def update_prompt_content_ndarray(parquet_file_path, new_system_prompt, output_file_path=None):
    """
    更新 parquet 文件中 prompt 字段的 system prompt 内容，保留原有的 Question 部分
    处理 numpy.ndarray 格式的 prompt 字段
    
    Args:
        parquet_file_path: 输入的 parquet 文件路径
        new_system_prompt: 新的 system prompt 内容
        output_file_path: 输出文件路径，如果为 None 则覆盖原文件
    """
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_file_path)
    
    print(f"原始数据集大小: {len(df)} 行")
    
    # 检查是否存在 prompt 字段
    if 'prompt' not in df.columns:
        print("错误: 数据集中没有找到 prompt 字段")
        return
    
    def extract_question_from_content(content):
        """从 content 中提取 Question 部分"""
        # 使用正则表达式匹配 Question: 后面的内容
        match = re.search(r'\n\nQuestion:\s*(.+?)$', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            # 如果没有找到标准格式，尝试其他可能的格式
            match = re.search(r'Question:\s*(.+?)$', content, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
    
    def process_prompt_array(prompt_array, new_system_prompt):
        """处理单个 numpy.ndarray 格式的 prompt"""
        try:
            # 如果是 numpy.ndarray，转换为 Python 对象
            if isinstance(prompt_array, np.ndarray):
                # 尝试不同的方式来获取数据
                if prompt_array.dtype == object:
                    # 如果是 object 类型，直接获取内容
                    prompt_data = prompt_array.item() if prompt_array.ndim == 0 else prompt_array.tolist()
                else:
                    prompt_data = prompt_array.tolist()
            else:
                prompt_data = prompt_array
            
            # 如果是字符串，尝试解析为 JSON
            if isinstance(prompt_data, str):
                try:
                    prompt_list = json.loads(prompt_data)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析 prompt 字符串为 JSON")
                    return prompt_array
            elif isinstance(prompt_data, list):
                prompt_list = prompt_data.copy()
            else:
                print(f"警告: prompt 数据格式不符合预期: {type(prompt_data)}")
                return prompt_array
            
            # 查找并更新 user role 的内容
            updated = False
            for item in prompt_list:
                if isinstance(item, dict) and item.get('role') == 'user':
                    original_content = item.get('content', '')
                    
                    # 提取原有的 Question 部分
                    question = extract_question_from_content(original_content)
                    
                    if question:
                        # 构建新的 content
                        new_content = f"{new_system_prompt}\n\nQuestion: {question}"
                        item['content'] = new_content
                        updated = True
                    else:
                        print(f"警告: 无法从内容中提取 Question 部分")
                        print(f"原内容: {original_content[:200]}...")
            
            if updated:
                # 转换回 numpy.ndarray 格式
                return np.array(prompt_list, dtype=object)
            else:
                return prompt_array
                
        except Exception as e:
            print(f"警告: 处理 prompt 时出错: {e}")
            return prompt_array
    
    # 应用更新函数
    updated_count = 0
    def apply_update(prompt_value):
        nonlocal updated_count
        original_value = prompt_value
        result = process_prompt_array(prompt_value, new_system_prompt)
        
        return result
    
    print("开始处理 prompt 字段...")
    df['prompt'] = df['prompt'].apply(apply_update)
    
    # 确定输出路径
    if output_file_path is None:
        output_file_path = parquet_file_path
    
    # 保存处理后的数据
    df.to_parquet(output_file_path, index=False)
    
    print(f"处理完成! ")
    print(f"数据已保存到: {output_file_path}")
    
    # 显示更新后的示例
    print("\n更新后的 prompt 示例 (前2行):")
    for i in range(min(2, len(df))):
        prompt_data = df.iloc[i]['prompt']
        try:
            if isinstance(prompt_data, np.ndarray):
                prompt_list = prompt_data.tolist() if prompt_data.dtype == object else prompt_data
                if isinstance(prompt_list, list) and len(prompt_list) > 0:
                    content = prompt_list[0].get('content', '') if isinstance(prompt_list[0], dict) else str(prompt_list[0])
                    print(f"行 {i} 内容预览: {content[:300]}...")
                else:
                    print(f"行 {i}: prompt 格式异常")
            else:
                print(f"行 {i}: prompt 不是 ndarray 格式: {type(prompt_data)}")
        except Exception as e:
            print(f"行 {i}: 显示时出错: {e}")
        print("-" * 50)

def verify_question_preservation_ndarray(parquet_file_path, num_samples=3):
    """
    验证 Question 部分是否被正确保留（处理 numpy.ndarray 格式）
    """
    df = pd.read_parquet(parquet_file_path)
    
    print(f"\n验证 Question 保留情况 (前 {num_samples} 行):")
    for i in range(min(num_samples, len(df))):
        try:
            prompt_data = df.iloc[i]['prompt']
            
            if isinstance(prompt_data, np.ndarray):
                prompt_list = prompt_data.tolist() if prompt_data.dtype == object else prompt_data
                if isinstance(prompt_list, list) and len(prompt_list) > 0:
                    content = prompt_list[0].get('content', '') if isinstance(prompt_list[0], dict) else str(prompt_list[0])
                    
                    # 提取 Question 部分
                    match = re.search(r'\n\nQuestion:\s*(.+?)$', content, re.DOTALL)
                    if match:
                        question = match.group(1).strip()
                        print(f"行 {i} Question: {question}")
                    else:
                        print(f"行 {i}: 未找到 Question 部分")
                else:
                    print(f"行 {i}: prompt 格式异常")
            else:
                print(f"行 {i}: prompt 不是 ndarray 格式")
                
        except Exception as e:
            print(f"行 {i}: 验证时出错: {e}")

def inspect_prompt_format(parquet_file_path, num_samples=2):
    """
    检查 prompt 字段的具体格式，用于调试
    """
    df = pd.read_parquet(parquet_file_path)
    
    print("检查 prompt 字段格式:")
    for i in range(min(num_samples, len(df))):
        prompt_data = df.iloc[i]['prompt']
        print(f"\n行 {i}:")
        print(f"  类型: {type(prompt_data)}")
        
        if isinstance(prompt_data, np.ndarray):
            print(f"  ndarray shape: {prompt_data.shape}")
            print(f"  ndarray dtype: {prompt_data.dtype}")
            print(f"  ndarray 内容类型: {type(prompt_data.item() if prompt_data.ndim == 0 else prompt_data[0])}")
            
            try:
                content = prompt_data.item() if prompt_data.ndim == 0 else prompt_data.tolist()
                print(f"  内容预览: {str(content)[:200]}...")
            except Exception as e:
                print(f"  获取内容时出错: {e}")

if __name__ == "__main__":
    # 新的 system prompt
    new_system_prompt = """You can respond to questions using the following 7 functions: think, plan, wiki_search, observation, reflection and answer.
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

Execution Rules:
1. Always use think before other functions.
2. Use dependency_plan to create a task dependency graph.
3. Execute independent tasks in parallel by using | separator in wiki_search.
4. Use think to synthesize and analyze results from multiple searches.
5. Tasks with dependencies execute only after prerequisites complete.
6. Use reflection if the plan needs major adjustment.
7. Multiple final answers should be separated by |.

Special Token Restriction: Function tags must not appear in free text content.

Example:
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
"""

    # 使用示例
    input_file = "/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-RL-Dataset/original_16w_column_filter_index.parquet"
    output_file = "/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-RL-Dataset/original_16w_column_filter_index_enhanced.parquet"  # 可以设置为 None 来覆盖原文件
    
    # 首先检查格式（可选）
    print("=== 检查原始格式 ===")
    inspect_prompt_format(input_file)
    
    # 更新 prompt 内容
    print("\n=== 开始更新 ===")
    update_prompt_content_ndarray(input_file, new_system_prompt, output_file)
    
    # 验证结果
    print("\n=== 验证结果 ===")
    verify_question_preservation_ndarray(output_file if output_file else input_file)
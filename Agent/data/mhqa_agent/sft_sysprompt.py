import json

# 定义新的instruction内容
new_instruction = """You can respond to questions using the following 7 functions: think, plan, wiki_search, observation, reflection and answer.
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

<answer>director</answer>"""

# 读取原始JSON文件
with open('/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-Agent-SFT-Dataset/hotpotQA_parallel.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # 加载整个JSON数据

# 修改所有instruction字段
if isinstance(data, list):
    # 如果JSON是列表格式（每个元素是一个字典）
    for item in data:
        if 'instruction' in item:
            item['instruction'] = new_instruction
elif isinstance(data, dict):
    # 如果JSON是字典格式（包含多个条目）
    for key in data:
        if isinstance(data[key], dict) and 'instruction' in data[key]:
            data[key]['instruction'] = new_instruction
else:
    print("无法识别的JSON格式")
    exit()

# 将修改后的数据写入新文件
with open('/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-Agent-SFT-Dataset/hotpotQA_parallel_sft.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 读取并输出修改后的文件
print("修改后的文件内容：")
with open('/home/jiaqi/Agent_Foundation_Models/AFM/data/mhqa_agent/AFM-MHQA-Agent-SFT-Dataset/hotpotQA_parallel_sft.json', 'r', encoding='utf-8') as f:
    modified_data = json.load(f)
    
    # 输出前3条数据作为参考
    if isinstance(modified_data, list):
        for i, item in enumerate(modified_data[:3], 1):
            print(f"第{i}条: {item}")
    elif isinstance(modified_data, dict):
        for i, key in enumerate(list(modified_data.keys())[:3], 1):
            print(f"第{i}条: {modified_data[key]}")
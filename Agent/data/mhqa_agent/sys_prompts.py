# System prompts (same as original)
SR1_SYS_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. \
For example, <answer> Beijing </answer>."""

MHQA_PROMPT = """You can only respond to a given question using the following 6 functions: think, plan, wiki_search, observation, reflection and answer. Below are the descriptions of these functions:
1.think: Before using any of the plan, wiki_search, reflection or answer functions, you must first use the think function to provide the reasoning, justification, and the procedural steps for the function you intend to use next. Begin with <think> and end with </think>.
2.plan: Based on the given question, you must break it down into very detailed, fine-grained sub-questions to facilitate execution using the wiki_search function. Begin with <plan> and end with </plan>.
3.wiki_search: You may use the wiki_search function retrieve external information for answering questions Begin with <wiki_search> and end with </wiki_search>. You should use the wiki_search function like this: <wiki_search>search_query</wiki_search>
4.observation: This function represents the result returned after using the tool function. Begin with <observation> and end with </observation>.
5.reflection: Evaluate and reflect on the current steps of the trajectory. Provide some suggestions for modifying the plan and wiki_search if necessary. Begin with <reflection> and end with </reflection>. \n6.answer: Your response must include answer function at the end, indicating that you are confident in the final answer. Begin with <answer> and end with </answer>.

Important Notes:
1.You can only use these functions to construct the correct reasoning path and arrive at the final answer to the given question.
2.Based on the results of the plan function, you can use the wiki_search function multiple times to gather sufficient external knowledge before formulating your response.
3.After reflection, you must use think function to guide your next step and the next step must be plan or wiki_search unless you are completely sure you get the answer. output format: 
<reflection>...</reflection><think>...</think><plan>...</plan><think>...</think> or <reflection>...</reflection><think>...</think><wiki_search>...</wiki_search><think>...</think>
4.Do not give the answer easily unless you are completely sure. The answer should be as concise as possible, avoiding detailed illustrations. For example, <answer> Beijing </answer>.
5.You must use the think function before each use of plan, wiki_search, reflection or answer function.
6.Special Token Restriction:
<think>, <plan>, <wiki_search>, <observation>, <reflection>, and <answer> are special tokens and must not appear in free text, especially not within the think function.
7.If the question requires multiple answers, your answers should be separated by | as a delimiter, for example <answer>answer1|answer2|answer3</answer>.
"""

MHQA_PROMPT_ONE_SHOT = """You are a helpful assistant that answers questions by reasoning and using available tools. You can only respond using the following 6 functions: think, plan, wiki_search, observation, reflection, and answer.
Function Descriptions:
1.think: The master reasoning function. You must use it at the beginning of your process to analyze the user's question and decide on a high-level strategy. You also use it after reflection to explain your new strategy. It should contain your reasoning and justification for the next major phase (e.g., creating a plan or giving the final answer). Begin with <think> and end with </think>.
2.plan: Based on the given question, you must break it down into very detailed, fine-grained sub-questions to facilitate execution using the wiki_search function. Begin with <plan> and end with </plan>. The plan can be adapted later if needed. Begin with <plan> and end with </plan>.
3.wiki_search: You may use the wiki_search function retrieve external information for answering questions Begin with <wiki_search> and end with </wiki_search>. You should use the wiki_search function like this: <wiki_search>search_query</wiki_search>
4.observation: Represents the result returned after using a tool. This is provided to you by the system. Begin with <observation> and end with </observation>.
5.reflection: Use this function to analyze what went wrong and propose a new course of action. **You should only use this function when your plan hits a dead end, an observation is empty or contradictory, or you realize your initial strategy was flawed.** Begin with <reflection> and end with </reflection>.
6.answer: Provide the final, concise answer to the user's question once you are confident. Begin with <answer> and end with </answer>.

Core Workflow & Rules:

1.Start with Strategy: Always begin your response with the <think> function to outline your initial strategy.
2.Always use tool: For simple questions where the answer is likely within your existing knowledge, DO NOT answer directly, use <wiki_search> to get the answer.
3.The Standard Path: For complex questions, the typical loop is <think> -> <plan> -> <wiki_search> -> <observation> -> ... -> <answer>.
4.Efficient Tool Use: After creating a <plan>, you can execute multiple <wiki_search> steps sequentially to follow your plan without needing a <think> before each one.
5.Error Correction: When you get stuck, use the <reflection> -> <think> -> <plan> (revised) sequence to get back on track.
6.Special Token Restriction: <think>, <plan>, <wiki_search>, <observation>, <reflection>, and <answer> are special tokens and must not appear in free text.
7.If the question requires multiple answers, your answers should be separated by | as a delimiter, for example <answer>answer1|answer2|answer3</answer>.

Example:
Question: What occupation was shared by both John Frankenheimer and Tiffanie DeBartolo?

<think>To answer the question about the shared occupation of John Frankenheimer and Tiffanie DeBartolo, I first need to search for the primary occupations of each individual. Then, I will compare their occupations to identify any overlap. To do this efficiently, I will break the question into two specific sub-questions: "What was John Frankenheimer's occupation?" and "What is Tiffanie DeBartolo's occupation?"</think>
<plan>
1. Find the occupation(s) of John Frankenheimer.
2. Find the occupation(s) of Tiffanie DeBartolo.
3. Identify any occupation that is common to both individuals.
</plan>
<think>I will begin by searching for John Frankenheimer's occupation.</think>
<wiki_search>John Frankenheimer occupation</wiki_search>

<observation> Doc 1(Title: "John Frankenheimer") John Frankenheimer John Michael Frankenheimer (February 19, 1930 – July 6, 2002) was an American film and television director known for social dramas and action/suspense films. Among his credits were ""Birdman of Alcatraz"" (1962), ""The Manchurian Candidate"" (1962), ""Seven Days in May"" (1964), ""The Train"" (1964), ""Seconds"" (1966), ""Grand Prix"" (1966), ""French Connection II"" (1975), ""Black Sunday"" (1977), and ""Ronin"" (1998). Frankenheimer won four Emmy Awards—three consecutive—in the 1990s for directing the television movies ""Against the Wall"", ""The Burning Season"", ""Andersonville"", and ""George Wallace"", the latter of which also received a Golden Globe Award for Best Miniseries or Television Film.
Doc 2(Title: "John Frankenheimer") He was considered one of the last remaining directors who insisted on having complete control over all elements of production, making his style unique in Hollywood. Frankenheimer's 30 feature films and over 50 plays for television were notable for their influence on contemporary thought. He became a pioneer of the ""modern-day political thriller,"" having begun his career at the peak of the Cold War. He was technically highly accomplished from his days in live television; many of his films were noted for creating ""psychological dilemmas"" for his male protagonists along with having a strong ""sense of environment,"" similar in style
Doc 3(Title: "John Frankenheimer") Sheedy. His father was of German Jewish descent, his mother was Irish Catholic, and Frankenheimer was raised in his mother's religion. He grew up in New York City and became interested in movies at an early age; he recalled going to the cinema every weekend. In 1947, he graduated from La Salle Military Academy in Oakdale, Long Island, New York. In 1951, he graduated from Williams College in Williamstown, Massachusetts, where he had studied English. He also developed an interest in acting as a career while in college but began thinking seriously about directing when he was in the Air
</observation>

<think>The first step of the plan—finding John Frankenheimer's occupation—was correctly executed. The wiki_search yielded clear evidence that John Frankenheimer was a film and television director. No errors were made in this trajectory step. The next step should be to similarly search for Tiffanie DeBartolo's occupation, as outlined in the plan, to facilitate comparison. No reflection-driven changes to the plan are needed at this point.</think>

<reflection>The plan and execution thus far are correct. The search for Frankenheimer's occupation was successful, yielding useful information. Continue with the plan by searching for Tiffanie DeBartolo's occupation before making comparisons. No modification is required at this step.</reflection>

<think>Having identified John Frankenheimer as a film and television director, I now need to determine Tiffanie DeBartolo's occupation. This will enable me to compare both individuals' professions and identify any shared occupation. The next logical step is to perform a wiki_search specifically for Tiffanie DeBartolo's occupation.</think>
<wiki_search>Tiffanie DeBartolo occupation</wiki_search>

<observation> Doc 1(Title: "Tiffanie DeBartolo") Tiffanie DeBartolo Tiffanie DeBartolo (born November 27, 1970) is an American novelist, filmmaker, and co-founder of independent record label Bright Antenna. She has written two novels, ""God-Shaped Hole"" and ""How To Kill a Rock Star"". She wrote and directed the film ""Dream for an Insomniac"", featuring Jennifer Aniston, but which had a very limited release in 1996. DeBartolo is the daughter of former San Francisco 49ers owner Edward J. DeBartolo, Jr.. She attended the all-girls Villa Maria High School and dropped out her senior year when they wouldn't let her graduate early. She later obtained her GED and graduated from
Doc 2(Title: "Tiffanie DeBartolo") UC Berkeley with a philosophy degree, then moved to Los Angeles. She has previously lived in New York City and Boulder, Colorado, and now resides in the San Francisco Bay Area. Her novel writing career began after her screenwriting/directorial debut, with ""God-Shaped Hole"" coming out in 2002, and ""How to Kill a Rock Star"" in 2005, both from Sourcebooks Landmark. Sidney Sheldon has praised her sophomore novel, saying it has ""Wonderful characters wrapped up in a story that moves like an express train"". ""God-Shaped Hole"" was published in the UK as ""The Shape of My Heart"". She has recently completed
Doc 3(Title: "Tiffanie DeBartolo") work on the text for a graphic novel about American musician Jeff Buckley. She is one of the owners of the Oakland, California based Indie music record label Bright Antenna, which has released music by Middle Class Rut, Jimmy Gnecco, The Wombats, Magic Wands, Orchestral Manoeuvres in the Dark, Flagship, Beware of Darkness, and Cheerleader. Tiffanie DeBartolo Tiffanie DeBartolo (born November 27, 1970) is an American novelist, filmmaker, and co-founder of independent record label Bright Antenna. She has written two novels, ""God-Shaped Hole"" and ""How To Kill a Rock Star"". She wrote and directed the film ""Dream for an Insomniac"",
</observation>

<think>The occupation information for Tiffanie DeBartolo is now clear from the search results. She is described as a novelist and filmmaker and is also referenced as a director of the film "Dream for an Insomniac." John Frankenheimer was previously identified as a film and television director. Both individuals have the occupation of "film director" (or a similar title such as filmmaker/director). Therefore, the shared occupation between John Frankenheimer and Tiffanie DeBartolo is "film director" (or simply: director/filmmaker). I am now ready to provide the final answer.</think>
<answer>director</answer>
"""

ENHANCED_MHQA_PROMPT = """You can respond to questions using the following 7 functions: think, plan, wiki_search, observation, reflection and answer.
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


WEB_AGENT_PROMPT = """
You can only use the following 8 functions to answer a given question: think, plan, tool, observation, reflection, suggested_answer, double_check, and answer. Here are the descriptions of these functions:

1. think: Before using any plan, tool, reflection, suggested_answer, double_check, and answer functions, you must use the think function to provide reasoning, arguments, and procedural steps for the functions you will use next. Start with <think> and end with </think>.
2. plan: Given a given question, you must break it down into very detailed, fine-grained sub-problems to execute using the tool function. After the reflection function, you can use the plan function to update the plan. Start with <plan> and end with </plan>.
3. tool: You can use any tool in the tool list below to find information relevant to answering the question. The tool label should be replaced with the exact tool name in the tool list below.
4. observation: The observation returned after using the tool.
5. reflection: Evaluate the trajectory of the historical algorithm to effectively guide the direction of your work towards the optimal path.
6. suggested_answer: Based on the historical trajectory, you can propose a suggested answer.
7. double_check: After giving the suggested answer, you will perform this step. You will reflect on your historical trajectory and task requirements, and give your reasoning and thinking based on the credibility of the suggested answer. If you are not confident in the suggested answer, you should rethink and update your plan to clarify the task; otherwise, you will come up with your answer.
8. answer: After the suggested answer meets the requirements during the double check process, you will give the answer.

Here are some tools you can use:
1. <wiki_search>Web search tools require a search query to get information from a web page</wiki_search>, for example: <wiki_search>Latest AI Development in 2023</wiki_search>

**Tool Usage Guide**
1) <wiki_search>: If the information is not relevant to the query, you should search again with another search query until you get enough information and are very confident in getting the final answer.

**Trail Notes**
1. You can only use these functions to build the correct reasoning path and get the final answer to the given question.
2. Based on the result of the plan function, you can use the tool function multiple times to collect enough external knowledge before formulating your answer.
3. Special tag restrictions: <think>, <plan>, <wiki_search>, <observation>, <reflection>, <double_check>, <suggested_answer> and <answer> are special tags and must not appear in free text, especially in the <think> function.

**Scoring Criteria Description**
1. <reflection>: Assigning PRM scores (good/average/poor) to current task progress to guide the completion of the task. If any criterion is scored as poor, the plan will be re-made.
    Criteria:
    (1) Information Conflict: Good (no contradictions, logical); Average (slight inconsistency, no major conflicts); Poor (direct contradictions/irreconcilable).
    (2) Tool Effectiveness: Good (highly relevant/effective); Average (relevant but inefficient/non-optimal); Poor (irrelevant, misused, or obstructive).
    (3) Trajectory Monitoring: Good (clear progress, no dead ends); Fair (vague/slow progress, reasonable path); Poor (stuck, looped, or unable to get answers).
2. <double_check>: The matching degree of the suggested answer to the task requirements is scored from 1 to 4: completely correct "4" means no errors; mostly correct "3" means slight defects; mostly wrong "2" means large errors, but with some validity; completely wrong "1" means completely incorrect/irrelevant. If the score is less than or equal to "2", the plan will be re-made; if the score is greater than or equal to "3", the task will be ended and the answer will be returned.

**Function Association Instructions**
1. Before using <plan>, <wiki_search>, <reflection>, <double_check> or <suggested_answer>, you must first use the <think> function.
2. After <plan>, you can only perform <wiki_search> and <suggested_answer>.
3. After <double_check>, you can only perform <plan>, <wiki_search> and <answer>.
4. After <reflection>, you can only perform <plan>, <wiki_search> and <suggested_answer>.
5. After <suggested_answer>, you can only perform <double_check>.
6. After <wiki_search>, you can only perform <wiki_search>, <reflection> and <suggested_answer>.
7. If any scoring criteria in <reflection> is poor, you need to re-plan.
8. Before getting <answer>, you should return to <suggested_answer> first, and then return the suggested answer with <double_check> score >= 3 as the answer. If your <double_check> score is < 3, you should re-plan and arrange your thinking and reasoning process until you get <suggested_answer> again.

**Answering Tips**
1. Don't give an answer easily unless you are absolutely sure. The answer should be as concise as possible and avoid detailed description. For example, <answer>Beijing</answer>.
2. You must give a clear answer. If you are unsure, you must think, re-plan, and try to find a clear answer based on the available information before giving a final answer. The final answer cannot be insufficient or uncertain. The question must have a clear answer. Therefore, your answer must be accurate and unambiguous.
3. If the question requires multiple answers, your answers need to be separated by | as a delimiter, for example <answer>answer1|answer2|answer3</answer>.
""".strip()


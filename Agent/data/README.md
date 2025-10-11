
# Data Preparation Guide

This guide explains how to prepare and process datasets for training using the `prepare_data.py` script.

## Original Datasets

The script processes question-answering datasets in JSONL or Parquet formats with the following fields:
- `question`: The query text
- `answer`: The answer (string or list of strings)
- `data_source`: Source identifier of the dataset

Example JSONL format:
```json
{"question": "total number of death row inmates in the us?", "answer": ["2,718"], "data_source": "nq"}
{"question": "big little lies season 2 how many episodes?", "answer": ["seven"], "data_source": "nq"}
```

## Data Processing

The script converts the raw JSONL/Parquet files into training-ready Parquet files by adding system prompts, tool configurations, and other required fields.

### Running the Script

```bash
python data/prepare_data.py "raw_jsonl_file_path" -o "output_dir" --tool-type "single|two|three|afm" --sys-prompt-key "DOUBLE_CHECK_RULE|AFM" --filter-none
```

Example:
```bash
python data/prepare_data.py data/raw_qa_data/raw_data_column_filter/nq_random_300.jsonl -o data/wiki_search --tool-type single --sys-prompt-key DOUBLE_CHECK_RULE
```

### Output Format

The script converts data to the following format:

```python
{
    "data_source": data_source,
    "prompt": [
        {"role": "user", "content": sys_prompt + question}
    ],
    "reward_model": {
        "ground_truth": "{"target": answer}"  # This is now a JSON string
    },
    "extra_info": {
        "need_tools_kwargs": True,
        "question": question,
        "answer": answer,
        "tools_kwargs": tools_kwargs
    }
}
```

### Customizing Data Processing

#### System Prompts
All prompts are defined in `data/sys_prompts.py`. You can add new prompts there and reference them in `prepare_data.py` by updating the `SYS_PROMPT_MAP` dictionary.

```python
SYS_PROMPT_MAP = {
    'SR1': SR1_SYS_PROMPT,
    'DOUBLE_CHECK_RULE': function_2_tool_reflection_doublecheck,
    'AFM': AFM_TOOLS
}
```

#### Tool Configuration

Three tool configurations are available:
- `single`: Uses only `wiki_search`
- `two`: Uses `web_search` and `crawl_page`
- `three`: Uses three tools: `wiki_search`, `web_search`, and `crawl_page`
- `afm`: Uses all tools: `web_search`, `crawl_page`, `code` and `visual_inspector`

To modify tool parameters, edit the `tools_kwargs` in the script:

```python
tools_kwargs = {
    name: {"create_kwargs": common_create_kwargs} for name in tool_names
}
```

#### Filtering Examples

Use the `--filter-none` flag to skip examples with missing or empty values instead of raising errors.

## Additional Notes

For Interleave Learning and Experience Buffer usage, you may need to add an `index` field (string type) to the `extra_info` dictionary.

For code and math train task, pls download the raw data parquet from huggingface directly. Because it contains testcases.

## Example Commands

```bash
# Sample 300 random entries from NQ dataset
cat raw_data_column_filter/test_seven_dataset_all_column_filter.jsonl | grep '"data_source": "nq"' | shuf -n 300 > raw_data_column_filter/nq_random_300.jsonl

# Extract all NQ entries
cat raw_data_column_filter/test_seven_dataset_all_column_filter.jsonl | grep '"data_source": "nq"' > raw_data_column_filter/nq_full.jsonl

# Process with WIKI_SEARCH prompt and single tool
python data/prepare_data.py data/raw_qa_data/raw_data_column_filter/nq_random_300.jsonl -o data/wiki_search --tool-type single --sys-prompt-key DOUBLE_CHECK_RULE
```

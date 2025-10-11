# Basic Protocol

The basic protocols are defined in `verl.tools.schemas.py`

### 1. Tool Call Schema

```mermaid
stateDiagram-v2
    
    state "role: str" as role
    state "content: str" as content
    state "tool_calls: List" as tool_calls
    
    Message --> tool_calls
    Message --> content
    Message --> role
    
    tool_calls --> OpenAIFunctionToolCall

    state "id: str" as id
    state "type: Literal['function']" as type
    
    OpenAIFunctionToolCall --> id
    OpenAIFunctionToolCall --> type
    OpenAIFunctionToolCall --> function

    function --> OpenAIFunctionCallSchema

    state "name: str" as name1
    state "arguments: dict" as arguments

    OpenAIFunctionCallSchema --> name1
    OpenAIFunctionCallSchema --> arguments
    OpenAIFunctionCallSchema --> from_openai_function_parsed_schema: staticmethod

    state "name: str" as name2
    state "arguments: json str" as arguments_2

    OpenAIFunctionParsedSchema --> name2
    OpenAIFunctionParsedSchema --> arguments_2

    OpenAIFunctionParsedSchema --> parse_arguments: as parameter input
    from_openai_function_parsed_schema --> parse_arguments: method
    
    parse_arguments --> OpenAIFunctionCallSchema: return object

    parse_arguments --> has_decode_error=False: successfully parse arguments to dict
    parse_arguments --> has_decode_error=True: unable to parse arguments to dict

    [*] --> OpenAIFunctionParsedSchema: Result returned from Tool execution

    style OpenAIFunctionParsedSchema stroke:#73A6FF,stroke-width:2px
    style OpenAIFunctionCallSchema stroke:#73A6FF,stroke-width:2px
    style OpenAIFunctionToolCall stroke:#73A6FF,stroke-width:2px
    style Message stroke:#73A6FF,stroke-width:2px
```

### 2. Tool Function Schema

```mermaid
stateDiagram-v2

    state "type: str" as type1
    
    OpenAIFunctionToolSchema --> type1
    OpenAIFunctionToolSchema --> function

    function --> OpenAIFunctionSchema

    state "name: str" as name1
    state "description: str" as description1
    state "strict=False" as strict
    
    OpenAIFunctionSchema --> name1
    OpenAIFunctionSchema --> description1
    OpenAIFunctionSchema --> strict
    OpenAIFunctionSchema --> parameters

    parameters --> OpenAIFunctionParametersSchema

    state "type: str" as type2
    state "properties: Dict" as properties
    state "required: List[str]" as required

    OpenAIFunctionParametersSchema --> type2
    OpenAIFunctionParametersSchema --> properties
    OpenAIFunctionParametersSchema --> required

    properties --> OpenAIFunctionPropertySchema: value

    state "type: str" as type3
    state "description: str" as description3

    OpenAIFunctionPropertySchema --> type3
    OpenAIFunctionPropertySchema --> description3

    style OpenAIFunctionPropertySchema stroke:#73A6FF,stroke-width:2px
    style OpenAIFunctionParametersSchema stroke:#73A6FF,stroke-width:2px
    style OpenAIFunctionSchema stroke:#73A6FF,stroke-width:2px
    style OpenAIFunctionToolSchema stroke:#73A6FF,stroke-width:2px
```

### 3. Base Tool

## Tool Instances in Rollout Process

### 1. Rollout

```mermaid
stateDiagram-v2
    state "SGLangRollout" as rollout
    state "self._tool_map: Dict" as tool_map
    state "self._tool_schemas: List" as tool_schemas1

    rollout --> tool_map
    rollout --> tool_schemas1

    state "name" as name1
    state "tool instance: BaseTool" as tool1

    tool_map --> name1: key
    tool_map --> tool1: value

    tool_schemas1 --> OpenAIFunctionToolSchema: converted to dictionary by internal method

    state "AsyncRolloutRequest" as request
    state "tools_kwargs: Dict" as kwargs1
    state "tool_schemas: List" as tool_schemas2

    request --> tool_schemas2
    request --> kwargs1

    state "batch.non_tensor_batch" as batch
    state "tools_kwargs: Dict" as kwargs2

    batch --> kwargs2: key
    kwargs2 --> kwargs1: copy

    state "tool instance" as point1
    tool_map --> point1: provides dictionary
    kwargs2 --> point1: provides key
    point1 --> tool_schemas2: internal method conversion

    style rollout stroke:#73A6FF,stroke-width:2px
    style request stroke:#73A6FF,stroke-width:2px
    tool1
    style tool1 stroke:#FF0000,stroke-width:2px
    style point1 stroke:#FF0000,stroke-width:2px
```

## Tools Overview

```mermaid
stateDiagram-v2

    state "verl/tools" as r
    state "code_executor.py" as code
    state "web_search_tool.py" as web_search
    
    r --> code: BaseTool
    r --> web_search: BaseTool

    state "./utils/code_executors" as code_utils
    state "./utils/web_search_tool" as search_utils

    code --> code_utils: helper functions
    web_search --> search_utils: helper functions

    state "./config/code_tool_config" as code_tool_config
    state "./config/search_tool_config" as search_tool_config

    code --> code_tool_config: related configuration
    web_search --> search_tool_config: related configuration

    style code stroke:#73A6FF,stroke-width:2px
    style web_search stroke:#73A6FF,stroke-width:2px
```
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Portions of this file are modifications by OPPO PersonalAI Team.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from typing import Any, Dict, List, NamedTuple, Tuple


class ParsedToolCall(NamedTuple):
    name: str
    parameters: str
    tool_index: str

class XMLToolParser:
    def __init__(self, tools: List[Dict[str, Any]]):
        """
        Initializes the XMLToolParser with a list of tool schemas.

        Args:
            tools: A list of tool schemas, where each schema is a dictionary
                   containing a 'function' key with a 'name' for the tool.
        """
        self.tool_names = [tool["function"]["name"] for tool in tools]
        self.tool_regexes = {}
        for tool_name in self.tool_names:
            escaped_name = re.escape(tool_name)
            pattern = rf"<{escaped_name}>((?:(?!<{escaped_name}>).)*?)</{escaped_name}>"
            self.tool_regexes[tool_name] = re.compile(pattern, re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """
        Checks if the given text contains any tool calls in XML format.
        """
        for regex in self.tool_regexes.values():
            for match in regex.finditer(text):
                arguments = match.group(1).strip()
                if arguments:
                    return True
        return False

    def get_stop_phrases(self) -> List[str]:
        """
        Returns a list of closing tags for each tool to be used as stop phrases.
        """
        return [f"</{name}>" for name in self.tool_names] + ["</answer>"]

    def _parse_wiki_search_queries(self, content: str) -> List[str]:
        """
        Parse wiki_search content to handle parallel queries separated by |.
        
        Args:
            content: The content inside wiki_search tags
            
        Returns:
            List of individual query strings
        """
        # print("content : ", content)
        if '|' in content:
            # Split by | and clean up each query
            queries = [query.strip() for query in content.split('|')]
            # Filter out empty queries
            return [q for q in queries if q]
        else:
            return [content.strip()] if content.strip() else []

    def parse_non_stream(self, text: str) -> Tuple[str, List[ParsedToolCall]]:
        """
        Parses tool calls from the text and returns the text with tool calls removed,
        along with a list of parsed tool call objects.
        
        For wiki_search, handles both single queries and parallel queries separated by |.
        """
        tool_calls = []
        all_matches = []
        
        for tool_name, regex in self.tool_regexes.items():
            for match in regex.finditer(text):
                all_matches.append((match.start(), match.end(), tool_name, match))
        
        all_matches.sort(key=lambda x: x[0])
        
        if all_matches:
            last_match = all_matches[-1]
            normed_text = text[:last_match[0]].strip()
        else:
            normed_text = text.strip()

        for i, (start, end, tool_name, match) in enumerate(all_matches):
            arguments = match.group(1).strip()
            
            if tool_name == "wiki_search":
                # Handle parallel wiki_search queries
                queries = self._parse_wiki_search_queries(arguments)
                
                for j, query in enumerate(queries):
                    parameters = {"query": query}
                    tool_calls.append(
                        ParsedToolCall(
                            name=tool_name,
                            parameters=json.dumps(parameters, ensure_ascii=False),
                            tool_index=f"call_{tool_name}_{i}_{j}",
                        )
                    )
            else:
                # Handle other tools normally
                parameters = {"query": arguments}
                tool_calls.append(
                    ParsedToolCall(
                        name=tool_name,
                        parameters=json.dumps(parameters, ensure_ascii=False),
                        tool_index=f"call_{tool_name}_{i}",
                    )
                )

        return normed_text, tool_calls

    def is_parallel_wiki_search(self, text: str) -> bool:
        """
        Check if the text contains a wiki_search with parallel queries (contains |).
        
        Args:
            text: The text to check
            
        Returns:
            True if contains parallel wiki_search, False otherwise
        """
        if "wiki_search" not in self.tool_names:
            return False
            
        regex = self.tool_regexes.get("wiki_search")
        if not regex:
            return False
            
        for match in regex.finditer(text):
            arguments = match.group(1).strip()
            if '|' in arguments:
                return True
        return False

    def get_wiki_search_count(self, text: str) -> int:
        """
        Get the total number of wiki_search queries (including parallel ones).
        
        Args:
            text: The text to analyze
            
        Returns:
            Total number of wiki_search queries
        """
        if "wiki_search" not in self.tool_names:
            return 0
            
        regex = self.tool_regexes.get("wiki_search")
        if not regex:
            return 0
            
        total_count = 0
        for match in regex.finditer(text):
            arguments = match.group(1).strip()
            queries = self._parse_wiki_search_queries(arguments)
            total_count += len(queries)
            
        return total_count
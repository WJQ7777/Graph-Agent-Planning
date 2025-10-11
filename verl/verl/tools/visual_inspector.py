#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. Personal AI team. All rights reserved.
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


import base64
import json
import logging
import mimetypes
import os
import uuid
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
from requests import RequestException
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

# logger config
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VISUALINSPECTOR_LOGGING_LEVEL", "WARN"))

load_dotenv(override=True)

# custom api key.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    raise ValueError(f'[Error] OPENAI_API_KEY is not set.')

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)
if OPENAI_API_BASE is None:
    raise ValueError(f'[Error] OPENAI_API_BASE is not set.')


class VisualInspector(BaseTool):
    """Process images from files or URLs using OpenAI's vision capabilities"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize VisualInspectorWrapper with configuration"""
        super().__init__(config, tool_schema)
        
        # Check required configuration
        if "model_name" not in config:
            raise ValueError("[Error] Missing 'model_name' parameter in visual_inspector.yaml")
        if "text_limit" not in config:
            raise ValueError("[Error] Missing 'text_limit' parameter in visual_inspector.yaml")
        if "download_path" not in config:
            raise ValueError("[Error] Missing 'download_path' parameter in visual_inspector.yaml")
        
        # Initialize configuration
        self.model_name = config["model_name"]
        self.text_limit = int(config["text_limit"])
        self.download_path = config["download_path"]
        self.gpt_key = OPENAI_API_KEY
        self.gpt_url = OPENAI_API_BASE
        
        # Instance state tracking
        self._instance_dict = {}
        
        # Print configuration
        logger.info(f"Initialized VisualInspector with config: {config}")
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        
        if self.tool_schema:
            return self.tool_schema
        
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "visual_inspector",
                "description": "Process image files or web image URLs to get descriptions or answer questions about them. Cannot load files directly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "File path or web image URL (e.g., 'https://example.com/image.jpg') to be read as an image. Must be in supported image formats (.jpg/.jpeg/.png/.gif/.bmp/.webp)."
                        }
                    },
                    "required": ["query"]
                }
            }
        )
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "results": []
        }
        return instance_id

    def _validate_file_type(self, file_path: str):
        # Allow URLs (rely on MIME type check later)
        if file_path.lower().startswith(('http://', 'https://')):
            return

        # Check file extensions for local files
        supported_extensions = [
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"
        ]
        if not any(file_path.lower().endswith(ext) for ext in supported_extensions):
            raise ValueError(f"Unsupported file type. Visual inspector only supports {supported_extensions}")

    def _resize_image(self, image_path: str) -> str:
        img = Image.open(image_path)
        width, height = img.size
        img = img.resize((int(width / 2), int(height / 2)))
        new_image_path = f"resized_{os.path.basename(image_path)}"
        img.save(new_image_path)
        return new_image_path

    def _encode_image(self, image_path: str) -> str:
        if image_path.startswith("http"):
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
            request_kwargs = {
                "headers": {"User-Agent": user_agent},
                "stream": True,
            }

            try:
                response = requests.get(image_path, **request_kwargs)
                response.raise_for_status()
            except RequestException as e:
                raise ValueError(f"Failed to download image: {str(e)}")
            
            content_type = response.headers.get("content-type", "")
            extension = mimetypes.guess_extension(content_type)
            if extension is None:
                extension = ".download"

            # Ensure downloads directory exists using the configurable path
            os.makedirs(self.download_path, exist_ok=True)
            
            fname = str(uuid.uuid4()) + extension
            download_path = os.path.abspath(os.path.join(self.download_path, fname))

            with open(download_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=512):
                    fh.write(chunk)

            image_path = download_path

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    def _clean_json_string(self, json_str):
        if isinstance(json_str, str):
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
        return json_str

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the visual inspection tool.
        
        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing a JSON string with file_path and optional question
            
        Returns:
            Tuple of (tool_response, tool_reward_score, tool_metrics)
        """
        
        try:
            # Parse the query parameter which is a JSON string
            query = json.loads(self._clean_json_string(parameters.get("query", "{}")))
            
            if "file_path" not in query:
                return """Error: file_path is required such as <visual_inspector>```json\n{""file_path"":""img path"", ""question"":""any question about the img""}\n</visual_inspector>""", 0.0, {"error": "Missing file_path"}
            
            file_path = query["file_path"]
            question = query.get("question", "")
            
            self._validate_file_type(file_path)
            
            # If no question is provided, only ask to describe the img details
            if not question:
                question = """Provide an extremely detailed analysis of this image in at least 5 sentences. May include:
1. Main subjects/people and their positions
2. Background elements and setting
3. Colors, lighting, and atmosphere
4. Text or numbers visible in the image
5. Objects and their relationships to each other
6. Any distinctive features or unusual elements
7. Textures and materials visible
8. Perspective and viewing angle
9. Size and scale relationships
10. Any temporal indicators (time of day, season, etc.)

Be as precise and comprehensive as possible. Stick strictly to what is observable in the image without making assumptions or adding information not visibly present."""

            # If question is specified, incoporate into prompt
            else:
                question = f"""Provide an extremely detailed analysis of this image in at least 5 sentences. Focus especially on aspects relevant to answering:
"{question}"

May include:
1. Main subjects/people and their positions
2. Background elements and setting
3. Colors, lighting, and atmosphere
4. Text or numbers visible in the image
5. Objects and their relationships to each other
6. Any distinctive features that might be relevant to the question
7. Specific details that could help answer the question

Be as precise and comprehensive as possible. Describe only what is observable in the image without making assumptions or adding information not visibly present. DO NOT attempt to answer the question directly - only provide detailed image description."""

            mime_type, _ = mimetypes.guess_type(file_path)
            base64_image = self._encode_image(file_path)
            
            payload = {
                "model": f"{self.model_name}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        ],
                    }
                ],
                "max_tokens": 1000,
                "top_p": 0.1,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gpt_key}"
            }

            response = requests.post(
                f"{self.gpt_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            description = response.json()["choices"][0]["message"]["content"]
            
            # Track the result for this instance
            self._instance_dict[instance_id]["results"].append(description)
            
            # Return result with placeholder reward and metrics
            return description, 0.0, {"mime_type": mime_type or "unknown"}
            
        except Exception as e:
            error_message = f"Visual processing failed: {str(e)}"
            logger.error(error_message)
            return error_message, 0.0, {"error": str(e)}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward for the tool instance."""
        # Implement custom reward calculation if needed
        return 0.0
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
import logging
import time
import uuid
import json
import base64
import tempfile
import re
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.controller.service import Service

logger = logging.getLogger(__name__)

app = FastAPI(title="TanitAI OpenAI Compatible API")
svc = Service()

class ChatMessage(BaseModel):
    role: str
    content: Any  # Can be string or list for multimodal

class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = None


def parse_content(content):
    text_input = ""
    file_paths = []
    audio_paths = []
    
    if isinstance(content, str):
        return content, [], []
        
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_input += item.get("text", "") + "\n"
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        match = re.match(r"data:(.+?);base64,(.*)", url)
                        if match:
                            mime_type = match.group(1)
                            b64_data = match.group(2)
                            ext = mime_type.split("/")[-1] if "/" in mime_type else "jpg"
                            # fix some common types if needed
                            if ext == "jpeg": ext = "jpg"
                            try:
                                img_data = base64.b64decode(b64_data)
                                tf = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
                                tf.write(img_data)
                                tf.close()
                                file_paths.append(tf.name)
                            except Exception as e:
                                logger.error(f"Failed to decode base64 image: {e}")
                    else:
                        # external url, we can optionally handle or leave as text
                        text_input += f"\n[Image URL: {url}]\n"
                elif item.get("type") == "input_audio":
                    audio_dict = item.get("input_audio", {})
                    data = audio_dict.get("data", "")
                    fmt = audio_dict.get("format", "wav")
                    try:
                        audio_data = base64.b64decode(data)
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}")
                        tf.write(audio_data)
                        tf.close()
                        audio_paths.append(tf.name)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 audio: {e}")

    return text_input.strip(), file_paths, audio_paths


@app.get("/v1/models")
async def list_models():
    model_id = "tanit_health_qwen3_8b"
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "tanitai",
            }
        ]
    })


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
        
    history = []
    for m in req.messages[:-1]:
        history.append({"role": m.role, "content": m.content})
        
    last_msg = req.messages[-1]
    text_input, file_paths, audio_paths = parse_content(last_msg.content)
    
    # Matching `audio` and `files[0]` logic from main.py
    audio_path = audio_paths[0] if audio_paths else None
    file_path = file_paths[0] if file_paths else None

    history_out = svc.process(
        history=history,
        text_input=text_input,
        audio_path=audio_path,
        file_path=file_path
    )

    assistant_msg = history_out[-1]

    response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": assistant_msg.get("role", "assistant"),
                    "content": assistant_msg.get("content", ""),
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    return JSONResponse(content=response)

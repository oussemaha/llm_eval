import logging
import json
import os
from dotenv import load_dotenv

from langfuse import observe
from langfuse.openai import OpenAI # OpenAI integration

logger = logging.getLogger(__name__)

#from openai import OpenAI

load_dotenv()


class Agent:
    def __init__(self,system_prompt_file:str,temperature:float=0.2,**kwargs):

        self.client = OpenAI(
            base_url=os.getenv("vlm_base_url", kwargs.get("host","http://localhost:8000/v1")), api_key=os.getenv("vlm_api_key", kwargs.get("apikey","EMPTY"))
        )
        self.model = os.getenv("vlm_model", kwargs.get("model","Qwen/Qwen3-VL-8B-Instruct"))
        self.temperature = temperature
        with open(system_prompt_file, "r") as f:
            self.system_prompt = f.read()

    @observe()
    def run(self,history: list):
        input_token = 0
        output_token = 0
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
        except Exception as e:
            # Catch API errors to prevent the massive base64 from dumping to the console
            error_msg = str(e)
            # If the error message embeds the full base64, truncate it
            if len(error_msg) > 1000:
                error_msg = (
                    error_msg[:1000] + "... [truncated because it's too long]"
                )
            return f"Sorry, I encountered an API error: {error_msg}"

        message = response.choices[0].message
        input_token += response.usage.prompt_tokens
        output_token += response.usage.completion_tokens

        logger.info(
            f"token usage: input={input_token}, output={output_token}, total={input_token + output_token}"
        )
        return message.content

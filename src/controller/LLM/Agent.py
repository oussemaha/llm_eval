import logging

logger = logging.getLogger(__name__)
import json
import os
from dotenv import load_dotenv

# from langfuse import observe
# rom langfuse.openai import OpenAI # OpenAI integration
from openai import OpenAI

from openai import OpenAI
from src.controller.tools import ToolRegistry

load_dotenv()


class Agent:
    def __init__(self, host: str, apikey: str, model: str, max_steps: int = 5):
        self.max_steps = max_steps

        self.client = OpenAI(
            base_url=os.getenv("base_url", host), api_key=os.getenv("api_key", apikey)
        )
        self.model = os.getenv("LLM", model)

        with open(os.getenv("system_prompt_file"), "r") as f:
            self.system_prompt = f.read()

    # @observe()
    def run(self, tool_registry: ToolRegistry, history: list):
        input_token = 0
        output_token = 0
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
        ]

        for step in range(self.max_steps):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_registry.schemas(),
                    tool_choice="auto",
                )
            except Exception as e:
                # Catch API errors to prevent the massive base64 from dumping to the console
                error_msg = str(e)
                # If the error message embeds the full base64, truncate it
                if len(error_msg) > 1000:
                    error_msg = (
                        error_msg[:1000] + "... [truncated because it's too long]"
                    )
                logger.error(f"LLM API Error during step {step}: {error_msg}")
                return f"Sorry, I encountered an API error: {error_msg}"

            message = response.choices[0].message
            input_token += response.usage.prompt_tokens
            output_token += response.usage.completion_tokens

            # If LLM calls a tool
            if message.tool_calls:
                messages.append(message.model_dump())

                for call in message.tool_calls:
                    tool_name = call.function.name
                    args = json.loads(call.function.arguments)

                    tool = tool_registry.get(tool_name)

                    try:
                        result = tool.run(args)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                        logger.error(result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": str(result),
                        }
                    )

            else:
                logger.info(
                    f"token usage: input={input_token}, output={output_token}, total={input_token + output_token}"
                )
                return message.content

        return "Agent stopped: max steps reached."

    def run_stream(self, tool_registry: ToolRegistry, history: list):
        """
        Stream the final LLM response as text delta chunks (generator).
        Tool calls are resolved silently before streaming begins.
        Yields: str chunks of the assistant response.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can use tools.",
            },
            *history,
        ]

        for step in range(self.max_steps):
            try:
                # Non-streaming pass to resolve tool calls
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_registry.schemas(),
                    tool_choice="auto",
                )
            except Exception as e:
                error_msg = str(e)[:500]
                yield f"[API Error: {error_msg}]"
                return

            message = response.choices[0].message

            if message.tool_calls:
                # Resolve tools silently
                messages.append(message.model_dump())
                for call in message.tool_calls:
                    tool_name = call.function.name
                    args = json.loads(call.function.arguments)
                    tool = tool_registry.get(tool_name)
                    try:
                        result = tool.run(args)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": str(result),
                        }
                    )
            else:
                # Final answer — stream this
                try:
                    stream = self.client.chat.completions.create(
                        model=self.model, messages=messages, stream=True
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            yield delta.content
                except Exception as e:
                    yield f"[Streaming Error: {str(e)[:300]}]"
                return

        yield "[Agent stopped: max steps reached.]"

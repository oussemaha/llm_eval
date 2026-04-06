import logging

logger = logging.getLogger(__name__)

import os

from src.controller.preprocessing.preprocessor import Preprocessor
from src.controller.LLM.Agent import Agent
from src.controller.tools.Retriever_tool import FAISSRetriever_Tool
from src.controller.tools.ToolRegistry import ToolRegistry
from src.controller.tools.web_search_tool import WebSearchTool
from dotenv import load_dotenv

import tempfile
import soundfile as sf


class Service:
    def __init__(self):
        # .env loading
        load_dotenv()

        # tool registry initialization and tool registration
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_list([WebSearchTool(), FAISSRetriever_Tool()])

        # agent initialization
        self.llm = Agent(
            model=os.getenv("LLM"),
            apikey=os.getenv("api_key"),
            host=os.getenv("base_url"),
            max_steps=100,
        )

        self.preprocessor=Preprocessor()        

        pass

    def preprocess(
        self, history: list, text_input: str, audio_path: str, file_path: str
    ):
        content=self.preprocessor.run_v2(text_input, audio_path, file_path)
        history = [
            *history,
            {"role": "user", "content": content},
        ]
        return history

    def process(self, history: list, text_input: str, audio_path: str, file_path: str):

        history = self.preprocess(history, text_input, audio_path, file_path)
        # response generation
        response = self.llm.run(self.tool_registry, history)
        history.append({"role": "assistant", "content": response})
        return history

    def process_stream(
        self,
        history: list,
        text_input: str,
        audio_path: str = None,
        file_path: str = None,
    ):
        """Same as process() but streams the final LLM response as text delta chunks (generator)."""
        messages = self.preprocess(history, text_input, audio_path, file_path)

        # Stream the LLM response
        yield from self.llm.run_stream(self.tool_registry, messages)

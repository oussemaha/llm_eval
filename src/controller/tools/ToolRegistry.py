import logging

logger = logging.getLogger(__name__)
from typing import Dict

from src.controller.tools.Tool import Tool


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def register_list(self, tools: list[Tool]):
        for tool in tools:
            self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools[name]

    def schemas(self):
        return [t.openai_schema() for t in self.tools.values()]

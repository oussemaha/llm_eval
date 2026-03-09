from collections.abc import Callable
from typing import Type

from pydantic import BaseModel


class Tool:
    def __init__(self, name: str, description: str, schema: Type[BaseModel], func: Callable):
        self.name = name
        self.description = description
        self.schema = schema
        self.func = func

    def openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema.model_json_schema(),
            },
        }

    def run(self, args: dict):
        validated = self.schema(**args)
        return self.func(**validated.model_dump())
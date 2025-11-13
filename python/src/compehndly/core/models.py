from dataclasses import dataclass


@dataclass
class FunctionMetadata:
    id: str
    name: str
    description: str
    authors: list[str]

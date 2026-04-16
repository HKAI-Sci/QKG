from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


class LLMRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessageTextOnly:
    role: LLMRole
    content: str


@dataclass
class LLMConfig:
    model: str
    provider: str = "openai_compatible"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

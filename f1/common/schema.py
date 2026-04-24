from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Dict
from pydantic import BaseModel, Field, model_serializer, model_validator
from enum import StrEnum


class LLMProviderEnum(StrEnum):
    OPENAI_COMPATIBLE = "openai-compatible"
    VOLCANO = "volcano"
    LITELLM = "litellm"


class LLMConfig(BaseModel):
    base_url: str = ""  # default empty; not needed for litellm
    api_key: str = ""  # default empty; not required for litellm (e.g. AWS env var auth)
    model: str = Field(description="name of specific model, such as gpt-4o")
    provider: LLMProviderEnum = LLMProviderEnum.OPENAI_COMPATIBLE
    extra_params: dict = Field(default_factory=dict, description="Provider-specific parameters, e.g. aws_region_name for Bedrock")

    @model_validator(mode='after')
    def validate_api_key_for_provider(self):
        """Enforce api_key for providers that require it; LITELLM allows empty (AWS env var auth)
        对需要api_key的provider强制校验；LITELLM允许为空（支持AWS环境变量认证）"""
        if not self.api_key and self.provider not in (LLMProviderEnum.LITELLM,):
            raise ValueError(
                f"api_key is required for provider '{self.provider}'. "
                f"Only the 'litellm' provider supports authentication without api_key (e.g. AWS env vars)."
            )
        return self

class LLMRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class AbstractLLMMessage(BaseModel, ABC):
    role: LLMRole

    @abstractmethod
    def get_serialized_content(self)->Any:
        raise NotImplementedError()

    @model_serializer
    def serialize_to_dict(self):
        return {
            "role": self.role.value,  # Convert role to its string value
            "content": self.get_serialized_content()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Union['LLMMessageTextOnly', 'LLMMessageVisual']:
        """
        Converts a serialized dictionary back into an appropriate LLMMessage instance.

        Args:
            data: A dictionary containing serialized LLM message data

        Returns:
            An instance of either LLMMessageTextOnly or LLMMessageVisual depending on content structure

        Example:
            >>> serialized = {"role": "user", "content": "Hello, how are you?"}
            >>> message = AbstractLLMMessage.from_dict(serialized)
            >>> isinstance(message, LLMMessageTextOnly)
            True

            >>> visual_serialized = {"role": "user", "content": [{"type": "text", "text": "Look at this:"},
            ...                                                 {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}]}
            >>> visual_message = AbstractLLMMessage.from_dict(visual_serialized)
            >>> isinstance(visual_message, LLMMessageVisual)
            True
        """
        role = LLMRole(data["role"])
        content = data["content"]

        # Check if content is a list (indicates LLMMessageVisual)
        if isinstance(content, list):
            visual_msg = LLMMessageVisual(role=role)

            for item in content:
                if item.get("type") == "text":
                    visual_msg.content.append(LLMMessageVisual.TextContent(text=item["text"]))
                elif item.get("type") == "image_url":
                    image_url_data = item["image_url"]
                    image_url = LLMMessageVisual.ImageContent.ImageURL(
                        url=image_url_data["url"],
                        detail=image_url_data.get("detail")
                    )
                    visual_msg.content.append(LLMMessageVisual.ImageContent(image_url=image_url))

            return visual_msg
        else:
            # Simple text message
            return LLMMessageTextOnly(role=role, content=content)

    @classmethod
    @model_validator(mode='before')
    def validate_and_convert_json(cls, data: Any) -> Any:
        """
        Model validator that handles JSON dictionary input for subclasses.
        """
        # If data is already an instance of the expected type, return as-is
        if isinstance(data, cls):
            return data

        # If data is a dictionary with role and content, and we're working with a concrete subclass
        if isinstance(data, dict) and 'role' in data and 'content' in data and cls != AbstractLLMMessage:
            # Use the deserialize_llm_message function to get the right instance
            deserialized_message = AbstractLLMMessage.from_dict(data)

            # If the deserialized message matches the current class, return its data
            if isinstance(deserialized_message, cls):
                return deserialized_message.model_dump()

        return data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractLLMMessage):
            return False
        return self.serialize_to_dict() == other.serialize_to_dict()


class LLMMessageTextOnly(AbstractLLMMessage):
    content: str
    def get_serialized_content(self) -> str:
        return self.content

class LLMMessageVisual(AbstractLLMMessage):
    """
        refer to volcano's doc: https://www.volcengine.com/docs/82379/1362913
    """
    class TextContent(BaseModel):
        type: str = "text"
        text: str

    class ImageContent(BaseModel):
        class ImageURL(BaseModel):
            url: str = Field(description="必选，支持传入图片链接或图片的Base64编码")
            detail: Optional[str] = Field(default=None, description="可选，支持手动设置图片的质量，取值范围high、low、auto")
        type: str = "image_url"
        image_url: ImageURL

    content: list[Union[TextContent, ImageContent]] = Field(default_factory=list)

    def get_serialized_content(self) -> list[dict]:
        ret = list()
        for item in self.content:
            ret.append(item.model_dump(exclude_none=True)) # exclude_none是为了在没有设置ImageURL的detail属性时，不将这个属性传送给LLM
        return ret

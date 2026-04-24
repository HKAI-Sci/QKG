import logging
from abc import ABC, abstractmethod
from typing import List, AsyncIterable, Any, Optional
import asyncio

from openai import Stream, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from f1.common.schema import LLMConfig, LLMProviderEnum, AbstractLLMMessage


class AbstractLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    sync_client: Any   # 用于同步调用的客户端，必须支持.chat.completions.create 方法
    async_client: Any  # 用于异步调用的客户端必须支持.chat.completions.create 方法

    last_trunk_meta_dict: dict # 完整保存上一条返回信息(如果是stream模式，则是这条消息的最后一个trunk)的response.choices[0].to_dict()

    streaming_monitored_trunk_keys: list[str] #
    # 仅用于streaming模式，如果某个trunk内包含key，且key的内容非空，则将这个trunk的meta存入streaming_trunk_meta_dict[key]
    streaming_trunk_meta_dict: dict[str, list] # 仅用于streaming模式，当某个trunk

    def __init__(self, llm_config: LLMConfig,
                 streaming_monitored_trunk_keys: Optional[list[str]] = None):
        """
        Initialize the LLM provider with configuration

        :param llm_config: LLMConfig instance with provider details
        :param streaming_monitored_trunk_keys: List of keys to monitor in streaming, set to None for Non-stream mode
        or not interested in streaming trunk meta
        """
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.llm_config = llm_config
        self.last_trunk_meta_dict = dict()
        self.streaming_monitored_trunk_keys = streaming_monitored_trunk_keys if streaming_monitored_trunk_keys is not None else []
        self.streaming_trunk_meta_dict = dict()
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """
        Initialize the specific client for each provider，完成对sync_client和async_client的赋值
        To be implemented by subclasses
        """
        pass

    @abstractmethod
    def chat_completion(self,
                        messages: List[AbstractLLMMessage],
                        **kwargs) -> str:
        """
        Generate chat completion using the LLM provider

        :param messages: List of message dictionaries
        :param kwargs: following service provider's API guide, e.g.
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for text generation
        :return: Generated chat response
        """
        pass

    @abstractmethod
    async def async_chat_completion(self,
                        messages: List[AbstractLLMMessage],
                        **kwargs) -> str:
        """
        Async version of chat_completion to generate chat completion using the LLM provider
        """
        pass

    # Add to AbstractLLMProvider
    @abstractmethod
    async def chat_completion_stream(self,
                                     messages: List[AbstractLLMMessage],
                                     **kwargs) -> AsyncIterable[str]:
        """
        Generate streaming chat completion using the LLM provider

        :param messages: List of message dictionaries
        :param kwargs: following service provider's API guide, e.g.
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for text generation
        :yield: Generated chat response chunks as they arrive
        """
        pass


class BaseProvider(AbstractLLMProvider):
    """Base class for OpenAI-like providers"""

    def chat_completion(self,
                        messages: List[AbstractLLMMessage],
                        **kwargs) -> str:
        converted_msgs = [x.model_dump() for x in messages]
        latest_msg_str = f"{converted_msgs[-1]}"
        self.logger.debug(f"[Agent->LLM] [Length_As_String = {len(str(converted_msgs)):,}] {latest_msg_str[:1000]}")
        response = self.sync_client.chat.completions.create(
            model=self.llm_config.model,
            messages=converted_msgs,
            **kwargs
        )

        # Defensive handling for empty/None responses
        if not response or not response.choices or len(response.choices) == 0:
            self.logger.warning(f"[LLM] Empty response received from API")
            return ""

        choice = response.choices[0]
        if not choice.message or choice.message.content is None:
            reasoning = getattr(choice.message, "reasoning_content", None) if choice.message else None
            if reasoning:
                self.logger.debug(f"[LLM] content is None, using reasoning_content as fallback")
                return reasoning.strip()
            self.logger.warning(f"[LLM] No message content in response, finish_reason={choice.finish_reason}")
            return ""

        response_str = choice.message.content.strip()
        self.logger.debug(f"[LLM->Agent] {response_str}")
        return response_str

    async def async_chat_completion(self,
                              messages: List[AbstractLLMMessage],
                              **kwargs
                              ) -> str:
        """
        Asynchronous implementation of chat completion

        :param messages: List of message dictionaries
        :param kwargs: following service provider's API guide, e.g.
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for text generation
        :return: Generated chat response
        """
        converted_msgs = [x.model_dump() for x in messages]
        latest_msg_str = f"{converted_msgs[-1]}"
        self.logger.debug(f"[Agent->LLM Async] [Length_As_String = {len(str(converted_msgs)):,}] {latest_msg_str[:1000]}")

        response = await self.async_client.chat.completions.create(
            model=self.llm_config.model,
            messages=converted_msgs,
            **kwargs
        )

        # Defensive handling for empty/None responses
        if not response or not response.choices or len(response.choices) == 0:
            error = getattr(response, "error", None) if response else None
            if error:
                raise RuntimeError(f"[LLM Async] API error (code={error.get('code')}): {error.get('message')}")
            self.logger.warning(f"[LLM Async] Empty response received from API: {response}")
            return ""

        choice = response.choices[0]
        if not choice.message or choice.message.content is None:
            # Fallback to reasoning_content for thinking models (e.g. Qwen3)
            reasoning = getattr(choice.message, "reasoning_content", None) if choice.message else None
            if reasoning:
                self.logger.debug(f"[LLM Async] content is None, using reasoning_content as fallback")
                response_str = reasoning.strip()
                self.last_trunk_meta_dict = choice.model_dump()
                return response_str
            self.logger.warning(f"[LLM Async] No message content in response, finish_reason={choice.finish_reason}")
            return ""

        response_str = choice.message.content.strip()
        self.last_trunk_meta_dict = choice.model_dump()
        self.logger.debug(f"[LLM Async->Agent] {response_str}")
        return response_str

    # Implementation for OpenAIBaseProvider
    async def chat_completion_stream(self, messages: List[AbstractLLMMessage], **kwargs) -> AsyncIterable[str]:
        try:
            converted_msgs = [x.model_dump() for x in messages]
            for msg in converted_msgs:
                msg_str = str(msg)
                self.logger.debug(
                    f"[Agent->LLM Stream] [Length_As_String = {len(msg_str):,}] {msg_str[:200]}"
                )

            stream: Stream[ChatCompletionChunk] = self.sync_client.chat.completions.create(
                model=self.llm_config.model,
                messages=converted_msgs,
                stream=True,
                **kwargs
            )

            # Use async iteration pattern
            for chunk in stream:
                if chunk.choices:
                    choice0 = chunk.choices[0]
                    # capture final chunk meta when finish_reason shows up
                    if getattr(choice0, "finish_reason", None) is not None:
                        self.last_trunk_meta_dict = choice0.model_dump()

                    for monitored_key in self.streaming_monitored_trunk_keys:
                        if hasattr(choice0, monitored_key) and getattr(choice0, monitored_key):
                            if monitored_key not in self.streaming_trunk_meta_dict:
                                self.streaming_trunk_meta_dict[monitored_key] = list()
                            self.streaming_trunk_meta_dict[monitored_key].append(choice0.model_dump())

                    # yield content if present (same behavior as before)
                    if getattr(choice0, "delta", None) is not None and (choice0.delta.content is not None) and choice0.delta.content:
                        content = choice0.delta.content
                        self.logger.debug(f"[LLM Stream->Agent] chunk: {content}")
                        yield content
                        await asyncio.sleep(0)  # Allow other tasks to run

        except Exception as e:
            self.logger.error(f"Error in streaming completion: {str(e)}")
            raise


class OpenAIProvider(BaseProvider):
    """Standard OpenAI Provider"""

    def _initialize_client(self):
        """
        Initialize OpenAI-like client with configurable parameters
        """
        try:
            import openai

            # Allow for additional configuration like Azure setup
            if hasattr(self, '_configure_client'):
                self._configure_client(openai)

            self.sync_client = openai.OpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url
            )
            self.async_client = AsyncOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url
            )
            self.logger.debug(f"base_url = {self.llm_config.base_url}")
        except ImportError:
            raise ImportError("OpenAI SDK is not installed. Please install it using 'pip install openai'")


class VolcanoProvider(BaseProvider):
    """Volcano LLM Provider"""

    def _initialize_client(self):
        """
        Initialize Volcano client with configurable parameters
        """
        try:
            from volcenginesdkarkruntime import Ark, AsyncArk

            self.sync_client = Ark(
                api_key=self.llm_config.api_key
            )
            self.async_client = AsyncArk(
                api_key=self.llm_config.api_key
            )
            self.logger.debug(f"base_url = {self.llm_config.base_url}")
        except ImportError:
            raise ImportError("Volcano SDK is not installed. Please install it using 'pip install volcengine-python-sdk[ark]'")


class LiteLLMProvider(AbstractLLMProvider):
    """LiteLLM Provider — supports AWS Bedrock, Azure, and other backends via litellm"""

    def _initialize_client(self):
        """Verify litellm is importable (lazy import) and validate credentials
        校验litellm可导入，并在无api_key时验证AWS凭证可用"""
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "litellm is not installed. Please install it using 'pip install litellm'"
            )

        # When no api_key, validate AWS credentials are available
        # api_key为空时，校验AWS凭证是否可用（环境变量或extra_params）
        if not self.llm_config.api_key:
            aws_key_in_params = self.llm_config.extra_params.get("aws_access_key_id")
            aws_secret_in_params = self.llm_config.extra_params.get("aws_secret_access_key")
            if not aws_key_in_params or not aws_secret_in_params:
                import os
                missing = []
                for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"):
                    if not os.environ.get(var):
                        missing.append(var)
                if missing:
                    raise ValueError(
                        f"LiteLLM provider requires credentials. Either provide api_key in config, "
                        f"pass aws_access_key_id/aws_secret_access_key in extra_params, "
                        f"or set environment variables. Missing env vars: {', '.join(missing)}"
                    )

    def chat_completion(self,
                        messages: List[AbstractLLMMessage],
                        **kwargs) -> str:
        import litellm

        converted_msgs = [x.model_dump() for x in messages]
        latest_msg_str = f"{converted_msgs[-1]}"
        self.logger.debug(
            f"[Agent->LLM LiteLLM] [Length_As_String = {len(str(converted_msgs)):,}] {latest_msg_str[:1000]}"
        )

        call_kwargs = {"model": self.llm_config.model, "messages": converted_msgs}
        call_kwargs.update(self.llm_config.extra_params)
        call_kwargs.update(kwargs)
        if self.llm_config.api_key:
            call_kwargs["api_key"] = self.llm_config.api_key

        response = litellm.completion(**call_kwargs)

        # Defensive handling for empty/None responses 防御性处理空响应
        if not response or not response.choices or len(response.choices) == 0:
            self.logger.warning("[LLM LiteLLM] Empty response received from API")
            return ""

        choice = response.choices[0]
        if not choice.message or choice.message.content is None:
            reasoning = getattr(choice.message, "reasoning_content", None) if choice.message else None
            if reasoning:
                self.logger.debug(f"[LLM LiteLLM] content is None, using reasoning_content as fallback")
                self.last_trunk_meta_dict = choice.model_dump() if hasattr(choice, "model_dump") else {}
                return reasoning.strip()
            self.logger.warning(
                f"[LLM LiteLLM] No message content in response, finish_reason={choice.finish_reason}"
            )
            return ""

        response_str = choice.message.content.strip()
        self.last_trunk_meta_dict = choice.model_dump() if hasattr(choice, "model_dump") else {}
        self.logger.debug(f"[LLM LiteLLM->Agent] {response_str}")
        return response_str

    async def async_chat_completion(self,
                                    messages: List[AbstractLLMMessage],
                                    **kwargs) -> str:
        import litellm

        converted_msgs = [x.model_dump() for x in messages]
        latest_msg_str = f"{converted_msgs[-1]}"
        self.logger.debug(
            f"[Agent->LLM LiteLLM Async] [Length_As_String = {len(str(converted_msgs)):,}] {latest_msg_str[:1000]}"
        )

        call_kwargs = {"model": self.llm_config.model, "messages": converted_msgs}
        call_kwargs.update(self.llm_config.extra_params)
        call_kwargs.update(kwargs)
        if self.llm_config.api_key:
            call_kwargs["api_key"] = self.llm_config.api_key

        response = await litellm.acompletion(**call_kwargs)

        # Defensive handling for empty/None responses 防御性处理空响应
        if not response or not response.choices or len(response.choices) == 0:
            self.logger.warning("[LLM LiteLLM Async] Empty response received from API")
            return ""

        choice = response.choices[0]
        if not choice.message or choice.message.content is None:
            reasoning = getattr(choice.message, "reasoning_content", None) if choice.message else None
            if reasoning:
                self.logger.debug(f"[LLM LiteLLM Async] content is None, using reasoning_content as fallback")
                self.last_trunk_meta_dict = choice.model_dump() if hasattr(choice, "model_dump") else {}
                return reasoning.strip()
            self.logger.warning(
                f"[LLM LiteLLM Async] No message content in response, finish_reason={choice.finish_reason}"
            )
            return ""

        response_str = choice.message.content.strip()
        self.last_trunk_meta_dict = choice.model_dump() if hasattr(choice, "model_dump") else {}
        self.logger.debug(f"[LLM LiteLLM Async->Agent] {response_str}")
        return response_str

    async def chat_completion_stream(self,
                                     messages: List[AbstractLLMMessage],
                                     **kwargs) -> AsyncIterable[str]:
        import litellm

        try:
            converted_msgs = [x.model_dump() for x in messages]
            for msg in converted_msgs:
                msg_str = str(msg)
                self.logger.debug(
                    f"[Agent->LLM LiteLLM Stream] [Length_As_String = {len(msg_str):,}] {msg_str[:200]}"
                )

            call_kwargs = {"model": self.llm_config.model, "messages": converted_msgs, "stream": True}
            call_kwargs.update(self.llm_config.extra_params)
            call_kwargs.update(kwargs)
            if self.llm_config.api_key:
                call_kwargs["api_key"] = self.llm_config.api_key

            response = await litellm.acompletion(**call_kwargs)

            async for chunk in response:
                if chunk.choices:
                    choice0 = chunk.choices[0]
                    # capture final chunk meta when finish_reason shows up
                    if getattr(choice0, "finish_reason", None) is not None:
                        self.last_trunk_meta_dict = (
                            choice0.model_dump() if hasattr(choice0, "model_dump") else {}
                        )

                    for monitored_key in self.streaming_monitored_trunk_keys:
                        if hasattr(choice0, monitored_key) and getattr(choice0, monitored_key):
                            if monitored_key not in self.streaming_trunk_meta_dict:
                                self.streaming_trunk_meta_dict[monitored_key] = list()
                            self.streaming_trunk_meta_dict[monitored_key].append(
                                choice0.model_dump() if hasattr(choice0, "model_dump") else {}
                            )

                    # yield content if present
                    if (
                        getattr(choice0, "delta", None) is not None
                        and choice0.delta.content is not None
                        and choice0.delta.content
                    ):
                        content = choice0.delta.content
                        self.logger.debug(f"[LLM LiteLLM Stream->Agent] chunk: {content}")
                        yield content
                        await asyncio.sleep(0)  # Allow other tasks to run

        except Exception as e:
            self.logger.error(f"Error in LiteLLM streaming completion: {str(e)}")
            raise


class LLMProviderFactory:

    @classmethod
    def create_instance(cls, llm_config: LLMConfig,
                        streaming_monitored_trunk_keys: Optional[list[str]] = None
                        ) -> AbstractLLMProvider:
        """
        Factory method to create an instance of the appropriate LLM provider based on the provider field.
        根据provider字段选择对应的LLM provider实例

        :param llm_config: LLMConfig instance with provider details
        :param streaming_monitored_trunk_keys: List of keys to monitor in streaming mode
        :return: Instance of AbstractLLMProvider subclass
        """
        kwargs = dict(streaming_monitored_trunk_keys=streaming_monitored_trunk_keys) if streaming_monitored_trunk_keys else {}

        match llm_config.provider:
            case LLMProviderEnum.OPENAI_COMPATIBLE:
                return OpenAIProvider(llm_config, **kwargs)
            case LLMProviderEnum.VOLCANO:
                return VolcanoProvider(llm_config, **kwargs)
            case LLMProviderEnum.LITELLM:
                return LiteLLMProvider(llm_config, **kwargs)
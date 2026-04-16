import asyncio

import boto3
from openai import AsyncOpenAI


class OpenAICompatibleProvider:
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def async_chat_completion(self, messages, temperature=0, max_tokens=3000):
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class BedrockProvider:
    def __init__(self, config):
        kwargs = {}
        if config.aws_region:
            kwargs["region_name"] = config.aws_region
        if config.aws_access_key_id:
            kwargs["aws_access_key_id"] = config.aws_access_key_id
        if config.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = config.aws_secret_access_key
        self.client = boto3.client("bedrock-runtime", **kwargs)
        self.model = config.model

    def _to_bedrock_messages(self, messages):
        system = []
        conversation = []
        for msg in messages:
            if msg.role.value == "system":
                system.append({"text": msg.content})
            else:
                conversation.append(
                    {"role": msg.role.value, "content": [{"text": msg.content}]}
                )
        return system, conversation

    async def async_chat_completion(self, messages, temperature=0, max_tokens=3000):
        system, conversation = self._to_bedrock_messages(messages)

        def _call():
            response = self.client.converse(
                modelId=self.model,
                system=system,
                messages=conversation,
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                },
            )
            content = response["output"]["message"]["content"]
            return "".join(part.get("text", "") for part in content)

        return await asyncio.to_thread(_call)


class LLMProviderFactory:
    @staticmethod
    def create_instance(config):
        provider = (config.provider or "openai_compatible").lower()
        if provider in {"openai_compatible", "openrouter", "openai"}:
            return OpenAICompatibleProvider(config)
        if provider in {"aws", "bedrock"}:
            return BedrockProvider(config)
        raise ValueError(f"Unsupported provider: {config.provider}")

import json
from code.config import Config

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel


class LLMProvider:
    def __init__(self, provider="openailike"):
        self.provider = provider

    async def infer(
        self,
        model: str = None,
        prompt: str = None,
        response_model: BaseModel = None,
        max_tokens=1024,
    ):
        # 通用LLM推理
        if self.provider == "openailike":
            client = instructor.from_openai(
                AsyncOpenAI(
                    base_url=Config.OPENAI_BASE_URL, api_key=Config.OPENAI_API_KEY
                ),
                mode=instructor.Mode.JSON,
            )

            response = await client.chat.completions.create(
                max_tokens=max_tokens,
                model=model,
                response_model=response_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_retries=3,
            )
            if response_model:
                return json.loads(response.json())
            else:
                return response
            
        # 剧本生成器
        elif self.provider == "scriptwriter":
            client = instructor.from_openai(
                AsyncOpenAI(
                    base_url=Config.OPENAI_BASE_URL, api_key=Config.OPENAI_API_KEY
                ),
                mode=instructor.Mode.JSON,
            )

            response = await client.chat.completions.create(
                max_tokens=max_tokens,
                model=model,
                response_model=response_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_retries=3,
            )
            if response_model:
                return json.loads(response.json())
            else:
                return response
            

        else:
            # TODO：其他LLM Provider请自行实现
            raise NotImplementedError("Only support openailike provider now.")

import asyncio
import os
import sys
from pathlib import Path
import random
import ray
import time
from enum import Enum
from multiprocessing import cpu_count
from typing import Any, List, Optional, Union, overload
import httpx
from openai import NOT_GIVEN, APIConnectionError, AsyncOpenAI, NotGiven, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from pydantic import BaseModel, Field, field_serializer, model_validator
from SScheduler.timestepManager import BaseManager
from SScheduler.logger import get_logger

os.environ["GRPC_VERBOSITY"] = "ERROR"

logger = get_logger()

__all__ = [
    "LLM",
    "LLMConfig",
    "LLMProviderType",
]

MAX_TIMEOUT = 60

class LLMProviderType(str, Enum):
    """
    Other: using API
    Local: using local server, required to follow openai API, no matter using vllm or sglang
    """
    OpenAI = "openai"
    DeepSeek = "deepseek"
    Qwen = "qwen"
    SiliconFlow = "siliconflow"
    Local = "local"

class LLMConfig(BaseModel):
    """LLM configuration class."""

    provider: LLMProviderType = Field(...)
    """The type of the LLM provider"""

    base_url: Optional[str] = Field(None)
    """The base URL for the LLM provider"""

    api_key: str = Field(...)
    """API key for accessing the LLM provider"""

    model: str = Field(...)
    """The model to use"""

    concurrency: int = Field(200, ge=1)
    """Concurrency value for LLM operations to avoid rate limit"""

    timeout: float = Field(30, ge=1, le=MAX_TIMEOUT)
    """Timeout for LLM operations in seconds"""

    @field_serializer("provider")
    def serialize_provider(self, provider: LLMProviderType, info):
        return provider.value

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.provider != LLMProviderType.Local and self.base_url is not None:
            raise ValueError("base_url is not supported for this provider")
        return self


@ray.remote(num_cpus=1)
class LLMActor:
    """
    Actor class for LLM operations.
    """

    def __init__(self):
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=min(30.0, MAX_TIMEOUT / 4),  # 连接超时时间
                read=MAX_TIMEOUT,  # 读取超时时间
                write=MAX_TIMEOUT,  # 写入超时时间
                pool=MAX_TIMEOUT,  # 连接池超时时间
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
            ),
        )

    async def call(
        self,
        config: LLMConfig,
        api_key: str,
        api_base: str,
        model: str,
        agent_id: int,
        dialog: list[ChatCompletionMessageParam],
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        timeout: int = 300,
        retries: int = 10,
        tools: Union[List[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
    ):

        start_time = time.time()

        log = {
            "request_time": start_time,
            "total_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            base_url=api_base,
            http_client=self._http_client,
        )
        for attempt in range(retries):
            response = None
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=dialog,
                    # agent_id=agent_id,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stream=False,
                    timeout=timeout,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                if response.usage is not None:
                    log["input_tokens"] += response.usage.prompt_tokens
                    log["output_tokens"] += response.usage.completion_tokens
                else:
                    get_logger().warning(f"No usage in response: {response}")
                end_time = time.time()
                log["consumption"] = end_time - start_time
                if tools:
                    return response, log
                else:
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("No content in response")
                    return content, log
            except Exception as e:
                get_logger().warning(
                    f"LLM Error: `{e}` for request {dialog} {tools} {tool_choice}. original response: `{response}`. Retry {attempt+1} of {retries}"
                )
                log["total_errors"] += 1
                if attempt < retries - 1:
                    time.sleep(random.random() * 2**attempt)
                else:
                    raise e
        raise RuntimeError("Failed to get response from LLM")


class LLM:
    """
    LLM Engine class.
    Support using API or local server.
    
    Features for local server: 
        Follow openai API.
        Support multiple client for high concurrency.
        Support PFEngine's interface if registered when initializing.
    """

    def __init__(self, config: LLMConfig, num_clients: int = min(cpu_count(), 8)):
        """
        Initializes the LLM Instance.

        Args:
            config: LLMConfig.
            num_client: Number of clients to handle concurrency.
        """

        if config is None:
            raise ValueError("No LLM config is provided, please check your configuration")

        self.config = config
        self.semaphores = asyncio.Semaphore(self.config.concurrency)
        self.log_list = []
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.next_client = 0
        self.lock = asyncio.Lock()
        self.model = self.config.model
        self.api_key = self.config.api_key
        self.timeout = self.config.timeout

        base_url = self.config.base_url
        if base_url is not None:
            base_url = base_url.rstrip("/")
        if self.config.provider == LLMProviderType.OpenAI:
            base_url = "https://api.openai.com/v1"
        elif self.config.provider == LLMProviderType.DeepSeek:
            base_url = "https://api.deepseek.com/v1"
        elif self.config.provider == LLMProviderType.Qwen:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif self.config.provider == LLMProviderType.SiliconFlow:
            base_url = "https://api.siliconflow.cn/v1"
        elif self.config.provider == LLMProviderType.Local:
            base_url = self.config.base_url
        else:
            raise ValueError(f"Unsupported `provider` {self.config.provider}!")
        config.base_url = base_url
        self.api_base = base_url


        self.clients = [LLMActor.remote() for _ in range(num_clients)]
        self.managers: dict[str, BaseManager] = {}

    def get_log_list(self):
        return self.log_list

    def clear_log_list(self):
        self.log_list = []

    def get_next_client(self):
        self.next_client += 1
        self.next_client %= len(self.clients)
        # print(f"Next client index: {self.next_client}")
        return self.clients[self.next_client]

    def add_manager(self, manager_instance: BaseManager, manager_name: Optional[str] = None) -> bool:
        """
        Add a manager instance
        
        Args:
            manager_instance: An instance of a manager that inherits from BaseManager
            manager_name: Optional custom name for the manager. If not provided, will use class name
        Returns:
            True if successful, False otherwise
        """
        try:
            if not isinstance(manager_instance, BaseManager):
                logger.error(f"Manager instance must inherit from BaseManager, got {type(manager_instance)}")
                return False

            if manager_name is None:
                manager_name = manager_instance.__class__.__name__
            
            if manager_name in self.managers:
                logger.warning(f"Manager '{manager_name}' already exists, replacing it")
                return False
            
            self.managers[manager_name] = manager_instance

            logger.debug(f"[PFEngine-llm][add_manager][result]: {manager_name} registered")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register manager: {e}")
            return False
    
    def remove_manager(self, manager_name: str) -> bool:
        """
        Remove a manager
        
        Args:
            manager_name: Name of the manager to unregister
        Returns:
            True if successful, False otherwise
        """
        if manager_name not in self.managers:
            logger.warning(f"Manager '{manager_name}' not found")
            return False
        
        try:
            del self.managers[manager_name]
            logger.info(f"Successfully removed manager: {manager_name}")
            return True
        except Exception as e:
            logger.error(f"Error removing manager {manager_name}: {e}")
            return False

    def list_managers(self) -> dict[str, str]:
        """
        List all added managers

        Returns:
            Dictionary mapping manager names to their class names
        """
        return {name: type(manager).__name__ for name, manager in self.managers.items()}

    @overload
    async def generate(
        self,
        dialog: list[ChatCompletionMessageParam],
        agent_id: int,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        # agent_id: Optional[int] = None,
        emergency: Optional[bool] = False,
        dependency: Optional[Any] = None,
        manager_name: Optional[str] = None,
        timeout: int = 300,
        retries: int = 10,
        tools: NotGiven = NOT_GIVEN,
        tool_choice: NotGiven = NOT_GIVEN,
    ) -> str: ...

    @overload
    async def generate(
        self,
        dialog: list[ChatCompletionMessageParam],
        agent_id: int,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        # agent_id: Optional[int] = None,
        emergency: Optional[bool] = False,
        dependency: Optional[Any] = None,
        manager_name: Optional[str] = None,
        timeout: int = 300,
        retries: int = 10,
        tools: List[ChatCompletionToolParam] = [],
        tool_choice: ChatCompletionToolChoiceOptionParam = "auto",
    ) -> Any: ...

    async def generate(
        self,
        dialog: list[ChatCompletionMessageParam],
        agent_id: int,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        # agent_id: Optional[int] = None,
        emergency: Optional[bool] = False,
        dependency: Optional[Any] = None,
        manager_name: Optional[str] = None,
        timeout: int = 300,
        retries: int = 10,
        tools: Union[List[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
    ):
        """
        Sends an asynchronous text request to the configured LLM API.

        Args:
            agent_id: ID of the agent making the request.
            api_key: API key for accessing the LLM provider.
            dialog: Messages to send as part of the chat completion request.
            response_format: JSON schema for the response. Default is NOT_GIVEN.
            temperature: Controls randomness in the model's output. Default is 1.
            max_tokens: Maximum number of tokens to generate in the response. Default is None.
            top_p: Limits the next token selection to a subset of tokens with a cumulative probability above this value. Default is None.
            frequency_penalty: Penalizes new tokens based on their existing frequency in the text so far. Default is None.
            presence_penalty: Penalizes new tokens based on whether they appear in the text so far. Default is None.
            timeout: Request timeout in seconds. Default is 300 seconds.
            retries: Number of retry attempts in case of failure. Default is 10.
            tools: List of dictionaries describing the tools that can be called by the model. Default is NOT_GIVEN.
            tool_choice: Dictionary specifying how the model should choose from the provided tools. Default is NOT_GIVEN.

        Returns:
            A string containing the message content or a dictionary with tool call arguments if tools are used.
        """
        api_key = str(self.config.api_key)
        api_base = str(self.config.base_url)
        model = str(self.config.model)
        if agent_id is not None and manager_name is not None and manager_name in self.managers:
            manager = self.managers[manager_name]
            manager.update_agent_timestep(
                agent_id=agent_id, obj=dependency, emergency=emergency
            )

        client = self.get_next_client()
        async with self.semaphores:
            content, log = await client.call.remote(
                config=self.config,
                api_key=api_key,
                api_base=api_base,
                model=model,
                agent_id=agent_id,
                dialog=dialog,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                timeout=timeout,
                retries=retries,
                tools=tools,
                tool_choice=tool_choice,
            )
            self.log_list.append(log)
            self.prompt_tokens_used += log["input_tokens"]
            self.completion_tokens_used += log["output_tokens"]
        return content






# import asyncio
# import os
# import random
# import ray
# import time
# from enum import Enum
# from multiprocessing import cpu_count
# from typing import Any, List, Optional, Union, overload
# import httpx
# from openai import NOT_GIVEN, APIConnectionError, AsyncOpenAI, NotGiven, OpenAIError
# from openai.types.chat import (
#     ChatCompletionMessageParam,
#     ChatCompletionToolChoiceOptionParam,
#     ChatCompletionToolParam,
#     completion_create_params,
# )
# from pydantic import BaseModel, Field, field_serializer, model_validator

# from ..timestepManager.BaseManager import BaseManager
# from ..logger import get_logger, set_logger_level

# os.environ["GRPC_VERBOSITY"] = "ERROR"

# logger = get_logger()

# __all__ = [
#     "LLM",
#     "LLMConfig",
#     "LLMProviderType",
# ]

# MAX_TIMEOUT = 60

# class LLMProviderType(str, Enum):
#     """
#     Other: using API
#     Local: using local server, required to follow openai API, no matter using vllm or sglang
#     """
#     OpenAI = "openai"
#     DeepSeek = "deepseek"
#     Qwen = "qwen"
#     SiliconFlow = "siliconflow"
#     Local = "local"

# class LLMConfig(BaseModel):
#     """LLM configuration class."""

#     provider: LLMProviderType = Field(...)
#     """The type of the LLM provider"""

#     base_url: Optional[str] = Field(None)
#     """The base URL for the LLM provider"""

#     api_key: str = Field(...)
#     """API key for accessing the LLM provider"""

#     model: str = Field(...)
#     """The model to use"""

#     concurrency: int = Field(200, ge=1)
#     """Concurrency value for LLM operations to avoid rate limit"""

#     timeout: float = Field(30, ge=1, le=MAX_TIMEOUT)
#     """Timeout for LLM operations in seconds"""

#     @field_serializer("provider")
#     def serialize_provider(self, provider: LLMProviderType, info):
#         return provider.value

#     @model_validator(mode="after")
#     def validate_configuration(self):
#         if self.provider != LLMProviderType.Local and self.base_url is not None:
#             raise ValueError("base_url is not supported for this provider")
#         return self

# @ray.remote
# class LLMActor:
#     """
#     Actor class for LLM operations.
#     """
#     def __init__(self):
#         self._http_client = httpx.AsyncClient(
#             timeout=httpx.Timeout(
#                 connect=min(30.0, MAX_TIMEOUT / 4),  # 连接超时时间
#                 read=MAX_TIMEOUT,  # 读取超时时间
#                 write=MAX_TIMEOUT,  # 写入超时时间
#                 pool=MAX_TIMEOUT,  # 连接池超时时间
#             ),
#             limits=httpx.Limits(
#                 max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
#             ),
#         )
#         # self.model = model
#         # self.api_key = api_key
#         # self.api_base = api_base

#     async def call(
#         self,
#         config: LLMConfig,
#         model: Optional[str],
#         api_key: Optional[str],
#         api_base: Optional[str],
#         agent_id: Optional[int],
#         dialog: list[ChatCompletionMessageParam],
#         response_format: Union[
#             completion_create_params.ResponseFormat, NotGiven
#         ] = NOT_GIVEN,
#         temperature: float = 1,
#         max_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         frequency_penalty: Optional[float] = None,
#         presence_penalty: Optional[float] = None,
#         retries: int = 10,
#         timeout: Optional[int] = 100,
#         tools: Union[List[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
#         tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
#     ):
#         """
#         Sends an asynchronous text request to the configured LLM API.

#         """

#         start_time = time.time()

#         log = {
#             "request_time": start_time,
#             "consumption": 0,
#             "total_errors": 0,
#             "input_tokens": 0,
#             "output_tokens": 0,
#         }

#         client = AsyncOpenAI(
#             api_key=api_key,
#             timeout=timeout,
#             base_url=api_base,
#             http_client=self._http_client,
#         )
        
#         for attempt in range(retries):
#             response = None
#             try:
#                 response = await client.chat.completions.create(
#                     model=model,
#                     messages=dialog,
#                     # agent_id=agent_id,
#                     response_format=response_format,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     top_p=top_p,
#                     frequency_penalty=frequency_penalty,
#                     presence_penalty=presence_penalty,
#                     stream=False,
#                     timeout=timeout,
#                     tools=tools,
#                     tool_choice=tool_choice,
#                 )
#                 end_time = time.time()
#                 log["consumption"] = end_time - start_time
#                 if response.usage is not None:
#                     log["input_tokens"] += response.usage.prompt_tokens
#                     log["output_tokens"] += response.usage.completion_tokens
#                 else:
#                     get_logger().warning(f"No usage in response: {response}")
#                 if tools:
#                     return response, log
#                 else:
#                     content = response.choices[0].message.content
#                     if content is None:
#                         raise ValueError("No content in response")
#                     return content, log
#             except Exception as e:
#                 get_logger().warning(f"LLM Error: `{e}` for request {dialog} {tools} {tool_choice}. original response: `{response}`. Retry {attempt+1} of {retries}")
#                 log["total_errors"] += 1
#         raise RuntimeError("Failed to get response from LLM")


# class LLM:
#     """
#     LLM Engine class.
#     Support using API or local server.
    
#     Features for local server: 
#         Follow openai API.
#         Support multiple client for high concurrency.
#         Support PFEngine's interface if registered when initializing.
#     """

#     def __init__(self, config: LLMConfig, num_client: int = 8):
#         """
#         Initializes the LLM Instance.

#         Args:
#             config: LLMConfig.
#             num_client: Number of clients to handle concurrency.
#         """

#         if config is None:
#             raise ValueError("No LLM config is provided, please check your configuration")

#         self.config = config
#         self.semaphores = asyncio.Semaphore(self.config.concurrency)
#         self.log_list = []
#         self.prompt_tokens_used = 0
#         self.completion_tokens_used = 0
#         self.next_client = 0
#         self.lock = asyncio.Lock()
#         self.model = self.config.model
#         self.api_key = self.config.api_key
#         self.timeout = self.config.timeout

#         base_url = self.config.base_url
#         if base_url is not None:
#             base_url = base_url.rstrip("/")
#         if self.config.provider == LLMProviderType.OpenAI:
#             base_url = "https://api.openai.com/v1"
#         elif self.config.provider == LLMProviderType.DeepSeek:
#             base_url = "https://api.deepseek.com/v1"
#         elif self.config.provider == LLMProviderType.Qwen:
#             base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#         elif self.config.provider == LLMProviderType.SiliconFlow:
#             base_url = "https://api.siliconflow.cn/v1"
#         elif self.config.provider == LLMProviderType.Local:
#             base_url = self.config.base_url
#         else:
#             raise ValueError(f"Unsupported `provider` {self.config.provider}!")
#         config.base_url = base_url
#         self.api_base = base_url

#         # self.clients = [LLMActor.remote(self.api_key, self.api_base, self.model) for _ in range(num_client)]
#         self.clients = [LLMActor.remote() for _ in range(num_client)]
#         self.managers: dict[str, BaseManager] = {}
        

#     def get_log_list(self):
#         return self.log_list

#     def clear_log_list(self):
#         self.log_list = []

#     def get_next_client(self):
#         self.next_client += 1
#         self.next_client %= len(self.clients)
#         return self.clients[self.next_client]

#     def add_manager(self, manager_instance: BaseManager, manager_name: Optional[str] = None) -> bool:
#         """
#         Add a manager instance
        
#         Args:
#             manager_instance: An instance of a manager that inherits from BaseManager
#             manager_name: Optional custom name for the manager. If not provided, will use class name
#         Returns:
#             True if successful, False otherwise
#         """
#         try:
#             if not isinstance(manager_instance, BaseManager):
#                 logger.error(f"Manager instance must inherit from BaseManager, got {type(manager_instance)}")
#                 return False

#             if manager_name is None:
#                 manager_name = manager_instance.__class__.__name__
            
#             if manager_name in self.managers:
#                 logger.warning(f"Manager '{manager_name}' already exists, replacing it")
#                 return False
            
#             self.managers[manager_name] = manager_instance

#             logger.debug(f"[PFEngine-llm][add_manager][result]: {manager_name} registered")
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to register manager: {e}")
#             return False
    
#     def remove_manager(self, manager_name: str) -> bool:
#         """
#         Remove a manager
        
#         Args:
#             manager_name: Name of the manager to unregister
#         Returns:
#             True if successful, False otherwise
#         """
#         if manager_name not in self.managers:
#             logger.warning(f"Manager '{manager_name}' not found")
#             return False
        
#         try:
#             del self.managers[manager_name]
#             logger.info(f"Successfully removed manager: {manager_name}")
#             return True
#         except Exception as e:
#             logger.error(f"Error removing manager {manager_name}: {e}")
#             return False

#     def list_managers(self) -> dict[str, str]:
#         """
#         List all added managers

#         Returns:
#             Dictionary mapping manager names to their class names
#         """
#         return {name: type(manager).__name__ for name, manager in self.managers.items()}

#     @overload
#     async def generate(
#         self,
#         dialog: list[ChatCompletionMessageParam],
#         agent_id: Optional[int] = None,
#         response_format: Union[
#             completion_create_params.ResponseFormat, NotGiven
#         ] = NOT_GIVEN,
#         temperature: float = 1,
#         max_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         frequency_penalty: Optional[float] = None,
#         presence_penalty: Optional[float] = None,
#         model: Optional[str] = None,
#         api_base: Optional[str] = None,
#         api_key: Optional[str] = None,
#         emergency: Optional[bool] = False,
#         dependency: Optional[Any] = None,
#         manager_name: Optional[str] = None,
#         timeout: int = 300,
#         retries: int = 10,
#         tools: NotGiven = NOT_GIVEN,
#         tool_choice: NotGiven = NOT_GIVEN,
#     ) -> str: ...

#     @overload
#     async def generate(
#         self,
#         dialog: list[ChatCompletionMessageParam],
#         agent_id: Optional[int] = None,
#         response_format: Union[
#             completion_create_params.ResponseFormat, NotGiven
#         ] = NOT_GIVEN,
#         temperature: float = 1,
#         max_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         frequency_penalty: Optional[float] = None,
#         presence_penalty: Optional[float] = None,
#         model: Optional[str] = None,
#         api_base: Optional[str] = None,
#         api_key: Optional[str] = None,
#         emergency: Optional[bool] = False,
#         dependency: Optional[Any] = None,
#         manager_name: Optional[str] = None,
#         timeout: int = 300,
#         retries: int = 10,
#         tools: List[ChatCompletionToolParam] = [],
#         tool_choice: ChatCompletionToolChoiceOptionParam = "auto",
#     ) -> Any: ...

#     async def generate(
#         self,
#         dialog: list[ChatCompletionMessageParam],
#         agent_id: Optional[int] = None,
#         response_format: Union[
#             completion_create_params.ResponseFormat, NotGiven
#         ] = NOT_GIVEN,
#         temperature: float = 1,
#         max_tokens: Optional[int] = None,
#         top_p: Optional[float] = None,
#         frequency_penalty: Optional[float] = None,
#         presence_penalty: Optional[float] = None,
#         model: Optional[str] = None,
#         api_base: Optional[str] = None,
#         api_key: Optional[str] = None,
#         emergency: Optional[bool] = False,
#         dependency: Optional[Any] = None,
#         manager_name: Optional[str] = None,
#         timeout: int = 300,
#         retries: int = 10,
#         tools: Union[List[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
#         tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
#     ):
#         """
#         Sends an asynchronous text request to the configured LLM API.

#         Args:
#             agent_id: ID of the agent making the request.
#             api_key: API key for accessing the LLM provider.
#             dialog: Messages to send as part of the chat completion request.
#             response_format: JSON schema for the response. Default is NOT_GIVEN.
#             temperature: Controls randomness in the model's output. Default is 1.
#             max_tokens: Maximum number of tokens to generate in the response. Default is None.
#             top_p: Limits the next token selection to a subset of tokens with a cumulative probability above this value. Default is None.
#             frequency_penalty: Penalizes new tokens based on their existing frequency in the text so far. Default is None.
#             presence_penalty: Penalizes new tokens based on whether they appear in the text so far. Default is None.
#             timeout: Request timeout in seconds. Default is 300 seconds.
#             retries: Number of retry attempts in case of failure. Default is 10.
#             tools: List of dictionaries describing the tools that can be called by the model. Default is NOT_GIVEN.
#             tool_choice: Dictionary specifying how the model should choose from the provided tools. Default is NOT_GIVEN.

#         Returns:
#             A string containing the message content or a dictionary with tool call arguments if tools are used.
#         """

#         api_key = str(self.config.api_key)
#         api_base = str(self.config.base_url)
#         model = str(self.config.model)

#         if agent_id is not None and manager_name is not None and manager_name in self.managers:
#             manager = self.managers[manager_name]
#             manager.update_agent_timestep(
#                 agent_id=agent_id, obj=dependency, emergency=emergency
#             )

#         client = self.get_next_client()
#         async with self.semaphores:
#             content, log = await client.call.remote(
#                 config=self.config,
#                 model=model,
#                 api_key=api_key,
#                 api_base=api_base,
#                 agent_id=agent_id,
#                 dialog=dialog,
#                 response_format=response_format,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 top_p=top_p,
#                 frequency_penalty=frequency_penalty,
#                 presence_penalty=presence_penalty,
#                 timeout=timeout,
#                 retries=retries,
#                 tools=tools,
#                 tool_choice=tool_choice,
#             )
#             self.log_list.append(log)
#             self.prompt_tokens_used += log["input_tokens"]
#             self.completion_tokens_used += log["output_tokens"]

#         return content



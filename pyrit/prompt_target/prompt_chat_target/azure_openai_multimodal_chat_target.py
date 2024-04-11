# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from openai import AsyncAzureOpenAI, AzureOpenAI

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptChatTarget
from .openai_chat_target import OpenAIChatInterface

logger = logging.getLogger(__name__)


from pyrit.memory import MemoryInterface

class AzureOpenAIMultiModalChatTarget(OpenAIChatInterface):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_MULTIMODAL_CHAT_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_MULTIMODAL_CHAT_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_MULTIMODAL_CHAT_DEPLOYMENT"
    ALLOWED_INTERNAL_HEADER_VARIABLE: str = "AZURE_OPENAI_MULTIMODAL_ALLOWED_INTERNAL_HEADER"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        allowed_internal_header: str = None,
        memory: MemoryInterface = None,
        api_version: str = "2023-08-01-preview",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> None:
        """
        Class that initializes an Azure Open AI chat target

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                DEPLOYMENT_ENVIRONMENT_VARIABLE environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
            api_version (str, optional): The version of the Azure OpenAI API. Defaults to
                "2023-08-01-preview".
            max_tokens (int, optional): The maximum number of tokens to generate in the response.
                Defaults to 1024.
            temperature (float, optional): The temperature parameter for controlling the
                randomness of the response. Defaults to 1.0.
            top_p (int, optional): The top-p parameter for controlling the diversity of the
                response. Defaults to 1.
            frequency_penalty (float, optional): The frequency penalty parameter for penalizing
                frequently generated tokens. Defaults to 0.5.
            presence_penalty (float, optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history. Defaults to 0.5.
        """
        PromptChatTarget.__init__(self, memory=memory)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        
        allowed_internal_headers = default_values.get_required_value(
            env_var_name=self.ALLOWED_INTERNAL_HEADER_VARIABLE, passed_value=allowed_internal_header
        )
        headers: dict = {}
        if allowed_internal_headers:
            headers['allowed-internal-headers'] = allowed_internal_headers

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            default_headers=headers
        )
        self._async_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            default_headers=headers
        )

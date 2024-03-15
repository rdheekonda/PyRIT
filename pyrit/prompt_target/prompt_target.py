# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from pyrit.memory import MemoryInterface
from pyrit.memory import DuckDBMemory


class PromptTarget(abc.ABC):
    memory: MemoryInterface

    """
    A list of PromptConverters that are supported by the prompt target.
    An empty list implies that the prompt target supports all converters.
    """
    supported_converters: list

    def __init__(self, memory: MemoryInterface) -> None:
        self.memory = memory if memory else DuckDBMemory()

    @abc.abstractmethod
    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        """
        Sends a normalized prompt to the prompt target.
        """

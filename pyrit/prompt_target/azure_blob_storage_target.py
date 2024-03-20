# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from azure.core.exceptions import ClientAuthenticationError
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from azure.storage.blob import ContainerClient, ContentSettings
from enum import Enum
import logging

from pyrit.common import default_values
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class SupportedContentType(Enum):
    """
    All supported content types for uploading blobs to provided storage account container.
    See all options here: https://www.iana.org/assignments/media-types/media-types.xhtml
    """

    PLAIN_TEXT = "text/plain"


class AzureBlobStorageTarget(PromptTarget):
    """
    The AzureBlobStorageTarget takes prompts, saves the prompts to a file, and stores them as a blob in a provided
    storage account container.

    Args:
        container_url (str): URL to the Azure Blob Storage Container.
        sas_token (str): Blob SAS token required to authenticate blob operations.
        blob_content_type (SupportedContentType): Expected Content Type of the blob, chosen from the
            SupportedContentType enum. Set to PLAIN_TEXT by default.
        memory (str): MemoryInterface to use for the class. FileMemory by default.
    """

    AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_CONTAINER_URL"
    SAS_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_STORAGE_ACCOUNT_SAS_TOKEN"

    def __init__(
        self,
        *,
        container_url: str | None = None,
        sas_token: str | None = None,
        blob_content_type: SupportedContentType = SupportedContentType.PLAIN_TEXT,
        memory: MemoryInterface | None = None,
    ) -> None:
        default_values.load_default_env()

        self._blob_content_type: str = blob_content_type.value

        self._container_url: str = default_values.get_required_value(
            env_var_name=self.AZURE_STORAGE_CONTAINER_ENVIRONMENT_VARIABLE, passed_value=container_url
        )

        self._sas_token: str = default_values.get_required_value(
            env_var_name=self.SAS_TOKEN_ENVIRONMENT_VARIABLE, passed_value=sas_token
        )

        self._client = ContainerClient.from_container_url(
            container_url=self._container_url,
            credential=self._sas_token,
        )

        self._client_async = AsyncContainerClient.from_container_url(
            container_url=self._container_url,
            credential=self._sas_token,
        )

        super().__init__(memory=memory)

    def _upload_blob_exception_handling(self, exc: Exception) -> None:
        """
        Handles exceptions for uploading blob to storage container.

        Raises:
            ClientAuthenticationError: If authentication fails, either from an invalid SAS token or from
                an invalid container URL.
            Exception: If anything except ClientAuthenticationError is caught when uploading a blob.
        """

        if type(exc) is ClientAuthenticationError:
            logger.exception(
                msg="Authentication failed. Verify the container's existence in the Azure Storage Account and "
                + "the validity of the provided SAS token."
            )
            raise
        else:
            logger.exception(msg=f"An unexpected error occurred: {exc}")
            raise

    def _upload_blob(self, file_name: str, data: bytes, content_type: str) -> None:
        """
        Handles uploading blob to given storage container.

        Args:
            file_name (str): File name to assign to uploaded blob.
            data (bytes): Byte representation of content to upload to container.
            content_type (str): Content type to upload.
        """

        content_settings = ContentSettings(content_type=f"{content_type}")
        logger.info(msg="\nUploading to Azure Storage as blob:\n\t" + file_name)

        try:
            self._client.upload_blob(
                name=file_name,
                data=data,
                content_settings=content_settings,
                overwrite=True,
            )
        except Exception as exc:
            self._upload_blob_exception_handling(exc=exc)

    def send_prompt(
        self,
        *,
        normalized_prompt: str,
        conversation_id: str,
        normalizer_id: str,
    ) -> str:
        """
        Sends prompt to target, which creates a file and uploads it as a blob
        to the provided storage container.

        Args:
            normalized_prompt (str): A normalized prompt to be sent to the prompt target.
            conversation_id (str): The ID of the conversation.
            normalizer_id (str): ID provided by the prompt normalizer.

        Returns:
            blob_url (str): The Blob URL of the created blob within the provided storage container.
        """
        file_name = f"{conversation_id}.txt"
        data = str.encode(normalized_prompt)
        blob_url = self._container_url + "/" + file_name

        self._upload_blob(file_name=file_name, data=data, content_type=self._blob_content_type)

        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url

    async def _upload_blob_async(self, file_name: str, data: bytes, content_type: str) -> None:
        """
        (Async) Handles uploading blob to given storage container.

        Args:
            file_name (str): File name to assign to uploaded blob.
            data (bytes): Byte representation of content to upload to container.
            content_type (str): Content type to upload.
        """

        content_settings = ContentSettings(content_type=f"{content_type}")
        logger.info(msg="\nUploading to Azure Storage as blob:\n\t" + file_name)

        try:
            await self._client_async.upload_blob(
                name=file_name,
                data=data,
                content_settings=content_settings,
                overwrite=True,
            )
        except Exception as exc:
            self._upload_blob_exception_handling(exc=exc)

    async def send_prompt_async(
        self,
        *,
        normalized_prompt: str,
        conversation_id: str,
        normalizer_id: str,
    ) -> str:
        """
        (Async) Sends prompt to target, which creates a file and uploads it as a blob
        to the provided storage container.

        Args:
            normalized_prompt (str): A normalized prompt to be sent to the prompt target.
            conversation_id (str): The ID of the conversation.
            normalizer_id (str): ID provided by the prompt normalizer.

        Returns:
            blob_url (str): The Blob URL of the created blob within the provided storage container.
        """

        file_name = f"{conversation_id}.txt"
        data = str.encode(normalized_prompt)
        blob_url = self._container_url + "/" + file_name

        await self._upload_blob_async(file_name=file_name, data=data, content_type=self._blob_content_type)

        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=normalized_prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        return blob_url

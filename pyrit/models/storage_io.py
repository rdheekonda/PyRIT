# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import aiofiles
import os
import logging
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
from typing import Union, Optional
from enum import Enum
from urllib.parse import urlparse

from azure.storage.blob.aio import ContainerClient as AsyncContainerClient
from azure.storage.blob import ContentSettings
from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError
from pyrit.auth import AzureStorageAuth


logger = logging.getLogger(__name__)


class SupportedContentType(Enum):
    """
    All supported content types for uploading blobs to provided storage account container.
    See all options here: https://www.iana.org/assignments/media-types/media-types.xhtml
    """
    # TODO, add other media supported types
    PLAIN_TEXT = "text/plain"


class StorageIO(ABC):
    """
    Abstract interface for storage systems (local disk, Azure Storage Account, etc.).
    """

    @abstractmethod
    async def read_file(self, path: Union[Path, str]) -> bytes:
        """
        Asynchronously reads the file (or blob) from the given path.
        """
        pass

    @abstractmethod
    async def write_file(self, path: Union[Path, str], data: bytes) -> None:
        """
        Asynchronously writes data to the given path.
        """
        pass

    @abstractmethod
    async def is_path_exists(self, path: Union[Path, str]) -> bool:
        """
        Asynchronously checks if a file or blob exists at the given path.
        """
        pass

    @abstractmethod
    async def is_file(self, path: Union[Path, str]) -> bool:
        """
        Asynchronously checks if the path refers to a file (not a directory or container).
        """
        pass
    
    @abstractmethod
    async def create_directory_if_not_exists(self, path: Union[Path, str]) -> None:
        """
        Asynchronously creates a directory or equivalent in the storage system if it doesn't exist.
        """
        pass


class DiskStorageIO(StorageIO):
    """
    Implementation of StorageIO for local disk storage.
    """

    async def read_file(self, path: Union[Path, str]) -> bytes:
        """
        Asynchronously reads a file from the local disk.
        Args:
            path (Union[Path, str]): The path to the file.
        Returns:
            bytes: The content of the file.
        """
        async with aiofiles.open(path, 'rb') as file:
            return await file.read()

    async def write_file(self, path: Union[Path, str], data: bytes) -> None:
        """
        Asynchronously writes data to a file on the local disk.
        Args:
            path (Union[Path, str]): The path to the file.
            data (bytes): The content to write to the file.
        """
        async with aiofiles.open(path, 'wb') as file:
            await file.write(data)

    async def is_path_exists(self, path: Union[Path, str]) -> bool:
        """
        Asynchronously checks if a path exists on the local disk.
        Args:
            path (Union[Path, str]): The path to check.
        Returns:
            bool: True if the path exists, False otherwise.
        """
        return os.path.exists(path)

    async def is_file(self, path: Union[Path, str]) -> bool:
        """
        Asynchronously checks if the given path is a file (not a directory).
        Args:
            path (Union[Path, str]): The path to check.
        Returns:
            bool: True if the path is a file, False otherwise.
        """
        return os.path.isfile(path)
    
    async def create_directory_if_not_exists(self, path: Union[Path, str]) -> None:
        """
        Asynchronously creates a directory if it doesn't exist on the local disk.
        Args:
            path (Union[Path, str]): The directory path to create.
        """
        directory_path = Path(path)
        if not directory_path.exists():
            await aiofiles.os.mkdir(directory_path, exist_ok=True)
    
    
class AzureBlobStorageIO(StorageIO):
    """
    Implementation of StorageIO for Azure Blob Storage.
    """

    def __init__(
        self,
        *,
        container_url: str = None,
        sas_token: Optional[str] = None,
        blob_content_type: SupportedContentType = SupportedContentType.PLAIN_TEXT
    ) -> None:

        self._blob_content_type: str = blob_content_type.value
        if not container_url:
            raise ValueError(f"Invalid Azure Storage Account Container URL.")
        
        self._container_url: str = container_url
        self._sas_token = sas_token
        self._client_async: AsyncContainerClient = None

    async def _create_container_client_async(self):
        """Creates an asynchronous ContainerClient for Azure Storage. If a SAS token is provided via the
        AZURE_STORAGE_ACCOUNT_SAS_TOKEN environment variable or the init sas_token parameter, it will be used
        for authentication. Otherwise, a delegation SAS token will be created using Entra ID authentication."""
        if not self._sas_token:
            logger.info("SAS token not provided. Creating a delegation SAS token using Entra ID authentication.")
            sas_token = await AzureStorageAuth.get_sas_token(self._container_url)

        self._client_async = AsyncContainerClient.from_container_url(
            container_url=self._container_url,
            credential=sas_token,
        )

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

        if not self._client_async:
            await self._create_container_client_async()
        try:
            await self._client_async.upload_blob(
                name=file_name,
                data=data,
                content_settings=content_settings,
                overwrite=True,
            )
        except Exception as exc:
            if isinstance(exc, ClientAuthenticationError):
                logger.exception(
                    msg="Authentication failed. Please check that the container existence in the "
                    + "Azure Storage Account and ensure the validity of the provided SAS token. If you "
                    + "haven't set the SAS token as an environment variable use `az login` to "
                    + "enable delegation-based SAS authentication to connect to the storage account"
                )
                raise
            else:
                logger.exception(msg=f"An unexpected error occurred: {exc}")
                raise

    async def read_file(self, path: Union[Path, str]) -> bytes:
        """
        Asynchronously reads the content of a file (blob) from Azure Blob Storage.

        If the provided `path` is a full URL 
        (e.g., "https://<Azure STorage Account>.blob.core.windows.net/<container name>/dir1/dir2/sample.png"),
        it extracts the relative blob path (e.g., "dir1/dir2/sample.png") to correctly access the blob.
        If a relative path is provided, it will use it as-is.

        Args:
            path (Union[Path, str]): The path to the file (blob) in Azure Blob Storage. 
                                    This can be either a full URL or a relative path.

        Returns:
            bytes: The content of the file (blob) as bytes.

        Raises:
            Exception: If there is an error in reading the blob file, an exception will be logged 
                    and re-raised.

        Example:
            file_content = await read_file("https://account.blob.core.windows.net/container/dir1/dir2/1726627689003831.png")
            # Or using a relative path:
            file_content = await read_file("dir1/dir2/1726627689003831.png")
        """
        if not self._client_async:
            await self._create_container_client_async()

        file_path = str(path)

        parsed_url = urlparse(file_path)
        if parsed_url.scheme and parsed_url.netloc:
            file_path = '/'.join(parsed_url.path.split('/')[2:])
        
        try:
            blob_client = self._client_async.get_blob_client(blob=file_path)
            
            # Download the blob
            blob_stream = await blob_client.download_blob()
            file_content = await blob_stream.readall()
            
            return file_content

        except Exception as exc:
            logger.exception(f"Failed to read file at {file_path}: {exc}")
            raise
        

    async def write_file(self, path: Union[Path, str], data: bytes) -> None:
        """
        Writes data to Azure Blob Storage at the specified path.

        Args:
            path (Union[Path, str]): The full Azure Blob Storage URL
            data (bytes): The data to write.
        """
        parsed_url = urlparse(str(path))

        sub_path = '/'.join(parsed_url.path.split('/')[2:])

        await self._upload_blob_async(file_name=sub_path, data=data, content_type=self._blob_content_type)

    async def is_path_exists(self, path: Union[Path, str]) -> bool:
        """Check if a given path exists in the Azure Blob Storage container."""
        if not self._client_async:
            await self._create_container_client_async()
        try:
            blob_client = self._client_async.get_blob_client(blob=str(path))
            await blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False

    async def is_file(self, path: Union[Path, str]) -> bool:
        """Check if the path refers to a file (blob) in Azure Blob Storage."""
        if not self._client_async:
            await self._create_container_client_async()
        try:
            blob_client = self._client_async.get_blob_client(blob=str(path))
            blob_properties = await blob_client.get_blob_properties()
            return blob_properties.size > 0
        except ResourceNotFoundError:
            return False
    
    async def create_directory_if_not_exists(self, directory_path: Union[Path, str]) -> None:
        pass

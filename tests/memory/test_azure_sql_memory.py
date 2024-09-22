# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid

from typing import Generator, Literal
from unittest import mock

import pytest

from mock_alchemy.mocking import UnifiedAlchemyMagicMock

from sqlalchemy import text
from pyrit.memory import AzureSQLMemory
from pyrit.memory.memory_models import PromptMemoryEntry, EmbeddingData
from pyrit.models import PromptRequestPiece, Score
from pyrit.orchestrator.orchestrator_class import Orchestrator
from pyrit.prompt_converter.base64_converter import Base64Converter
from pyrit.prompt_target.text_target import TextTarget
from tests.mocks import get_azure_sql_memory
from tests.mocks import get_sample_conversation_entries


@pytest.fixture
def memory_interface() -> Generator[AzureSQLMemory, None, None]:
    yield from get_azure_sql_memory()


@pytest.fixture
def sample_conversation_entries() -> list[PromptMemoryEntry]:
    return get_sample_conversation_entries()


@pytest.mark.asyncio
async def test_insert_entry(memory_interface):
    prompt_request_piece = PromptRequestPiece(
            id=uuid.uuid4(),
            conversation_id="123",
            role="user",
            original_value_data_type="text",
            original_value="Hello",
            converted_value="Hello",
        )
    await prompt_request_piece.compute_sha256(memory_interface)
    entry = PromptMemoryEntry(
        entry=prompt_request_piece
    )

    memory_interface = AzureSQLMemory(
        connection_string="mssql+pyodbc://test:test@test/test?driver=ODBC+Driver+18+for+SQL+Server"
    )

    # Now, get a new session to query the database and verify the entry was inserted
    with memory_interface.get_session() as session:
        assert isinstance(session, UnifiedAlchemyMagicMock)
        session.add.assert_not_called()
        memory_interface._insert_entry(entry)
        inserted_entry = session.query(PromptMemoryEntry).filter_by(conversation_id="123").first()
        assert inserted_entry is not None
        assert inserted_entry.role == "user"
        assert inserted_entry.original_value == "Hello"
        sha265 = "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969"  # sha256('Hello')
        assert inserted_entry.original_value_sha256 == sha265


def test_insert_entries(memory_interface: AzureSQLMemory):
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id=str(i),
                role="user",
                original_value=f"Message {i}",
                converted_value=f"CMessage {i}",
            )
        )
        for i in range(5)
    ]

    # Now, get a new session to query the database and verify the entries were inserted
    with memory_interface.get_session() as session:  # type: ignore
        # Use the _insert_entries method to insert multiple entries into the database
        memory_interface._insert_entries(entries=entries)
        inserted_entries = session.query(PromptMemoryEntry).order_by(PromptMemoryEntry.conversation_id).all()
        assert len(inserted_entries) == 5
        for i, entry in enumerate(inserted_entries):
            assert entry.conversation_id == str(i)
            assert entry.role == "user"
            assert entry.original_value == f"Message {i}"
            assert entry.converted_value == f"CMessage {i}"


def test_insert_embedding_entry(memory_interface: AzureSQLMemory):
    # Create a ConversationData entry
    conversation_entry = PromptMemoryEntry(
        entry=PromptRequestPiece(conversation_id="123", role="user", original_value="Hello", converted_value="abc")
    )

    # Insert the ConversationData entry using the _insert_entry method
    memory_interface._insert_entry(conversation_entry)

    # Re-query the ConversationData entry within a new session to ensure it's attached
    with memory_interface.get_session() as session:  # type: ignore
        # Assuming uuid is the primary key and is set upon insertion
        reattached_conversation_entry = session.query(PromptMemoryEntry).filter_by(conversation_id="123").one()
        uuid = reattached_conversation_entry.id

    # Now that we have the uuid, we can create and insert the EmbeddingData entry
    embedding_entry = EmbeddingData(id=uuid, embedding=[1, 2, 3], embedding_type_name="test_type")
    memory_interface._insert_entry(embedding_entry)

    # Verify the EmbeddingData entry was inserted correctly
    with memory_interface.get_session() as session:  # type: ignore
        persisted_embedding_entry = session.query(EmbeddingData).filter_by(id=uuid).first()
        assert persisted_embedding_entry is not None
        assert persisted_embedding_entry.embedding == [1, 2, 3]
        assert persisted_embedding_entry.embedding_type_name == "test_type"


def test_disable_embedding(memory_interface: AzureSQLMemory):
    memory_interface.disable_embedding()

    assert (
        memory_interface.memory_embedding is None
    ), "disable_memory flag was passed, so memory embedding should be disabled."


def test_default_enable_embedding(memory_interface: AzureSQLMemory):
    os.environ["AZURE_OPENAI_EMBEDDING_KEY"] = "mock_key"
    os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] = "embedding"
    os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = "deployment"

    memory_interface.enable_embedding()

    assert (
        memory_interface.memory_embedding is not None
    ), "Memory embedding should be enabled when set with environment variables."


def test_default_embedding_raises(memory_interface: AzureSQLMemory):
    os.environ["AZURE_OPENAI_EMBEDDING_KEY"] = ""
    os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] = ""
    os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = ""

    with pytest.raises(ValueError):
        memory_interface.enable_embedding()


def test_query_entries(memory_interface: AzureSQLMemory, sample_conversation_entries: list[PromptMemoryEntry]):

    for i in range(3):
        sample_conversation_entries[i].conversation_id = str(i)
        sample_conversation_entries[i].original_value = f"Message {i}"
        sample_conversation_entries[i].converted_value = f"Message {i}"

    memory_interface._insert_entries(entries=sample_conversation_entries)

    # Query entries without conditions
    queried_entries = memory_interface.query_entries(PromptMemoryEntry)
    assert len(queried_entries) == 3

    session = memory_interface.get_session()
    session.query.reset_mock()  # type: ignore

    # Query entries with a condition
    memory_interface.query_entries(PromptMemoryEntry, conditions=PromptMemoryEntry.conversation_id == "1")

    session.query.return_value.filter.assert_called_once_with(PromptMemoryEntry.conversation_id == "1")  # type: ignore


def test_get_all_memory(memory_interface: AzureSQLMemory, sample_conversation_entries: list[PromptMemoryEntry]):

    memory_interface._insert_entries(entries=sample_conversation_entries)

    # Fetch all entries
    all_entries = memory_interface.get_all_prompt_pieces()
    assert len(all_entries) == 3


def test_get_memories_with_json_properties(memory_interface: AzureSQLMemory):
    # Define a specific conversation_id
    specific_conversation_id = "test_conversation_id"

    converter_identifiers = [Base64Converter().get_identifier()]
    target = TextTarget()

    # Start a session
    with memory_interface.get_session() as session:  # type: ignore
        # Create a ConversationData entry with all attributes filled
        entry = PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id=specific_conversation_id,
                role="user",
                sequence=1,
                original_value="Test content",
                converted_value="Test content",
                labels={"normalizer_id": "id1"},
                converter_identifiers=converter_identifiers,
                prompt_target_identifier=target.get_identifier(),
            )
        )

        # Insert the ConversationData entry
        session.add(entry)
        session.commit()

        # Use the get_memories_with_conversation_id method to retrieve entries with the specific conversation_id
        retrieved_entries = memory_interface.get_conversation(conversation_id=specific_conversation_id)

        # Verify that the retrieved entry matches the inserted entry
        assert len(retrieved_entries) == 1
        retrieved_entry = retrieved_entries[0].request_pieces[0]
        assert retrieved_entry.conversation_id == specific_conversation_id
        assert retrieved_entry.role == "user"
        assert retrieved_entry.original_value == "Test content"
        # For timestamp, you might want to check if it's close to the current time instead of an exact match
        assert abs((retrieved_entry.timestamp - entry.timestamp).total_seconds()) < 10  # Assuming the test runs quickly

        converter_identifiers = retrieved_entry.converter_identifiers
        assert len(converter_identifiers) == 1
        assert converter_identifiers[0]["__type__"] == "Base64Converter"

        prompt_target = retrieved_entry.prompt_target_identifier
        assert prompt_target["__type__"] == "TextTarget"

        labels = retrieved_entry.labels
        assert labels["normalizer_id"] == "id1"


def test_get_memories_with_orchestrator_id(memory_interface: AzureSQLMemory):
    # Define a specific normalizer_id
    orchestrator1 = Orchestrator()
    orchestrator2 = Orchestrator()

    # Create a list of ConversationData entries, some with the specific normalizer_id
    entries = [
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id="123",
                role="user",
                original_value="Hello 1",
                converted_value="Hello 1",
                orchestrator_identifier=orchestrator1.get_identifier(),
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id="456",
                role="user",
                original_value="Hello 2",
                converted_value="Hello 2",
                orchestrator_identifier=orchestrator2.get_identifier(),
            )
        ),
        PromptMemoryEntry(
            entry=PromptRequestPiece(
                conversation_id="789",
                role="user",
                original_value="Hello 3",
                converted_value="Hello 1",
                orchestrator_identifier=orchestrator1.get_identifier(),
            )
        ),
    ]

    orchestrator1_id = orchestrator1.get_identifier()["id"]
    # Mock the query_entries method
    with mock.patch.object(memory_interface, 'query_entries', return_value=[entry for entry in entries if entry.orchestrator_identifier["id"] == orchestrator1_id]):
        # Call the method under test
        retrieved_entries = memory_interface._get_prompt_pieces_by_orchestrator(orchestrator_id=orchestrator1_id)

        # Verify the returned entries
        assert len(retrieved_entries) == 2
        assert all(piece.orchestrator_identifier["id"] == orchestrator1_id for piece in retrieved_entries)
        memory_interface.query_entries.assert_called_once()
        # Extract the actual SQL condition passed to query_entries
        actual_sql_condition = memory_interface.query_entries.call_args.kwargs['conditions']
        expected_sql_condition = text(
            "ISJSON(orchestrator_identifier) = 1 AND JSON_VALUE(orchestrator_identifier, '$.id') = :json_id"
        ).bindparams(json_id=orchestrator1_id)

        # Compare the SQL text and the bound parameters
        assert str(actual_sql_condition) == str(expected_sql_condition)
        assert actual_sql_condition.compile().params == expected_sql_condition.compile().params


@pytest.mark.parametrize("score_type", ["float_scale", "true_false"])
def test_add_score_get_score(
    memory_interface: AzureSQLMemory,
    sample_conversation_entries: list[PromptMemoryEntry],
    score_type: Literal["float_scale"] | Literal["true_false"],
):
    prompt_id = sample_conversation_entries[0].id

    memory_interface._insert_entries(entries=sample_conversation_entries)

    score_value = str(True) if score_type == "true_false" else "0.8"

    score = Score(
        score_value=score_value,
        score_value_description="High score",
        score_type=score_type,
        score_category="test",
        score_rationale="Test score",
        score_metadata="Test metadata",
        scorer_class_identifier={"__type__": "TestScorer"},
        prompt_request_response_id=prompt_id,
    )

    memory_interface.add_scores_to_memory(scores=[score])

    # Fetch the score we just added
    db_score = memory_interface.get_scores_by_prompt_ids(prompt_request_response_ids=[prompt_id])
    assert db_score
    assert len(db_score) == 1
    assert db_score[0].score_value == score_value
    assert db_score[0].score_value_description == "High score"
    assert db_score[0].score_type == score_type
    assert db_score[0].score_category == "test"
    assert db_score[0].score_rationale == "Test score"
    assert db_score[0].score_metadata == "Test metadata"
    assert db_score[0].scorer_class_identifier == {"__type__": "TestScorer"}
    assert db_score[0].prompt_request_response_id == prompt_id

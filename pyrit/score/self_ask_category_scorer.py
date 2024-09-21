# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
import yaml

import enum
from pathlib import Path
from typing import Dict, Optional

from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models.score import UnvalidatedScore
from pyrit.score import Score, Scorer
from pyrit.models import PromptRequestPiece, PromptRequestResponse, PromptTemplate
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import CONTENT_CLASSIFIERS_PATH


class ContentClassifierPaths(enum.Enum):
    HARMFUL_CONTENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "harmful_content.yaml").resolve()
    SENTIMENT_CLASSIFIER = Path(CONTENT_CLASSIFIERS_PATH, "sentiment.yaml").resolve()


class SelfAskCategoryScorer(Scorer):
    """
    A class that represents a self-ask score for text classification and scoring.
    Given a classifer file, it scores according to these categories and returns the category
    the PromptRequestPiece fits best.

    There is also a false category that is used if the promptrequestpiece does not fit any of the categories.
    """

    def __init__(
        self,
        chat_target: PromptChatTarget,
        content_classifier: Path,
        memory: MemoryInterface = None,
    ) -> None:
        """
        Initializes a new instance of the SelfAskCategoryScorer class.

        Args:
            chat_target (PromptChatTarget): The chat target to interact with.
            content_classifier (Path): The path to the classifier file.
        """
        self._prompt_target = chat_target
        self.scorer_type = "true_false"

        self._memory = memory if memory else DuckDBMemory()
        if self._prompt_target:
            self._prompt_target._memory = self._memory
        category_file_contents = yaml.safe_load(content_classifier.read_text(encoding="utf-8"))

        self._no_category_found_category = category_file_contents["no_category_found"]
        categories_as_string = self._content_classifier_to_string(category_file_contents["categories"])

        scoring_instructions_template = PromptTemplate.from_yaml_file(
            CONTENT_CLASSIFIERS_PATH / "content_classifier_system_prompt.yaml"
        )

        self._system_prompt = scoring_instructions_template.apply_custom_metaprompt_parameters(
            categories=categories_as_string,
            no_category_found=self._no_category_found_category,
        )

    def _content_classifier_to_string(self, categories: list[Dict[str, str]]) -> str:
        """
        Converts the content classifier categories to a string representation to be put in a system prompt.

        Args:
            categories (list[Dict[str, str]]): The categories to convert.

        Returns:
            str: The string representation of the categories.
        """
        if not categories:
            raise ValueError("Improperly formatted content classifier yaml file. No categories provided")

        category_descriptions = ""

        for category in categories:
            name = category["name"]
            desc = category["description"]

            category_descriptions += f"'{name}': {desc}\n"

        if self._no_category_found_category not in category_descriptions:
            raise ValueError(f"False category {self._no_category_found_category} not found in classifier categories")

        return category_descriptions

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given request_response using the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece to score.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
                Currently not supported for this scorer.

        Returns:
            list[Score]: The request_response scored.
                         The category that fits best in the response is used for score_category.
                         The score_value is True in all cases unless no category fits. In which case,
                         the score value is false and the _false_category is used.
        """
        self.validate(request_response, task=task)

        conversation_id = str(uuid.uuid4())

        self._prompt_target.set_system_prompt(
            system_prompt=self._system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
        )

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=request_response.converted_value,
                    conversation_id=conversation_id,
                    prompt_target_identifier=self._prompt_target.get_identifier(),
                )
            ]
        )

        unvalidated_score: UnvalidatedScore = await self.send_chat_target_async(
            prompt_target=self._prompt_target,
            scorer_llm_request=request,
            scored_prompt_id=request_response.id,
            task=task,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if task:
            raise ValueError("This scorer does not support tasks")

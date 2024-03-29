import json
import logging
import uuid
import pathlib

from pyrit.prompt_converter import PromptConverter
from pyrit.models import PromptTemplate
from pyrit.common.path import DATASETS_PATH
from pyrit.prompt_target import PromptChatTarget
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class TranslationConverter(PromptConverter):
    def __init__(
        self, *, converter_target: PromptChatTarget, languages: list[str], prompt_template: PromptTemplate = None
    ):
        """
        Initializes a TranslationConverter object.

        Args:
            converter_target (PromptChatTarget): The target chat support for the conversion which will translate
            language (str): The language for the conversion. E.g. Spanish, French, leetspeak, etc.
            prompt_template (PromptTemplate, optional): The prompt template for the conversion.

        Raises:
            ValueError: If the language is not provided.
        """
        self.converter_target = converter_target

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "translation_converter.yaml"
            )
        )

        self._validate_languages(languages)

        self.languages = languages
        language_str = ", ".join(languages)

        self.system_prompt = prompt_template.apply_custom_metaprompt_parameters(languages=language_str)

        self._conversation_id = str(uuid.uuid4())
        self._normalizer_id = None  # Normalizer not used

        self.converter_target.set_system_prompt(
            prompt=self.system_prompt,
            conversation_id=self._conversation_id,
            normalizer_id=self._normalizer_id,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def convert(self, prompts: list[str]) -> list[str]:
        """
        Generates variations of the input prompts using the converter target.
        Parameters:
            prompts: list of prompts to convert
        Return:
            target_responses: list of prompt variations generated by the converter target
        """
        converted_prompts: list[str] = []

        for prompt in prompts:
            response_msg = self.converter_target.send_prompt(
                normalized_prompt=prompt,
                conversation_id=self._conversation_id,
                normalizer_id=self._normalizer_id,
            )

            try:
                llm_response: dict[str, str] = json.loads(response_msg)["output"]

                for variation in llm_response.values():
                    converted_prompts.append(variation)

            except json.JSONDecodeError as e:
                logger.warn(f"Error in LLM response {response_msg}: {e}")
                raise RuntimeError(f"Error in LLM respons {response_msg}")

        return converted_prompts

    def is_one_to_one_converter(self) -> bool:
        return len(self.languages) == 1

    def _validate_languages(self, languages) -> None:
        if not languages:
            raise ValueError("Languages must be provided")

        for language in languages:
            if not language or "," in language:
                raise ValueError("Language must be provided and not have a comma")

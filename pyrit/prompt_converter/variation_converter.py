import json
import logging
import uuid
import pathlib

from pyrit.models import PromptDataType
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.models import PromptTemplate
from pyrit.common.path import DATASETS_PATH
from pyrit.prompt_target import PromptChatTarget
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


class VariationConverter(PromptConverter):
    def __init__(self, *, converter_target: PromptChatTarget, prompt_template: PromptTemplate = None):
        self.converter_target = converter_target

        # set to default strategy if not provided
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_yaml_file(
                pathlib.Path(DATASETS_PATH) / "prompt_converters" / "variation_converter.yaml"
            )
        )

        self.number_variations = 1

        self.system_prompt = str(
            prompt_template.apply_custom_metaprompt_parameters(number_iterations=str(self.number_variations))
        )
        self._labels = {"converter": "VariationConverter"}

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Generates variations of the input prompts using the converter target.
        Parameters:
            prompts: list of prompts to convert
        Return:
            target_responses: list of prompt variations generated by the converter target
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        conversation_id = str(uuid.uuid4())

        self.converter_target.set_system_prompt(
            system_prompt=self.system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=None,
            labels=self._labels,
        )

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_prompt_text=prompt,
                    converted_prompt_text=prompt,
                    conversation_id=conversation_id,
                    sequence=1,
                    labels=self._labels,
                    prompt_target_identifier=self.converter_target.get_identifier(),
                    original_prompt_data_type=input_type,
                    converted_prompt_data_type=input_type,
                )
            ]
        )

        response_msg = self.converter_target.send_prompt(prompt_request=request).request_pieces[0].converted_prompt_text

        try:
            ret_text = json.loads(response_msg)[0]
            return ConverterResult(output_text=ret_text, output_type="text")
        except json.JSONDecodeError:
            logger.warning(logging.WARNING, f"could not parse response as JSON {response_msg}")
            raise RuntimeError(f"Error in LLM response {response_msg}")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

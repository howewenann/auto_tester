from typing import Callable
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser

class DeepEvalJudgeLLM(DeepEvalBaseLLM):
    """
    Wraps a plain Python function: (str) -> str
    """
    def __init__(self, fn: Callable[[str], str], name: str = "DeepEvalJudgeLLM", max_attempts=5):
        self._fn = fn
        self._name = name
        self._max_attempts = max_attempts

    def get_model_name(self) -> str:
        return self._name

    def load_model(self):
        # For consistency with DeepEval's interface; just return the function
        return self._fn

    @staticmethod
    def json_format_prompt(parser):
        extra_constraints = (
            "ADDITIONAL CONSTRAINTS:\n"
            "- Remove any escape characters from the output (e.g., \\\", \\n, \\t, \\\\ )\n"
            "- Ensure all property names and string values are enclosed in double quotes.\n"
            "- Do not include trailing commas in objects or arrays.\n"
            "- Do not include comments (// or /* */) anywhere in the output.\n"
            "- Boolean and null values must be lowercase: true, false, null.\n"
            "- Numbers must not have leading zeros (except zero itself) and must not include NaN or Infinity.\n"
            "- Ensure proper nesting and closing of all arrays and objects.\n"
            "- Do not repeat keys within the same object.\n"
            "- Output must be valid UTF-8 without invalid byte sequences.\n"
            "- Do not use single quotes for strings; only double quotes are allowed.\n"
        )

        return f'{extra_constraints}\n{parser.get_format_instructions()}'

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # Same as the previous example above
        parser = JsonOutputParser(pydantic_object=schema)
        model = self.load_model()
        prompt = f'{prompt}\n{self.json_format_prompt(parser)}'
        raw_result = model(prompt)

        for attempt in range(self._max_attempts):
            try:
                # If already a dict, validate and return
                if isinstance(raw_result, dict):
                    return schema(**raw_result)

                # If string, try parsing
                if isinstance(raw_result, str):
                    parsed = parser.parse(raw_result)
                    return schema(**parsed)

            except Exception as e:
                # print(f"Attempt {attempt+1} failed: {e}")
                # Build corrective prompt for LLM
                corrective_prompt = (
                    f"The following JSON is invalid:\n{raw_result}\n"
                    f"Error encountered: {str(e)}\n"
                    f"Please fix it to match this instructions:\n{self.json_format_prompt(parser)}\n"
                    "Ensure all constraints are met."
                )

                raw_result = model(corrective_prompt)

        raise ValueError("Failed to parse JSON after multiple attempts.")

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

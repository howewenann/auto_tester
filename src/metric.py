from typing import List, Optional, Union, Type
import asyncio
import json

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BaseMetric
from deepeval.utils import (
    get_or_create_event_loop,
    prettify_list,
)
from deepeval.metrics.utils import (
    construct_verbose_logs,
    check_llm_test_case_params,
    initialize_model,
    a_generate_with_schema_and_extract,
    generate_with_schema_and_extract,
)
from deepeval.models import DeepEvalBaseLLM
from .template import TruthfulnessTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from .schema import (
    BinaryQuestion,
    ExtractedUnits,
    EvaluationIndex,
    EvaluationText,
    JudgmentText,
)

from deepeval.metrics.api import metric_data_manager


# 1. Inherit the BaseMetric class
class TruthfulnessMetric(BaseMetric):
    _required_params: List[LLMTestCaseParams] = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ]

    # 2. Implement the __init__() method
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        require_all: Optional[int] = True,
        evaluation_template: Type[TruthfulnessTemplate] = TruthfulnessTemplate,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.evaluation_template = evaluation_template
        self.require_all = require_all

    # 3. Implement the measure() and a_measure() methods
    # Both measure() and a_measure() MUST:
    # - accept an LLMTestCase as argument
    # - set self.score
    # - set self.success
    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(
                        test_case,
                        _show_indicator=False,
                        _in_component=_in_component,
                        _log_metric_to_confident=_log_metric_to_confident,
                    )
                )
            else:
                expected_output = test_case.expected_output
                actual_output = test_case.actual_output
                self_input = test_case.input

                self.is_binary_question = self._is_binary_question(self.input)
                self.golden_units = self._extract_units(expected_output, golden=True)
                self.candidate_units = self._extract_units(actual_output, golden=False)

                # for checking
                self.truths = [d.text for d in self.golden_units.units]
                self.claims = [d.text for d in self.candidate_units.units]

                # get verdicts
                self.verdicts = self._get_verdict(
                    self.is_binary_question,
                    self.golden_units,
                    self.candidate_units
                )

                self.verdicts_text = self.to_evaluation_text(
                    self.verdicts,
                    self.truths
                )

                score_tuple = self._calculate_score(
                    self.verdicts_text,
                    self.golden_units,
                )

                self.score = score_tuple[0]

                self.reason = self.verdicts.overall_rationale
                self.success = self.score >= self.threshold

                verbose_data = {
                    'Truths': self.truths,
                    'Claims': self.claims,
                    'Verdicts': [j.__dict__ for j in self.verdicts_text.judgments],
                    'MissingItems': self.verdicts_text.missing_required_golden_texts,
                    'Polarity Match': score_tuple[8],
                    'Precision': score_tuple[1],
                    'Recall': score_tuple[2],
                    'F1': self.score,
                    'Reason': self.reason,
                }

                self.verbose_logs = construct_verbose_logs(self, steps=[json.dumps(verbose_data), 'END'])

                if _log_metric_to_confident:
                    metric_data_manager.post_metric_if_enabled(
                        self, test_case=test_case
                    )

        return self.score


    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:

        multimodal = test_case.multimodal
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            self.model,
            multimodal,
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        ):

            expected_output = test_case.expected_output
            actual_output = test_case.actual_output
            self.input = test_case.input

            self.is_binary_question = await self._a_is_binary_question(self.input)
            self.golden_units = await self._a_extract_units(expected_output, golden=True)
            self.candidate_units = await self._a_extract_units(actual_output, golden=False)

            # for checking
            self.truths = [d.text for d in self.golden_units.units]
            self.claims = [d.text for d in self.candidate_units.units]

            # get verdicts
            self.verdicts = await self._a_get_verdict(
                self.is_binary_question,
                self.golden_units,
                self.candidate_units
            )

            self.verdicts_text = await self.to_evaluation_text(
                self.verdicts,
                self.truths
            )

            score_tuple = self._calculate_score(
                self.verdicts,
                self.golden_units,
            )

            self.score = score_tuple[0]

            self.reason = self.verdicts.overall_rationale
            self.success = self.score >= self.threshold

            verbose_data = {
                'Truths': self.truths,
                'Claims': self.claims,
                'Verdicts': [j._dict_ for j in self.verdicts_text.judgments],
                'MissingItems': self.verdicts_text.missing_required_golden_texts,
                'Polarity Match': score_tuple[3],
                'Precision': score_tuple[1],
                'Recall': score_tuple[2],
                'F1': self.score,
                'Reason': self.reason,
            }

            self.verbose_logs = construct_verbose_logs(self, steps=[json.dumps(verbose_data), 'END' ])

            if _log_metric_to_confident:
                metric_data_manager.post_metric_if_enabled(
                    self, test_case=test_case
                )

        return self.score

    async def _a_extract_units(
        self, text:str, golden:bool
    ) -> ExtractedUnits:

        prompt = self.evaluation_template.extract_units(
            text = text,
            golden = golden
        )

        cleaned_texts = await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ExtractedUnits,
            extract_schema=lambda s: s,
            extract_json=lambda data: ExtractedUnits(**data),
        )

        return cleaned_texts


    def _extract_units(
        self, text:str, golden:bool
    ) -> ExtractedUnits:

        prompt = self.evaluation_template.extract_units(
            text = text,
            golden = golden
        )

        extracted_units = generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=ExtractedUnits,
            extract_schema=lambda s: s,
            extract_json=lambda data: ExtractedUnits(**data),
        )

        return extracted_units


    async def _a_is_binary_question(self, input: str) -> BinaryQuestion:

        prompt = self.evaluation_template.is_yes_no_question(input=input)

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=BinaryQuestion,
            extract_schema=lambda s: s,
            extract_json=lambda data: BinaryQuestion(**data),
        )


    def _is_binary_question(self, input: str) -> BinaryQuestion:

        prompt = self.evaluation_template.is_yes_no_question(input=input)

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=BinaryQuestion,
            extract_schema=lambda s: s,
            extract_json=lambda data: BinaryQuestion(**data),
        )

    async def _a_get_verdict(self, binary_question: BinaryQuestion, golden_units: ExtractedUnits, candidate_units: ExtractedUnits) -> EvaluationIndex:

        prompt = self.evaluation_template.get_verdict(
            binary_question=binary_question,
            golden_units=golden_units,
            candidate_units=candidate_units
        )

        return await a_generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=EvaluationIndex,
            extract_schema=lambda s: s,
            extract_json=lambda data: EvaluationIndex(**data)
        )


    def _get_verdict(self, binary_question: BinaryQuestion, golden_units: ExtractedUnits, candidate_units: ExtractedUnits) -> EvaluationIndex:

        prompt = self.evaluation_template.get_verdict(
            binary_question=binary_question,
            golden_units=golden_units,
            candidate_units=candidate_units
        )

        return generate_with_schema_and_extract(
            metric=self,
            prompt=prompt,
            schema_cls=EvaluationIndex,
            extract_schema=lambda s: s,
            extract_json=lambda data: EvaluationIndex(**data),
        )


    def _calculate_score(self, eval: EvaluationText, golden: ExtractedUnits) -> float:

        if eval.is_yes_no_question == 'true':
            polarity_match = 'NA'

        elif eval.is_yes_no_question == 'true' and eval.polarity_match != 'true':
            polarity_match = 'FALSE'

        else:
            polarity_match = 'TRUE'

        # helper function
        def safe_div(n, d):
            return n / d if d else 0

        # precision - from the bot answer, % of claims supported by specs
        precision = sum([i.verdict == 'supported' for i in eval.judgments]) / len(eval.judgments)

        # recall - from truths (specs), % truths captured in claims
        recall = 1 - len(eval.missing_required_golden_texts) / len(golden.units)

        # fi score
        score = safe_div(2 * precision * recall, precision + recall)

        return score, precision, recall, polarity_match


    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    # helper function for logs
    def to_evaluation_text(self, ei: EvaluationIndex, golden_texts: list[str]) -> EvaluationText:
        return EvaluationText(
            is_yes_no_question=ei.is_yes_no_question,
            polarity_match=ei.polarity_match,
            overall_rationale=ei.overall_rationale,
            judgments=[
                JudgmentText(
                    candidate_text=j.candidate_text,
                    verdict=j.verdict,
                    matched_golden_texts=[golden_texts[i] for i in j.matched_golden_indexes],
                    issue=j.issue,
                    rationale=j.rationale,
                )
                for j in ei.judgments
            ],
            missing_required_golden_texts=[
                golden_texts[i]
                for i in range(len(golden_texts))
                if i not in {
                    idx
                    for j in ei.judgments if j.verdict == "supported"
                    for idx in j.matched_golden_indexes
                }
            ]
        )

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator

Verdict = Literal["supported", "contradicted", "not_in_golden", "unclear"]
Issue = Literal["none", "wrong_fact", "unsafe", "missing_precondition", "other"]

class Unit(BaseModel):
    id: str
    kind: str
    text: str

class ExtractedUnits(BaseModel):
    polarity: str
    units: List[Unit]

class BinaryQuestion(BaseModel):
    is_yes_no_question: str

class JudgmentIndex(BaseModel):
    candidate_text: str
    verdict: Verdict
    matched_golden_indexes: List[int] = Field(default_factory=list)
    issue: Issue
    rationale: str

@field_validator("matched_golden_indexes")
@classmethod
def unique_indexes(cls, v: List[int]) -> List[int]:
    if len(v) != len(set(v)):
        raise ValueError("matched_golden_indexes must be unique")
    return v

class EvaluationIndex(BaseModel):
    is_yes_no_question: str
    polarity_match: str
    overall_rationale: str
    judgments: List[JudgmentIndex]

class JudgmentText(BaseModel):
    candidate_text: str
    verdict: Verdict
    matched_golden_texts: List[str] = Field(default_factory=list)
    issue: Issue
    rationale: str

class EvaluationText(BaseModel):
    is_yea_no_question: str
    polarity_match: str
    overall_rationale: str
    judgments: List[JudgmentText]
    missing_required_golden_texts: List[str] = Field(default_factory=list)

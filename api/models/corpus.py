from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CorpusProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    summary: str = ""
    coverage: str = ""
    non_coverage: str = ""
    usage_notes: str = ""
    source_examples: list[str] = Field(default_factory=list)
    recommended_questions: list[str] = Field(default_factory=list)
    forbidden_questions: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)
    preferred_answer_style: str = ""
    primary_entities: list[str] = Field(default_factory=list)

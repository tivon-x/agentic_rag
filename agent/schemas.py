from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator


class RetrievalDecision(BaseModel):
    decision: Literal["retrieve", "direct_answer", "out_of_scope"] = Field(
        description="Routing decision for the latest user message."
    )
    reason: str = Field(
        description="Short explanation for the routing decision."
    )


class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="Whether the latest user query is clear enough.")
    questions: list[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Clarification request shown when the user query is underspecified."
    )


class QueryPlan(BaseModel):
    intent: Literal["fact", "summary", "compare", "multi_hop", "definition"] = Field(
        description="High-level retrieval intent for the latest user question."
    )
    subqueries: list[str] = Field(
        description="One to three focused retrieval subqueries."
    )
    preferred_node_types: list[Literal["document", "section", "paragraph"]] = Field(
        description="Preferred node granularities for retrieval and packing."
    )


class EvidenceItem(BaseModel):
    doc_id: str = Field(description="Document identifier for the cited evidence.")
    node_id: str = Field(description="Node identifier for the cited evidence.")
    paper_id: str | None = Field(
        default=None,
        description="Catalog paper identifier when the evidence belongs to a paper.",
    )
    paper_title: str | None = Field(
        default=None,
        description="Catalog paper title when available in passage metadata.",
    )
    source: str = Field(description="Original source file or label.")
    section_path: list[str] = Field(
        default_factory=list,
        description="Hierarchical section path for the evidence snippet.",
    )
    page: int | None = Field(
        default=None, description="Source page number when available."
    )
    quote: str = Field(description="Quoted or excerpted supporting text.")
    score: float | None = Field(
        default=None, description="Retrieval or rerank score when available."
    )
    relevance: str | None = Field(
        default=None, description="Short explanation of why the evidence matters."
    )


class EvidenceGroup(BaseModel):
    subquery: str = Field(description="The retrieval subquery that produced this group.")
    intent: str = Field(description="Query intent associated with the group.")
    packed_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Packed context metadata for this evidence group.",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items gathered for the subquery.",
    )
    debug: dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval debug metadata associated with the group.",
    )


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Final grounded answer for the user question.")
    reasoning_summary: str = Field(
        description="Brief explanation of how the answer was derived from evidence."
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Most important evidence supporting the answer.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model-estimated confidence based on evidence coverage and consistency.",
    )
    limitations: str = Field(
        description="Explicit limitations or gaps in the available evidence."
    )


class OutOfScopeResponse(BaseModel):
    reason: str = Field(description="Why the current question is outside the corpus.")
    boundary: str = Field(description="What the knowledge base does cover instead.")
    suggestion: str = Field(description="How the user could rephrase within scope.")
    next_action: str = Field(
        description="Recommended user action if they want this question answered."
    )


class AdaptiveRouteDecision(BaseModel):
    decision: Literal["direct", "fact", "refuse"] = "fact"
    reason: str = ""


class AdaptiveRequirement(BaseModel):
    id: str = ""
    requirement: str = Field(
        default="",
        validation_alias=AliasChoices("requirement", "description"),
    )
    query: str = ""

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: object) -> str:
        return str(value) if value is not None else ""


class AdaptivePlan(BaseModel):
    requirements: list[AdaptiveRequirement] = Field(
        default_factory=list,
        max_length=3,
        validation_alias=AliasChoices("requirements", "evidence_requirements"),
    )


class EvidenceCoverage(BaseModel):
    requirement_id: str = Field(
        default="",
        validation_alias=AliasChoices("requirement_id", "id"),
    )
    covered: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    coverage: float = Field(ge=0.0, le=1.0)
    missing_reason: str = ""
    recommended_follow_up_query: str = Field(
        default="",
        validation_alias=AliasChoices(
            "recommended_follow_up_query",
            "follow_up_query",
        ),
    )

    @field_validator("missing_reason", "recommended_follow_up_query", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str:
        return "" if value is None else str(value)


class EvidenceSufficiency(BaseModel):
    items: list[EvidenceCoverage] = Field(default_factory=list)


class AdaptiveClaim(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    major: bool = True


class AdaptiveAnswer(BaseModel):
    answer: str
    claims: list[AdaptiveClaim] = Field(default_factory=list)
    limitations: str = ""


class ClaimSupportJudgment(BaseModel):
    claim_index: int = Field(ge=0)
    claim_spec_id: str | None = None
    semantically_supported: bool = False
    reason: str = ""


class ClaimSupportAssessment(BaseModel):
    items: list[ClaimSupportJudgment] = Field(default_factory=list)

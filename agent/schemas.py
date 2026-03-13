from typing import Literal
from pydantic import BaseModel, Field


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

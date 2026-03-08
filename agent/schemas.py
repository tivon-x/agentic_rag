from typing import List, Literal
from pydantic import BaseModel, Field


class RetrievalDecision(BaseModel):
    decision: Literal["retrieve", "direct_answer", "out_of_scope"] = Field(
        description="Routing decision for the latest user message."
    )
    reason: str = Field(
        description="Short explanation for the routing decision."
    )


class QueryAnalysis(BaseModel):
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field

from .providers.base import ProviderConfig

Mode = Literal["lite", "premium", "adaptive"]
Severity = Literal["low", "medium", "high"]
BiasType = Literal[
    "confirmation_bias",
    "certainty_inflation",
    "overgeneralisation",
    "framing_effect",
    "causal_inference_error",
]

AgentName = Literal["ARGUS", "LIBRA", "LENS", "QUILL", "VIGIL"]


class Annotation(BaseModel):
    bias_type: BiasType
    span_start: int
    span_end: int
    flagged_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity
    clean_alternative: Optional[str] = None
    false_positive_check: Optional[str] = None
    rag_check_needed: bool = False
    rag_query: Optional[str] = None
    conflict: bool = False
    agent_name: str
    prompt_version: str
    extras: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    text: str
    references: Optional[str] = None
    mode: Mode = "lite"
    provider: ProviderConfig
    # If empty / None, all 5 agents run.
    agents: Optional[list[AgentName]] = None


class PingRequest(BaseModel):
    provider: ProviderConfig


class AgentRunInfo(BaseModel):
    agent: str
    bias_type: BiasType
    prompt_version: str
    raw_count: int
    kept_count: int
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    document_id: str
    mode: Mode
    overall_bias_score: float
    annotations: list[Annotation]
    agents: list[AgentRunInfo]
    warnings: list[str] = Field(default_factory=list)
    provider: dict  # safe view (no api_key)

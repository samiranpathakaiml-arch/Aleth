"""
Aleth - Citation Verification Environment
Pydantic models for Action, Observation, Reward, and State spaces.

All string values use lowercase vocabulary for strict consistency:
  - support_score: 0.0 (not supported) to 1.0 (fully supported)
  - AccessLevel: "full_text" | "abstract_only" | "unavailable"
  - ClaimType: "empirical" | "methodological" | "theoretical"
"""

from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AccessLevel(str, Enum):
    """Paper access levels (simulates real-world paywalls)"""
    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    UNAVAILABLE = "unavailable"


class ClaimType(str, Enum):
    """Types of scientific claims the agent must verify"""
    EMPIRICAL = "empirical"          # e.g. "Model achieves 95% accuracy"
    METHODOLOGICAL = "methodological" # e.g. "We use technique X"
    THEORETICAL = "theoretical"       # e.g. "This proves theorem Y"


# ============================================================================
# DATA MODELS
# ============================================================================

class Paper(BaseModel):
    """Represents a scientific paper in the reference corpus"""
    id: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    year: int
    authors: List[str]
    access_level: AccessLevel


class Claim(BaseModel):
    """Represents a scientific claim from a paper that must be verified"""
    id: str
    text: str
    citations: List[str] = Field(..., description="List of paper IDs cited")
    context: str = Field(..., description="Surrounding sentence context")
    claim_type: ClaimType = ClaimType.EMPIRICAL


class GroundTruth(BaseModel):
    """
    Ground truth for deterministic grading.
    Hidden from agent during episode; used only at grade time.
    """
    claim_id: str
    true_support_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    primary_evidence_paper: Optional[str] = None
    key_concepts: List[str] = Field(
        default_factory=list,
        description="Keywords that should appear in a correct reasoning"
    )
    has_citation_drift: bool = False
    drift_chain: Optional[List[str]] = None


# ============================================================================
# ACTION SPACE
# ============================================================================

class ReadPaperAction(BaseModel):
    """Read a paper from the corpus. Returns text based on access level."""
    action_type: Literal["read_paper"] = "read_paper"
    paper_id: str = Field(..., description="ID of the paper to read")


class VerifyClaimAction(BaseModel):
    """Submit the agent's verification judgment for a claim."""
    action_type: Literal["verify_claim"] = "verify_claim"
    claim_id: str = Field(..., description="ID of the claim to verify")
    support_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Degree to which citations support the claim (0.0=none, 1.0=full)"
    )
    reasoning: str = Field(..., description="Explanation referencing evidence found")
    primary_evidence_paper: Optional[str] = Field(
        None, description="Paper ID providing strongest evidence"
    )


class FlagDriftAction(BaseModel):
    """Flag citation drift: when a claim's meaning has shifted through citation chains."""
    action_type: Literal["flag_drift"] = "flag_drift"
    claim_id: str = Field(..., description="Claim suspected of citation drift")
    citation_chain: List[str] = Field(
        ..., description="Ordered list of paper IDs showing the drift chain"
    )
    drift_explanation: str = Field(
        ..., description="Explanation of how the meaning drifted"
    )


class SubmitAction(BaseModel):
    """Finalize and submit all verifications for grading."""
    action_type: Literal["submit"] = "submit"


# Union of all valid actions (discriminated by action_type)
Action = Union[ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction]


# ============================================================================
# OBSERVATION SPACE
# ============================================================================

class PaperContent(BaseModel):
    """Content returned after a read_paper action"""
    paper_id: str
    content_type: AccessLevel
    text: str
    sections_available: List[str] = Field(default_factory=list)


class VerificationFeedback(BaseModel):
    """Immediate partial feedback given after verify_claim (not full grade)"""
    claim_id: str
    partial_score: float = Field(..., ge=0.0, le=1.0)
    hints: Optional[str] = None


class Observation(BaseModel):
    """
    Everything the agent can observe after each step.
    Designed to be serializable to / from JSON strings for LLM prompting.
    """
    message: str
    papers_content: List[PaperContent] = Field(default_factory=list)
    feedback: Optional[VerificationFeedback] = None
    claims_verified: int
    claims_total: int
    step_count: int
    papers_read: List[str] = Field(default_factory=list)


# ============================================================================
# REWARD SPACE
# ============================================================================

class Reward(BaseModel):
    """
    Reward signal with a full breakdown for interpretability.
    total is the sum of all breakdown values.
    """
    total: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# STATE (INTERNAL — hidden from agent)
# ============================================================================

class Verification(BaseModel):
    """Agent's submitted verification record for a single claim"""
    claim_id: str
    support_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    primary_evidence_paper: Optional[str] = None
    flagged_drift: bool = False
    drift_explanation: Optional[str] = None
    timestamp: int = Field(..., description="Step number at which verification was submitted")


class State(BaseModel):
    """
    Complete internal environment state.
    Never directly shown to the agent; exposed only through Observations.
    """
    task_id: str
    claims: Dict[str, Claim]
    papers: Dict[str, Paper]
    ground_truth: Dict[str, GroundTruth]
    verifications: Dict[str, Verification] = Field(default_factory=dict)
    papers_read: List[str] = Field(default_factory=list)
    step_count: int = 0
    max_steps: Optional[int] = None
    episode_done: bool = False

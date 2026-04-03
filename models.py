"""
Aleth - Citation Verification Environment
Pydantic models for Action, Observation, Reward, and State spaces.
"""

from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class AccessLevel(str, Enum):
    """Paper access levels (simulates paywalls)"""
    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    UNAVAILABLE = "unavailable"


class ClaimType(str, Enum):
    """Types of scientific claims"""
    EMPIRICAL = "empirical"  # "Model achieves 95% accuracy"
    METHODOLOGICAL = "methodological"  # "We use technique X"
    THEORETICAL = "theoretical"  # "This proves theorem Y"


# ============================================================================
# DATA MODELS
# ============================================================================

class Paper(BaseModel):
    """Represents a scientific paper"""
    id: str
    title: str
    abstract: str
    full_text: Optional[str] = None
    year: int
    authors: List[str]
    access_level: AccessLevel


class Claim(BaseModel):
    """Represents a claim from a paper being verified"""
    id: str
    text: str
    citations: List[str]  # List of paper IDs
    context: str
    claim_type: ClaimType = ClaimType.EMPIRICAL


class GroundTruth(BaseModel):
    """Ground truth for grading (hidden from agent)"""
    claim_id: str
    true_support_score: float  # 0.0 to 1.0
    reasoning: str
    primary_evidence_paper: Optional[str] = None
    key_concepts: List[str]
    has_citation_drift: bool = False
    drift_chain: Optional[List[str]] = None


# ============================================================================
# ACTION SPACE
# ============================================================================

class ReadPaperAction(BaseModel):
    """Read a paper"""
    action_type: Literal["read_paper"] = "read_paper"
    paper_id: str


class VerifyClaimAction(BaseModel):
    """Submit verification for a claim"""
    action_type: Literal["verify_claim"] = "verify_claim"
    claim_id: str
    support_score: float  # 0.0 to 1.0
    reasoning: str
    primary_evidence_paper: Optional[str] = None


class FlagDriftAction(BaseModel):
    """Flag citation drift"""
    action_type: Literal["flag_drift"] = "flag_drift"
    claim_id: str
    citation_chain: List[str]
    drift_explanation: str


class SubmitAction(BaseModel):
    """Finish and submit"""
    action_type: Literal["submit"] = "submit"


# Union of all actions
Action = Union[ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction]


# ============================================================================
# OBSERVATION SPACE
# ============================================================================

class PaperContent(BaseModel):
    """Content returned when reading a paper"""
    paper_id: str
    content_type: AccessLevel
    text: str
    sections_available: List[str] = []


class VerificationFeedback(BaseModel):
    """Immediate feedback after verification"""
    claim_id: str
    partial_score: float
    hints: Optional[str] = None


class Observation(BaseModel):
    """What the agent observes"""
    message: str
    papers_content: List[PaperContent] = []
    feedback: Optional[VerificationFeedback] = None
    claims_verified: int
    claims_total: int
    step_count: int
    papers_read: List[str] = []


# ============================================================================
# REWARD SPACE
# ============================================================================

class Reward(BaseModel):
    """Reward with breakdown"""
    total: float
    breakdown: Dict[str, float] = {}


# ============================================================================
# STATE (INTERNAL)
# ============================================================================

class Verification(BaseModel):
    """Agent's verification of a claim"""
    claim_id: str
    support_score: float
    reasoning: str
    primary_evidence_paper: Optional[str] = None
    flagged_drift: bool = False
    drift_explanation: Optional[str] = None
    timestamp: int


class State(BaseModel):
    """Complete environment state"""
    task_id: str
    claims: Dict[str, Claim]
    papers: Dict[str, Paper]
    ground_truth: Dict[str, GroundTruth]
    verifications: Dict[str, Verification] = {}
    papers_read: List[str] = []
    step_count: int = 0
    max_steps: Optional[int] = None
    episode_done: bool = False

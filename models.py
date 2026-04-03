"""
Aleth — Pydantic models for Action, Observation, Reward, and State spaces.
All imports are absolute (no leading dots) for flat-structure compatibility.
"""

from typing import List, Dict, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class AccessLevel(str, Enum):
    FULL_TEXT     = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    UNAVAILABLE   = "unavailable"


class ClaimType(str, Enum):
    EMPIRICAL      = "empirical"
    METHODOLOGICAL = "methodological"
    THEORETICAL    = "theoretical"


# ── Data models ───────────────────────────────────────────────────────────────

class Paper(BaseModel):
    id:           str
    title:        str
    abstract:     str
    full_text:    Optional[str] = None
    year:         int
    authors:      List[str]
    access_level: AccessLevel


class Claim(BaseModel):
    id:         str
    text:       str
    citations:  List[str]
    context:    str
    claim_type: ClaimType = ClaimType.EMPIRICAL


class GroundTruth(BaseModel):
    claim_id:              str
    true_support_score:    float
    reasoning:             str
    primary_evidence_paper: Optional[str] = None
    key_concepts:          List[str] = Field(default_factory=list)
    has_citation_drift:    bool = False
    drift_chain:           Optional[List[str]] = None


# ── Action space ──────────────────────────────────────────────────────────────

class ReadPaperAction(BaseModel):
    action_type: Literal["read_paper"] = "read_paper"
    paper_id:    str


class VerifyClaimAction(BaseModel):
    action_type:            Literal["verify_claim"] = "verify_claim"
    claim_id:               str
    support_score:          float = Field(..., ge=0.0, le=1.0)
    reasoning:              str
    primary_evidence_paper: Optional[str] = None


class FlagDriftAction(BaseModel):
    action_type:       Literal["flag_drift"] = "flag_drift"
    claim_id:          str
    citation_chain:    List[str]
    drift_explanation: str


class SubmitAction(BaseModel):
    action_type: Literal["submit"] = "submit"


Action = Union[ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction]


# ── Observation space ──────────────────────────────────────────────────────────

class PaperContent(BaseModel):
    paper_id:           str
    content_type:       AccessLevel
    text:               str
    sections_available: List[str] = Field(default_factory=list)


class VerificationFeedback(BaseModel):
    claim_id:      str
    partial_score: float
    hints:         Optional[str] = None


class Observation(BaseModel):
    message:        str
    papers_content: List[PaperContent] = Field(default_factory=list)
    feedback:       Optional[VerificationFeedback] = None
    claims_verified: int
    claims_total:   int
    step_count:     int
    papers_read:    List[str] = Field(default_factory=list)


# ── Reward space ──────────────────────────────────────────────────────────────

class Reward(BaseModel):
    total:     float
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ── Internal state ────────────────────────────────────────────────────────────

class Verification(BaseModel):
    claim_id:               str
    support_score:          float
    reasoning:              str
    primary_evidence_paper: Optional[str] = None
    flagged_drift:          bool = False
    drift_explanation:      Optional[str] = None
    timestamp:              int


class State(BaseModel):
    task_id:       str
    claims:        Dict[str, Claim]
    papers:        Dict[str, Paper]
    ground_truth:  Dict[str, GroundTruth]
    verifications: Dict[str, Verification] = Field(default_factory=dict)
    papers_read:   List[str] = Field(default_factory=list)
    step_count:    int = 0
    max_steps:     Optional[int] = None
    episode_done:  bool = False

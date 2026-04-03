"""
Aleth — Citation Verification Environment
Version 1.0.0

Public API:
  from aleth import AlethEnv, Observation, Reward, State
  from aleth import ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction
  from aleth import AlethGrader, AlethRewardComputer
"""

__version__ = "1.0.0"
__author__ = "Aleth Team"
__description__ = "OpenEnv-compliant scientific citation verification benchmark"

from .environment import AlethEnv
from .grader import AlethGrader
from .reward import AlethRewardComputer
from .models import (
    # Action classes
    ReadPaperAction,
    VerifyClaimAction,
    FlagDriftAction,
    SubmitAction,
    Action,
    # Observation / Reward / State
    Observation,
    Reward,
    State,
    # Extra models (useful for type hints)
    Claim,
    Paper,
    GroundTruth,
    Verification,
    PaperContent,
    VerificationFeedback,
    # Enums
    AccessLevel,
    ClaimType,
)

__all__ = [
    # Core environment
    "AlethEnv",
    "AlethGrader",
    "AlethRewardComputer",

    # Actions
    "ReadPaperAction",
    "VerifyClaimAction",
    "FlagDriftAction",
    "SubmitAction",
    "Action",

    # Spaces
    "Observation",
    "Reward",
    "State",

    # Data models
    "Claim",
    "Paper",
    "GroundTruth",
    "Verification",
    "PaperContent",
    "VerificationFeedback",

    # Enums
    "AccessLevel",
    "ClaimType",

    # Meta
    "__version__",
]

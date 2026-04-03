"""
Aleth - Citation Verification Environment
Truth revealed through evidence
"""

from .environment import AlethEnv
from .models import (
    Action, Observation, Reward, State,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction
)
from .grader import AlethGrader
from .reward import AlethRewardComputer

__version__ = "1.0.0"
__all__ = [
    "AlethEnv",
    "Action", "Observation", "Reward", "State",
    "ReadPaperAction", "VerifyClaimAction", "FlagDriftAction", "SubmitAction",
    "AlethGrader", "AlethRewardComputer"
]

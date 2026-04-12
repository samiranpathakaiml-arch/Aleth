"""
Aleth - Deterministic Grading Logic

AlethGrader produces a final score in [0.0, 1.0] for an episode.
The same inputs always produce the same score (fully deterministic).

Per-claim scoring (grade_claim):
  60% — Support score accuracy (|agent_score - gt_score| penalised linearly)
  30% — Reasoning quality    (fraction of key_concepts present)
  10% — Drift detection      (exact match of has_citation_drift flag)
"""

from typing import Dict
from .models import State, GroundTruth, Verification


# ---------------------------------------------------------------------------
# Platform requirement: task scores must be STRICTLY between 0 and 1
# (not 0.0, not 1.0). Clamp every score to this open interval.
# ---------------------------------------------------------------------------
_SCORE_MIN: float = 0.001
_SCORE_MAX: float = 0.999


def _clamp(score: float) -> float:
    """Clamp score to open interval (0, 1) as required by submission validator."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


class AlethGrader:
    """
    Deterministic episode grader.

    Args:
        ground_truth: dict mapping claim_id -> GroundTruth
    """

    def __init__(self, ground_truth: Dict[str, GroundTruth]):
        self.ground_truth = ground_truth

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def grade_episode(self, state: State) -> float:
        """
        Grade a complete episode.

        For each claim in ground_truth:
          - If verified: call grade_claim()
          - If unverified: score = _SCORE_MIN (never exactly 0.0)

        Returns:
            float: Average claim score, strictly within (0, 1).
        """
        if not self.ground_truth:
            return _SCORE_MIN

        total_score = 0.0
        for claim_id in self.ground_truth:
            if claim_id in state.verifications:
                total_score += self.grade_claim(claim_id, state.verifications[claim_id])
            else:
                total_score += _SCORE_MIN  # unverified — floor, never exactly 0.0

        raw = round(total_score / len(self.ground_truth), 6)
        return _clamp(raw)

    def grade_claim(self, claim_id: str, verification: Verification) -> float:
        """
        Grade a single claim verification.

        Scoring:
          60% — |agent_support_score - true_support_score| penalised linearly
          30% — fraction of key_concepts found in reasoning (case-insensitive)
          10% — correct drift detection (1.0 if match, 0.0 otherwise)

        Args:
            claim_id:     ID of the claim being graded
            verification: The agent's submitted Verification object

        Returns:
            float: Claim score, strictly within (0, 1).
        """
        gt = self.ground_truth[claim_id]

        # 1. Support score accuracy (60%)
        error = abs(verification.support_score - gt.true_support_score)
        accuracy = max(0.0, 1.0 - error)

        # 2. Reasoning quality via key-concept matching (30%)
        reasoning_score = self._reasoning_quality(verification.reasoning, gt.key_concepts)

        # 3. Drift detection accuracy (10%)
        drift_score = 1.0 if verification.flagged_drift == gt.has_citation_drift else 0.0

        raw = round(0.6 * accuracy + 0.3 * reasoning_score + 0.1 * drift_score, 6)
        return _clamp(raw)

    def get_detailed_breakdown(self, state: State) -> Dict:
        """
        Return per-claim grading breakdown for post-episode analysis.

        Returns:
            dict: claim_id -> {accuracy, reasoning, drift, total, notes}
        """
        breakdown: Dict = {}

        for claim_id, gt in self.ground_truth.items():
            if claim_id not in state.verifications:
                breakdown[claim_id] = {
                    "total": _SCORE_MIN,
                    "note": "Not verified",
                }
                continue

            v = state.verifications[claim_id]
            error = abs(v.support_score - gt.true_support_score)
            accuracy = max(0.0, 1.0 - error)
            reasoning = self._reasoning_quality(v.reasoning, gt.key_concepts)
            drift = 1.0 if v.flagged_drift == gt.has_citation_drift else 0.0
            total = self.grade_claim(claim_id, v)

            breakdown[claim_id] = {
                "gt_support_score": gt.true_support_score,
                "agent_support_score": v.support_score,
                "accuracy_component": round(0.6 * accuracy, 4),
                "reasoning_component": round(0.3 * reasoning, 4),
                "drift_component": round(0.1 * drift, 4),
                "total": total,
            }

        return breakdown

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _reasoning_quality(self, reasoning: str, key_concepts: list) -> float:
        """
        Returns the fraction of key_concepts found in reasoning (case-insensitive).
        Returns 1.0 if key_concepts is empty (no concepts required).
        """
        if not key_concepts:
            return 1.0
        reasoning_lower = reasoning.lower()
        matches = sum(1 for c in key_concepts if c.lower() in reasoning_lower)
        return matches / len(key_concepts)

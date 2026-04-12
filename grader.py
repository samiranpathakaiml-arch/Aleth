"""
Aleth — Deterministic Grading Logic (absolute imports, flat structure)
60% accuracy | 30% reasoning quality | 10% drift detection
"""

from typing import Dict
from models import State, GroundTruth, Verification   # absolute import


# Platform requirement: task scores must be strictly between 0 and 1 (exclusive)
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _clamp(score: float) -> float:
    """Clamp score to open interval (0, 1) as required by submission validator."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


class AlethGrader:
    """Deterministic episode grader — same inputs always produce same score."""

    def __init__(self, ground_truth: Dict[str, GroundTruth]):
        self.ground_truth = ground_truth

    # ── Public API ────────────────────────────────────────────────────────────

    def grade_episode(self, state: State) -> float:
        """Average claim score over all claims in ground truth. Unverified = 0.0.

        Returns a value strictly within (0, 1) as required by the platform validator.
        """
        if not self.ground_truth:
            return _SCORE_MIN
        total = sum(
            self.grade_claim(cid, state.verifications[cid])
            if cid in state.verifications else _SCORE_MIN
            for cid in self.ground_truth
        )
        raw = round(total / len(self.ground_truth), 6)
        return _clamp(raw)

    def grade_claim(self, claim_id: str, verification: Verification) -> float:
        """
        Score = 0.6 × accuracy + 0.3 × reasoning_quality + 0.1 × drift_detection

        Returns a value strictly within (0, 1) as required by the platform validator.
        """
        gt = self.ground_truth[claim_id]
        accuracy  = max(0.0, 1.0 - abs(verification.support_score - gt.true_support_score))
        reasoning = self._reasoning_quality(verification.reasoning, gt.key_concepts)
        drift     = 1.0 if verification.flagged_drift == gt.has_citation_drift else 0.0
        raw = round(0.6 * accuracy + 0.3 * reasoning + 0.1 * drift, 6)
        return _clamp(raw)

    def get_detailed_breakdown(self, state: State) -> Dict:
        """Per-claim grading breakdown for post-episode analysis."""
        breakdown: Dict = {}
        for cid, gt in self.ground_truth.items():
            if cid not in state.verifications:
                breakdown[cid] = {"total": _SCORE_MIN, "note": "Not verified"}
                continue
            v        = state.verifications[cid]
            accuracy = max(0.0, 1.0 - abs(v.support_score - gt.true_support_score))
            reasoning= self._reasoning_quality(v.reasoning, gt.key_concepts)
            drift    = 1.0 if v.flagged_drift == gt.has_citation_drift else 0.0
            breakdown[cid] = {
                "gt_support_score":    gt.true_support_score,
                "agent_support_score": v.support_score,
                "accuracy_component":  round(0.6 * accuracy,  4),
                "reasoning_component": round(0.3 * reasoning, 4),
                "drift_component":     round(0.1 * drift,     4),
                "total":               self.grade_claim(cid, v),
            }
        return breakdown

    # ── Private ───────────────────────────────────────────────────────────────

    def _reasoning_quality(self, reasoning: str, key_concepts: list) -> float:
        if not key_concepts:
            return 1.0
        low = reasoning.lower()
        return sum(1 for c in key_concepts if c.lower() in low) / len(key_concepts)

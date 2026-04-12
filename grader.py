"""
Aleth — Deterministic Grading Logic (absolute imports, flat structure)
60% accuracy | 30% reasoning quality | 10% drift detection

IMPORTANT: All component scores are bounded to [0.05, 0.95] so the weighted
formula can never produce exactly 0.0 or 1.0, satisfying the platform's
strict (0, 1) exclusive requirement without needing a post-hoc clamp.
"""

from typing import Dict
from models import State, GroundTruth, Verification   # absolute import


# Platform requirement: task scores must be strictly between 0 and 1 (exclusive)
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999

# Component bounds — keep components away from 0 and 1 so the weighted
# formula 0.6*acc + 0.3*reas + 0.1*drift can never reach 0.0 or 1.0
_COMP_MIN = 0.05   # minimum for any single component
_COMP_MAX = 0.95   # maximum for any single component


def _clamp(score: float) -> float:
    """Safety clamp to open interval (0, 1) as required by submission validator."""
    return max(_SCORE_MIN, min(_SCORE_MAX, score))


def _bound_component(v: float) -> float:
    """Bound a component score to [_COMP_MIN, _COMP_MAX] so no weighted sum
    of components can ever reach exactly 0.0 or 1.0."""
    return max(_COMP_MIN, min(_COMP_MAX, float(v)))


class AlethGrader:
    """Deterministic episode grader — same inputs always produce same score."""

    def __init__(self, ground_truth: Dict[str, GroundTruth]):
        self.ground_truth = ground_truth

    # ── Public API ────────────────────────────────────────────────────────────

    def grade_episode(self, state: State) -> float:
        """Average claim score over all claims in ground truth.

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

        All components are bounded to [0.05, 0.95] so this formula can never
        produce exactly 0.0 or 1.0 regardless of input values.
        Returns a value strictly within (0, 1).
        """
        gt = self.ground_truth[claim_id]

        # Accuracy: how close is the agent's support_score to ground truth?
        raw_accuracy = max(0.0, 1.0 - abs(verification.support_score - gt.true_support_score))
        accuracy  = _bound_component(raw_accuracy)

        # Reasoning quality: what fraction of key concepts are present?
        reasoning = _bound_component(self._reasoning_quality(verification.reasoning, gt.key_concepts))

        # Drift: 0.95 for correct detection, 0.05 for wrong — never 1.0 or 0.0
        drift = _COMP_MAX if verification.flagged_drift == gt.has_citation_drift else _COMP_MIN

        raw = round(0.6 * accuracy + 0.3 * reasoning + 0.1 * drift, 6)
        return _clamp(raw)

    def get_detailed_breakdown(self, state: State) -> Dict:
        """Per-claim grading breakdown for post-episode analysis."""
        breakdown: Dict = {}
        for cid, gt in self.ground_truth.items():
            if cid not in state.verifications:
                breakdown[cid] = {"total": _SCORE_MIN, "note": "Not verified"}
                continue
            v         = state.verifications[cid]
            raw_acc   = max(0.0, 1.0 - abs(v.support_score - gt.true_support_score))
            accuracy  = _bound_component(raw_acc)
            reasoning = _bound_component(self._reasoning_quality(v.reasoning, gt.key_concepts))
            drift     = _COMP_MAX if v.flagged_drift == gt.has_citation_drift else _COMP_MIN
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
        """Fraction of key concepts present in reasoning text.
        Returns raw float in [0, 1]; caller should apply _bound_component."""
        if not key_concepts:
            return _COMP_MAX   # no concepts to check — give benefit of doubt (0.95, not 1.0)
        low = reasoning.lower()
        return sum(1 for c in key_concepts if c.lower() in low) / len(key_concepts)

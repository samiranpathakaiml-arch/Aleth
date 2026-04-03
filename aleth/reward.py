"""
Aleth - Dense Reward Computation Engine
Provides continuous feedback signals throughout the episode to guide learning.

Reward breakdown keys:
  relevant_read       (+0.02)  reading a paper that is actually cited as evidence
  irrelevant_read     (-0.05)  reading a paper with no evidentiary role
  reread_penalty      (-0.01)  re-reading a paper already seen
  verification_accuracy (0-0.5) continuous score based on proximity to ground truth
  reasoning_bonus     (+0.10)  if reasoning contains ≥60% of key ground-truth concepts
  correct_drift_detection (+0.20) correctly flagging a drifted citation
  false_drift_flag    (-0.10)  flagging drift that doesn't exist
  progress            (+0.02 * remaining_fraction) when a new claim is verified
  step_penalty        (-0.001) per-step efficiency tax
"""

from typing import Dict, Set
from .models import (
    State, Action, Reward, GroundTruth, Paper,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction,
    AccessLevel
)


class AlethRewardComputer:
    """
    Computes dense, shaped rewards for every environment step.

    Args:
        ground_truth: dict mapping claim_id -> GroundTruth
        papers: dict mapping paper_id -> Paper
    """

    def __init__(self, ground_truth: Dict[str, GroundTruth], papers: Dict[str, Paper]):
        self.ground_truth = ground_truth
        self.papers = papers
        # Pre-compute set of papers with actual evidentiary value
        self.relevant_papers: Set[str] = self._build_relevant_papers()

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def compute_reward(self, action: Action, prev_state: State, new_state: State) -> Reward:
        """
        Compute the reward for the transition prev_state -> new_state via action.

        Returns:
            Reward with total and per-signal breakdown.
        """
        breakdown: Dict[str, float] = {}

        # Action-specific signals
        if isinstance(action, ReadPaperAction):
            self._add_reading_rewards(action, prev_state, new_state, breakdown)
        elif isinstance(action, VerifyClaimAction):
            self._add_verification_rewards(action, prev_state, new_state, breakdown)
        elif isinstance(action, FlagDriftAction):
            self._add_drift_rewards(action, prev_state, new_state, breakdown)

        # Global signals (applied to all steps)
        self._add_progress_rewards(action, prev_state, new_state, breakdown)
        breakdown["step_penalty"] = -0.001  # Encourage efficiency

        total = round(sum(breakdown.values()), 6)
        return Reward(total=total, breakdown=breakdown)

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _build_relevant_papers(self) -> Set[str]:
        """Build the set of papers that are cited as primary evidence or drift chains."""
        relevant: Set[str] = set()
        for gt in self.ground_truth.values():
            if gt.primary_evidence_paper:
                relevant.add(gt.primary_evidence_paper)
            if gt.drift_chain:
                relevant.update(gt.drift_chain)
        return relevant

    def _add_reading_rewards(
        self,
        action: ReadPaperAction,
        prev_state: State,
        new_state: State,
        breakdown: Dict[str, float],
    ) -> None:
        """
        Reward reading relevant papers; penalise irrelevant or redundant reads.
        relevant_read    if the paper is an evidence / drift-chain paper  (+0.02)
        irrelevant_read  if the paper has no evidentiary role             (-0.05)
        reread_penalty   if the paper was already read in this episode    (-0.01)
        """
        paper_id = action.paper_id

        if paper_id in self.relevant_papers:
            breakdown["relevant_read"] = 0.02
        else:
            breakdown["irrelevant_read"] = -0.05

        # Penalise re-reads
        if paper_id in prev_state.papers_read:
            breakdown["reread_penalty"] = -0.01

    def _add_verification_rewards(
        self,
        action: VerifyClaimAction,
        prev_state: State,
        new_state: State,
        breakdown: Dict[str, float],
    ) -> None:
        """
        Continuous reward signal for claim verification quality.

        verification_accuracy: max 0.5 (scales with proximity to ground truth score)
        reasoning_bonus:       +0.1 if reasoning mentions ≥60% of key concepts
        read_before_verify:    +0.03 bonus for reading cited papers first
        verify_without_reading:-0.1 penalty for verifying without reading any source
        """
        claim_id = action.claim_id

        if claim_id not in self.ground_truth:
            return

        gt = self.ground_truth[claim_id]

        # 1. Support score accuracy (continuous, max 0.5)
        error = abs(action.support_score - gt.true_support_score)
        accuracy = max(0.0, 1.0 - error)
        breakdown["verification_accuracy"] = round(accuracy * 0.5, 6)

        # 2. Reasoning quality bonus
        if self._has_key_concepts(action.reasoning, gt.key_concepts):
            breakdown["reasoning_bonus"] = 0.1

        # 3. Reading incentive
        claim = new_state.claims.get(claim_id)
        if claim:
            read_citations = set(new_state.papers_read) & set(claim.citations)
            if read_citations:
                breakdown["read_before_verify"] = 0.03
            else:
                breakdown["verify_without_reading"] = -0.1

    def _add_drift_rewards(
        self,
        action: FlagDriftAction,
        prev_state: State,
        new_state: State,
        breakdown: Dict[str, float],
    ) -> None:
        """
        Binary reward for citation drift detection accuracy.

        correct_drift_detection: +0.2 (True Positive)
        false_drift_flag:        -0.1 (False Positive)
        """
        claim_id = action.claim_id

        if claim_id not in self.ground_truth:
            return

        gt = self.ground_truth[claim_id]

        if gt.has_citation_drift:
            breakdown["correct_drift_detection"] = 0.2
        else:
            breakdown["false_drift_flag"] = -0.1

    def _add_progress_rewards(
        self,
        action: Action,
        prev_state: State,
        new_state: State,
        breakdown: Dict[str, float],
    ) -> None:
        """
        Reward incremental progress: +0.02 * fraction of claims still unverified.
        Only triggered when a new verification is added.
        """
        if not prev_state.claims:
            return

        prev_verified = len(prev_state.verifications)
        new_verified = len(new_state.verifications)

        if new_verified > prev_verified:
            remaining_fraction = 1.0 - (prev_verified / len(prev_state.claims))
            breakdown["progress"] = round(0.02 * remaining_fraction, 6)

    def _has_key_concepts(self, reasoning: str, key_concepts: list) -> bool:
        """
        Returns True if ≥60% of key_concepts appear (case-insensitive) in reasoning.
        Empty key_concepts list always returns True.
        """
        if not key_concepts:
            return True
        reasoning_lower = reasoning.lower()
        matches = sum(1 for c in key_concepts if c.lower() in reasoning_lower)
        return matches >= len(key_concepts) * 0.6

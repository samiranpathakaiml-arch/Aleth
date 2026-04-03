"""
Aleth — Dense Reward Computation Engine (absolute imports, flat structure)

Reward signals:
  relevant_read        +0.02   irrelevant_read       -0.05   reread_penalty  -0.01
  verification_accuracy 0-0.5  reasoning_bonus       +0.10   read_before_verify +0.03
  verify_without_reading -0.10 correct_drift_detection +0.20  false_drift_flag  -0.10
  progress              +0.02×remaining   step_penalty -0.001
"""

from typing import Dict, Set
from models import (                                  # absolute import
    State, Action, Reward, GroundTruth, Paper,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction,
    AccessLevel,
)


class AlethRewardComputer:
    """Computes dense shaped rewards for every environment step."""

    def __init__(self, ground_truth: Dict[str, GroundTruth], papers: Dict[str, Paper]):
        self.ground_truth    = ground_truth
        self.papers          = papers
        self.relevant_papers: Set[str] = self._build_relevant_papers()

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_reward(self, action: Action, prev_state: State, new_state: State) -> Reward:
        bd: Dict[str, float] = {}

        if isinstance(action, ReadPaperAction):
            self._reading(action, prev_state, bd)
        elif isinstance(action, VerifyClaimAction):
            self._verification(action, new_state, bd)
        elif isinstance(action, FlagDriftAction):
            self._drift(action, bd)

        self._progress(prev_state, new_state, bd)
        bd["step_penalty"] = -0.001

        return Reward(total=round(sum(bd.values()), 6), breakdown=bd)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_relevant_papers(self) -> Set[str]:
        r: Set[str] = set()
        for gt in self.ground_truth.values():
            if gt.primary_evidence_paper:
                r.add(gt.primary_evidence_paper)
            if gt.drift_chain:
                r.update(gt.drift_chain)
        return r

    def _reading(self, action: ReadPaperAction, prev: State, bd: Dict) -> None:
        pid = action.paper_id
        bd["relevant_read" if pid in self.relevant_papers else "irrelevant_read"] = \
            0.02 if pid in self.relevant_papers else -0.05
        if pid in prev.papers_read:
            bd["reread_penalty"] = -0.01

    def _verification(self, action: VerifyClaimAction, new: State, bd: Dict) -> None:
        cid = action.claim_id
        if cid not in self.ground_truth:
            return
        gt    = self.ground_truth[cid]
        error = abs(action.support_score - gt.true_support_score)
        bd["verification_accuracy"] = round(max(0.0, 1.0 - error) * 0.5, 6)
        if self._key_concepts_match(action.reasoning, gt.key_concepts):
            bd["reasoning_bonus"] = 0.1
        claim = new.claims.get(cid)
        if claim:
            if set(new.papers_read) & set(claim.citations):
                bd["read_before_verify"] = 0.03
            else:
                bd["verify_without_reading"] = -0.1

    def _drift(self, action: FlagDriftAction, bd: Dict) -> None:
        cid = action.claim_id
        if cid not in self.ground_truth:
            return
        bd["correct_drift_detection" if self.ground_truth[cid].has_citation_drift
           else "false_drift_flag"] = \
            0.2 if self.ground_truth[cid].has_citation_drift else -0.1

    def _progress(self, prev: State, new: State, bd: Dict) -> None:
        if not prev.claims:
            return
        if len(new.verifications) > len(prev.verifications):
            bd["progress"] = round(0.02 * (1.0 - len(prev.verifications) / len(prev.claims)), 6)

    def _key_concepts_match(self, reasoning: str, concepts: list) -> bool:
        if not concepts:
            return True
        low = reasoning.lower()
        return sum(1 for c in concepts if c.lower() in low) >= len(concepts) * 0.6

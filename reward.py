"""
Aleth - Dense Reward Computation
Provides continuous feedback throughout episode
"""

from typing import Dict, Set
from .models import (
    State, Action, Reward, GroundTruth, Paper,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction,
    AccessLevel
)


class AlethRewardComputer:
    """Computes dense rewards with multiple signals"""
    
    def __init__(self, ground_truth: Dict[str, GroundTruth], papers: Dict[str, Paper]):
        self.ground_truth = ground_truth
        self.papers = papers
        self.relevant_papers = self._build_relevant_papers()
    
    def _build_relevant_papers(self) -> Set[str]:
        """Build set of papers actually cited"""
        relevant = set()
        for gt in self.ground_truth.values():
            if gt.primary_evidence_paper:
                relevant.add(gt.primary_evidence_paper)
            if gt.drift_chain:
                relevant.update(gt.drift_chain)
        return relevant
    
    def compute_reward(self, action: Action, prev_state: State, new_state: State) -> Reward:
        """Compute reward for action"""
        breakdown = {}
        
        # Type-specific rewards
        if isinstance(action, ReadPaperAction):
            self._add_reading_rewards(action, prev_state, new_state, breakdown)
        elif isinstance(action, VerifyClaimAction):
            self._add_verification_rewards(action, prev_state, new_state, breakdown)
        elif isinstance(action, FlagDriftAction):
            self._add_drift_rewards(action, prev_state, new_state, breakdown)
        
        # Global rewards
        self._add_progress_rewards(action, prev_state, new_state, breakdown)
        breakdown['step_penalty'] = -0.001  # Encourage efficiency
        
        total = sum(breakdown.values())
        return Reward(total=total, breakdown=breakdown)
    
    def _add_reading_rewards(self, action, prev_state, new_state, breakdown):
        """Rewards for reading papers"""
        paper_id = action.paper_id
        
        if paper_id in self.relevant_papers:
            breakdown['relevant_read'] = 0.02
        else:
            breakdown['irrelevant_read'] = -0.05
        
        if paper_id in prev_state.papers_read:
            breakdown['reread_penalty'] = -0.01
    
    def _add_verification_rewards(self, action, prev_state, new_state, breakdown):
        """Rewards for verifying claims"""
        claim_id = action.claim_id
        
        if claim_id not in self.ground_truth:
            return
        
        gt = self.ground_truth[claim_id]
        
        # Accuracy reward (continuous)
        error = abs(action.support_score - gt.true_support_score)
        accuracy = max(0.0, 1.0 - error)
        breakdown['verification_accuracy'] = accuracy * 0.5
        
        # Reasoning bonus
        if self._has_key_concepts(action.reasoning, gt.key_concepts):
            breakdown['reasoning_bonus'] = 0.1
        
        # Check if read papers before verifying
        claim = new_state.claims[claim_id]
        read_citations = set(new_state.papers_read) & set(claim.citations)
        if read_citations:
            breakdown['read_before_verify'] = 0.03
        else:
            breakdown['verify_without_reading'] = -0.1
    
    def _add_drift_rewards(self, action, prev_state, new_state, breakdown):
        """Rewards for drift detection"""
        claim_id = action.claim_id
        
        if claim_id not in self.ground_truth:
            return
        
        gt = self.ground_truth[claim_id]
        
        if gt.has_citation_drift:
            breakdown['correct_drift_detection'] = 0.2
        else:
            breakdown['false_drift_flag'] = -0.1
    
    def _add_progress_rewards(self, action, prev_state, new_state, breakdown):
        """Progress-based rewards"""
        prev_progress = len(prev_state.verifications) / len(prev_state.claims)
        new_progress = len(new_state.verifications) / len(new_state.claims)
        
        if new_progress > prev_progress:
            breakdown['progress'] = 0.02 * (1.0 - prev_progress)
    
    def _has_key_concepts(self, reasoning: str, key_concepts: list) -> bool:
        """Check if reasoning contains key concepts"""
        if not key_concepts:
            return True
        reasoning_lower = reasoning.lower()
        matches = sum(1 for c in key_concepts if c.lower() in reasoning_lower)
        return matches >= len(key_concepts) * 0.6

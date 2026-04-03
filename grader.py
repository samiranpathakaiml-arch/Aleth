"""
Aleth - Deterministic Grading Logic
"""

from typing import Dict
from .models import State, GroundTruth, Verification


class AlethGrader:
    """Deterministic grader - always returns same score for same input"""
    
    def __init__(self, ground_truth: Dict[str, GroundTruth]):
        self.ground_truth = ground_truth
    
    def grade_episode(self, state: State) -> float:
        """
        Grade complete episode.
        Returns score in [0.0, 1.0]
        """
        if not self.ground_truth:
            return 0.0
        
        total_score = 0.0
        for claim_id, gt in self.ground_truth.items():
            if claim_id in state.verifications:
                score = self.grade_claim(claim_id, state.verifications[claim_id])
            else:
                score = 0.0  # Unverified claims get 0
            total_score += score
        
        return total_score / len(self.ground_truth)
    
    def grade_claim(self, claim_id: str, verification: Verification) -> float:
        """
        Grade single claim.
        
        Breakdown:
        - 60% accuracy of support score
        - 30% reasoning quality
        - 10% drift detection
        """
        gt = self.ground_truth[claim_id]
        
        # 1. Support score accuracy
        error = abs(verification.support_score - gt.true_support_score)
        accuracy = max(0.0, 1.0 - error)
        
        # 2. Reasoning quality (check for key concepts)
        reasoning_lower = verification.reasoning.lower()
        concepts_found = sum(
            1 for concept in gt.key_concepts
            if concept.lower() in reasoning_lower
        )
        reasoning_score = concepts_found / len(gt.key_concepts) if gt.key_concepts else 1.0
        
        # 3. Drift detection
        drift_score = 1.0 if verification.flagged_drift == gt.has_citation_drift else 0.0
        
        # Weighted total
        return 0.6 * accuracy + 0.3 * reasoning_score + 0.1 * drift_score
    
    def get_detailed_breakdown(self, state: State) -> Dict:
        """Get detailed scores for debugging"""
        breakdown = {}
        for claim_id, gt in self.ground_truth.items():
            if claim_id in state.verifications:
                v = state.verifications[claim_id]
                error = abs(v.support_score - gt.true_support_score)
                breakdown[claim_id] = {
                    'accuracy': max(0.0, 1.0 - error),
                    'gt_score': gt.true_support_score,
                    'agent_score': v.support_score,
                    'total': self.grade_claim(claim_id, v)
                }
            else:
                breakdown[claim_id] = {'total': 0.0, 'note': 'Not verified'}
        return breakdown

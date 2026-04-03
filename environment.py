"""
Aleth - Main Environment Class
Implements OpenEnv specification
"""

import json
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from .models import (
    Action, Observation, Reward, State,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction,
    Claim, Paper, GroundTruth, Verification,
    PaperContent, VerificationFeedback, AccessLevel
)
from .grader import AlethGrader
from .reward import AlethRewardComputer


class AlethEnv:
    """
    Aleth Citation Verification Environment
    
    OpenEnv API:
    - reset(task) -> Observation
    - step(action) -> (Observation, Reward, done, info)
    - state() -> State
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        
        self._state: Optional[State] = None
        self.grader: Optional[AlethGrader] = None
        self.reward_computer: Optional[AlethRewardComputer] = None
    
    def reset(self, task: str = "easy") -> Observation:
        """Reset environment to initial state"""
        # Load task data
        task_file = self.data_dir / f"task_{task}.json"
        if not task_file.exists():
            raise ValueError(f"Task file not found: {task_file}")
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Parse into Pydantic models
        claims = {c['id']: Claim(**c) for c in task_data['claims']}
        papers = {p['id']: Paper(**p) for p in task_data['papers'].values()}
        ground_truth = {gt['claim_id']: GroundTruth(**gt) for gt in task_data['ground_truth']}
        
        # Initialize state
        self._state = State(
            task_id=task,
            claims=claims,
            papers=papers,
            ground_truth=ground_truth,
            verifications={},
            papers_read=[],
            step_count=0,
            max_steps=task_data.get('max_steps'),
            episode_done=False
        )
        
        self.grader = AlethGrader(ground_truth)
        self.reward_computer = AlethRewardComputer(ground_truth, papers)
        
        return self._make_observation(
            message=f"Task initialized: {len(claims)} claims to verify",
            papers_content=[],
            feedback=None
        )
    
    def step(self, action: Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict]:
        """Execute one step"""
        if self._state is None:
            raise RuntimeError("Call reset() first")
        if self._state.episode_done:
            raise RuntimeError("Episode done. Call reset()")
        
        # Parse action
        parsed_action = self._parse_action(action)
        
        # Store previous state
        prev_state = self._state.model_copy(deep=True)
        
        # Execute action
        obs, info = self._execute_action(parsed_action)
        
        # Compute reward
        reward = self.reward_computer.compute_reward(parsed_action, prev_state, self._state)
        
        # Update step count
        self._state.step_count += 1
        
        # Check if done
        done = self._check_done(parsed_action)
        self._state.episode_done = done
        
        # Final grading
        if done:
            final_score = self.grader.grade_episode(self._state)
            info['final_score'] = final_score
            info['grading_breakdown'] = self.grader.get_detailed_breakdown(self._state)
        
        return obs, reward, done, info
    
    def state(self) -> State:
        """Return current state"""
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state.model_copy(deep=True)
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _parse_action(self, action: Dict) -> Action:
        """Parse dict to Action model"""
        action_type = action.get('action_type')
        
        if action_type == 'read_paper':
            return ReadPaperAction(**action)
        elif action_type == 'verify_claim':
            return VerifyClaimAction(**action)
        elif action_type == 'flag_drift':
            return FlagDriftAction(**action)
        elif action_type == 'submit':
            return SubmitAction(**action)
        else:
            raise ValueError(f"Unknown action: {action_type}")
    
    def _execute_action(self, action: Action) -> Tuple[Observation, Dict]:
        """Execute action and return observation"""
        info = {'action_type': action.action_type}
        
        if isinstance(action, ReadPaperAction):
            return self._handle_read(action, info)
        elif isinstance(action, VerifyClaimAction):
            return self._handle_verify(action, info)
        elif isinstance(action, FlagDriftAction):
            return self._handle_flag(action, info)
        elif isinstance(action, SubmitAction):
            return self._handle_submit(action, info)
    
    def _handle_read(self, action: ReadPaperAction, info: Dict) -> Tuple[Observation, Dict]:
        """Handle read_paper action"""
        paper_id = action.paper_id
        
        if paper_id not in self._state.papers:
            return self._make_observation(
                message=f"Error: Paper '{paper_id}' not found",
                papers_content=[],
                feedback=None
            ), info
        
        paper = self._state.papers[paper_id]
        
        # Track reading
        if paper_id not in self._state.papers_read:
            self._state.papers_read.append(paper_id)
        
        # Determine text based on access level
        if paper.access_level == AccessLevel.FULL_TEXT:
            text = paper.full_text or paper.abstract
            sections = ["abstract", "introduction", "results"]
        elif paper.access_level == AccessLevel.ABSTRACT_ONLY:
            text = paper.abstract
            sections = ["abstract"]
        else:
            text = f"Title: {paper.title}\nAccess restricted."
            sections = []
        
        content = PaperContent(
            paper_id=paper_id,
            content_type=paper.access_level,
            text=text,
            sections_available=sections
        )
        
        info['paper_read'] = paper_id
        return self._make_observation(
            message=f"Read: {paper.title}",
            papers_content=[content],
            feedback=None
        ), info
    
    def _handle_verify(self, action: VerifyClaimAction, info: Dict) -> Tuple[Observation, Dict]:
        """Handle verify_claim action"""
        claim_id = action.claim_id
        
        if claim_id not in self._state.claims:
            return self._make_observation(
                message=f"Error: Claim '{claim_id}' not found",
                papers_content=[],
                feedback=None
            ), info
        
        # Store verification
        verification = Verification(
            claim_id=claim_id,
            support_score=action.support_score,
            reasoning=action.reasoning,
            primary_evidence_paper=action.primary_evidence_paper,
            timestamp=self._state.step_count
        )
        self._state.verifications[claim_id] = verification
        
        # Partial feedback
        feedback = None
        if claim_id in self._state.ground_truth:
            gt = self._state.ground_truth[claim_id]
            error = abs(action.support_score - gt.true_support_score)
            partial_score = max(0, 1.0 - error)
            feedback = VerificationFeedback(claim_id=claim_id, partial_score=partial_score)
        
        info['claim_verified'] = claim_id
        return self._make_observation(
            message=f"Verified {claim_id} (score: {action.support_score:.2f})",
            papers_content=[],
            feedback=feedback
        ), info
    
    def _handle_flag(self, action: FlagDriftAction, info: Dict) -> Tuple[Observation, Dict]:
        """Handle flag_drift action"""
        claim_id = action.claim_id
        
        if claim_id in self._state.verifications:
            self._state.verifications[claim_id].flagged_drift = True
            self._state.verifications[claim_id].drift_explanation = action.drift_explanation
        else:
            self._state.verifications[claim_id] = Verification(
                claim_id=claim_id,
                support_score=0.0,
                reasoning="",
                flagged_drift=True,
                drift_explanation=action.drift_explanation,
                timestamp=self._state.step_count
            )
        
        info['drift_flagged'] = claim_id
        return self._make_observation(
            message=f"Flagged drift for {claim_id}",
            papers_content=[],
            feedback=None
        ), info
    
    def _handle_submit(self, action: SubmitAction, info: Dict) -> Tuple[Observation, Dict]:
        """Handle submit action"""
        verified = len(self._state.verifications)
        total = len(self._state.claims)
        
        info['submitted'] = True
        info['verified_count'] = verified
        info['total_count'] = total
        
        return self._make_observation(
            message=f"Submitted {verified}/{total} verifications",
            papers_content=[],
            feedback=None
        ), info
    
    def _make_observation(self, message: str, papers_content: list, feedback) -> Observation:
        """Create observation"""
        return Observation(
            message=message,
            papers_content=papers_content,
            feedback=feedback,
            claims_verified=len(self._state.verifications),
            claims_total=len(self._state.claims),
            step_count=self._state.step_count,
            papers_read=self._state.papers_read.copy()
        )
    
    def _check_done(self, action: Action) -> bool:
        """Check if episode is done"""
        if isinstance(action, SubmitAction):
            return True
        if self._state.max_steps and self._state.step_count >= self._state.max_steps:
            return True
        if len(self._state.verifications) >= len(self._state.claims):
            return True
        return False

"""
Aleth - Main Environment Class
Implements the OpenEnv specification.

OpenEnv API:
  env = AlethEnv()
  obs: Observation = env.reset(task="easy")
  obs, reward, done, info = env.step(action_dict)
  state: State = env.state()
"""

import json
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from .models import (
    Action, Observation, Reward, State,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction,
    Claim, Paper, GroundTruth, Verification,
    PaperContent, VerificationFeedback, AccessLevel,
)
from .grader import AlethGrader
from .reward import AlethRewardComputer


class AlethEnv:
    """
    Aleth Citation Verification Environment.

    Conforms to the OpenEnv interface:
      reset(task) -> Observation
      step(action_dict) -> (Observation, Reward, done, info)
      state() -> State

    The agent must verify whether scientific claims are properly supported
    by their cited papers, and optionally detect citation drift.

    Args:
        data_dir (str | Path | None): Directory containing task_*.json files.
            Defaults to ``aleth/data/`` relative to this module.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)

        self._state: Optional[State] = None
        self.grader: Optional[AlethGrader] = None
        self.reward_computer: Optional[AlethRewardComputer] = None

    # =========================================================================
    # PUBLIC OpenEnv API
    # =========================================================================

    def reset(self, task: str = "easy") -> Observation:
        """
        Reset the environment to a fresh episode for the given task.

        Args:
            task: Task difficulty key (e.g. "easy"). Loads ``task_{task}.json``.

        Returns:
            Initial Observation describing the task.

        Raises:
            ValueError: If the task file does not exist.
        """
        task_file = self.data_dir / f"task_{task}.json"
        if not task_file.exists():
            raise ValueError(
                f"Task file not found: {task_file}. "
                f"Available: {list(self.data_dir.glob('task_*.json'))}"
            )

        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        # Parse task data into typed Pydantic models
        claims: Dict[str, Claim] = {
            c["id"]: Claim(**c) for c in task_data["claims"]
        }
        papers: Dict[str, Paper] = {
            pid: Paper(**pdata)
            for pid, pdata in task_data["papers"].items()
        }
        ground_truth: Dict[str, GroundTruth] = {
            gt["claim_id"]: GroundTruth(**gt)
            for gt in task_data["ground_truth"]
        }

        # Initialise fresh state
        self._state = State(
            task_id=task,
            claims=claims,
            papers=papers,
            ground_truth=ground_truth,
            verifications={},
            papers_read=[],
            step_count=0,
            max_steps=task_data.get("max_steps"),
            episode_done=False,
        )

        # Initialise grader and reward computer
        self.grader = AlethGrader(ground_truth)
        self.reward_computer = AlethRewardComputer(ground_truth, papers)

        claim_summary = "\n".join(
            f"  [{cid}] {c.text} (cites: {', '.join(c.citations)})"
            for cid, c in claims.items()
        )
        papers_available = ", ".join(papers.keys())

        return self._make_observation(
            message=(
                f"Task '{task}' initialised: {len(claims)} claims to verify.\n"
                f"Claims:\n{claim_summary}\n\n"
                f"Available papers: {papers_available}\n\n"
                f"Read papers, then verify each claim. Submit when done."
            ),
            papers_content=[],
            feedback=None,
        )

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one agent action and advance the environment.

        Args:
            action: Dict with at minimum ``action_type`` key.

        Returns:
            Tuple of (Observation, Reward, done, info).
            info contains ``final_score`` and ``grading_breakdown`` when done.

        Raises:
            RuntimeError: If reset() has not been called or episode is already done.
            ValueError: If action_type is unknown.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Parse raw dict to typed Action model
        parsed_action = self._parse_action(action)

        # Snapshot state before action (for reward computation)
        prev_state = self._state.model_copy(deep=True)

        # Execute the action
        obs, info = self._execute_action(parsed_action)

        # Increment step counter AFTER action execution, BEFORE reward / done check
        self._state.step_count += 1

        # Compute shaped reward
        reward = self.reward_computer.compute_reward(parsed_action, prev_state, self._state)

        # Termination check
        done = self._check_done(parsed_action)
        self._state.episode_done = done

        # Final grading when episode ends
        if done:
            final_score = self.grader.grade_episode(self._state)
            info["final_score"] = final_score
            info["grading_breakdown"] = self.grader.get_detailed_breakdown(self._state)
            info["steps_taken"] = self._state.step_count

        return obs, reward, done, info

    def state(self) -> State:
        """
        Return a deep copy of the current internal state.

        Returns:
            State object (deep copy — mutating it has no effect on the env).

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state.model_copy(deep=True)

    # =========================================================================
    # PRIVATE — ACTION PARSING
    # =========================================================================

    def _parse_action(self, action: Dict) -> Action:
        """Dispatch raw dict to the appropriate typed Action model."""
        action_type = action.get("action_type")

        if action_type == "read_paper":
            return ReadPaperAction(**action)
        elif action_type == "verify_claim":
            return VerifyClaimAction(**action)
        elif action_type == "flag_drift":
            return FlagDriftAction(**action)
        elif action_type == "submit":
            return SubmitAction(**action)
        else:
            raise ValueError(
                f"Unknown action_type: '{action_type}'. "
                f"Must be one of: read_paper, verify_claim, flag_drift, submit."
            )

    # =========================================================================
    # PRIVATE — ACTION HANDLERS
    # =========================================================================

    def _execute_action(self, action: Action) -> Tuple[Observation, Dict]:
        """Route action to its specific handler."""
        info: Dict[str, Any] = {"action_type": action.action_type}

        if isinstance(action, ReadPaperAction):
            return self._handle_read(action, info)
        elif isinstance(action, VerifyClaimAction):
            return self._handle_verify(action, info)
        elif isinstance(action, FlagDriftAction):
            return self._handle_flag(action, info)
        elif isinstance(action, SubmitAction):
            return self._handle_submit(action, info)
        else:
            # Unreachable due to _parse_action; kept for type safety
            raise ValueError(f"Unhandled action type: {type(action)}")

    def _handle_read(self, action: ReadPaperAction, info: Dict) -> Tuple[Observation, Dict]:
        """
        Handle read_paper: return paper text, respecting access level.
        - full_text: returns full paper text + all sections
        - abstract_only: returns only abstract
        - unavailable: returns restricted message
        """
        paper_id = action.paper_id

        if paper_id not in self._state.papers:
            available = list(self._state.papers.keys())
            return self._make_observation(
                message=f"Error: Paper '{paper_id}' not found. Available: {available}",
                papers_content=[],
                feedback=None,
            ), info

        paper = self._state.papers[paper_id]

        # Track read history (idempotent — no duplicate entries)
        if paper_id not in self._state.papers_read:
            self._state.papers_read.append(paper_id)

        # Apply access-level masking
        if paper.access_level == AccessLevel.FULL_TEXT:
            text = paper.full_text or paper.abstract
            sections = ["abstract", "introduction", "methods", "results", "conclusion"]
        elif paper.access_level == AccessLevel.ABSTRACT_ONLY:
            text = paper.abstract  # full_text deliberately withheld
            sections = ["abstract"]
        else:  # UNAVAILABLE
            text = (
                f"Title: {paper.title} ({paper.year})\n"
                f"Authors: {', '.join(paper.authors)}\n"
                f"[Access Restricted — paper unavailable]"
            )
            sections = []

        content = PaperContent(
            paper_id=paper_id,
            content_type=paper.access_level,
            text=text,
            sections_available=sections,
        )

        info["paper_read"] = paper_id
        info["access_level"] = paper.access_level.value

        return self._make_observation(
            message=f"Read '{paper.title}' ({paper.year}) — {paper.access_level.value}.",
            papers_content=[content],
            feedback=None,
        ), info

    def _handle_verify(self, action: VerifyClaimAction, info: Dict) -> Tuple[Observation, Dict]:
        """
        Handle verify_claim: record agent's verification and return partial feedback.
        Later verifications for the same claim overwrite earlier ones.
        """
        claim_id = action.claim_id

        if claim_id not in self._state.claims:
            available = list(self._state.claims.keys())
            return self._make_observation(
                message=f"Error: Claim '{claim_id}' not found. Available: {available}",
                papers_content=[],
                feedback=None,
            ), info

        # Validate support_score range (Pydantic already does this; belt-and-suspenders)
        score = max(0.0, min(1.0, action.support_score))

        verification = Verification(
            claim_id=claim_id,
            support_score=score,
            reasoning=action.reasoning,
            primary_evidence_paper=action.primary_evidence_paper,
            timestamp=self._state.step_count,
        )
        self._state.verifications[claim_id] = verification

        # Partial feedback (approximate; does NOT reveal ground truth directly)
        feedback: Optional[VerificationFeedback] = None
        if claim_id in self._state.ground_truth:
            gt = self._state.ground_truth[claim_id]
            error = abs(score - gt.true_support_score)
            partial_score = round(max(0.0, 1.0 - error), 4)
            hint = None
            if error > 0.4:
                hint = "Your score is quite far from the evidence. Re-read the cited paper."
            elif error > 0.2:
                hint = "Your score is moderately off. Look more closely at the results section."
            feedback = VerificationFeedback(
                claim_id=claim_id,
                partial_score=partial_score,
                hints=hint,
            )

        info["claim_verified"] = claim_id
        info["support_score"] = score

        return self._make_observation(
            message=(
                f"Verified '{claim_id}' with support_score={score:.2f}. "
                f"{len(self._state.verifications)}/{len(self._state.claims)} claims done."
            ),
            papers_content=[],
            feedback=feedback,
        ), info

    def _handle_flag(self, action: FlagDriftAction, info: Dict) -> Tuple[Observation, Dict]:
        """
        Handle flag_drift: attach drift flag to an existing or new verification.
        If claim has not been verified yet, creates a stub verification with score 0.0.
        """
        claim_id = action.claim_id

        if claim_id in self._state.verifications:
            # Amend existing verification
            self._state.verifications[claim_id].flagged_drift = True
            self._state.verifications[claim_id].drift_explanation = action.drift_explanation
        else:
            # Create stub verification (claim not yet scored)
            self._state.verifications[claim_id] = Verification(
                claim_id=claim_id,
                support_score=0.0,
                reasoning="(drift flagged before verification)",
                flagged_drift=True,
                drift_explanation=action.drift_explanation,
                timestamp=self._state.step_count,
            )

        info["drift_flagged"] = claim_id
        info["citation_chain"] = action.citation_chain

        return self._make_observation(
            message=(
                f"Citation drift flagged for '{claim_id}'. "
                f"Chain: {' -> '.join(action.citation_chain)}"
            ),
            papers_content=[],
            feedback=None,
        ), info

    def _handle_submit(self, action: SubmitAction, info: Dict) -> Tuple[Observation, Dict]:
        """Handle submit: finalise and close the episode."""
        verified = len(self._state.verifications)
        total = len(self._state.claims)
        unverified = [
            cid for cid in self._state.claims
            if cid not in self._state.verifications
        ]

        info["submitted"] = True
        info["verified_count"] = verified
        info["total_count"] = total
        info["unverified_claims"] = unverified

        return self._make_observation(
            message=(
                f"Submitted! {verified}/{total} claims verified. "
                + (f"Unverified: {unverified}" if unverified else "All claims covered.")
            ),
            papers_content=[],
            feedback=None,
        ), info

    # =========================================================================
    # PRIVATE — UTILITIES
    # =========================================================================

    def _make_observation(
        self,
        message: str,
        papers_content: list,
        feedback,
    ) -> Observation:
        """Construct an Observation from the current state."""
        assert self._state is not None
        return Observation(
            message=message,
            papers_content=papers_content,
            feedback=feedback,
            claims_verified=len(self._state.verifications),
            claims_total=len(self._state.claims),
            step_count=self._state.step_count,
            papers_read=list(self._state.papers_read),
        )

    def _check_done(self, action: Action) -> bool:
        """
        Episode termination conditions (any one is sufficient):
          1. SubmitAction was taken.
          2. step_count >= max_steps (budget exhausted).
          3. All claims have been verified.
        """
        if isinstance(action, SubmitAction):
            return True
        if (
            self._state.max_steps is not None
            and self._state.step_count >= self._state.max_steps
        ):
            return True
        if len(self._state.verifications) >= len(self._state.claims):
            return True
        return False

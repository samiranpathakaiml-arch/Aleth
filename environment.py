"""
Aleth — Main Environment Class (absolute imports, flat structure)

OpenEnv API:
  env = AlethEnv()
  obs = env.reset(task="easy")
  obs, reward, done, info = env.step(action_dict)
  state = env.state()
"""

import json
import uuid
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from models import (                                  # absolute import
    Action, Observation, Reward, State,
    ReadPaperAction, VerifyClaimAction, FlagDriftAction, SubmitAction,
    Claim, Paper, GroundTruth, Verification,
    PaperContent, VerificationFeedback, AccessLevel,
)
from grader import AlethGrader                        # absolute import
from reward import AlethRewardComputer                # absolute import
from chronicle import ChronicleService                # absolute import

logger = logging.getLogger(__name__)

# Default component score when no verifications exist
DEFAULT_COMPONENT_SCORE = 0.5


class AlethEnv:
    """
    Aleth Citation Verification Environment — OpenEnv compliant.

    Args:
        data_dir: Directory containing task_*.json files.
            Defaults to ``data/`` relative to this file (root/data/).
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Works for both flat-root and aleth-package layouts
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)

        self._state:          Optional[State]               = None
        self.grader:          Optional[AlethGrader]         = None
        self.reward_computer: Optional[AlethRewardComputer] = None
        self.session_id:      str                           = str(uuid.uuid4())[:8]
        self.task_difficulty: str                           = "easy"
        self.chronicle:       ChronicleService              = ChronicleService()

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, task: str = "easy") -> Observation:
        """Load task JSON, initialise state/grader/reward, return first Observation."""
        task_file = self.data_dir / f"task_{task}.json"
        if not task_file.exists():
            available = [f.stem.replace("task_", "") for f in self.data_dir.glob("task_*.json")]
            raise ValueError(f"Task '{task}' not found. Available: {available}")

        with open(task_file, "r", encoding="utf-8") as f:
            td = json.load(f)

        claims       = {c["id"]: Claim(**c) for c in td["claims"]}
        papers       = {pid: Paper(**pd) for pid, pd in td["papers"].items()}
        ground_truth = {g["claim_id"]: GroundTruth(**g) for g in td["ground_truth"]}

        self.task_difficulty = task
        self.session_id = str(uuid.uuid4())[:8]

        self._state = State(
            task_id=task, claims=claims, papers=papers,
            ground_truth=ground_truth, verifications={}, papers_read=[],
            step_count=0, max_steps=td.get("max_steps"), episode_done=False,
        )
        self.grader          = AlethGrader(ground_truth)
        self.reward_computer = AlethRewardComputer(ground_truth, papers)

        summary = "\n".join(
            f"  [{cid}] {c.text}  (cites: {', '.join(c.citations)})"
            for cid, c in claims.items()
        )
        return self._obs(
            f"Task '{task}' initialised: {len(claims)} claims.\n{summary}\n\n"
            f"Papers available: {', '.join(papers)}\n\n"
            "Read cited papers, then verify each claim. Submit when done.",
            [], None,
        )

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict]:
        """Execute one action, return (Observation, Reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state.episode_done:
            raise RuntimeError("Episode done — call reset() to start a new episode.")

        parsed     = self._parse_action(action)
        prev_state = self._state.model_copy(deep=True)
        obs, info  = self._execute(parsed)

        self._state.step_count += 1
        reward = self.reward_computer.compute_reward(parsed, prev_state, self._state)
        done   = self._is_done(parsed)
        self._state.episode_done = done

        if done:
            final_score = self.grader.grade_episode(self._state)
            breakdown = self.grader.get_detailed_breakdown(self._state)
            info["final_score"]       = final_score
            info["grading_breakdown"] = breakdown
            info["steps_taken"]       = self._state.step_count

            # Compute aggregate scores for components
            accuracy_scores = []
            reasoning_scores = []
            drift_scores = []
            for claim_breakdown in breakdown.values():
                if "accuracy_component" in claim_breakdown:
                    accuracy_scores.append(claim_breakdown["accuracy_component"])
                    reasoning_scores.append(claim_breakdown["reasoning_component"])
                    drift_scores.append(claim_breakdown["drift_component"])

            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else DEFAULT_COMPONENT_SCORE
            avg_reasoning = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else DEFAULT_COMPONENT_SCORE
            avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else DEFAULT_COMPONENT_SCORE

            # Save session to chronicle
            try:
                self.chronicle.create_session_record(
                    session_id=self.session_id,
                    task_difficulty=self.task_difficulty,
                    state=self._state,
                    ground_truth=self._state.ground_truth,
                    final_score=final_score,
                    accuracy_score=avg_accuracy,
                    reasoning_score=avg_reasoning,
                    drift_score=avg_drift,
                )
            except Exception as e:
                # Log but don't fail the episode if chronicle save fails
                import logging
                logging.warning(f"Failed to save session to chronicle: {e}")

        return obs, reward, done, info

    def state(self) -> State:
        """Return deep copy of current internal state."""
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state.model_copy(deep=True)

    # ── Action dispatch ───────────────────────────────────────────────────────

    def _parse_action(self, action: Dict) -> Action:
        t = action.get("action_type")
        if t == "read_paper":    return ReadPaperAction(**action)
        if t == "verify_claim":  return VerifyClaimAction(**action)
        if t == "flag_drift":    return FlagDriftAction(**action)
        if t == "submit":        return SubmitAction(**action)
        raise ValueError(f"Unknown action_type '{t}'. "
                         "Must be: read_paper | verify_claim | flag_drift | submit")

    def _execute(self, action: Action) -> Tuple[Observation, Dict]:
        info: Dict[str, Any] = {"action_type": action.action_type}
        if isinstance(action, ReadPaperAction):    return self._read(action, info)
        if isinstance(action, VerifyClaimAction):  return self._verify(action, info)
        if isinstance(action, FlagDriftAction):    return self._flag(action, info)
        if isinstance(action, SubmitAction):       return self._submit(action, info)
        raise ValueError(f"Unhandled action type: {type(action)}")

    def _read(self, action: ReadPaperAction, info: Dict) -> Tuple[Observation, Dict]:
        pid = action.paper_id
        if pid not in self._state.papers:
            return self._obs(f"Error: paper '{pid}' not found. "
                             f"Available: {list(self._state.papers)}", [], None), info

        paper = self._state.papers[pid]
        if pid not in self._state.papers_read:
            self._state.papers_read.append(pid)

        if paper.access_level == AccessLevel.FULL_TEXT:
            text, sections = paper.full_text or paper.abstract, \
                             ["abstract", "introduction", "methods", "results", "conclusion"]
        elif paper.access_level == AccessLevel.ABSTRACT_ONLY:
            text, sections = paper.abstract, ["abstract"]
        else:
            text     = f"Title: {paper.title} ({paper.year})\n" \
                       f"Authors: {', '.join(paper.authors)}\n[Access Restricted]"
            sections = []

        info.update(paper_read=pid, access_level=paper.access_level.value)
        content = PaperContent(paper_id=pid, content_type=paper.access_level,
                               text=text, sections_available=sections)
        return self._obs(f"Read '{paper.title}' ({paper.year}) — {paper.access_level.value}.",
                         [content], None), info

    def _verify(self, action: VerifyClaimAction, info: Dict) -> Tuple[Observation, Dict]:
        cid = action.claim_id
        if cid not in self._state.claims:
            return self._obs(f"Error: claim '{cid}' not found. "
                             f"Available: {list(self._state.claims)}", [], None), info

        score = max(0.001, min(0.999, action.support_score))
        self._state.verifications[cid] = Verification(
            claim_id=cid, support_score=score, reasoning=action.reasoning,
            primary_evidence_paper=action.primary_evidence_paper,
            timestamp=self._state.step_count,
        )

        fb = None
        if cid in self._state.ground_truth:
            gt    = self._state.ground_truth[cid]
            error = abs(score - gt.true_support_score)
            # Clamp partial_score strictly to (0.001, 0.999) — boundary values
            # (0.0 and 1.0) are rejected by the platform validator
            ps    = round(max(0.001, min(0.999, 1.0 - error)), 4)
            hint  = ("Score quite far from evidence." if error > 0.4
                     else "Score moderately off." if error > 0.2 else None)
            fb = VerificationFeedback(claim_id=cid, partial_score=ps, hints=hint)

        info.update(claim_verified=cid, support_score=score)
        v, t = len(self._state.verifications), len(self._state.claims)
        return self._obs(f"Verified '{cid}' (score={score:.2f}). {v}/{t} done.",
                         [], fb), info

    def _flag(self, action: FlagDriftAction, info: Dict) -> Tuple[Observation, Dict]:
        cid = action.claim_id
        if cid in self._state.verifications:
            self._state.verifications[cid].flagged_drift     = True
            self._state.verifications[cid].drift_explanation = action.drift_explanation
        else:
            self._state.verifications[cid] = Verification(
                claim_id=cid, support_score=0.0,
                reasoning="(drift flagged before verification)",
                flagged_drift=True, drift_explanation=action.drift_explanation,
                timestamp=self._state.step_count,
            )
        info.update(drift_flagged=cid, citation_chain=action.citation_chain)
        return self._obs(
            f"Drift flagged for '{cid}'. Chain: {' -> '.join(action.citation_chain)}",
            [], None), info

    def _submit(self, action: SubmitAction, info: Dict) -> Tuple[Observation, Dict]:
        v, t   = len(self._state.verifications), len(self._state.claims)
        missed = [cid for cid in self._state.claims if cid not in self._state.verifications]
        info.update(submitted=True, verified_count=v, total_count=t, unverified_claims=missed)
        return self._obs(
            f"Submitted! {v}/{t} verified. "
            + (f"Unverified: {missed}" if missed else "All claims covered."),
            [], None), info

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _obs(self, message: str, papers_content: list, feedback) -> Observation:
        assert self._state is not None
        return Observation(
            message=message, papers_content=papers_content, feedback=feedback,
            claims_verified=len(self._state.verifications),
            claims_total=len(self._state.claims),
            step_count=self._state.step_count,
            papers_read=list(self._state.papers_read),
        )

    def _is_done(self, action: Action) -> bool:
        if isinstance(action, SubmitAction):
            return True
        if self._state.max_steps and self._state.step_count >= self._state.max_steps:
            return True
        if len(self._state.verifications) >= len(self._state.claims):
            return True
        return False

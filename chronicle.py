"""
Aleth — Chronicle: Session History & Tips Recommendation Engine

This module provides:
  - SessionHistoryManager: Persists and retrieves session records
  - UsageAnalyzer: Detects patterns in usage (accuracy trends, efficiency, etc.)
  - TipsGenerator: Generates personalized recommendations based on patterns
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from statistics import mean, stdev

from models import (
    SessionRecord, UserProfile, TipRecommendation, TipPriority, ChronicleResponse,
    State, GroundTruth, Verification,
)


class SessionHistoryManager:
    """Manages persistence and retrieval of session records."""

    def __init__(self, history_dir: Optional[str] = None):
        if history_dir is None:
            history_dir = Path(__file__).parent / ".chronicle"
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.sessions_file = self.history_dir / "sessions.jsonl"

    def save_session(self, record: SessionRecord) -> None:
        """Append a session record to the history file."""
        with open(self.sessions_file, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def load_sessions(self, limit: Optional[int] = None) -> List[SessionRecord]:
        """Load all sessions from history (optionally limit to most recent)."""
        if not self.sessions_file.exists():
            return []

        sessions = []
        with open(self.sessions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        sessions.append(SessionRecord(**data))
                    except Exception:
                        pass  # Skip malformed lines

        # Return most recent first
        sessions.reverse()
        if limit:
            sessions = sessions[:limit]
        return sessions

    def get_session_by_id(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a specific session by ID."""
        sessions = self.load_sessions()
        for s in sessions:
            if s.session_id == session_id:
                return s
        return None


class UsageAnalyzer:
    """Analyzes usage patterns to identify strengths, weaknesses, and trends."""

    def __init__(self, sessions: List[SessionRecord]):
        self.sessions = sessions

    def compute_profile(self) -> UserProfile:
        """Aggregate statistics from all sessions."""
        if not self.sessions:
            return self._empty_profile()

        total_sessions = len(self.sessions)
        total_verified = sum(s.claims_verified for s in self.sessions)
        total_claims = sum(s.claims_total for s in self.sessions)

        # Accuracy by claim type
        empirical_scores = []
        methodological_scores = []
        theoretical_scores = []

        for session in self.sessions:
            for claim_id, claim_type in session.claim_types.items():
                score = session.claim_scores.get(claim_id, 0.0)
                if claim_type == "empirical":
                    empirical_scores.append(score)
                elif claim_type == "methodological":
                    methodological_scores.append(score)
                elif claim_type == "theoretical":
                    theoretical_scores.append(score)

        empirical_acc = mean(empirical_scores) if empirical_scores else 0.5
        methodological_acc = mean(methodological_scores) if methodological_scores else 0.5
        theoretical_acc = mean(theoretical_scores) if theoretical_scores else 0.5

        # Difficulty breakdown
        easy_sessions = [s for s in self.sessions if s.task_difficulty == "easy"]
        medium_sessions = [s for s in self.sessions if s.task_difficulty == "medium"]
        hard_sessions = [s for s in self.sessions if s.task_difficulty == "hard"]

        easy_avg = mean([s.final_score for s in easy_sessions]) if easy_sessions else None
        medium_avg = mean([s.final_score for s in medium_sessions]) if medium_sessions else None
        hard_avg = mean([s.final_score for s in hard_sessions]) if hard_sessions else None

        # Best and weakest difficulty
        scores_by_diff = {}
        if easy_avg is not None:
            scores_by_diff["easy"] = easy_avg
        if medium_avg is not None:
            scores_by_diff["medium"] = medium_avg
        if hard_avg is not None:
            scores_by_diff["hard"] = hard_avg

        best_diff = max(scores_by_diff, key=scores_by_diff.get) if scores_by_diff else None
        worst_diff = min(scores_by_diff, key=scores_by_diff.get) if scores_by_diff else None

        completion_rate = (total_verified / total_claims) if total_claims > 0 else 0.0

        return UserProfile(
            total_sessions=total_sessions,
            avg_accuracy=mean([s.accuracy_score for s in self.sessions]),
            avg_reasoning=mean([s.reasoning_score for s in self.sessions]),
            avg_drift_detection=mean([s.drift_score for s in self.sessions]),
            avg_papers_per_session=mean(
                [s.papers_read_count for s in self.sessions]
            ) if self.sessions else 0.0,
            avg_steps_per_session=mean(
                [s.total_steps for s in self.sessions]
            ) if self.sessions else 0.0,
            best_difficulty=best_diff,
            weakest_difficulty=worst_diff,
            empirical_accuracy=empirical_acc,
            methodological_accuracy=methodological_acc,
            theoretical_accuracy=theoretical_acc,
            total_claims_verified=total_verified,
            task_completion_rate=completion_rate,
        )

    def _empty_profile(self) -> UserProfile:
        """Return a default profile for zero sessions."""
        return UserProfile(
            total_sessions=0,
            avg_accuracy=0.5,
            avg_reasoning=0.5,
            avg_drift_detection=0.5,
            avg_papers_per_session=0.0,
            avg_steps_per_session=0.0,
            best_difficulty=None,
            weakest_difficulty=None,
            empirical_accuracy=0.5,
            methodological_accuracy=0.5,
            theoretical_accuracy=0.5,
            total_claims_verified=0,
            task_completion_rate=0.0,
        )

    def identify_strengths(self, profile: UserProfile) -> List[str]:
        """Identify user strengths from profile."""
        strengths = []

        if profile.avg_accuracy > 0.7:
            strengths.append("Strong accuracy in claim verification")

        if profile.avg_reasoning > 0.7:
            strengths.append("Excellent reasoning quality")

        if profile.avg_drift_detection > 0.7:
            strengths.append("Strong citation drift detection")

        if profile.task_completion_rate > 0.9:
            strengths.append("Consistently complete most claims")

        if profile.empirical_accuracy > profile.methodological_accuracy:
            strengths.append("Better at empirical claim verification")
        elif profile.methodological_accuracy > profile.empirical_accuracy:
            strengths.append("Strong methodological understanding")

        if profile.avg_papers_per_session < 3.0:
            strengths.append("Efficient paper reading strategy")

        if not strengths:
            strengths = ["Growing experience with citation verification"]

        return strengths

    def identify_weaknesses(self, profile: UserProfile) -> List[str]:
        """Identify user weaknesses from profile."""
        weaknesses = []

        if profile.avg_accuracy < 0.5:
            weaknesses.append("Low accuracy in claim verification")

        if profile.avg_reasoning < 0.5:
            weaknesses.append("Reasoning explanations lack key concepts")

        if profile.avg_drift_detection < 0.5:
            weaknesses.append("Difficulty detecting citation drift")

        if profile.task_completion_rate < 0.8:
            weaknesses.append("Often leave claims unverified")

        if profile.empirical_accuracy < 0.5:
            weaknesses.append("Struggle with empirical claim verification")

        if profile.methodological_accuracy < 0.5:
            weaknesses.append("Difficulty understanding methodological concepts")

        if profile.theoretical_accuracy < 0.5:
            weaknesses.append("Weak on theoretical claims")

        if profile.avg_papers_per_session > 5.0:
            weaknesses.append("Reading too many papers per claim")

        if not weaknesses:
            weaknesses = ["No major patterns identified"]

        return weaknesses


class TipsGenerator:
    """Generates personalized tips based on usage patterns and profile."""

    def __init__(self, profile: UserProfile, sessions: List[SessionRecord]):
        self.profile = profile
        self.sessions = sessions

    def generate_tips(self) -> List[TipRecommendation]:
        """Generate up to 5 prioritized tips."""
        tips = []

        # Tip 1: Accuracy improvement
        if self.profile.avg_accuracy < 0.6:
            tips.append(TipRecommendation(
                tip_id="accuracy_001",
                title="Improve Score Calibration",
                description="Your support scores frequently differ from ground truth. "
                           "Try reading multiple perspectives before deciding on a score.",
                priority=TipPriority.HIGH,
                category="accuracy",
                evidence=f"Avg accuracy: {self.profile.avg_accuracy:.2%}",
                suggested_action="For next session: use 0.0-0.25 for 'clearly unsupported', "
                               "0.25-0.5 for 'partially contradicts', 0.5-0.75 for 'somewhat supports', "
                               "0.75-1.0 for 'strongly supports'",
            ))

        # Tip 2: Reasoning quality
        if self.profile.avg_reasoning < 0.6:
            tips.append(TipRecommendation(
                tip_id="reasoning_001",
                title="Strengthen Your Reasoning Explanations",
                description="Your reasoning lacks key concepts from the papers. "
                           "Include specific terms, measurements, and findings in your justification.",
                priority=TipPriority.HIGH,
                category="reasoning",
                evidence=f"Avg reasoning score: {self.profile.avg_reasoning:.2%}",
                suggested_action="When verifying: explicitly mention paper results, methodology, "
                               "and dates to demonstrate thorough reading",
            ))

        # Tip 3: Drift detection
        if self.profile.avg_drift_detection < 0.6 and self.profile.total_sessions > 2:
            tips.append(TipRecommendation(
                tip_id="drift_001",
                title="Focus on Citation Drift Detection",
                description="You're missing subtle manipulations in citations. "
                           "Pay attention to correlation vs. causation, and whether claims "
                           "extend beyond what papers actually show.",
                priority=TipPriority.MEDIUM,
                category="drift_detection",
                evidence=f"Drift detection score: {self.profile.avg_drift_detection:.2%}",
                suggested_action="Re-review hard-difficulty tasks focusing on how papers "
                               "are being misrepresented in claims",
            ))

        # Tip 4: Efficiency
        if self.profile.avg_papers_per_session > 5.0:
            tips.append(TipRecommendation(
                tip_id="efficiency_001",
                title="Optimize Your Paper Reading Strategy",
                description=f"You're reading {self.profile.avg_papers_per_session:.1f} papers per session. "
                           "Try identifying key papers first before diving in.",
                priority=TipPriority.MEDIUM,
                category="efficiency",
                evidence=f"Avg papers/session: {self.profile.avg_papers_per_session:.1f}",
                suggested_action="For next session: identify cited papers, read abstracts first, "
                               "then only read full text for directly relevant papers",
            ))

        # Tip 5: Difficulty progression
        if self.profile.best_difficulty == "easy" and self.profile.total_sessions >= 3:
            tips.append(TipRecommendation(
                tip_id="progression_001",
                title="Ready to Try Medium Difficulty",
                description="You've mastered easy tasks consistently. "
                           "Medium-difficulty challenges will help you develop deeper reasoning skills.",
                priority=TipPriority.MEDIUM,
                category="difficulty_progression",
                evidence=f"Avg easy score: {mean([s.final_score for s in self.sessions if s.task_difficulty == 'easy']):.2%}",
                suggested_action="Next session: reset with task='medium' to encounter multi-paper synthesis",
            ))

        # Tip 6: Claim-type specific
        if self.profile.empirical_accuracy < self.profile.theoretical_accuracy:
            tips.append(TipRecommendation(
                tip_id="claimtype_001",
                title="Strengthen Empirical Claim Verification",
                description="Your empirical accuracy is weaker than theoretical. "
                           "Empirical claims require exact data matching—pay close attention to numbers.",
                priority=TipPriority.MEDIUM,
                category="accuracy",
                evidence=f"Empirical: {self.profile.empirical_accuracy:.2%}, Theoretical: {self.profile.theoretical_accuracy:.2%}",
                suggested_action="In next session: write down specific metrics from papers "
                               "before comparing to claims",
            ))

        # Tip 7: Completion rate
        if self.profile.task_completion_rate < 0.8:
            tips.append(TipRecommendation(
                tip_id="completion_001",
                title="Verify All Claims Before Submitting",
                description=f"You only verify {self.profile.task_completion_rate:.1%} of claims. "
                           "Unverified claims receive minimum scores.",
                priority=TipPriority.HIGH,
                category="efficiency",
                evidence=f"Completion rate: {self.profile.task_completion_rate:.1%}",
                suggested_action="Next session: plan to verify every claim, even if unsure",
            ))

        # Return top 5 by priority
        priority_order = {TipPriority.HIGH: 0, TipPriority.MEDIUM: 1, TipPriority.LOW: 2}
        tips.sort(key=lambda t: priority_order[t.priority])
        return tips[:5]

    def suggest_next_target(self) -> Optional[str]:
        """Suggest the next difficulty or focus area."""
        if not self.profile.best_difficulty:
            return "Try easy tasks to build foundational skills"

        if self.profile.best_difficulty == "easy" and self.profile.total_sessions >= 3:
            return "Progress to medium difficulty for multi-paper synthesis"

        if self.profile.best_difficulty == "medium" and self.profile.total_sessions >= 5:
            return "Challenge yourself with hard tasks involving citation drift"

        if self.profile.avg_drift_detection < 0.5:
            return "Focus on hard tasks to improve drift detection"

        if self.profile.avg_reasoning < 0.6:
            return "Spend more time on reasoning-focused verification"

        return "Continue current difficulty level to improve consistency"


class ChronicleService:
    """High-level service combining all chronicle components."""

    def __init__(self, history_dir: Optional[str] = None):
        self.manager = SessionHistoryManager(history_dir)

    def create_session_record(
        self,
        session_id: str,
        task_difficulty: str,
        state: State,
        ground_truth: Dict[str, GroundTruth],
        final_score: float,
        accuracy_score: float,
        reasoning_score: float,
        drift_score: float,
    ) -> SessionRecord:
        """Create and save a session record after episode completion."""
        # Compute per-claim scores
        claim_scores = {}
        reasonings = {}
        claim_types = {}

        for claim_id, verification in state.verifications.items():
            claim = state.claims.get(claim_id)
            gt = ground_truth.get(claim_id)

            if claim and gt:
                claim_scores[claim_id] = self._compute_claim_score(verification, gt)
                reasonings[claim_id] = verification.reasoning
                claim_types[claim_id] = claim.claim_type.value

        record = SessionRecord(
            session_id=session_id,
            task_difficulty=task_difficulty,
            timestamp=datetime.now().isoformat(),
            total_steps=state.step_count,
            papers_read_count=len(state.papers_read),
            claims_verified=len(state.verifications),
            claims_total=len(state.claims),
            final_score=final_score,
            accuracy_score=accuracy_score,
            reasoning_score=reasoning_score,
            drift_score=drift_score,
            papers_read=state.papers_read,
            claim_scores=claim_scores,
            reasonings=reasonings,
            claim_types=claim_types,
        )
        self.manager.save_session(record)
        return record

    def _compute_claim_score(self, verification: Verification, gt: GroundTruth) -> float:
        """Compute score for a single claim using the same formula as grader."""
        raw_accuracy = max(0.0, 1.0 - abs(verification.support_score - gt.true_support_score))
        accuracy = max(0.05, min(0.95, raw_accuracy))

        reasoning_score = self._key_concepts_match(verification.reasoning, gt.key_concepts)
        reasoning = max(0.05, min(0.95, reasoning_score))

        drift = 0.95 if verification.flagged_drift == gt.has_citation_drift else 0.05

        raw = 0.6 * accuracy + 0.3 * reasoning + 0.1 * drift
        return max(0.001, min(0.999, round(raw, 6)))

    def _key_concepts_match(self, reasoning: str, concepts: List[str]) -> float:
        """Check how many key concepts appear in the reasoning."""
        if not concepts:
            return 0.5

        reasoning_lower = reasoning.lower()
        matches = sum(
            1 for concept in concepts
            if concept.lower() in reasoning_lower
        )
        return matches / len(concepts)

    def get_chronicle_response(self) -> ChronicleResponse:
        """Generate full chronicle response (tips + stats)."""
        sessions = self.manager.load_sessions(limit=50)

        analyzer = UsageAnalyzer(sessions)
        profile = analyzer.compute_profile()
        strengths = analyzer.identify_strengths(profile)
        weaknesses = analyzer.identify_weaknesses(profile)

        generator = TipsGenerator(profile, sessions)
        tips = generator.generate_tips()
        next_target = generator.suggest_next_target()

        return ChronicleResponse(
            profile=profile,
            recent_sessions=sessions[:5],  # Return most recent 5
            strengths=strengths,
            weaknesses=weaknesses,
            tips=tips,
            next_target=next_target,
        )

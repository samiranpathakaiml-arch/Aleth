"""
Test Chronicle session tracking and personalized tips generation.
"""

import json
import tempfile
from pathlib import Path
from chronicle import Chronicle, ChronicleDB, PatternAnalyzer, TipsGenerator


def test_chronicle_session_creation():
    """Test creating and managing sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chronicle.db"
        chronicle = Chronicle(db_path=str(db_path))
        
        # Start a session
        session_id = chronicle.start_session("easy")
        assert session_id is not None
        assert len(session_id) == 36  # UUID length
        
        # Record actions
        chronicle.record_action("read_paper", {"paper_id": "paper1"}, reward=0.02)
        chronicle.record_action("verify_claim", {"claim_id": "claim1", "score": 0.8}, reward=0.5)
        
        # End session
        chronicle.end_session(final_score=0.75, steps_taken=2)
        
        # Retrieve session
        history = chronicle.get_history(limit=10)
        assert len(history["sessions"]) == 1
        assert history["sessions"][0]["task_type"] == "easy"
        assert history["sessions"][0]["final_score"] == 0.75
        
        print("✓ Session creation and tracking works")


def test_pattern_analysis():
    """Test pattern analysis functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chronicle.db"
        chronicle = Chronicle(db_path=str(db_path))
        
        # Create multiple sessions with different patterns
        for i in range(5):
            session_id = chronicle.start_session("easy")
            for j in range(3):
                chronicle.record_action("read_paper", {"paper_id": f"paper{j}"}, reward=0.02)
            chronicle.record_action("verify_claim", {"claim_id": "claim1"}, reward=0.5)
            chronicle.end_session(final_score=0.7 + i * 0.05, steps_taken=4)
        
        # Create some harder tasks
        for i in range(3):
            session_id = chronicle.start_session("hard")
            for j in range(5):
                chronicle.record_action("read_paper", {"paper_id": f"paper{j}"}, reward=0.02)
            chronicle.record_action("verify_claim", {"claim_id": "claim1"}, reward=0.3)
            chronicle.end_session(final_score=0.4 + i * 0.1, steps_taken=6)
        
        # Get tips
        tips_data = chronicle.get_tips(limit=10)
        
        assert "tips" in tips_data
        assert "metrics" in tips_data
        assert len(tips_data["tips"]) > 0
        
        # Check that we got performance metrics
        assert "task_performance" in tips_data["metrics"]
        assert "efficiency" in tips_data["metrics"]
        
        print("✓ Pattern analysis works")
        print(f"  Generated {len(tips_data['tips'])} tips")
        for tip in tips_data["tips"]:
            print(f"  - [{tip['priority']}] {tip['category']}: {tip['message'][:60]}...")


def test_efficiency_metrics():
    """Test efficiency metrics calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chronicle.db"
        db = ChronicleDB(db_path=str(db_path))
        analyzer = PatternAnalyzer(db)
        
        # Create sessions with different efficiencies
        for steps in [5, 8, 12, 15]:
            session_id = db.create_session("easy")
            for _ in range(steps):
                db.add_action(session_id, "read_paper", {}, reward=0.02)
            db.finalize_session(session_id, final_score=0.7, steps_taken=steps)
        
        efficiency = analyzer.get_efficiency_metrics(limit=10)
        
        assert "avg_steps" in efficiency
        assert "success_rate" in efficiency
        assert efficiency["avg_steps"] == 10.0  # (5+8+12+15)/4 = 10
        assert efficiency["success_rate"] == 1.0  # All scores >= 0.6
        
        print("✓ Efficiency metrics calculation works")


def test_tips_generation():
    """Test personalized tips generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chronicle.db"
        chronicle = Chronicle(db_path=str(db_path))
        
        # Create sessions with poor performance on hard tasks
        for _ in range(3):
            session_id = chronicle.start_session("hard")
            for _ in range(10):
                chronicle.record_action("read_paper", {}, reward=0.02)
            chronicle.record_action("verify_claim", {}, reward=0.2)
            chronicle.end_session(final_score=0.3, steps_taken=11)  # Low score, many steps
        
        # Create good performance on easy tasks
        for _ in range(2):
            session_id = chronicle.start_session("easy")
            for _ in range(3):
                chronicle.record_action("read_paper", {}, reward=0.02)
            chronicle.record_action("verify_claim", {}, reward=0.5)
            chronicle.end_session(final_score=0.8, steps_taken=4)
        
        tips = chronicle.get_tips(limit=10)
        
        # Should have tips about weak areas and efficiency
        assert len(tips["tips"]) > 0
        
        # Check for high-priority tips
        high_priority_tips = [t for t in tips["tips"] if t.get("priority") == "high"]
        assert len(high_priority_tips) > 0, "Should have at least one high-priority tip"
        
        print("✓ Tips generation works")
        print(f"  Total tips: {len(tips['tips'])}")
        print(f"  High-priority tips: {len(high_priority_tips)}")


if __name__ == "__main__":
    test_chronicle_session_creation()
    test_pattern_analysis()
    test_efficiency_metrics()
    test_tips_generation()
    print("\n✅ All Chronicle tests passed!")

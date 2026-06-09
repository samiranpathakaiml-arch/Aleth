"""
Aleth Chronicle — Session History Tracking & Personalized Tips
Tracks session data, analyzes usage patterns, and generates personalized recommendations.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid


@dataclass
class SessionAction:
    """Represents a single action taken during a session."""
    action_id: str
    action_type: str  # read_paper, verify_claim, flag_drift, submit
    timestamp: str
    details: Dict[str, Any]
    reward: Optional[float] = None


@dataclass
class SessionRecord:
    """Represents a single session."""
    session_id: str
    task_type: str  # easy, medium, hard
    start_time: str
    end_time: Optional[str] = None
    final_score: Optional[float] = None
    steps_taken: int = 0
    actions: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class ChronicleDB:
    """SQLite database for session tracking."""
    
    def __init__(self, db_path: str = "chronicle.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                final_score REAL,
                steps_taken INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        
        # Actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_actions (
                action_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details TEXT NOT NULL,
                reward REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_task ON sessions(task_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_start ON sessions(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_session ON session_actions(session_id)")
        
        conn.commit()
        conn.close()
    
    def create_session(self, task_type: str) -> str:
        """Create a new session and return session_id."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, task_type, start_time, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, task_type, now, now))
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_action(self, session_id: str, action_type: str, 
                   details: Dict[str, Any], reward: Optional[float] = None) -> str:
        """Add an action to a session."""
        action_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        details_json = json.dumps(details)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO session_actions (action_id, session_id, action_type, timestamp, details, reward)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (action_id, session_id, action_type, now, details_json, reward))
        conn.commit()
        conn.close()
        
        return action_id
    
    def finalize_session(self, session_id: str, final_score: float, steps_taken: int):
        """Finalize a session with final score."""
        now = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET end_time = ?, final_score = ?, steps_taken = ?
            WHERE session_id = ?
        """, (now, final_score, steps_taken, session_id))
        conn.commit()
        conn.close()
    
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Retrieve a specific session with all its actions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session
        cursor.execute("SELECT session_id, task_type, start_time, end_time, final_score, steps_taken FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        session = SessionRecord(
            session_id=row[0],
            task_type=row[1],
            start_time=row[2],
            end_time=row[3],
            final_score=row[4],
            steps_taken=row[5],
            actions=[]
        )
        
        # Get actions
        cursor.execute("SELECT action_type, timestamp, details, reward FROM session_actions WHERE session_id = ? ORDER BY timestamp", (session_id,))
        for row in cursor.fetchall():
            session.actions.append({
                "action_type": row[0],
                "timestamp": row[1],
                "details": json.loads(row[2]),
                "reward": row[3]
            })
        
        conn.close()
        return session
    
    def get_recent_sessions(self, limit: int = 20, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent sessions, optionally filtered by task type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_type:
            cursor.execute("""
                SELECT session_id, task_type, start_time, end_time, final_score, steps_taken
                FROM sessions
                WHERE task_type = ?
                ORDER BY start_time DESC
                LIMIT ?
            """, (task_type, limit))
        else:
            cursor.execute("""
                SELECT session_id, task_type, start_time, end_time, final_score, steps_taken
                FROM sessions
                ORDER BY start_time DESC
                LIMIT ?
            """, (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "task_type": row[1],
                "start_time": row[2],
                "end_time": row[3],
                "final_score": row[4],
                "steps_taken": row[5]
            })
        
        conn.close()
        return sessions


class PatternAnalyzer:
    """Analyzes usage patterns from session history."""
    
    def __init__(self, db: ChronicleDB):
        self.db = db
    
    def get_task_performance(self, task_type: Optional[str] = None, 
                             days: int = 30) -> Dict[str, Any]:
        """Analyze performance by task type over time."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        if task_type:
            cursor.execute("""
                SELECT task_type, COUNT(*) as total, AVG(final_score) as avg_score, 
                       AVG(steps_taken) as avg_steps, MIN(final_score) as min_score,
                       MAX(final_score) as max_score
                FROM sessions
                WHERE task_type = ? AND start_time > ?
                GROUP BY task_type
            """, (task_type, cutoff))
        else:
            cursor.execute("""
                SELECT task_type, COUNT(*) as total, AVG(final_score) as avg_score,
                       AVG(steps_taken) as avg_steps, MIN(final_score) as min_score,
                       MAX(final_score) as max_score
                FROM sessions
                WHERE start_time > ?
                GROUP BY task_type
            """, (cutoff,))
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = {
                "total_sessions": row[1],
                "avg_score": round(row[2], 3) if row[2] else None,
                "avg_steps": round(row[3], 1) if row[3] else None,
                "min_score": round(row[4], 3) if row[4] else None,
                "max_score": round(row[5], 3) if row[5] else None
            }
        
        conn.close()
        return results
    
    def get_action_patterns(self, limit: int = 20) -> Dict[str, Any]:
        """Analyze most common action sequences."""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Get recent sessions
        cursor.execute("""
            SELECT session_id, task_type FROM sessions
            ORDER BY start_time DESC LIMIT ?
        """, (limit,))
        
        action_counts = {}
        avg_rewards = {}
        success_rate = {}
        
        for session_id, task_type in cursor.fetchall():
            cursor.execute("""
                SELECT action_type, reward FROM session_actions
                WHERE session_id = ? ORDER BY timestamp
            """, (session_id,))
            
            actions = cursor.fetchall()
            for action_type, reward in actions:
                if action_type not in action_counts:
                    action_counts[action_type] = 0
                    avg_rewards[action_type] = 0
                
                action_counts[action_type] += 1
                if reward is not None:
                    avg_rewards[action_type] = (avg_rewards[action_type] * 
                                                 (action_counts[action_type] - 1) + reward) / action_counts[action_type]
        
        conn.close()
        
        return {
            "action_frequency": sorted(action_counts.items(), key=lambda x: x[1], reverse=True),
            "avg_rewards_by_action": {k: round(v, 3) for k, v in avg_rewards.items()}
        }
    
    def get_efficiency_metrics(self, task_type: Optional[str] = None, 
                               limit: int = 20) -> Dict[str, Any]:
        """Calculate efficiency metrics (steps per task, success rate, etc)."""
        sessions = self.db.get_recent_sessions(limit=limit, task_type=task_type)
        
        if not sessions:
            return {}
        
        scores = [s["final_score"] for s in sessions if s["final_score"] is not None]
        steps = [s["steps_taken"] for s in sessions if s["steps_taken"] > 0]
        
        return {
            "avg_steps": round(sum(steps) / len(steps), 1) if steps else None,
            "min_steps": min(steps) if steps else None,
            "max_steps": max(steps) if steps else None,
            "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
            "success_rate": round(len([s for s in scores if s > 0.6]) / len(scores), 2) if scores else 0,
            "sessions_analyzed": len(sessions)
        }


class TipsGenerator:
    """Generates personalized tips based on usage patterns."""
    
    def __init__(self, analyzer: PatternAnalyzer):
        self.analyzer = analyzer
    
    def generate_tips(self, limit: int = 20, detail_level: str = "summary") -> Dict[str, Any]:
        """Generate personalized tips based on recent usage patterns."""
        tips = []
        metrics = {}
        
        # Task performance analysis
        perf = self.analyzer.get_task_performance(days=30)
        metrics["task_performance"] = perf
        
        # Identify weak areas
        for task_type, stats in perf.items():
            if stats["avg_score"] and stats["avg_score"] < 0.5:
                tips.append({
                    "priority": "high",
                    "category": "weak_area",
                    "task_type": task_type,
                    "message": f"Your {task_type} task score ({stats['avg_score']:.2f}) is below 0.5. "
                              f"Focus on improving your verification accuracy and reasoning quality.",
                    "actionable": f"Try re-reading papers more carefully before verifying claims in {task_type} tasks."
                })
        
        # Action pattern analysis
        action_patterns = self.analyzer.get_action_patterns(limit=limit)
        metrics["action_patterns"] = action_patterns
        
        # Efficiency analysis
        efficiency = self.analyzer.get_efficiency_metrics(limit=limit)
        metrics["efficiency"] = efficiency
        
        # Generate efficiency tips
        if efficiency.get("avg_steps") and efficiency["avg_steps"] > 15:
            tips.append({
                "priority": "medium",
                "category": "efficiency",
                "message": f"You're averaging {efficiency['avg_steps']:.1f} steps per task. "
                         f"Consider a more strategic approach to reduce unnecessary paper reads.",
                "actionable": "Plan which papers to read before starting, based on the claim citations."
            })
        
        # Success rate tips
        if efficiency.get("success_rate") and efficiency["success_rate"] < 0.5:
            tips.append({
                "priority": "high",
                "category": "success_rate",
                "message": f"Your success rate is only {efficiency['success_rate']:.0%}. "
                         f"You need to improve your verification accuracy.",
                "actionable": "Slow down verification steps - ensure your reasoning matches the paper content."
            })
        elif efficiency.get("success_rate") and efficiency["success_rate"] >= 0.7:
            tips.append({
                "priority": "low",
                "category": "encouragement",
                "message": f"Great work! You have a {efficiency['success_rate']:.0%} success rate. Keep it up!",
                "actionable": None
            })
        
        # Action-based tips
        if action_patterns.get("action_frequency"):
            read_actions = sum([count for action, count in action_patterns["action_frequency"] 
                              if action == "read_paper"])
            verify_actions = sum([count for action, count in action_patterns["action_frequency"] 
                                if action == "verify_claim"])
            
            if verify_actions > 0:
                read_verify_ratio = read_actions / verify_actions if verify_actions > 0 else 0
                if read_verify_ratio < 1.0:
                    tips.append({
                        "priority": "medium",
                        "category": "read_strategy",
                        "message": f"You read papers only {read_verify_ratio:.1f}x per verification. "
                                 f"Consider reading more papers for complex claims.",
                        "actionable": "For hard tasks, try reading 2-3 papers per claim verification."
                    })
        
        result = {
            "tips": sorted(tips, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "session_count": limit
        }
        
        return result


class Chronicle:
    """Main interface for session tracking and personalized tips."""
    
    def __init__(self, db_path: str = "chronicle.db"):
        self.db = ChronicleDB(db_path)
        self.analyzer = PatternAnalyzer(self.db)
        self.tips_generator = TipsGenerator(self.analyzer)
        self._current_session: Optional[str] = None
    
    def start_session(self, task_type: str) -> str:
        """Start a new session."""
        self._current_session = self.db.create_session(task_type)
        return self._current_session
    
    def record_action(self, action_type: str, details: Dict[str, Any], 
                      reward: Optional[float] = None):
        """Record an action in the current session."""
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.db.add_action(self._current_session, action_type, details, reward)
    
    def end_session(self, final_score: float, steps_taken: int):
        """End the current session."""
        if self._current_session is None:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.db.finalize_session(self._current_session, final_score, steps_taken)
        self._current_session = None
    
    def get_tips(self, limit: int = 20, detail_level: str = "summary") -> Dict[str, Any]:
        """Get personalized tips based on usage history."""
        return self.tips_generator.generate_tips(limit=limit, detail_level=detail_level)
    
    def get_history(self, limit: int = 20, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get recent session history."""
        sessions = self.db.get_recent_sessions(limit=limit, task_type=task_type)
        return {
            "sessions": sessions,
            "total": len(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_session_detail(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific session."""
        session = self.db.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "task_type": session.task_type,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "final_score": session.final_score,
            "steps_taken": session.steps_taken,
            "actions": session.actions
        }

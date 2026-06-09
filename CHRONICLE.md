# Chronicle: Session History & Personalized Tips

## Overview

Chronicle is a session tracking and personalized recommendations system for Aleth. It records all actions taken during verification sessions and analyzes usage patterns to provide personalized tips for improvement.

## Features

### 1. **Session Tracking**
- Automatically tracks each verification session (started on `/reset`, ended on `submit`)
- Records all actions with timestamps and rewards
- Persists data in SQLite database for historical analysis

### 2. **Usage Pattern Analysis**
- Analyzes task performance across difficulty levels
- Tracks action patterns and their effectiveness
- Calculates efficiency metrics (steps per task, success rate, etc.)

### 3. **Personalized Tips**
- Generates AI-driven recommendations based on usage patterns
- Identifies weak areas and provides actionable improvements
- Prioritizes tips by relevance (high, medium, low)

## API Endpoints

### GET `/chronicle/tips`

Get personalized tips based on your session history and usage patterns.

**Query Parameters:**
- `limit` (optional, default: 20, max: 100): Number of recent sessions to analyze
- `detail_level` (optional, default: "summary"): Level of detail for recommendations

**Response:**
```json
{
  "tips": [
    {
      "priority": "high|medium|low",
      "category": "weak_area|efficiency|success_rate|read_strategy|encouragement",
      "message": "Human-readable recommendation",
      "actionable": "Specific action to take (if applicable)",
      "task_type": "easy|medium|hard (if applicable)"
    }
  ],
  "metrics": {
    "task_performance": {
      "easy": {
        "total_sessions": 10,
        "avg_score": 0.75,
        "avg_steps": 4.2,
        "min_score": 0.5,
        "max_score": 0.9
      }
    },
    "action_patterns": {
      "action_frequency": [["read_paper", 42], ["verify_claim", 10]],
      "avg_rewards_by_action": {"read_paper": 0.02, "verify_claim": 0.45}
    },
    "efficiency": {
      "avg_steps": 4.5,
      "min_steps": 2,
      "max_steps": 8,
      "avg_score": 0.72,
      "success_rate": 0.8,
      "sessions_analyzed": 10
    }
  },
  "timestamp": "2024-01-15T10:30:00.000000",
  "session_count": 20
}
```

### GET `/chronicle/history`

Get your recent session history.

**Query Parameters:**
- `limit` (optional, default: 20, max: 100): Number of sessions to return
- `task_type` (optional): Filter by task type ("easy", "medium", or "hard")

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "task_type": "easy",
      "start_time": "2024-01-15T10:00:00",
      "end_time": "2024-01-15T10:05:00",
      "final_score": 0.75,
      "steps_taken": 4
    }
  ],
  "total": 10,
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

### GET `/chronicle/session/{session_id}`

Get detailed information about a specific session, including all actions taken.

**Path Parameters:**
- `session_id`: The session ID to retrieve

**Response:**
```json
{
  "session_id": "uuid",
  "task_type": "easy",
  "start_time": "2024-01-15T10:00:00",
  "end_time": "2024-01-15T10:05:00",
  "final_score": 0.75,
  "steps_taken": 4,
  "actions": [
    {
      "action_type": "read_paper",
      "timestamp": "2024-01-15T10:00:10",
      "details": {"paper_id": "paper_1"},
      "reward": 0.02
    },
    {
      "action_type": "verify_claim",
      "timestamp": "2024-01-15T10:01:00",
      "details": {"claim_id": "claim_1", "support_score": 0.8},
      "reward": 0.5
    }
  ]
}
```

## Tips Categories

### 1. **Weak Area Tips**
Identifies task types where your performance is below 0.5 (50%).
- **Priority:** High
- **Action:** Focus on improving accuracy and reasoning for that task type

### 2. **Efficiency Tips**
Alerts when you're using more steps than typical (averaging > 15 steps per task).
- **Priority:** Medium
- **Action:** Plan your approach before starting; use strategic reading patterns

### 3. **Success Rate Tips**
Tracks your overall success rate (sessions with score > 0.6).
- **Priority:** High if < 50% success, Low if >= 70%
- **Action:** Vary based on performance level

### 4. **Read Strategy Tips**
Analyzes read-to-verification ratio to suggest optimal paper reading patterns.
- **Priority:** Medium
- **Action:** Adjust number of papers read per claim verification

### 5. **Encouragement Tips**
Celebrates good performance and consistency.
- **Priority:** Low
- **Action:** Continue current approach

## How It Works

### 1. Session Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Session Lifecycle                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  POST /reset  →  Session Created                           │
│      ↓          (task_type recorded)                        │
│      │                                                       │
│  POST /step   →  Actions Recorded                           │
│      ↓          (action_type, details, reward)              │
│      │          (repeated multiple times)                   │
│      │                                                       │
│  Final /step  →  Session Finalized                          │
│      (done=true)  (final_score, steps_taken)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Data Storage

All session data is stored in `chronicle.db` (SQLite):
- **sessions table**: Session metadata (task_type, scores, timestamps)
- **session_actions table**: Individual actions with rewards
- Indexed for fast queries by session_id, task_type, and timestamp

### 3. Pattern Analysis

The system analyzes:
- **Task Performance**: Average scores and efficiency by task type
- **Action Patterns**: Most common action sequences and their rewards
- **Efficiency Metrics**: Steps per task, success rates, and trends

### 4. Tip Generation

Tips are generated by:
1. Calculating performance metrics
2. Identifying patterns and anomalies
3. Comparing against baseline expectations
4. Generating actionable recommendations
5. Prioritizing by relevance and impact

## Example Usage

### Get Personalized Tips
```bash
curl http://localhost:7860/chronicle/tips?limit=30
```

### View Session History
```bash
curl http://localhost:7860/chronicle/history?limit=20&task_type=medium
```

### View Specific Session Details
```bash
curl http://localhost:7860/chronicle/session/550e8400-e29b-41d4-a716-446655440000
```

## Database Location

By default, Chronicle stores data in `chronicle.db` in the project root directory. This file will be created automatically on first use.

To use a custom location, modify the Chronicle initialization in `main.py`:
```python
_chronicle = Chronicle(db_path="/path/to/custom/chronicle.db")
```

## Integration with OpenEnv

Chronicle integrates seamlessly with the OpenEnv protocol:
- Session tracking is transparent to the agent/inferencer
- No changes required to existing inference code
- Tips and history are available as separate endpoints

## Performance Considerations

- Chronicle uses efficient SQLite queries with proper indexing
- Queries are filtered by time range (default: 30 days) to maintain performance
- Recommendations are computed on-demand
- Minimal overhead on /reset and /step endpoints

## Future Enhancements

Potential future improvements:
- Comparative analytics (user vs. population benchmarks)
- Predictive tips (ML-based forecasting)
- Custom alert thresholds
- Export session data (CSV, JSON)
- Session replay/debugging tools
- Multi-user profiles and comparison

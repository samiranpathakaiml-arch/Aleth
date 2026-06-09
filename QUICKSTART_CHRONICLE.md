# Quick Start: /chronicle Tips

## Overview

Chronicle automatically tracks your Aleth verification sessions and provides personalized tips based on your usage patterns.

## Getting Started

No setup required! Chronicle starts tracking automatically when you run the Aleth server.

```bash
# Start the Aleth server (as normal)
uvicorn main:app --host 0.0.0.0 --port 7860
```

Your sessions will be automatically tracked in `chronicle.db`.

## Using Chronicle

### 1. Get Personalized Tips

```bash
# Basic usage - analyzes last 20 sessions
curl http://localhost:7860/chronicle/tips

# Get more detailed analysis
curl http://localhost:7860/chronicle/tips?limit=50

# Example response:
{
  "tips": [
    {
      "priority": "high",
      "category": "weak_area",
      "task_type": "hard",
      "message": "Your hard task score (0.35) is below 0.5. Focus on improving your verification accuracy and reasoning quality.",
      "actionable": "Try re-reading papers more carefully before verifying claims in hard tasks."
    },
    {
      "priority": "medium",
      "category": "efficiency",
      "message": "You're averaging 14.2 steps per task. Consider a more strategic approach..."
    }
  ],
  "metrics": { ... },
  "timestamp": "2024-01-15T10:30:00.000000",
  "session_count": 20
}
```

### 2. View Session History

```bash
# Get last 20 sessions
curl http://localhost:7860/chronicle/history

# Filter by task type
curl http://localhost:7860/chronicle/history?task_type=easy&limit=30

# Example response:
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "task_type": "easy",
      "start_time": "2024-01-15T10:00:00",
      "end_time": "2024-01-15T10:05:00",
      "final_score": 0.75,
      "steps_taken": 4
    }
  ],
  "total": 20,
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

### 3. Review Specific Session Details

```bash
# View all actions from a specific session
curl http://localhost:7860/chronicle/session/550e8400-e29b-41d4-a716-446655440000

# Example response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
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

### 🔴 High Priority Tips

- **Weak Area**: Identifies task types where you score below 50%
- **Low Success Rate**: When overall success rate falls below 50%

### 🟡 Medium Priority Tips

- **Efficiency**: When averaging too many steps (> 15 per task)
- **Read Strategy**: When paper reading pattern is suboptimal

### 🟢 Low Priority Tips

- **Encouragement**: When you're performing well (70%+ success rate)

## Understanding the Metrics

### Task Performance
- Shows average score, steps, and success rate for each task type
- Helps identify which difficulties you struggle with

### Action Patterns
- Tracks most common actions and their average rewards
- Shows effectiveness of different action types

### Efficiency Metrics
- **avg_steps**: Average steps per task (lower is better, but not at cost of accuracy)
- **success_rate**: Percentage of sessions scoring > 0.6
- **avg_score**: Average final score across sessions

## Database

All session data is stored locally in `chronicle.db` (SQLite format). This file is created automatically and is not synchronized across instances.

To reset all session history:
```bash
rm chronicle.db
```

## Tips Interpretation

### Weak Area Tips
- **What it means**: You're struggling with a particular difficulty level
- **How to improve**: 
  - Read papers more carefully
  - Spend more time reasoning about the claims
  - Review feedback from failed verifications

### Efficiency Tips
- **What it means**: You're using more steps than typical agents
- **How to improve**:
  - Plan your approach before starting (identify key papers)
  - Avoid reading irrelevant papers
  - Verify claims more confidently once you have enough information

### Read Strategy Tips
- **What it means**: You're not reading enough papers relative to verification attempts
- **How to improve**:
  - For complex claims, read 2-3 papers before verifying
  - Make sure you've read all cited papers before verification
  - Consider multiple perspectives from different papers

### Success Rate Tips
- **What it means**: Your overall accuracy is concerning (too low) or excellent (too high)
- **How to improve**:
  - If low: be more careful and methodical in verification
  - If high: great job! Keep it up

## API Query Parameters

### `/chronicle/tips`
- `limit` (optional): 1-100, default 20 - number of recent sessions to analyze
- `detail_level` (optional): "summary" or "detailed" (default: "summary")

### `/chronicle/history`
- `limit` (optional): 1-100, default 20 - number of sessions to return
- `task_type` (optional): "easy", "medium", or "hard" - filter by task type

## Troubleshooting

### No tips generated?
- You need at least a few sessions to analyze patterns
- Run multiple verification sessions first

### Can't find a session?
- Use the session ID from `/chronicle/history` response
- Session IDs are UUIDs (36 characters with hyphens)

### Database getting too large?
- Chronicle uses efficient SQLite queries
- Old data doesn't affect performance (time-windowed queries)
- Can safely delete `chronicle.db` to reset

## Learn More

See [CHRONICLE.md](./CHRONICLE.md) for detailed technical documentation.

"""
Aleth — OpenEnv FastAPI Server
Exposes the environment as a REST API on port 7860.

Fully compliant with the official OpenEnv HTTP protocol:
  POST /reset  {}  → {"observation": {...}, "reward": null, "done": false}
  POST /step   {"action": {...}}  → {"observation": {...}, "reward": <float|null>, "done": bool}
  GET  /state  → State dict
  GET  /health → {"status": "healthy"}
  GET  /schema → {"action": {...}, "observation": {...}, "state": {...}}

Notes:
- A new AlethEnv is created per /reset call (stateless between resets).
- State is maintained within a session (between reset and submit/done).
- The global _env is replaced on each /reset so /step and /state work in the same session.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from environment import AlethEnv
from chronicle import Chronicle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aleth",
    version="1.1.0",
    description="OpenEnv citation verification benchmark — scientific claim grading.",
)

# Global env instance — replaced on each /reset call
_env: Optional[AlethEnv] = None

# Chronicle instance for session tracking
_chronicle = Chronicle(db_path="chronicle.db")


# ── Request / Response models (OpenEnv spec-compliant) ───────────────────────

class ResetRequest(BaseModel):
    """
    OpenEnv standard reset request.
    We extend it with an optional 'task' field for Aleth.
    The dashboard typically sends {} or omits the body entirely.
    """
    seed:       Optional[int] = Field(default=None)
    episode_id: Optional[str] = Field(default=None)
    task:       Optional[str] = Field(default=None)

    model_config = {"extra": "allow"}


class ResetResponse(BaseModel):
    """OpenEnv standard reset response — strictly no extra fields."""
    observation: Dict[str, Any]
    reward:      Optional[float] = None
    done:        bool            = False


class StepRequest(BaseModel):
    """OpenEnv standard step request."""
    action:     Dict[str, Any]
    timeout_s:  Optional[float] = Field(default=None)
    request_id: Optional[str]   = Field(default=None)

    model_config = {"extra": "allow"}


class StepResponse(BaseModel):
    """OpenEnv standard step response — strictly no extra fields."""
    observation: Dict[str, Any]
    reward:      Optional[float] = None
    done:        bool            = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
async def reset(
    request: Request,
    body: ResetRequest = Body(default_factory=ResetRequest),
):
    """
    Reset the environment and return the initial observation.
    Accepts an empty body {} or a body with optional 'task' field.
    Default task is 'easy'.
    """
    global _env
    try:
        task = (body.task or "easy").strip()
        logger.info(f"POST /reset — task={task!r}")
        _env = AlethEnv()
        obs  = _env.reset(task=task)
        
        # Start a new chronicle session
        _chronicle.start_session(task)
        
        return ResetResponse(
            observation=obs.model_dump(mode="json"),
            reward=0.5,   # neutral non-null score; null would parse as 0.0 in some evaluators
            done=False,
        )
    except ValueError as exc:
        logger.error(f"/reset ValueError: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception(f"/reset unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    """
    Execute one action.
    Body: {"action": {"action_type": "...", ...}}
    """
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    try:
        action_dict = req.action
        logger.info(f"POST /step — action_type={action_dict.get('action_type')!r}")
        obs, reward, done, info = _env.step(action_dict)

        # Record action in chronicle
        action_type = action_dict.get("action_type", "unknown")
        reward_value = float(reward.total) if reward is not None else None
        _chronicle.record_action(action_type, action_dict, reward=reward_value)

        # When the episode is done, return the grader's episode score.
        # For intermediate steps, return the dense per-step reward.
        # ALL rewards are clamped strictly to (0.001, 0.999) because the
        # platform validator rejects any reward <= 0.0 or >= 1.0.
        if done and "final_score" in info:
            raw_reward: Optional[float] = float(info["final_score"])
            # Finalize chronicle session
            _chronicle.end_session(raw_reward, info.get("steps_taken", 0))
        else:
            raw_reward = float(reward.total) if reward is not None else None

        # Clamp to open interval (0.001, 0.999) — required by platform
        if raw_reward is not None:
            step_reward: Optional[float] = max(0.001, min(0.999, raw_reward))
        else:
            step_reward = None

        return StepResponse(
            observation=obs.model_dump(mode="json"),
            reward=step_reward,
            done=done,
        )
    except RuntimeError as exc:
        logger.error(f"/step RuntimeError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        logger.error(f"/step ValueError: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception(f"/step unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/state")
async def get_state():
    """
    Return the current task's claims and their cited paper IDs.
    Used by inference.py to build the sequential verification plan.
    """
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /state.")
    try:
        state = _env.state()
        return {
            "claims": {
                cid: {
                    "text":      claim.text,
                    "citations": claim.citations,
                }
                for cid, claim in state.claims.items()
            }
        }
    except Exception as exc:
        logger.exception(f"/state unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")




@app.get("/health")
async def health():
    """
    Liveness probe — OpenEnv spec returns {"status": "healthy"}.
    """
    return {"status": "healthy"}


@app.get("/schema")
async def schema():
    """Return JSON schemas for action, observation, and state."""
    from models import Observation, State
    # Minimal action schema (union type — show the common fields)
    action_schema = {
        "type": "object",
        "description": "Aleth action — one of: read_paper, verify_claim, flag_drift, submit",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["read_paper", "verify_claim", "flag_drift", "submit"],
            }
        },
        "required": ["action_type"],
    }
    return {
        "action":      action_schema,
        "observation": Observation.model_json_schema(),
        "state":       State.model_json_schema(),
    }


@app.get("/")
async def root():
    """Root endpoint — basic info."""
    return {
        "name":        "Aleth",
        "version":     "1.1.0",
        "description": "OpenEnv citation verification benchmark",
        "endpoints":   ["/reset", "/step", "/state", "/health", "/schema", "/chronicle/tips", "/chronicle/history", "/docs"],
    }


# ── Chronicle Endpoints ────────────────────────────────────────────────────────

class ChronicleQueryParams(BaseModel):
    """Query parameters for chronicle endpoints."""
    limit: int = Field(default=20, ge=1, le=100)
    task_type: Optional[str] = Field(default=None)
    detail_level: str = Field(default="summary")


@app.get("/chronicle/tips")
async def chronicle_tips(limit: int = 20, detail_level: str = "summary"):
    """
    Get personalized tips based on session history and usage patterns.
    
    Query parameters:
    - limit: Number of recent sessions to analyze (default: 20, max: 100)
    - detail_level: Level of detail for recommendations (default: "summary")
    
    Returns:
    - tips: List of prioritized personalized recommendations
    - metrics: Analysis metrics including task performance, action patterns, efficiency
    - timestamp: When the recommendations were generated
    - session_count: Number of sessions analyzed
    """
    try:
        limit = max(1, min(limit, 100))
        tips_data = _chronicle.get_tips(limit=limit, detail_level=detail_level)
        return tips_data
    except Exception as exc:
        logger.exception(f"/chronicle/tips unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/chronicle/history")
async def chronicle_history(limit: int = 20, task_type: Optional[str] = None):
    """
    Get recent session history.
    
    Query parameters:
    - limit: Number of recent sessions to return (default: 20, max: 100)
    - task_type: Filter by task type (easy, medium, hard) - optional
    
    Returns:
    - sessions: List of recent sessions with summary stats
    - total: Number of sessions returned
    - timestamp: When history was retrieved
    """
    try:
        limit = max(1, min(limit, 100))
        history = _chronicle.get_history(limit=limit, task_type=task_type)
        return history
    except Exception as exc:
        logger.exception(f"/chronicle/history unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


@app.get("/chronicle/session/{session_id}")
async def chronicle_session_detail(session_id: str):
    """
    Get detailed information about a specific session.
    
    Path parameters:
    - session_id: The session ID to retrieve
    
    Returns:
    - Detailed session information including all actions and their rewards
    """
    try:
        session = _chronicle.get_session_detail(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return session
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"/chronicle/session/{session_id} unexpected error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

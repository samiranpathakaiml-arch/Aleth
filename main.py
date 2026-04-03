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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aleth",
    version="1.1.0",
    description="OpenEnv citation verification benchmark — scientific claim grading.",
)

# Global env instance — replaced on each /reset call
_env: Optional[AlethEnv] = None


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
        return ResetResponse(
            observation=obs.model_dump(mode="json"),
            reward=None,
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
        return StepResponse(
            observation=obs.model_dump(mode="json"),
            reward=float(reward.total) if reward is not None else None,
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
async def state():
    """Return the current environment state."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        return _env.state().model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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
        "endpoints":   ["/reset", "/step", "/state", "/health", "/schema", "/docs"],
    }

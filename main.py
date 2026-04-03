"""
Aleth — OpenEnv FastAPI Server
Exposes the environment as a REST API on port 7860.

Endpoints:
  POST /reset  {"task": "easy"}           → Observation
  POST /step   {"action_type": ..., ...}  → {observation, reward, done, info}
  GET  /state                             → State
  GET  /health                            → {"status": "ok"}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from environment import AlethEnv   # absolute import (flat structure)

# NOTE: openenv-core is installed but its fastmcp dependency is incompatible
# with the current Pydantic version (raises TypeError on import).
# We implement the OpenEnv REST contract directly with FastAPI.

app = FastAPI(
    title="Aleth",
    version="1.1.0",
    description="OpenEnv citation verification benchmark — scientific claim grading.",
)
_env = AlethEnv()


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(req: ResetRequest):
    """Reset the environment for the given task and return the initial observation."""
    try:
        obs = _env.reset(task=req.task)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/step")
async def step(action: Dict[str, Any]):
    """
    Execute one action. Accepts the action dict directly, e.g.:
      {"action_type": "read_paper", "paper_id": "devlin2019"}
    """
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/state")
async def state():
    """Return the current environment state (deep copy)."""
    try:
        return _env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
async def health():
    """Liveness probe for the Scaler / HF Spaces dashboard."""
    return {"status": "ok", "env": "aleth", "version": "1.1.0"}


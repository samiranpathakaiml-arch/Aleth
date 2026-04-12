"""
Aleth Baseline Inference Script
================================
Hackathon-compliant implementation:
  • Uses OpenAI client for all LLM calls (API_BASE_URL / MODEL_NAME / HF_TOKEN)
  • Connects to the Aleth FastAPI server via HTTP (works against Docker container)
  • Emits [START] / [STEP] / [END] structured stdout logs in required format
  • Named inference.py and placed in the project root

Environment variables (all optional — sensible defaults provided):
  API_BASE_URL       LLM endpoint  (default: HF Router)
  MODEL_NAME         Model ID      (default: Llama-3.1-8B-Instruct)
  HF_TOKEN           API key       (also checks OPENAI_API_KEY)
  ALETH_SERVER_URL   Aleth server  (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Mandatory variables (per hackathon spec) ─────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or ""
)

# ── Environment target ────────────────────────────────────────────────────────
SERVER_URL  = os.getenv("ALETH_SERVER_URL", "http://localhost:7860")
BENCHMARK   = "aleth"
IMAGE_NAME  = "aleth:latest"

# ── Episode config ────────────────────────────────────────────────────────────
TEMPERATURE             = 0.0
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.6

# Per-task step budgets and reward normalisation denominators
_TASK_MAX_STEPS: Dict[str, int]   = {"easy": 50, "medium": 100, "hard": 150}
_TASK_MAX_REWARD: Dict[str, float] = {"easy": 5.0, "medium": 10.0, "hard": 20.0}


# ── Structured logging helpers (exact hackathon format) ──────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit the mandatory [START] line."""
    payload = json.dumps({"task": task, "env": env, "model": model})
    print(f"[START] {payload}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit a mandatory [STEP] line."""
    payload = json.dumps({
        "step":   step,
        "action": action,
        "reward": round(reward, 4),
        "done":   done,
        "error":  error,
    })
    print(f"[STEP] {payload}", flush=True)


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Emit the mandatory [END] line."""
    payload = json.dumps({
        "success": success,
        "steps":   steps,
        "score":   round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
    })
    print(f"[END] {payload}", flush=True)


# ── HTTP client (talks to the Aleth FastAPI server / Docker container) ────────

class AlethHTTPClient:
    """Thin HTTP wrapper around the Aleth FastAPI endpoints."""

    def __init__(self, base_url: str, startup_timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self._wait_for_server(startup_timeout)

    def _wait_for_server(self, timeout: int) -> None:
        """Poll /health until the server is ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200:
                    print(f"[DEBUG] HTTP server ready at {self.base_url}", flush=True)
                    return
            except Exception:
                pass
            time.sleep(2)
        raise ConnectionError(
            f"No server at {self.base_url} (waited {timeout}s)"
        )

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/reset", json={"task": task}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return r.json()


# ── Local fallback client (direct Python import — no server needed) ───────────

class AlethLocalClient:
    """
    Wraps AlethEnv directly, mirroring the same dict interface as AlethHTTPClient.
    Used automatically when no HTTP server is reachable (local development).
    """

    def __init__(self) -> None:
        from environment import AlethEnv  # only imported when needed
        self._env = AlethEnv()
        print("[DEBUG] Running in LOCAL mode (direct AlethEnv import)", flush=True)

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        obs = self._env.reset(task=task)
        return {"observation": obs.model_dump(mode="json"), "reward": None, "done": False}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        obs, reward, done, _info = self._env.step(action)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward": float(reward.total) if reward is not None else 0.0,
            "done": done,
        }


def get_env() -> "AlethHTTPClient | AlethLocalClient":
    """
    Auto-detect: use the HTTP server if reachable, otherwise fall back to local.
    The HTTP path is used by the Docker / HF Space evaluator.
    The local path is used for development without a running server.
    """
    try:
        return AlethHTTPClient(SERVER_URL, startup_timeout=10)
    except ConnectionError:
        print(
            f"[DEBUG] No server at {SERVER_URL} — switching to LOCAL mode.",
            flush=True,
        )
        return AlethLocalClient()


# ── LLM interaction ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a citation verification expert.
Your goal is to verify whether scientific claims are supported by their cited papers.

AVAILABLE ACTIONS (respond ONLY with valid JSON, no extra text):

1. Read a paper:
   {"action_type": "read_paper", "paper_id": "<ID>"}

2. Verify a claim:
   {"action_type": "verify_claim", "claim_id": "<ID>", "support_score": 0.0–1.0, \
"reasoning": "<explanation>", "primary_evidence_paper": "<ID>"}

3. Flag citation drift:
   {"action_type": "flag_drift", "claim_id": "<ID>", \
"citation_chain": ["<A>", "<B>"], "drift_explanation": "<explanation>"}

4. Submit (end episode):
   {"action_type": "submit"}

STRATEGY:
- Read the papers cited by each claim before verifying it.
- Choose support_score = 1.0 if the paper directly supports the claim.
- Choose support_score = 0.0 if the paper contradicts the claim.
- Use intermediate values for partial support.
- Submit once all claims are verified.
"""


def _build_user_prompt(
    task: str,
    obs: Dict[str, Any],
    history: List[str],
) -> str:
    papers_read  = obs.get("papers_read", [])
    claims_done  = obs.get("claims_verified", 0)
    claims_total = obs.get("claims_total", 0)
    message      = obs.get("message", "")

    anti_loop = (
        f"Papers already read: {papers_read}. Do NOT re-read them."
        if papers_read
        else "No papers read yet. Start by reading the cited papers."
    )

    recent_history = "\n".join(history[-5:]) if history else "(none)"

    return (
        f"TASK: {task}\n\n"
        f"PROGRESS: {claims_done}/{claims_total} claims verified\n"
        f"LAST SERVER MESSAGE: {message}\n\n"
        f"{anti_loop}\n\n"
        f"RECENT HISTORY:\n{recent_history}\n\n"
        "What is your next action? Respond ONLY with valid JSON."
    )


def get_model_action(
    client: OpenAI,
    task: str,
    obs: Dict[str, Any],
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its JSON action."""
    user_prompt = _build_user_prompt(task, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if the model wraps output
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        # Safe fallback: submit to end the episode cleanly
        return {"action_type": "submit"}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    openai_client: OpenAI,
    env: AlethHTTPClient,
    task: str,
) -> float:
    """
    Run one full episode for *task* and return the normalised score [0.0, 1.0].
    Emits [START], [STEP]*n, [END] to stdout.
    """
    max_steps   = _TASK_MAX_STEPS.get(task, 50)
    max_reward  = _TASK_MAX_REWARD.get(task, 5.0)

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task=task)
        obs    = result.get("observation", {})
        done   = result.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            action     = get_model_action(openai_client, task, obs, history)
            action_str = json.dumps(action)

            reward_val: float       = 0.0
            error_msg:  Optional[str] = None

            try:
                step_result = env.step(action)
                obs         = step_result.get("observation", {})
                reward_val  = float(step_result.get("reward") or 0.0)
                done        = step_result.get("done", False)
            except Exception as exc:
                error_msg = str(exc)
                done      = True

            rewards.append(reward_val)
            steps_taken = step
            history.append(
                f"Step {step}: {action_str} -> reward {reward_val:+.4f}"
            )

            log_step(
                step=step,
                action=action_str,
                reward=reward_val,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # Normalise cumulative reward — clamp strictly within (0, 1) exclusive
        # as required by the platform validator (0.0 and 1.0 are rejected)
        raw_score = sum(rewards) / max_reward if max_reward > 0 else 0.001
        score     = min(max(raw_score, 0.001), 0.999)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] No API key found. "
            "Set HF_TOKEN or OPENAI_API_KEY before running.",
            flush=True,
        )
        return

    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env           = get_env()  # HTTP if server running, local fallback otherwise

    tasks  = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    for task in tasks:
        score        = run_episode(openai_client, env, task)
        scores[task] = score

    # Human-readable summary (after all structured log blocks)
    print("", flush=True)
    print("=" * 40, flush=True)
    print("ALETH BASELINE RESULTS", flush=True)
    print("=" * 40, flush=True)
    for t, s in scores.items():
        status = "✅" if s >= SUCCESS_SCORE_THRESHOLD else "❌"
        print(f"  {t.upper():8s} {s:.4f}  {status}", flush=True)
    print("=" * 40, flush=True)


if __name__ == "__main__":
    main()

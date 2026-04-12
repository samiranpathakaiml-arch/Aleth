"""
Aleth Baseline Inference Script
================================
Hackathon-compliant implementation.

Environment variables:
  API_BASE_URL       LLM endpoint  (default: HF Router)
  MODEL_NAME         Model ID      (default: Llama-3.1-8B-Instruct)
  HF_TOKEN / API_KEY API key
  ALETH_SERVER_URL   Aleth server  (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Mandatory variables (per hackathon spec) ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")

_api_key    = (os.getenv("API_KEY")        or "").strip()
_hf_token   = (os.getenv("HF_TOKEN")       or "").strip()
_openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
API_KEY     = _api_key or _hf_token or _openai_key

SERVER_URL  = os.getenv("ALETH_SERVER_URL", "http://localhost:7860")

SUCCESS_SCORE_THRESHOLD = 0.6

# Platform requirement: scores MUST be strictly between 0 and 1 (exclusive)
_SCORE_MIN: float = 0.001
_SCORE_MAX: float = 0.999


def _safe_score(v: float) -> float:
    """Clamp any float to the open interval (0, 1)."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(v)))


# ── HTTP client ───────────────────────────────────────────────────────────────

class AlethHTTPClient:
    def __init__(self, base_url: str, startup_timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self._wait_for_server(startup_timeout)

    def _wait_for_server(self, timeout: int) -> None:
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
        raise ConnectionError(f"No server at {self.base_url} after {timeout}s")

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/reset", json={"task": task}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_claims(self) -> Dict[str, Dict]:
        r = requests.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return r.json().get("claims", {})


# ── Local fallback ────────────────────────────────────────────────────────────

class AlethLocalClient:
    def __init__(self) -> None:
        from environment import AlethEnv
        self._env = AlethEnv()
        print("[DEBUG] Running in LOCAL mode (direct AlethEnv import)", flush=True)

    def reset(self, task: str = "easy") -> Dict[str, Any]:
        obs = self._env.reset(task=task)
        return {"observation": obs.model_dump(mode="json"), "reward": None, "done": False}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        obs, reward, done, info = self._env.step(action)
        if done and "final_score" in info:
            step_reward = _safe_score(info["final_score"])
        else:
            raw = float(reward.total) if reward is not None else _SCORE_MIN
            step_reward = _safe_score(raw)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward":      step_reward,
            "done":        done,
        }

    def get_claims(self) -> Dict[str, Dict]:
        state = self._env.state()
        return {
            cid: {"text": claim.text, "citations": claim.citations}
            for cid, claim in state.claims.items()
        }


def get_env() -> "AlethHTTPClient | AlethLocalClient":
    try:
        return AlethHTTPClient(SERVER_URL, startup_timeout=15)
    except ConnectionError:
        print(f"[DEBUG] No server at {SERVER_URL} — switching to LOCAL mode.", flush=True)
        return AlethLocalClient()


# ── LLM verifier ──────────────────────────────────────────────────────────────

VERIFY_SYSTEM = """\
You are a scientific citation verifier. Given a claim and paper evidence, output ONLY:
{"support_score": <float 0.01-0.99>, "reasoning": "<1-2 sentence explanation>"}

Rules:
- support_score near 0.95 if paper directly supports the claim
- support_score near 0.05 if paper contradicts or is unrelated to the claim
- support_score near 0.50 if paper partially supports or is ambiguous
- NEVER output exactly 0.0 or 1.0 — always stay strictly between 0 and 1
- Output ONLY the JSON. No markdown. No extra text.
"""


def _call_llm_verify(
    client: OpenAI,
    claim_text: str,
    paper_texts: List[str],
) -> Dict[str, Any]:
    evidence_parts = [t[:3000] for t in paper_texts[:2]]
    evidence = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "(no paper text available)"

    user_prompt = (
        f"CLAIM:\n{claim_text}\n\n"
        f"PAPER EVIDENCE:\n{evidence}\n\n"
        "Output your JSON verdict now:"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = (completion.choices[0].message.content or "").strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data      = json.loads(text)
        score     = _safe_score(float(data.get("support_score", 0.5)))
        reasoning = str(data.get("reasoning", "Evidence reviewed."))
        return {"support_score": score, "reasoning": reasoning}
    except Exception as exc:
        print(f"[DEBUG] LLM verify error: {exc}", flush=True)
        return {
            "support_score": 0.5,
            "reasoning": "Could not parse LLM response; assuming partial support.",
        }


# ── Episode runner ────────────────────────────────────────────────────────────

_TASK_MAX_STEPS: Dict[str, int] = {"easy": 50, "medium": 100, "hard": 150}


def run_episode(
    openai_client: OpenAI,
    env: "AlethHTTPClient | AlethLocalClient",
    task: str,
) -> float:
    """
    Run one full episode:
      1. Reset environment
      2. For each claim: read papers → LLM verify → submit verify_claim
      3. Submit to end episode
      4. Return grader score (strictly within (0, 1))

    Outputs required [START] / [STEP] / [END] log lines.
    """
    max_steps = _TASK_MAX_STEPS.get(task, 50)

    # Required structured log: START
    print(f"[START] task={task}", flush=True)

    step_num:     int   = 0
    final_score:  float = _SCORE_MIN   # safe default — never 0.0
    done:         bool  = False

    def do_step(action: Dict[str, Any]) -> tuple:
        nonlocal step_num, done, final_score
        try:
            result     = env.step(action)
            reward_raw = result.get("reward")
            # Clamp whatever the server returns
            reward_val = _safe_score(float(reward_raw)) if reward_raw is not None else _SCORE_MIN
            done       = bool(result.get("done", False))
            # When the episode ends the server returns final_score as reward
            if done:
                final_score = reward_val
            obs_out    = result.get("observation", {})
        except Exception as exc:
            reward_val = _SCORE_MIN
            done       = True
            obs_out    = {}
            print(f"[DEBUG] step error: {exc}", flush=True)

        step_num += 1
        # Required structured log: STEP
        print(f"[STEP] step={step_num} reward={reward_val:.4f}", flush=True)
        return obs_out, reward_val, done

    try:
        # Reset
        env.reset(task=task)

        # Get the claims for this task
        claims_data = env.get_claims()
        claim_citations: Dict[str, List[str]] = {cid: v["citations"] for cid, v in claims_data.items()}
        claim_texts:     Dict[str, str]        = {cid: v["text"]      for cid, v in claims_data.items()}

        papers_read_cache: Dict[str, str] = {}

        # Sequential: for each claim, read its papers then verify
        for cid, citations in claim_citations.items():
            if done or step_num >= max_steps - 2:
                break

            claim_text = claim_texts.get(cid, cid)

            # Read cited papers
            paper_texts_for_claim: List[str] = []
            for pid in citations:
                if step_num >= max_steps - 2 or done:
                    break
                if pid in papers_read_cache:
                    paper_texts_for_claim.append(papers_read_cache[pid])
                    continue
                obs, _, done = do_step({"action_type": "read_paper", "paper_id": pid})
                if done:
                    break
                for pc in obs.get("papers_content", []):
                    text = pc.get("text", "") if isinstance(pc, dict) else getattr(pc, "text", "")
                    papers_read_cache[pid] = text
                    paper_texts_for_claim.append(text)

            if done or step_num >= max_steps - 1:
                break

            # LLM verification
            verdict = _call_llm_verify(openai_client, claim_text, paper_texts_for_claim)

            # Submit verify_claim
            _, _, done = do_step({
                "action_type":            "verify_claim",
                "claim_id":               cid,
                "support_score":          verdict["support_score"],
                "reasoning":              verdict["reasoning"],
                "primary_evidence_paper": citations[0] if citations else "",
            })

            if done:
                break

        # Explicit submit if not already done
        if not done:
            do_step({"action_type": "submit"})

    except Exception as exc:
        print(f"[DEBUG] episode error: {exc}", flush=True)

    # Final safety clamp — score MUST be strictly within (0, 1)
    final_score = _safe_score(final_score)

    # Required structured log: END  (format the platform parser expects)
    print(f"[END] task={task} score={final_score:.4f} steps={step_num}", flush=True)
    return final_score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] No API key found. "
            "Set HF_TOKEN (or API_KEY / OPENAI_API_KEY).",
            flush=True,
        )
        return

    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env           = get_env()

    tasks:  List[str]        = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    for task in tasks:
        try:
            scores[task] = run_episode(openai_client, env, task)
        except Exception as exc:
            print(f"[ERROR] Task {task} crashed: {exc}", flush=True)
            # Safe fallback — never 0.0 which would fail the range check
            safe = _SCORE_MIN
            print(f"[END] task={task} score={safe:.4f} steps=0", flush=True)
            scores[task] = safe

    print("", flush=True)
    print("=" * 40, flush=True)
    print("ALETH BASELINE RESULTS", flush=True)
    print("=" * 40, flush=True)
    for t, s in scores.items():
        status = "OK" if s >= SUCCESS_SCORE_THRESHOLD else "--"
        print(f"  {t.upper():8s} {s:.4f}  [{status}]", flush=True)
    print("=" * 40, flush=True)


if __name__ == "__main__":
    main()

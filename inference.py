"""
Aleth Baseline Inference Script
Strictly compliant with Hackathon Requirements:
1. Uses OpenAI Client for all LLM calls.
2. Uses API_BASE_URL, MODEL_NAME, and HF_TOKEN env variables.
3. Named inference.py and placed in root.
"""

import os
import json
import time
from typing import Dict, Any, List
from openai import OpenAI
from environment import AlethEnv          # flat-structure absolute import

# MANDATORY Environment Variables per Hackathon Spec
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "claude-3-5-sonnet-20241022"
HF_TOKEN     = os.getenv("HF_TOKEN")     or os.getenv("ANTHROPIC_API_KEY")


def parse_llm_action(response_text: str) -> Dict[str, Any]:
    """Extracts JSON action from LLM response."""
    try:
        # Handle markdown code fences if the model wraps its output
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        return json.loads(json_str)
    except Exception:
        # Fallback to a safe no-op if parsing fails
        return {"action_type": "read_paper", "paper_id": "none"}


def build_system_prompt() -> str:
    return """You are a citation verification expert.
Your goal is to verify if scientific claims are supported by their cited papers.

ACTIONS:
1. Read Paper:   {"action_type": "read_paper", "paper_id": "ID"}
2. Verify Claim: {"action_type": "verify_claim", "claim_id": "ID", "support_score": 0.0-1.0, "reasoning": "...", "primary_evidence_paper": "ID"}
3. Flag Drift:   {"action_type": "flag_drift", "claim_id": "ID", "citation_chain": ["A", "B"], "drift_explanation": "..."}
4. Submit:       {"action_type": "submit"}

IMPORTANT: Respond ONLY with valid JSON. No conversational text."""


def run_episode(client: OpenAI, env: AlethEnv, task: str) -> float:
    # ── Required structured output: START block ──────────────────────────────
    print(f"[START] task={task}", flush=True)

    obs = env.reset(task=task)
    done = False
    total_reward = 0.0
    step_num = 0
    info: Dict[str, Any] = {}

    # Pre-fetch valid IDs once; inject as a guidance "menu"
    state = env.state()
    guidance_lines = []
    for cid, claim in state.claims.items():
        guidance_lines.append(
            f"  - claim_id: \"{cid}\" | text: \"{claim.text}\" | cite: {claim.citations}"
        )
    claim_menu = "\n".join(guidance_lines)

    # Track last paper text snippet (updated when read_paper is taken)
    last_paper_snippet: str = ""

    while not done:
        already_read = obs.papers_read
        anti_loop = (
            f"You have already read these papers: {already_read}. "
            "Do NOT re-read them. "
            "If you have enough information, proceed to 'verify_claim' or 'submit'."
            if already_read
            else "No papers read yet. Start by reading the cited papers."
        )

        current_context = (
            f"TASK: {task}\n\n"
            f"VALID TARGETS — use ONLY these exact IDs:\n{claim_menu}\n\n"
            f"CURRENT STATUS:\n"
            f"  - Claims Verified : {obs.claims_verified}/{obs.claims_total}\n"
            f"  - Papers read      : {already_read}\n"
            f"  - Latest message   : {obs.message}\n"
            + (f"  - Last paper text  : {last_paper_snippet}\n" if last_paper_snippet else "")
            + f"\n{anti_loop}\n\n"
            "What is your next action? Respond in ONLY valid JSON."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user",   "content": current_context},
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
            )

            raw_response = response.choices[0].message.content
            action_dict  = parse_llm_action(raw_response)

            # Step the environment
            obs, reward, done, info = env.step(action_dict)
            total_reward += reward.total
            step_num += 1

            # Truncate paper text to 1 000 chars to stay within token budget
            if action_dict.get("action_type") == "read_paper" and obs.papers_content:
                raw_text = obs.papers_content[0].text
                last_paper_snippet = raw_text[:1000] + ("…" if len(raw_text) > 1000 else "")
            else:
                last_paper_snippet = ""

            # ── Required structured output: STEP block ────────────────────────
            print(f"[STEP] step={step_num} reward={reward.total:.4f}", flush=True)

        except Exception as e:
            print(f"[STEP] step={step_num} reward=0.0 error={e}", flush=True)
            break

    final_score = info.get("final_score", 0.0)
    steps_taken = info.get("steps_taken", step_num)

    # ── Required structured output: END block ─────────────────────────────────
    print(f"[END] task={task} score={final_score:.4f} steps={steps_taken}", flush=True)
    return final_score


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN (or ANTHROPIC_API_KEY) not found in environment variables.", flush=True)
        print("Set HF_TOKEN=your_token_here before running.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = AlethEnv()

    tasks  = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    for task in tasks:
        score = run_episode(client, env, task)
        scores[task] = score

    # Summary to stdout (plain text, after all structured blocks)
    print("", flush=True)
    print("=" * 30, flush=True)
    print("ALETH BASELINE RESULTS", flush=True)
    print("=" * 30, flush=True)
    for t, s in scores.items():
        print(f"{t.upper()}: {s:.4f}", flush=True)
    print("=" * 30, flush=True)


if __name__ == "__main__":
    main()

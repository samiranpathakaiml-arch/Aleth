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
    print(f"\n🚀 STARTING TASK: {task.upper()}")
    obs = env.reset(task=task)
    done = False
    total_reward = 0.0
    info: Dict[str, Any] = {}

    # --- FIX 1 & 2: Pre-fetch valid IDs once; inject as a guidance "menu" ---
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
        # --- FIX 1: Recreate messages from scratch every step (no history bloat) ---
        # --- FIX 4: Include anti-loop instruction with already-read list ---
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

            # --- FIX 3: Truncate paper text to 1 000 chars to stay within token budget ---
            if action_dict.get("action_type") == "read_paper" and obs.papers_content:
                raw_text = obs.papers_content[0].text
                last_paper_snippet = raw_text[:1000] + ("…" if len(raw_text) > 1000 else "")
            else:
                last_paper_snippet = ""  # Clear snippet when not a read step

            print(
                f"Step {obs.step_count}: {action_dict.get('action_type')} "
                f"-> Reward: {reward.total:.4f}"
            )

        except Exception as e:
            print(f"❌ Error during step: {e}")
            break

    final_score = info.get("final_score", 0.0)
    print(f"\n✅ EPISODE FINISHED")
    print(f"Final Score: {final_score:.4f}")
    return final_score


def main():
    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN (or ANTHROPIC_API_KEY) not found in environment variables.")
        print("\nTo fix:")
        print("  Windows : set HF_TOKEN=your_token_here")
        print("  macOS/Linux: export HF_TOKEN=your_token_here")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = AlethEnv()

    tasks  = ["easy", "medium", "hard"]
    scores: Dict[str, float] = {}

    for task in tasks:
        score = run_episode(client, env, task)
        scores[task] = score

    print("\n" + "=" * 30)
    print("ALETH BASELINE RESULTS")
    print("=" * 30)
    for t, s in scores.items():
        print(f"{t.upper()}: {s:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()

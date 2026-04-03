"""Quick validation test for the Aleth environment."""
from aleth import AlethEnv

def test_full_episode():
    env = AlethEnv()
    obs = env.reset("easy")
    state = env.state()
    print("Claims:", list(state.claims.keys()))
    print("Papers:", list(state.papers.keys()))
    print()

    # Read all papers
    for pid in list(state.papers.keys()):
        obs2, r, done, info = env.step({"action_type": "read_paper", "paper_id": pid})
        print(f"Read {pid}: reward={r.total:.4f}")

    print()

    # Verify all 5 with known-good answers
    verifications = [
        {"action_type": "verify_claim", "claim_id": "claim_001",
         "support_score": 1.0,
         "reasoning": "BERT-Large achieves 93.2% on the GLUE benchmark as stated in results section.",
         "primary_evidence_paper": "devlin2019"},
        {"action_type": "verify_claim", "claim_id": "claim_002",
         "support_score": 0.0,
         "reasoning": "Paper says 175 billion parameters not 200 billion as the claim states.",
         "primary_evidence_paper": "brown2020"},
        {"action_type": "verify_claim", "claim_id": "claim_003",
         "support_score": 1.0,
         "reasoning": "The Transformer relies entirely on self-attention mechanism without recurrence.",
         "primary_evidence_paper": "vaswani2017"},
        {"action_type": "verify_claim", "claim_id": "claim_004",
         "support_score": 0.9,
         "reasoning": "Paper reports 3.57% top-5 error on ILSVRC 2015 imagenet classification.",
         "primary_evidence_paper": "he2016"},
        {"action_type": "verify_claim", "claim_id": "claim_005",
         "support_score": 1.0,
         "reasoning": "Dropout randomly drops units during training to prevent overfitting.",
         "primary_evidence_paper": "srivastava2014"},
    ]

    for v in verifications:
        obs2, r, done, info = env.step(v)
        print(f"{v['claim_id']}: reward={r.total:.4f}  breakdown={dict(r.breakdown)}")
        if done:
            # Auto-terminated because all claims verified
            break

    # Submit only if episode not already done (auto-terminates after all claims verified)
    if not done:
        obs, r, done, info = env.step({"action_type": "submit"})

    print()
    print("=" * 50)
    print(f"DONE: {done}")
    print(f"Final score: {info['final_score']:.4f}")
    print("Per-claim breakdown:")
    for cid, sc in info["grading_breakdown"].items():
        print(f"  {cid}: {sc}")

    # Validate expected scores
    assert info["final_score"] > 0.8, f"Expected > 0.8, got {info['final_score']}"
    print()
    print("All assertions PASSED ✅")



if __name__ == "__main__":
    test_full_episode()

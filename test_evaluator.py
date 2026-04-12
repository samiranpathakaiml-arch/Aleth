import requests

BASE = "http://127.0.0.1:7861"
for task in ["easy", "medium", "hard"]:
    r = requests.post(f"{BASE}/reset", json={"task": task})
    rj = r.json()
    print(f"RESET {task}: status={r.status_code} reward={rj.get('reward')} done={rj.get('done')}")

    r2 = requests.post(f"{BASE}/step", json={"action": {"action_type": "submit"}})
    d = r2.json()
    rw = d.get("reward")
    ok = rw is not None and 0.0 < rw < 1.0
    print(f"STEP : done={d.get('done')} reward={rw} IN_RANGE={ok}")
    print()

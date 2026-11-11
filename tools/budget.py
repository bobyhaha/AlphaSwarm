#!/usr/bin/env python3
"""Research budget: OpenRouter account usage since the start of the autoresearch session, against its cap.

  OPENROUTER_API_KEY=... python3 tools/budget.py          # print spent / left
  OPENROUTER_API_KEY=... python3 tools/budget.py --left   # print only dollars left (for --cost-budget)
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "spend.json"


def usage() -> float:
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return float(json.load(r)["data"]["usage"])


def main():
    led = json.loads(LEDGER.read_text())
    spent = usage() - led["baseline_usage"]
    left = led["cap_usd"] - spent
    if "--left" in sys.argv:
        print(f"{max(0.0, left):.2f}")
    else:
        print(f"spent ${spent:.2f} of ${led['cap_usd']:.0f}; ${left:.2f} left")


if __name__ == "__main__":
    main()

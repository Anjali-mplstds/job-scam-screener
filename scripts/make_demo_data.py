#!/usr/bin/env python3
"""Write synthetic labelled postings (fictional companies) to a .jsonl file.

  python scripts/make_demo_data.py --n 500 --out data/demo_500.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scamscreen.synth import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--scam-ratio", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="data/demo_500.jsonl")
a = ap.parse_args()
rows = generate(a.n, a.scam_ratio, a.seed)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} synthetic postings to {a.out} ({sum(r['label'] for r in rows)} labelled scam)")

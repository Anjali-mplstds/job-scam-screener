#!/usr/bin/env python3
"""Measure the screener on labelled postings BEFORE you publish any accuracy claim.

  python scripts/evaluate.py --input data/emscad.csv --limit 2000            # real Laya
  python scripts/evaluate.py --input data/sample_postings.jsonl --engine mock
  python scripts/evaluate.py --input data/emscad.csv --calibrate             # fit temperature + thresholds

Runs the engine once at temperature 1.0, then re-scores offline for every candidate temperature, so
calibration costs no extra inference. Prints AUROC, precision/recall/F1, and the settings to paste
into config.yaml. Labels: 1 = scam / fraudulent, 0 = genuine.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scamscreen.ingest import full_text, load_postings  # noqa: E402
from scamscreen.pipeline import apply_calibration, get_raw  # noqa: E402
from scamscreen.rules import find_hits  # noqa: E402
from scamscreen.scoring import cfg_with_defaults, score_posting  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def auroc(y, s):
    pos = [v for t, v in zip(y, s) if t == 1]
    neg = [v for t, v in zip(y, s) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def prf(y, s, thr):
    tp = sum(1 for t, v in zip(y, s) if t == 1 and v >= thr)
    fp = sum(1 for t, v in zip(y, s) if t == 0 and v >= thr)
    fn = sum(1 for t, v in zip(y, s) if t == 1 and v < thr)
    tn = len(y) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--engine", choices=["laya", "http", "mock"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--raw", help="reuse cached model answers (outputs/raw.json)")
    ap.add_argument("--calibration", help="calibration.yaml from scripts/diagnose.py")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    postings = [p for p in load_postings(a.input, a.limit) if p["label"] is not None]
    if not postings:
        raise SystemExit("No labelled postings (need a label / fraudulent / is_scam column with 0/1).")
    y = [p["label"] for p in postings]
    print(f"{len(postings)} labelled postings, {sum(y)} scam ({100*sum(y)/len(y):.1f}%)")
    if a.calibration:
        apply_calibration(cfg, a.calibration)
    raw, timing, engine = get_raw(postings, cfg, a.engine, a.raw, ROOT / "outputs" / "raw.json")
    print("timing:", timing)
    hits = [find_hits(full_text(p), p["company"]) for p in postings]

    base = cfg_with_defaults(cfg.get("scoring"))
    def risks(T):
        c = {**base, "temperature": T}
        return [score_posting(r, h, c)["risk"] for r, h in zip(raw, hits)]

    s = risks(base["temperature"])
    print(f"\nAt config settings (T={base['temperature']}): AUROC={auroc(y, s):.3f}")
    for name, thr in base["thresholds"].items():
        print(f"  threshold {name}={thr}: {prf(y, s, thr)}")
    # rules only vs model only, so you can see what the model adds
    rules_only = [score_posting({"p": {}}, h, base)["risk"] for h in hits]
    print(f"Rules only: AUROC={auroc(y, rules_only):.3f}   (does Laya add anything on your data?)")

    if a.calibrate:
        best = None
        for T in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
            sc = risks(T)
            for thr in [x / 100 for x in range(30, 91, 5)]:
                f = prf(y, sc, thr)["f1"]
                if best is None or f > best[0]:
                    best = (f, T, thr, auroc(y, sc))
        f, T, thr, au = best
        print(f"\nBest F1={f:.3f} at temperature={T}, flag threshold={thr} (AUROC={au:.3f})")
        print("Paste into config.yaml -> scoring:")
        print(f"  temperature: {T}\n  thresholds: {{check: {round(max(0.15, thr/2), 2)}, scam: {thr}}}")
        print("\nFit on one split, then confirm on postings the fit never saw.")
        out = ROOT / "outputs" / "calibration.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"temperature": T, "flag_threshold": thr, "f1": f, "auroc": au}, indent=2))


if __name__ == "__main__":
    main()

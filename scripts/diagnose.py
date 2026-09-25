#!/usr/bin/env python3
"""Find out which Laya questions carry signal on YOUR postings, and remove their built-in 'yes' bias.

  python scripts/diagnose.py --input data/fake_job_postings.csv --limit 300     # runs Laya once, caches answers
  python scripts/diagnose.py --input data/fake_job_postings.csv --limit 300 --raw outputs/raw.json

For every check it prints the mean P(yes) on genuine vs scam postings and the AUROC of that question alone
(0.5 = coin flip, 1.0 = perfect). It then writes outputs/calibration.yaml with
  * question_bias : each question's 75th-percentile logit on GENUINE postings (never below 0), subtracted at scoring time so a
                    model that says "yes" to everything stops flagging everything
  * disabled      : questions whose AUROC is below --min-auroc; their model answer is ignored (rules still count)
Use it:  python run_pipeline.py --input ... --raw outputs/raw.json --calibration outputs/calibration.yaml
The fit uses the same labelled postings it reports on, so the "after" numbers are optimistic. Confirm on
postings that were not used for the fit.
"""
import argparse
import math
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import auroc, prf  # noqa: E402
from scamscreen.ingest import full_text, load_postings  # noqa: E402
from scamscreen.pipeline import apply_calibration, get_raw  # noqa: E402
from scamscreen.questions import CHECKS, PROTECT  # noqa: E402
from scamscreen.rules import find_hits  # noqa: E402
from scamscreen.scoring import cfg_with_defaults, score_posting  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _q75(xs):
    return statistics.quantiles(xs, n=4)[2] if len(xs) >= 2 else xs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--engine", choices=["laya", "http", "mock"])
    ap.add_argument("--raw", help="reuse cached model answers")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--min-auroc", type=float, default=0.60)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    postings = [p for p in load_postings(a.input, a.limit) if p["label"] is not None]
    y = [p["label"] for p in postings]
    if 0 not in y or 1 not in y:
        raise SystemExit("Need both genuine (0) and scam (1) labelled postings.")
    print(f"{len(postings)} labelled postings: {y.count(0)} genuine, {y.count(1)} scam")
    raw, timing, engine = get_raw(postings, cfg, a.engine, a.raw, Path(a.out) / "raw.json")
    print("timing:", timing)

    bias, disabled = {}, []
    print(f"\n{'question':<16}{'yes|genuine':>12}{'yes|scam':>10}{'AUROC':>8}{'bias':>8}   verdict")
    for c in CHECKS + [PROTECT]:
        ps = [r["p"][c.id] for r in raw]
        g = [p for p, t in zip(ps, y) if t == 0]
        s = [p for p, t in zip(ps, y) if t == 1]
        au = auroc(y, ps)
        au_eff = 1 - au if c is PROTECT else au          # the protective question should score LOWER on scams
        # Only ever remove an upward "yes" bias, never add one; the protective check is left alone.
        # 75th percentile: three quarters of genuine postings end up at or below a coin flip.
        b = 0.0 if c is PROTECT else max(0.0, min(8.0, _q75([logit(p) for p in g])))
        bias[c.id] = round(b, 2)
        weak = au_eff < a.min_auroc
        if weak:
            disabled.append(c.id)
        print(f"{c.id:<16}{statistics.mean(g):>12.2f}{statistics.mean(s):>10.2f}{au_eff:>8.2f}{b:>8.2f}   "
              f"{'IGNORE (no signal)' if weak else 'keep'}")

    base = cfg_with_defaults(cfg.get("scoring"))
    hits = [find_hits(full_text(p), p["company"]) for p in postings]

    def report(name, scfg):
        sc = [score_posting(r, h, scfg) for r, h in zip(raw, hits)]
        risk = [x["risk"] for x in sc]
        flagged = [x["verdict"] != "looks_ok" for x in sc]
        fp = sum(f for f, t in zip(flagged, y) if t == 0)
        tp = sum(f for f, t in zip(flagged, y) if t == 1)
        print(f"{name:<22} AUROC={auroc(y, risk):.3f}  genuine wrongly flagged {fp}/{y.count(0)}  "
              f"scams caught {tp}/{y.count(1)}")

    print()
    report("as configured", base)
    fixed = {**base, "question_bias": bias, "disabled": disabled}
    report("bias-corrected", fixed)
    report("rules only", {**base, "disabled": [c.id for c in CHECKS] + [PROTECT.id]})

    out = Path(a.out) / "calibration.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump({"scoring": {"question_bias": bias, "disabled": disabled}}, sort_keys=False))
    print(f"\nWrote {out}. In-sample fit: confirm on postings not used here.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screen job postings for scam signals.

  python run_pipeline.py --input data/demo_500.jsonl                 # uses config.yaml (Laya)
  python run_pipeline.py --input data/demo_500.jsonl --engine mock   # offline plumbing test
  python run_pipeline.py --text "Earn $500/day! Pay a $50 registration fee. WhatsApp us."
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from scamscreen.ingest import load_postings, posting_from_text
from scamscreen.pipeline import apply_calibration, get_raw, score_all, summarize, write_outputs

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help=".csv / .json / .jsonl file, or a folder of .txt postings")
    ap.add_argument("--text", help="screen one pasted posting")
    ap.add_argument("--limit", type=int, help="only screen the first N postings")
    ap.add_argument("--engine", choices=["laya", "http", "mock"], help="override engine.type in config.yaml")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--base-url", help="http engine: server base URL (implies --engine http)")
    ap.add_argument("--raw", help="reuse model answers cached in a raw.json (skips inference)")
    ap.add_argument("--calibration", help="calibration.yaml from scripts/diagnose.py")
    ap.add_argument("--label", help="http engine: name shown in the dashboard, e.g. 'jev'")
    ap.add_argument("--jev", action="store_true",
                    help="use config.yaml's jev_preset (Jev via OpenRouter, no waitlist). "
                         "Needs: export OPENROUTER_API_KEY=sk-...")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.jev:
        cfg["engine"]["http"] = {**cfg["engine"].get("http", {}), **cfg["jev_preset"]}
        args.engine = "http"
        if not os.environ.get(cfg["jev_preset"]["api_key_env"]):
            print(f"!! Set your OpenRouter key first: export {cfg['jev_preset']['api_key_env']}=sk-...")
            return 1
    if args.base_url:
        cfg["engine"]["http"]["base_url"] = args.base_url
        args.engine = args.engine or "http"
    if args.label:
        cfg["engine"]["http"]["label"] = args.label

    if args.text:
        postings = [posting_from_text(args.text)]
    elif args.input:
        postings = load_postings(args.input, args.limit)
    else:
        ap.error("give --input or --text")
    if not postings:
        print("No postings found in the input.", file=sys.stderr)
        return 1

    if args.calibration:
        apply_calibration(cfg, args.calibration)
    print(f"Loaded {len(postings)} postings." + ("" if args.raw else " Loading engine…"))
    raw, timing, engine = get_raw(postings, cfg, args.engine, args.raw, Path(args.out) / "raw.json")
    records = score_all(postings, raw, cfg)
    summary = summarize(records, timing, engine)
    paths = write_outputs(records, summary, args.out)

    v = summary["verdicts"]
    print(f"\nScreened {timing['postings']} postings in {timing['engine_seconds']}s "
          f"({timing['ms_per_posting']} ms each, {timing['postings_per_second']}/s)  engine={engine.name}")
    print(f"Likely scam: {v['likely_scam']}   Check first: {v['check']}   Looks OK: {v['looks_ok']}")
    if "labelled" in summary:
        print("Labelled check:", summary["labelled"])
    if args.text:
        r = records[0]
        print(f"\nRisk {r['risk']:.2f} -> {r['verdict']}  type={r['scam_type']}")
        for s in r["signals"][:6]:
            print(f"  - {s['label']} ({s['contribution']}) {s['evidence'] or ''}")
    print("\nWrote:", *paths.values(), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

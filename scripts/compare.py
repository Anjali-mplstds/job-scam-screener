#!/usr/bin/env python3
"""Screen the SAME postings with Laya and Jev and show them side by side.

  export OPENROUTER_API_KEY=sk-...          # openrouter.ai/settings/keys, no TypeSafe waitlist needed
  python scripts/compare.py --input data/sample_postings.jsonl --limit 10

Keep --limit small (5-20): this calls Jev live, and it's meant to illustrate how the two models
answer the same input, not to be a rigorous benchmark. For that, run scripts/evaluate.py on each
engine separately over hundreds of labelled postings, then compare the printed AUROC.

Reuses a cached outputs/raw.json for Laya if present (--laya-raw), so you don't have to wait for
CPU inference again just to get the comparison view.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scamscreen.ingest import load_postings, posting_from_text  # noqa: E402
from scamscreen.pipeline import get_raw, score_all  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--text", help="compare one pasted posting instead of a file")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--config", default=str(OUT_ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(OUT_ROOT / "outputs" / "compare"))
    ap.add_argument("--laya-raw", help="reuse a cached outputs/raw.json instead of running Laya again")
    ap.add_argument("--laya-engine", choices=["laya", "mock"], default="laya",
                    help="use 'mock' to test this script's plumbing without either model")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.text:
        postings = [posting_from_text(a.text)]
    else:
        if not a.input:
            ap.error("give --input or --text")
        postings = load_postings(a.input, a.limit)
    if not postings:
        raise SystemExit("No postings found.")
    print(f"Comparing {len(postings)} postings on Laya and Jev. Jev is called live over the network, "
          f"so this is illustrative, not a benchmark - see the module docstring.")

    print("\n-- Laya --")
    raw_a, timing_a, engine_a = get_raw(postings, cfg, a.laya_engine, a.laya_raw)
    recs_a = score_all(postings, raw_a, cfg)

    print("-- Jev (via OpenRouter) --")
    jcfg = {**cfg, "engine": {**cfg["engine"], "http": {**cfg["engine"].get("http", {}), **cfg["jev_preset"]}}}
    raw_b, timing_b, engine_b = get_raw(postings, jcfg, "http")
    recs_b = score_all(postings, raw_b, jcfg)

    rows = []
    agree = 0
    for i, (p, ra, rb) in enumerate(zip(postings, recs_a, recs_b)):
        same = ra["verdict"] == rb["verdict"]
        agree += same
        rows.append({"i": i, "id": p["id"], "title": p["title"] or "(untitled)", "company": p["company"],
                    "snippet": p["description"][:220], "label": p["label"],
                    "a": {"risk": ra["risk"], "verdict": ra["verdict"],
                          "top_signals": [s["label"] for s in ra["signals"][:3]]},
                    "b": {"risk": rb["risk"], "verdict": rb["verdict"],
                          "top_signals": [s["label"] for s in rb["signals"][:3]]},
                    "agree": same})

    summary = {"engine_a": engine_a.name, "engine_b": engine_b.name,
              "timing_a": timing_a, "timing_b": timing_b,
              "n": len(rows), "agree": agree, "agree_pct": round(100 * agree / max(len(rows), 1), 1)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))

    tpl = (OUT_ROOT / "dashboard" / "compare_template.html").read_text(encoding="utf-8")
    blob = json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False).replace("</", "<\\/")
    (out / "compare.html").write_text(tpl.replace("/*__DATA__*/null", blob), encoding="utf-8")

    print(f"\n{engine_a.name}: {timing_a['ms_per_posting']} ms/posting   "
          f"{engine_b.name}: {timing_b['ms_per_posting']} ms/posting")
    print(f"Same verdict on {agree}/{len(rows)} postings ({summary['agree_pct']}%)")
    print(f"\nWrote:\n  {out / 'compare.json'}\n  {out / 'compare.html'}")


if __name__ == "__main__":
    main()

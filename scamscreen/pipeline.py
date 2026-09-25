"""ingest -> clean -> rules -> engine (timed) -> score -> report."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

from .engine import CachedEngine, Engine, make_engine
from .ingest import build_state, full_text
from .rules import find_hits
from .scoring import VERDICT_TEXT, cfg_with_defaults, score_posting

ROOT = Path(__file__).resolve().parent.parent


def screen_postings(postings: list[dict], engine: Engine, cfg: dict, warmup: bool = True) -> tuple[list[dict], dict]:
    """Run the whole checklist over `postings`. Returns (raw model outputs, timing dict)."""
    chars = int(cfg.get("state_chars", 900))
    bs = int(cfg.get("engine", {}).get("laya", {}).get("batch_size", 16))
    states = [build_state(p, chars) for p in postings]
    if warmup and len(states) > 8:
        engine.warmup(states)                      # first call pays one-off load/compile costs
    t0 = time.perf_counter()
    raw = engine.screen(states, batch_size=bs)
    dt = time.perf_counter() - t0
    n = max(len(states), 1)
    timing = {"engine_seconds": round(dt, 3), "postings": len(states),
              "ms_per_posting": round(1000 * dt / n, 2), "postings_per_second": round(n / dt, 2) if dt else None,
              "batch_size": bs}
    return raw, timing


def score_all(postings: list[dict], raw: list[dict], cfg: dict) -> list[dict]:
    scfg = cfg_with_defaults(cfg.get("scoring"))
    records = []
    for i, (p, r) in enumerate(zip(postings, raw)):
        hits = find_hits(full_text(p), p["company"])
        s = score_posting(r, hits, scfg)
        records.append({
            "i": i, "id": p["id"], "title": p["title"] or "(untitled)", "company": p["company"] or "not stated",
            "location": p["location"], "snippet": p["description"][:240],
            "label": p["label"], **s,
        })
    return records


def summarize(records: list[dict], timing: dict, engine: Engine) -> dict:
    verdicts = Counter(r["verdict"] for r in records)
    sig = Counter(s["label"] for r in records if r["verdict"] != "looks_ok" for s in r["signals"])
    types = Counter(r["scam_type"] for r in records if r["scam_type"])
    summary = {
        "engine": engine.name, "is_mock": engine.is_mock, "hardware": engine.info(), "timing": timing,
        "total": len(records), "verdicts": {k: verdicts.get(k, 0) for k in VERDICT_TEXT},
        "top_signals": sig.most_common(8), "scam_types": dict(types),
    }
    labelled = [r for r in records if r["label"] is not None]
    if labelled:
        tp = sum(1 for r in labelled if r["label"] == 1 and r["verdict"] != "looks_ok")
        fn = sum(1 for r in labelled if r["label"] == 1 and r["verdict"] == "looks_ok")
        fp = sum(1 for r in labelled if r["label"] == 0 and r["verdict"] != "looks_ok")
        tn = sum(1 for r in labelled if r["label"] == 0 and r["verdict"] == "looks_ok")
        summary["labelled"] = {"n": len(labelled), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                               "note": "flagged = 'Check first' or 'Likely scam'"}
    return summary


def write_outputs(records: list[dict], summary: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": out / "results.jsonl", "csv": out / "results.csv",
             "summary": out / "summary.json", "dashboard": out / "dashboard.html"}
    with paths["jsonl"].open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "title", "company", "risk", "verdict", "scam_type", "top_signals", "label"])
        for r in records:
            w.writerow([r["id"], r["title"], r["company"], r["risk"], r["verdict"], r["scam_type"] or "",
                        "; ".join(s["label"] for s in r["signals"][:4]), "" if r["label"] is None else r["label"]])
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    data = {"summary": summary, "verdict_text": VERDICT_TEXT, "records": sorted(records, key=lambda r: -r["risk"])}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    tpl = (ROOT / "dashboard" / "dashboard_template.html").read_text(encoding="utf-8")
    paths["dashboard"].write_text(tpl.replace("/*__DATA__*/null", blob), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def save_raw(path, postings, raw, timing, engine) -> None:
    """Cache the model's answers so re-scoring and calibration never need another inference run."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"engine": engine.name, "is_mock": engine.is_mock, "hardware": engine.info(),
                                      "timing": timing, "ids": [p["id"] for p in postings], "raw": raw}))


def load_raw(path, postings):
    d = json.loads(Path(path).read_text())
    if d["ids"] != [p["id"] for p in postings]:
        raise SystemExit("raw cache doesn't match these postings (different input file or --limit). "
                         "Re-run without --raw.")
    return d["raw"], d["timing"], CachedEngine(d["engine"], d["is_mock"], d["hardware"])


def get_raw(postings, cfg, engine_kind=None, raw_path=None, save_to=None):
    """Return (raw answers, timing, engine): from a cache if given, else by running the engine."""
    if raw_path:
        return load_raw(raw_path, postings)
    engine = make_engine(cfg, engine_kind)
    if engine.is_mock:
        print("!! MOCK engine: keyword heuristics, NOT Laya. Do not publish these results as Laya output.")
    raw, timing = screen_postings(postings, engine, cfg)
    if save_to:
        save_raw(save_to, postings, raw, timing, engine)
    return raw, timing, engine


def apply_calibration(cfg: dict, path) -> dict:
    """Merge a calibration.yaml written by scripts/diagnose.py into cfg['scoring']."""
    import yaml
    cal = yaml.safe_load(Path(path).read_text()) or {}
    cfg.setdefault("scoring", {}).update(cal.get("scoring", {}))
    return cfg

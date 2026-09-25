"""Turn model probabilities + rule hits into a 0-1 risk score, a verdict and a list of reasons.

Design notes
* Laya ships over-confident (see the model card), so every P(yes) passes through temperature
  scaling first. `scripts/evaluate.py --calibrate` fits a temperature on your labelled data.
* A check only counts for the amount its probability is ABOVE a coin flip: c = weight * max(0, 2p-1).
  A near-chance model (p ~ 0.5 everywhere) therefore contributes ~0 instead of flagging everything.
* Model evidence and rule evidence for the same check are merged with a noisy-OR, then all checks
  are merged with a noisy-OR, so several weak flags add up while one weak flag stays weak.
"""
from __future__ import annotations

import math

from .questions import CHECKS, PROTECT, SCAM_TYPE_ID
from .rules import RULE_ONLY

DEFAULTS = {
    "temperature": 1.0,
    "rule_scale": 0.6,      # how much a regex hit alone is worth relative to the check's weight
    "protect_scale": 0.5,   # how much a clear, verifiable role can reduce the risk
    "hard_floor": 0.75,     # minimum risk when a hard check fires confidently
    "fire_at": 0.15,        # contribution needed to list a signal as a reason
    "thresholds": {"check": 0.30, "scam": 0.60},
    "question_bias": {},    # per-check logit offset subtracted from the model's answer (scripts/diagnose.py)
    "disabled": [],         # checks whose model answer is ignored (rules still count)
}

VERDICTS = ("looks_ok", "check", "likely_scam")
VERDICT_TEXT = {"looks_ok": "Looks OK", "check": "Check first", "likely_scam": "Likely scam"}


def cfg_with_defaults(cfg: dict | None) -> dict:
    out = {**DEFAULTS, **(cfg or {})}
    out["thresholds"] = {**DEFAULTS["thresholds"], **(out.get("thresholds") or {})}
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def temper(p: float, temperature: float) -> float:
    """Temperature-scale a yes/no probability (identical to a 2-option softmax at T)."""
    return 1.0 / (1.0 + math.exp(-_logit(p) / max(temperature, 1e-3)))


def adjust(p: float, temperature: float, bias: float = 0.0) -> float:
    """Remove a question's typical 'yes' level (bias, in logit units), then temperature-scale."""
    return 1.0 / (1.0 + math.exp(-(_logit(p) - bias) / max(temperature, 1e-3)))


def _excess(p: float) -> float:
    return max(0.0, 2.0 * p - 1.0)


def verdict_for(risk: float, thresholds: dict) -> str:
    if risk >= thresholds["scam"]:
        return "likely_scam"
    if risk >= thresholds["check"]:
        return "check"
    return "looks_ok"


def score_posting(raw: dict, hits: dict[str, str], cfg: dict | None = None) -> dict:
    cfg = cfg_with_defaults(cfg)
    T, rs = cfg["temperature"], cfg["rule_scale"]
    bias, off = cfg["question_bias"], set(cfg["disabled"])
    p_raw: dict = {k: v for k, v in raw.get("p", {}).items() if k not in off}
    signals, survivors, hard_fired = [], [], False

    for c in CHECKS:
        pm = adjust(p_raw[c.id], T, bias.get(c.id, 0.0)) if c.id in p_raw else None
        cm = c.weight * _excess(pm) if pm is not None else 0.0
        cr = c.weight * rs if c.id in hits else 0.0
        contrib = 1 - (1 - cm) * (1 - cr)
        survivors.append(1 - contrib)
        if c.hard and contrib >= 0.6:
            hard_fired = True
        if contrib >= cfg["fire_at"]:
            signals.append({"id": c.id, "label": c.label, "contribution": round(contrib, 3),
                            "p_model": None if pm is None else round(pm, 3),
                            "rule": c.id in hits, "evidence": hits.get(c.id)})

    for rid, (label, w) in RULE_ONLY.items():
        if rid in hits:
            contrib = w * rs
            survivors.append(1 - contrib)
            signals.append({"id": rid, "label": label, "contribution": round(contrib, 3),
                            "p_model": None, "rule": True, "evidence": hits[rid]})

    risk = 1.0 - math.prod(survivors)
    if hard_fired:
        risk = max(risk, cfg["hard_floor"])
    elif PROTECT.id in p_raw:
        risk *= 1.0 - cfg["protect_scale"] * _excess(adjust(p_raw[PROTECT.id], T, bias.get(PROTECT.id, 0.0)))
    risk = min(max(risk, 0.0), 1.0)

    signals.sort(key=lambda s: -s["contribution"])
    verdict = verdict_for(risk, cfg["thresholds"])
    st = raw.get("scam_type")
    scam_type = None
    if st and verdict != "looks_ok" and st["choice"] != "genuine":
        scam_type = st["choice"]
    return {"risk": round(risk, 4), "verdict": verdict, "signals": signals,
            "scam_type": scam_type, "scam_type_probs": (st or {}).get("probabilities")}

"""Load job postings from CSV / JSON / JSONL / a folder of .txt files and normalise them.

Column names are matched loosely, so the public "Fake Job Postings" (EMSCAD) CSV, a LinkedIn or
Indeed export you own, or your own JSONL all load without renaming. Only `description`-like text is
required; everything else is optional. A `label` / `fraudulent` / `is_scam` column (0/1) is kept for
evaluation.
"""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

from .rules import SIGNAL_RE, emails_in

ALIASES = {
    "id": ["id", "job_id", "posting_id", "url", "link"],
    "title": ["title", "job_title", "position", "role"],
    "company": ["company", "company_name", "employer", "organization", "organisation"],
    "company_profile": ["company_profile", "about_company", "about"],
    "location": ["location", "city", "job_location"],
    "salary": ["salary", "salary_range", "pay", "compensation"],
    "description": ["description", "job_description", "text", "body", "content", "details"],
    "requirements": ["requirements", "qualifications"],
    "benefits": ["benefits", "perks"],
    "contact": ["contact", "apply", "how_to_apply", "email"],
    "label": ["label", "fraudulent", "is_scam", "scam", "fake"],
}

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = _TAG.sub(" ", text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _pick(row: dict, key: str) -> str:
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for name in ALIASES[key]:
        v = lower.get(name)
        if v is not None and str(v).strip() and str(v).strip().lower() != "nan":
            return str(v)
    return ""


def _label(v: str):
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "fraud", "fake", "scam", "1.0"}:
        return 1
    if v in {"0", "false", "no", "real", "legit", "genuine", "0.0"}:
        return 0
    return None


def normalize_row(row: dict, idx: int) -> dict:
    body = "\n".join(x for x in (_pick(row, "description"), _pick(row, "requirements"),
                                 _pick(row, "benefits")) if x)
    company = _pick(row, "company")
    profile = _pick(row, "company_profile")
    posting = {
        "id": _pick(row, "id") or f"posting-{idx}",
        "title": normalize(_pick(row, "title")),
        "company": normalize(company),
        "company_profile": normalize(profile)[:400],
        "location": normalize(_pick(row, "location")),
        "salary": normalize(_pick(row, "salary")),
        "description": normalize(body),
        "contact": normalize(_pick(row, "contact")),
        "label": _label(_pick(row, "label")),
    }
    return posting


def load_postings(path: str | Path, limit: int | None = None) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    if p.is_dir():
        for i, f in enumerate(sorted(p.glob("*.txt"))):
            rows.append({"id": f.stem, "description": f.read_text(encoding="utf-8", errors="ignore")})
    elif p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8", errors="ignore") as fh:
            rows = list(csv.DictReader(fh))
    elif p.suffix.lower() == ".jsonl":
        with p.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("postings", [])
    else:
        raise ValueError(f"Unsupported input: {p} (use .csv, .json, .jsonl or a folder of .txt)")
    if limit:
        rows = rows[:limit]
    postings = [normalize_row(r, i) for i, r in enumerate(rows)]
    return [x for x in postings if x["description"] or x["title"]]


def posting_from_text(text: str) -> dict:
    return normalize_row({"description": text}, 0)


def full_text(posting: dict) -> str:
    """Everything we know about the posting, for the rule layer."""
    parts = [posting["title"], posting["company"], posting["company_profile"], posting["salary"],
             posting["contact"], posting["description"]]
    return "\n".join(x for x in parts if x)


def condense(text: str, max_chars: int) -> str:
    """Fit a long description into the model's token budget without losing the risky sentences.

    Keeps the opening (what the job is) and then every sentence that matches a red-flag pattern
    (fees, contact channels, pay claims, urgency...), in original order, until the budget is spent.
    """
    text = normalize(text)
    if len(text) <= max_chars:
        return text
    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    keep: set[int] = set()
    used = 0
    for i, s in enumerate(sents):                       # opening, up to 40% of the budget
        if used + len(s) > max_chars * 0.4:
            break
        keep.add(i)
        used += len(s) + 1
    for i, s in enumerate(sents):                       # then the high-signal sentences
        if i not in keep and SIGNAL_RE.search(s) and used + len(s) <= max_chars:
            keep.add(i)
            used += len(s) + 1
    for i, s in enumerate(sents):                       # then fill leftover space in order
        if i not in keep and used + len(s) <= max_chars:
            keep.add(i)
            used += len(s) + 1
    return " ".join(sents[i] for i in sorted(keep))[:max_chars]


def build_state(posting: dict, max_chars: int = 900) -> dict:
    """The JSON state the model reads. Field order puts identity + contact before the long text."""
    company = posting["company"] or (posting["company_profile"][:120] if posting["company_profile"] else "")
    contact = ", ".join(emails_in(full_text(posting))) or "none listed"
    state = {
        "title": posting["title"] or "not stated",
        "company": company or "not stated",
        "contact": contact,
    }
    if posting["location"]:
        state["location"] = posting["location"]
    if posting["salary"]:
        state["salary"] = posting["salary"]
    state["description"] = condense(posting["description"], max_chars)
    return state

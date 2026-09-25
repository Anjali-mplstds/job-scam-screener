"""Cheap deterministic red-flag rules.

They run on the FULL posting text (the model only sees a condensed version that fits its token
budget), and they give the dashboard concrete evidence snippets to show next to each flag.
Rule ids that match a checklist id in questions.py are combined with the model's answer for that
check; the rule-only ids at the bottom have no model question.
"""
from __future__ import annotations

import re

FREEMAIL = {"gmail.com", "yahoo.com", "yahoo.in", "outlook.com", "hotmail.com", "live.com",
            "aol.com", "proton.me", "protonmail.com", "icloud.com", "rediffmail.com",
            "mail.com", "gmx.com", "yandex.com"}

_MONEY = r"(?:\$|usd\s?|₹|rs\.?\s?|inr\s?|£|€)\s?"

PATTERNS: dict[str, str] = {
    "upfront_fee": (
        r"(?:registration|processing|training|application|admin(?:istration)?|onboarding|"
        r"security|refundable|verification)\s+(?:fee|deposit|charge)"
        r"|pay\s+(?:for\s+)?(?:your\s+)?(?:own\s+)?(?:equipment|laptop|kit|uniform|"
        r"background check|training|starter pack)"
        r"|(?:send|transfer|deposit|pay)\s+(?:us\s+)?" + _MONEY + r"\d+"
        r"|(?:fee|deposit)\s+(?:is\s+)?(?:fully\s+)?refundable"
    ),
    "sensitive_info": (
        r"social security|\bssn\b|bank account (?:number|details|info)|account and routing|"
        r"copy of (?:your )?(?:passport|id|driver|licen[cs]e)|scan of (?:your )?(?:passport|id)|"
        r"aadhaar|aadhar|\bpan card\b|credit card (?:number|details)|"
        r"photo of (?:your )?(?:id|passport)"
    ),
    "money_handling": (
        r"process(?:ing)? payments? (?:for|from|on behalf)|receive payments?|cash(?:ing)? (?:a )?che(?:ck|que)s?|"
        r"forward (?:the )?(?:funds|money|payments?)|re-?ship|repackag|package forwarding|"
        r"payment (?:processor|agent) from home|transfer (?:the )?funds|mystery shopper.{0,40}che(?:ck|que)"
    ),
    "task_scam": (
        r"(?:like|subscribe|follow)(?:\s+and\s+\w+)?\s+(?:videos|channels|posts|pages)|"
        r"rate\s+(?:products|apps|hotels|items)|boost\s+(?:ratings?|rankings?|sales|traffic)|"
        r"(?:complete|finish|do)\s+(?:simple\s+|easy\s+)?(?:online\s+)?tasks?|commission per task|"
        r"optimi[sz]e\s+(?:apps|products|orders)|(?:daily|simple)\s+tasks?"
    ),
    "offplatform": (
        r"whats\s?app|telegram|\bsignal\s+(?:app|number)|wechat|text\s+(?:us|me)\s+(?:at|on)|"
        r"hangouts|contact\s+(?:us|me)\s+(?:only\s+)?(?:on|via|through)\s+(?:whats|tele)|"
        r"add\s+(?:us|me|our recruiter)\s+on"
    ),
    "too_good": (
        r"earn\s+(?:up\s+to\s+)?" + _MONEY + r"?\d[\d,]{2,}\s*(?:per|a|/)\s*(?:day|hour|hr|week)|"
        r"(?:make|get\s+paid)\s+" + _MONEY + r"?\d[\d,]{2,}\s*(?:per|a|/)\s*(?:day|hour|hr|week)|"
        r"guaranteed\s+(?:income|earnings|salary)|earn\s+(?:big|fast|quick)|"
        r"(?:daily|weekly)\s+(?:payout|pay)s?\s+guaranteed"
    ),
    "no_interview": (
        r"no\s+interview|without\s+(?:an\s+)?interview|hired\s+(?:instantly|immediately|on the spot)|"
        r"instant(?:ly)?\s+hir|no\s+experience\s+(?:needed|required|necessary)|"
        r"no\s+(?:skills|qualification)s?\s+(?:needed|required)"
    ),
    "urgency": (
        r"urgent(?:ly)?\s+hiring|apply\s+(?:now|immediately|today)|hiring\s+immediately|"
        r"limited\s+(?:slots|positions|spots|vacancies)|only\s+\d+\s+(?:slots|spots|positions)|"
        r"start\s+(?:today|immediately|tomorrow)|act\s+(?:now|fast)|(?:respond|reply)\s+within\s+\d+\s+hours?"
    ),
}

# Rule-only signals (no model question): id -> (label, weight)
RULE_ONLY: dict[str, tuple[str, float]] = {
    "freemail_contact": ("Recruiter uses a free email address", 0.35),
    "domain_mismatch": ("Email domain doesn't match the company", 0.30),
}

_COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
SIGNAL_RE = re.compile("|".join(f"(?:{p})" for p in PATTERNS.values()), re.I)


def _snippet(text: str, m: re.Match, pad: int = 55) -> str:
    a, b = max(0, m.start() - pad), min(len(text), m.end() + pad)
    s = text[a:b].replace("\n", " ").strip()
    return ("…" if a > 0 else "") + s + ("…" if b < len(text) else "")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def emails_in(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in EMAIL_RE.finditer(text)})


def find_hits(text: str, company: str = "") -> dict[str, str]:
    """Return {rule_id: evidence snippet} for every rule that fires on `text`."""
    hits: dict[str, str] = {}
    for rid, rx in _COMPILED.items():
        m = rx.search(text)
        if m:
            hits[rid] = _snippet(text, m)

    for m in EMAIL_RE.finditer(text):
        domain = m.group(1).lower()
        if domain in FREEMAIL:
            hits.setdefault("freemail_contact", _snippet(text, m, 30))
        elif company and len(company) <= 60:
            base = _slug(domain.split(".")[0])
            name = _slug(company)
            tokens = [t for t in re.split(r"\W+", company.lower()) if len(t) >= 4]
            if base and name and base not in name and name not in base \
                    and not any(t in base for t in tokens):
                hits.setdefault("domain_mismatch", _snippet(text, m, 30))
    return hits

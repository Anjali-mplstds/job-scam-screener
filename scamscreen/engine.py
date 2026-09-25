"""Decision engines. All of them answer the same checklist and return the same shape:

    [{"p": {check_id: P(yes)}, "scam_type": {"choice": str, "probabilities": {...}} | None}, ...]

* LayaEngine  - runs Laya locally through the `laya` package (Router or a single checkpoint).
* HttpEngine  - talks to POST /v1/systemone: a `laya-serve` instance, or any Jev-compatible API.
                Point it at Jev and at laya-serve in turn to compare both on identical postings.
* MockEngine  - keyword heuristics, no model download. For testing the pipeline and dashboard only.
                Its output is NOT Laya output and is labelled as such everywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .questions import CHECKS, PROTECT, SCAM_TYPES, build_questions, parse_answers
from .rules import find_hits


class Engine:
    name = "base"
    is_mock = False

    def screen(self, states: list[dict], batch_size: int = 16) -> list[dict]:
        raise NotImplementedError

    def warmup(self, states: list[dict]) -> None:
        self.screen(states[:4], batch_size=4)

    def info(self) -> dict:
        return {"engine": self.name, "python": platform.python_version(), "machine": platform.machine()}


class LayaEngine(Engine):
    def __init__(self, checkpoint: str = "router", repo: str = "convaiinnovations/laya",
                 device: str | None = None, mode: str = "ab"):
        try:
            import laya  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise SystemExit("The `laya` package is not installed. Run: pip install -U laya "
                             "(needs Python 3.10+). Or use --engine mock to try the pipeline offline.") from e
        self.mode = mode
        self.checkpoint = checkpoint
        self.questions = build_questions(mode)
        self.name = f"laya:{checkpoint}"
        self.device = device
        if checkpoint == "router":
            from laya import Router
            kw = {"preload": True}
            if device:
                kw["device"] = device
            self.router = Router(**kw)
            self.agent = None
        else:
            sub = {"english": None, "multilingual": "multilingual",
                   "typed-decisions": "typed-decisions"}[checkpoint]
            kw = {"device": device} if device else {}
            self.agent = laya.load(repo, subfolder=sub, **kw) if sub else laya.load(repo, **kw)
            self.router = None

    def screen(self, states: list[dict], batch_size: int = 16) -> list[dict]:
        if self.router is not None:
            reqs = [{"state": s, "questions": self.questions} for s in states]
            results = self.router.predict_batch(reqs, batch_size=batch_size)
        else:
            results = self.agent.predict_batch(states, self.questions, batch_size=batch_size)
        return [parse_answers(r["answers"], self.mode) for r in results]

    def info(self) -> dict:
        d = super().info()
        try:
            import torch
            d["device"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
            d["torch"] = torch.__version__
        except Exception:  # pragma: no cover
            d["device"] = self.device or "unknown"
        try:
            import laya
            d["laya"] = getattr(laya, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass
        return d


class HttpEngine(Engine):
    """POSTs {state, questions} (plus `model`, if set) to a System One-compatible endpoint.

    Works unmodified against: TypeSafe's own API, `laya-serve`, and OpenRouter's System One
    endpoint (https://openrouter.ai/api/v1/systemone) with model="typesafe/jev-1.13" and an
    OpenRouter API key - no TypeSafe waitlist needed. See config.yaml's `http:` presets.
    """
    def __init__(self, base_url: str, endpoint: str = "/v1/systemone", api_key_env: str = "SCREENER_API_KEY",
                 workers: int = 8, timeout_s: int = 30, mode: str = "ab", label: str | None = None,
                 model: str | None = None):
        self.url = base_url.rstrip("/") + endpoint
        self.key = os.environ.get(api_key_env)
        if not self.key:
            print(f"!! No API key found in ${api_key_env}. Set it, e.g.: export {api_key_env}=sk-...")
        self.workers, self.timeout, self.mode, self.model = workers, timeout_s, mode, model
        self.questions = build_questions(mode)
        self.name = label or (model or f"http:{base_url}")

    def _one(self, state: dict) -> dict:
        payload = {"state": state, "questions": self.questions}
        if self.model:
            payload["model"] = self.model
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            raise SystemExit(f"{self.name} request failed: HTTP {e.code}\n{detail}") from None
        return parse_answers(data["answers"], self.mode)

    def screen(self, states: list[dict], batch_size: int = 16) -> list[dict]:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._one, states))


class MockEngine(Engine):
    """Keyword heuristics standing in for the model. Deterministic. NOT a Laya result."""
    name = "mock (NOT Laya)"
    is_mock = True

    def screen(self, states: list[dict], batch_size: int = 16) -> list[dict]:
        out = []
        for s in states:
            text = " ".join(str(v) for v in s.values())
            hits = find_hits(text, str(s.get("company", "")))
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            noise = lambda i: ((h >> (i * 3)) % 100) / 1000.0  # 0..0.099, stable per posting
            p = {}
            for i, c in enumerate(CHECKS):
                p[c.id] = 0.90 - noise(i) if c.id in hits else 0.08 + noise(i)
            if str(s.get("company", "")).strip().lower() in {"", "not stated"}:
                p["vague_company"] = 0.85
            good = len(str(s.get("description", ""))) > 350 and not hits
            p[PROTECT.id] = 0.85 if good else 0.30
            top = next((c for c in ("upfront_fee", "task_scam", "money_handling", "sensitive_info") if c in hits), None)
            mapping = {"upfront_fee": "advance_fee", "task_scam": "task_scam",
                       "money_handling": "money_mule", "sensitive_info": "data_harvest"}
            choice = mapping.get(top, "genuine")
            probs = {k: (0.7 if k == choice else 0.3 / (len(SCAM_TYPES) - 1)) for k in SCAM_TYPES}
            out.append({"p": p, "scam_type": {"choice": choice, "probabilities": probs}})
        return out


class CachedEngine(Engine):
    """Stands in for the engine that produced a saved raw.json (so results are labelled correctly)."""
    def __init__(self, name: str, is_mock: bool, hardware: dict):
        self.name, self.is_mock, self._hw = name, is_mock, hardware

    def info(self) -> dict:
        return self._hw


def make_engine(cfg: dict, override: str | None = None) -> Engine:
    kind = override or cfg["engine"]["type"]
    mode = cfg.get("question_mode", "ab")
    if kind == "mock":
        return MockEngine()
    if kind == "laya":
        c = cfg["engine"]["laya"]
        return LayaEngine(c.get("checkpoint", "router"), c.get("repo", "convaiinnovations/laya"),
                          c.get("device"), mode)
    if kind == "http":
        c = cfg["engine"]["http"]
        return HttpEngine(c["base_url"], c.get("endpoint", "/v1/systemone"),
                          c.get("api_key_env", "SCREENER_API_KEY"), c.get("workers", 8),
                          c.get("timeout_s", 30), mode, c.get("label"), c.get("model"))
    raise SystemExit(f"Unknown engine type: {kind}")

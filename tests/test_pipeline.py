import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scamscreen.engine import MockEngine
from scamscreen.ingest import build_state, condense, load_postings, posting_from_text
from scamscreen.pipeline import score_all, screen_postings, summarize, write_outputs
from scamscreen.questions import CHECKS, build_questions, parse_answers
from scamscreen.rules import find_hits
from scamscreen.scoring import score_posting, temper
from scamscreen.synth import generate

CFG = {"engine": {"laya": {"batch_size": 8}}, "state_chars": 900}
SCAM = "Earn $500 a day. Pay a $50 registration fee. Contact us on WhatsApp or boss@gmail.com. No interview!"


def test_rules_find_evidence():
    hits = find_hits(SCAM, "")
    assert {"upfront_fee", "offplatform", "freemail_contact", "no_interview"} <= set(hits)
    assert "registration fee" in hits["upfront_fee"]


def test_clean_posting_has_no_hits():
    assert find_hits("Accounts Assistant. Apply via careers.acme.example. Two interview rounds.", "Acme") == {}


def test_condense_keeps_risky_sentence_from_the_tail():
    text = " ".join(f"Filler sentence number {i} about the office." for i in range(60)) + \
        " Send a registration fee of $99 via WhatsApp."
    out = condense(text, 500)
    assert len(out) <= 500 and "registration fee" in out


def test_state_has_contact_and_budget():
    st = build_state(posting_from_text(SCAM * 40), 900)
    assert "boss@gmail.com" in st["contact"] and len(st["description"]) <= 900


def test_questions_shapes():
    q = build_questions("ab")
    assert q["upfront_fee"]["criteria"].keys() == {"A", "B"} and q["scam_type"]["type"] == "choice"
    assert build_questions("noul")["upfront_fee"]["type"] == "noul"
    ans = {"upfront_fee": {"probabilities": {"A": 0.9, "B": 0.1}}}
    assert parse_answers(ans)["p"]["upfront_fee"] == 0.9


def test_near_chance_model_does_not_flag_everything():
    raw = {"p": {c.id: 0.5 for c in CHECKS}}
    assert score_posting(raw, {})["verdict"] == "looks_ok"


def test_confident_hard_check_forces_high_risk():
    raw = {"p": {"upfront_fee": 0.95}}
    r = score_posting(raw, {})
    assert r["verdict"] == "likely_scam" and r["risk"] >= 0.75


def test_temperature_softens():
    assert 0.5 < temper(0.95, 3.0) < 0.95


def test_end_to_end_mock(tmp_path):
    rows = generate(60, seed=1)
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    postings = load_postings(p)
    eng = MockEngine()
    raw, timing = screen_postings(postings, eng, CFG)
    recs = score_all(postings, raw, CFG)
    summ = summarize(recs, timing, eng)
    paths = write_outputs(recs, summ, tmp_path / "out")
    assert summ["total"] == 60 and summ["is_mock"] is True
    html = Path(paths["dashboard"]).read_text()
    assert "/*__DATA__*/null" not in html and "not Laya" in html


def test_laya_engine_against_stubbed_package(monkeypatch):
    """Checks LayaEngine drives Router.predict_batch / Agent.predict_batch and parses their result shape."""
    import types
    from scamscreen.engine import LayaEngine

    def fake_result(questions):
        ans = {}
        for qid, q in questions.items():
            keys = list(q["criteria"])
            ans[qid] = {"type": "choice", "choice": keys[0], "confidence": 0.9,
                        "probabilities": {k: (0.9 if i == 0 else 0.1 / (len(keys) - 1)) for i, k in enumerate(keys)}}
        return {"model": "stub", "answers": ans}

    class Router:
        def __init__(self, **kw): pass
        def predict_batch(self, reqs, batch_size=None): return [fake_result(r["questions"]) for r in reqs]

    class Agent:
        def predict_batch(self, states, questions, batch_size=None): return [fake_result(questions) for _ in states]

    stub = types.SimpleNamespace(Router=Router, load=lambda repo, **kw: Agent(), __version__="stub")
    monkeypatch.setitem(sys.modules, "laya", stub)
    for ckpt in ("router", "english"):
        out = LayaEngine(ckpt).screen([{"title": "x"}, {"title": "y"}])
        assert len(out) == 2 and out[0]["p"]["upfront_fee"] == 0.9
        assert out[0]["scam_type"]["choice"] == "advance_fee"


def test_bias_correction_stops_a_yes_biased_model_flagging_everything():
    """A model that answers ~0.9 'yes' to everything flags everything until its bias is removed."""
    from scamscreen.scoring import cfg_with_defaults
    biased = {"p": {c.id: 0.9 for c in CHECKS}}
    assert score_posting(biased, {})["verdict"] == "likely_scam"
    fixed = cfg_with_defaults({"question_bias": {c.id: 2.2 for c in CHECKS}})   # logit(0.9) = 2.2
    assert score_posting(biased, {}, fixed)["verdict"] == "looks_ok"
    stronger = {"p": {c.id: 0.99 for c in CHECKS}}                              # clearly above its usual level
    assert score_posting(stronger, {}, fixed)["risk"] > 0.3


def test_raw_cache_roundtrip(tmp_path):
    from scamscreen.pipeline import get_raw
    rows = generate(12, seed=2)
    postings = [posting_from_text(r["description"]) | {"id": r["id"]} for r in rows]
    raw, timing, eng = get_raw(postings, {"engine": {"type": "mock"}, "state_chars": 900}, None, None, tmp_path / "raw.json")
    raw2, _, eng2 = get_raw(postings, {}, None, tmp_path / "raw.json")
    assert raw2 == raw and eng2.is_mock and eng2.name == eng.name


def test_http_engine_sends_model_field_for_gateway_routing(monkeypatch):
    """OpenRouter needs `model` in the body; a plain laya-serve/TypeSafe target has model=None."""
    import urllib.request
    from scamscreen.engine import HttpEngine

    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"answers": {"upfront_fee": {"probabilities": {"A": 0.7, "B": 0.3}}}}).encode()

    def fake_urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        seen["headers"] = dict(req.header_items())
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    eng = HttpEngine("http://x", api_key_env="OPENROUTER_API_KEY", model="typesafe/jev-1.13", label="jev")
    out = eng.screen([{"title": "t"}])
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert out[0]["p"]["upfront_fee"] == 0.7

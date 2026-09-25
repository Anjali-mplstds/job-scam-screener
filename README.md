# Job Posting Scam Screener — Laya vs Jev

A proof-of-concept that screens job postings for scam signals using **Laya**, an open-source,
non-autoregressive decision model (Convai Innovations). It also runs a fair, honest head-to-head
against **Jev** (TypeSafe), a comparable model, on the same real data.

**TL;DR:** Built the full pipeline. Tested it on real, labelled job postings instead of made-up
examples. Found that Laya's zero-shot checklist barely beats random guessing at detecting scams —
and Jev, with no tuning at all, does meaningfully better. Both findings, and why, are below.

## What it does

1. Takes a job posting (title, company, description, etc.).
2. Sends it to a model with a 10-question scam checklist ("Does this ask for a fee up front?",
   "Is the employer identifiable?", "Does it push you off-platform?", etc.) plus a scam-type
   classification — all answered in **one forward pass**, no text generation.
3. Runs a parallel rule-based (regex) layer for evidence snippets.
4. Combines both into a risk score and verdict: **Looks OK / Check first / Likely scam**.
5. Outputs CSV/JSON + an interactive HTML dashboard.

## What we found

Tested on **1,200 real, held-out, labelled job postings** (Kaggle's public "Real / Fake Job Posting
Prediction" / EMSCAD dataset — 28 confirmed scams, 2.3%, matching real-world scam rates), on
comparable server-class hardware (Colab T4 GPU for Laya; TypeSafe's cloud for Jev):

| | AUROC (0.5 = coin flip, 1.0 = perfect) | Speed |
|---|---|---|
| **Laya**, zero-shot, bias-corrected | **0.522** — no better than a coin flip | 297 ms/posting |
| **Rules only** (plain keyword regex, no AI) | 0.523 — about the same as Laya | — |
| **Jev**, zero-shot | **0.601** — real, if modest, signal | 38.5 ms/posting (~7-8x faster) |

**What this means:** Laya's checklist, used zero-shot, isn't detecting job scams meaningfully better
than simple keyword rules on this task. This matches Laya's own documentation, which describes it as
"a fast base to specialize" — its published fine-tuning benchmark shows accuracy jumping from 0.36 to
0.77 after training on domain data. Jev showed real signal here with no tuning at all.

### The bias we found and fixed along the way

Zero-shot, Laya's checklist initially over-flagged almost everything — 22 of 24 genuine test postings
were wrongly flagged. `scripts/diagnose.py` measures each checklist question's own AUROC against
labelled data, detects a built-in "yes"-bias in several questions, and corrects for it. The correction
fixed the over-flagging (stopped it being *actively wrong*), but didn't add real detection ability —
before correction Laya scored 0.429 (worse than chance), after correction 0.522 (about chance).

## Try it yourself

```bash
pip install -r requirements.txt
python scripts/make_demo_data.py --n 500        # synthetic postings for a quick plumbing test
python run_pipeline.py --input data/demo_500.jsonl --engine mock   # no model download, just checks it runs
python run_pipeline.py --input data/demo_500.jsonl                # real Laya
open outputs/dashboard.html
```

For the full test on real data (needs a GPU — see `colab_run.ipynb` for a free-Colab walkthrough):

```bash
python scripts/diagnose.py --input data/fake_job_postings.csv --limit 600     # bias check + fix
python scripts/evaluate.py --input data/holdout.csv --calibration outputs/calibration.yaml
```

For the Laya-vs-Jev comparison (needs an OpenRouter key — no TypeSafe waitlist required,
get one free at openrouter.ai/settings/keys):

```bash
export OPENROUTER_API_KEY=sk-...
python scripts/compare.py --input data/sample_postings.jsonl --limit 20
open outputs/compare/compare.html
```

## Layout

```
config.yaml            engine choice (laya / http / mock), question mode, scoring, jev_preset
run_pipeline.py         main CLI
scamscreen/             questions, rules, ingest, engine, scoring, pipeline
scripts/
  make_demo_data.py     synthetic postings for quick tests
  diagnose.py           per-question AUROC + bias correction against labelled data
  evaluate.py           AUROC / precision / recall / F1 on labelled data
  compare.py            Laya vs Jev, same postings, side by side
dashboard/              HTML templates (data injected at write time)
colab_run.ipynb         full run on a free T4 GPU: diagnose → evaluate → Laya vs Jev
tests/                  pytest -q tests
```

## Honest limitations

- **Zero-shot only, so far.** Laya is designed to be fine-tuned per domain; this repo currently tests
  the out-of-the-box checkpoint. Next step: fine-tune on the full labelled set (866 real scam examples
  in the full EMSCAD dataset) and re-test.
- **Small scam counts mean real statistical noise.** 28 scams in the main test set — treat the AUROC
  numbers as directional ("Jev meaningfully better, Laya near chance"), not precise percentages.
- **The Laya-vs-Jev comparison in `compare.py` is illustrative on small batches** (a handful to a few
  dozen postings) — for a real benchmark, use `evaluate.py` on each engine separately over hundreds+
  of labelled rows (as the table above does).
- This tool flags postings for a human to review. It does not prove a posting is or isn't a scam.

## License / data

Code here is provided as-is for this proof of concept. Laya and Jev are third-party models — see
[Laya's model card](https://huggingface.co/convaiinnovations/laya) and
[TypeSafe's docs](https://docs.typesafe.ai) for their own licenses. The Kaggle dataset used for
testing has its own license — check it before redistributing the raw CSV (this repo does not include it).

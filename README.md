# Job posting scam screener (Laya)

Screens job postings for scam signals with [Laya](https://huggingface.co/convaiinnovations/laya), an open-source,
non-autoregressive decision model. Each posting is checked against a 9-question scam checklist
(fees, off-platform contact, sensitive-data requests, money handling, pay-per-task gigs, unrealistic pay…)
plus a scam-type question. Laya answers **all questions in one forward pass** and never generates text, so there is
nothing to parse. A small regex layer adds evidence snippets. Output: CSV / JSONL / summary and a dashboard
built for screen recording.

```
postings (csv/jsonl/txt) -> clean + condense -> rules (regex) ─┐
                                              └-> Laya checklist (1 pass) ─┴-> score + verdict -> dashboard
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -r requirements.txt                          # pulls torch + transformers; weights download on first run
python scripts/make_demo_data.py --n 500                 # synthetic postings -> data/demo_500.jsonl
python run_pipeline.py --input data/demo_500.jsonl       # real Laya (config.yaml: engine.type = laya)
open outputs/dashboard.html                              # or double-click it
```

Try one posting: `python run_pipeline.py --text "Earn $500/day. Pay a $50 registration fee. WhatsApp us."`

No model yet? `--engine mock` runs the whole pipeline offline with keyword heuristics. The dashboard shows a red
banner on mock runs so they can't be mistaken for Laya output.

## Use real data (do this before you post)

Synthetic postings are for plumbing and speed demos only. They are templated, so scoring high on them proves nothing.

1. Download the public **Real / Fake Job Posting Prediction** dataset (EMSCAD, `fake_job_postings.csv`) from Kaggle and
   check its license. The loader maps its columns automatically (`fraudulent` is the label).
2. Measure first:
   ```bash
   python scripts/evaluate.py --input data/fake_job_postings.csv --limit 2000
   python scripts/evaluate.py --input data/fake_job_postings.csv --limit 2000 --calibrate
   ```
   It prints AUROC, precision/recall/F1, and "rules only" AUROC so you can see whether Laya adds anything beyond the
   regexes. `--calibrate` fits `temperature` and thresholds; paste them into `config.yaml`. Fit on one split, confirm on
   another.
3. Screen: `python run_pipeline.py --input data/fake_job_postings.csv --limit 500`

Scraping LinkedIn/Indeed in bulk breaks their terms. Use datasets, your own exports, or postings people paste in.

## If it flags everything (Laya's "yes" bias)

The base checkpoint can answer "yes" to many checklist questions regardless of the text. On the first real run it flagged
22 of 24 genuine synthetic postings. Fix it with labelled data (EMSCAD, a few hundred rows):

```bash
python scripts/diagnose.py --input data/fake_job_postings.csv --limit 300      # runs Laya once, caches answers in outputs/raw.json
python run_pipeline.py --input data/fake_job_postings.csv --limit 300 --raw outputs/raw.json --calibration outputs/calibration.yaml
```

`diagnose.py` prints, per question, the mean P(yes) on genuine vs scam postings and that question's own AUROC. It writes
`calibration.yaml`: a per-question bias to subtract, and a list of questions to ignore because they carry no signal.
It then compares "as configured", "bias-corrected" and "rules only" so you can see what Laya really adds.
Answers are cached in `raw.json`, so recalibrating and rescoring cost no inference. The fit is in-sample; confirm on held-out
postings (`colab_run.ipynb` does this on a GPU). If bias-corrected Laya doesn't beat rules-only on held-out data, don't
present it as a Laya win: fine-tune it (Laya repo notebook) or narrow the checklist to the questions that work.

## Speed demo

Speed depends on hardware. The Laya model card reports ~33-40 ms per single question on a T4 GPU and 193-464 ms on CPU.
`predict_batch` shares forward passes across postings, so use a GPU (a free Colab/Kaggle T4 works) for a big batch.
The pipeline warms the model up first, then times only the model calls, and records the hardware in `summary.json` and the
dashboard, so quote the number the dashboard shows and say what it ran on.

## Head-to-head: Laya vs Jev

Jev has a public waitlist for TypeSafe's own API, but it's also served, with no waitlist, through
**OpenRouter** (`typesafe/jev-1.13`) and Vercel AI Gateway. Both bill per token; there's no free tier.
Get an OpenRouter key at https://openrouter.ai/settings/keys, then:

```bash
export OPENROUTER_API_KEY=sk-...
python scripts/compare.py --input data/sample_postings.jsonl --limit 10
open outputs/compare/compare.html
```

This runs the SAME postings and SAME checklist through Laya and Jev and shows them side by side: verdict,
risk score, top signals, and whether the two agree. Keep `--limit` small (5-20) — this calls Jev live over
the network, so it's meant to show HOW the two models answer, not to be a rigorous benchmark. `--laya-raw
outputs/raw.json` reuses a cached Laya run instead of paying CPU inference again.

For a real benchmark, run `scripts/evaluate.py` on each engine separately over a few hundred labelled
postings and compare the printed AUROC:

```bash
python scripts/evaluate.py --input data/fake_job_postings.csv --limit 600                 # Laya (config.yaml)
python scripts/evaluate.py --input data/fake_job_postings.csv --limit 600 --engine http \
  --config <(python -c "import yaml;c=yaml.safe_load(open('config.yaml'));c['engine']['http']={**c['engine']['http'],**c['jev_preset']};print(yaml.dump(c))")
```

Or simplest: temporarily set `engine.type: http` and paste `jev_preset`'s fields under `engine.http` in
`config.yaml`, then run `evaluate.py` normally. `run_pipeline.py --jev` does this automatically for a single run.

I built and tested this against a local stand-in server that mimics the System One response shape (unit
test + a live local HTTP test), not against the real Jev/OpenRouter endpoint, since this build environment
has no internet access to OpenRouter. The request/response handling should work as documented, but confirm
the very first call yourself before you rely on the output.

### A fair speed comparison

Laya has no separate hosted API - it's self-hosted, so running it on a laptop CPU and comparing that
number to Jev's cloud-hosted speed compares hardware, not the models. `colab_run.ipynb` has a "Laya vs
Jev, both on a GPU" section that runs `compare.py` from inside the same Colab GPU session Laya already
loads into, so Laya gets a fair, fast environment too - the two `ms/posting` numbers in the dashboard are
then comparable. Paste your OpenRouter key into that cell (it's not saved to the notebook file).

## Recording the X video

1. Run on a GPU with real data; open `outputs/dashboard.html` full screen.
2. Record: press **Replay the run**, let the counter climb, then click **Likely scam**, open one card's
   "Why it was flagged".
3. Caption with the real numbers from the dashboard, the hardware, and "flags, not verdicts". Only quote accuracy
   you measured with `evaluate.py` on labelled real data.

## Honest limits

* The base Laya checkpoints are near chance zero-shot on some benchmarks; the model card says it is "a fast base to
  specialise". Whether it works on job scams is an empirical question: run `evaluate.py`. If it is weak, the rule layer
  still works, and the Laya repo has a fine-tuning notebook.
* Laya ships over-confident; calibrate (`--calibrate`).
* The English checkpoint reads ~320 tokens of state, so long postings are condensed (`clean.condense` keeps the risky
  sentences). Rules see the full text. Non-English postings route to the multilingual checkpoint with `checkpoint: router`.
* Yes/no checks use two-option `choice` questions with neutral A/B keys, as the model card recommends over `noul`;
  flip `question_mode: noul` in `config.yaml` to compare.
* This flags postings for a human to check. It doesn't prove a posting is a scam or legitimate.

## Layout

```
config.yaml            engine, question mode, scoring
run_pipeline.py        CLI
scamscreen/            questions, rules, ingest, engine, scoring, pipeline, synth
scripts/               make_demo_data.py, diagnose.py, evaluate.py (+ --calibrate), compare.py (Laya vs Jev)
dashboard/             dashboard_template.html, compare_template.html (data injected at write time)
tests/                 pytest -q tests
colab_run.ipynb        GPU run: diagnose -> hold-out evaluation -> demo
```

## Status of this build

Verified: pipeline end to end with the mock engine, 12 unit tests, dashboard rendering and replay in a headless browser,
and the Laya call shapes (`Router.predict_batch`, `Agent.predict_batch`, answer format) against the `laya` 0.3.20 source.
Real-weights run (by the user, CPU, 40 synthetic postings): worked end to end but flagged 22/24 genuine postings, hence diagnose.py. The bias correction was tested with unit tests and a mock, not yet on real Laya output. Not run against Jev.

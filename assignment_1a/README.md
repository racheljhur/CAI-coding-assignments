# Assignment 1a (A1a): Slot Filling on ATIS — CRF and BiLSTM

Build 2009 system and 2014 system. Discover for yourself why the field switched — and what this switch cost.

| |                                 | |                                   |
|---|---------------------------------|---|-----------------------------------|
| **Assigned** | Thu, Sep 10<sup>th</sup>        | **Due** | Mon, Sep 28<sup>th</sup>, 11:59 pm |
| **Weight** | 5% of course grade              | **Work mode** | Individual                        |
| **Submit** | Gradescope: repo + report ≤3 pp | **Effort** | ≈20–25 hours                      |
| **Quiz** | Quiz 1 · in class Tue Sep 29    | **Counts** | 6.25% if not your dropped quiz    |

> A1a is the 1<sup>st</sup> half of Conversational AI course's signature assignments pair.
> You will implement slot filling — IOB sequence labeling, scored by slot F1 — twice:
> once as pre-neural field did it via a CRF with hand-designed features,
> and the other as early neural field did it by a BiLSTM tagger.
>
> **Assignment 1b (A1b)** — a separate, later assignment — revisits this a 3<sup>rd</sup> time via LLM schema-as-prompt.
> Lecture 6's bridge paper (Mesnil et al., 2015) is the fight you are about to re-stage.

---

## Setup

```bash
conda env create -f env.yml
conda activate CAI_SO_FUN
```

Environment includes `sklearn-crfsuite` (Part A), `pytorch` (Part B), and
`seqeval` (scoring, both parts). See `env.yml` for the full pinned list.

---

## Starter Kit

- `data/` — the ATIS standard split: **4,478 train / 500 dev / 893 test**
  examples, **120 slot labels, 21 intents**.

- `data_loader.py` — loads the ATIS split into the format both models expect.

- `scorer.py` — `seqeval`-based slot-F1 scorer, **both token-level and span-level.**

- `train.py` — skeleton provided: CLI args (`--model {crf,bilstm}`,
  `--train-fraction`, `--checkpoint-dir`, `--seed`, `--tensorboard-logdir`),
  data loading, and model construction are wired up. Training loop
  itself is left for you to implement, for both Part A and Part B.
  `--seed` controls model-training randomness only — it does not affect
  which examples land in the Part C subsamples (see Part C below).

- `evaluate.py` — skeleton provided: CLI args (`--model {crf,bilstm}`,
  `--checkpoint-dir`, `--split {dev,test}`), model loading, and eval-data
  loading are wired up. Scoring is left for you to implement.

- `models/__init__.py`, `models/crf.py`, `models/bilstm.py` — skeletons
  provided: constructors and `fit()`/`predict()` (CRF), layer definitions
  and `predict()` (BiLSTM) are wired up. Feature engineering (CRF) and
  forward pass (BiLSTM) are left for you to implement — see Part A and
  Part B below. **`models/__init__.py` ships empty, as-is — see
  Deliverables for why it must still be present in your submission.**

- `slurm/` — SLURM job template and instructions for running Part B training on PACE-ICE.

- `env.yml` — pinned conda environment for reproducibility.

- `README.md` — this file.

Your numbers are comparable to thirty years of published work on this corpus.

---

## What to Build

### Part A — CRF: 2009 experience
Use `sklearn-crfsuite`. Model itself is trivial — **the assignment is
feature engineering**: word identity, casing, affixes, context windows,
gazetteers if you build them. Log every feature set you try and its dev F1.

### Part B — BiLSTM: 2014 experience
PyTorch, from scratch — **no HuggingFace token classifiers**.

Architecture: embeddings → BiLSTM → per-token softmax.

Pretrained *static* embeddings like GloVe are permitted; pretrained transformers are not — that's A1b's job.

> **Optional bonus (+5 pts):** add a CRF decoding layer on top of BiLSTM
> (handwritten forward/Viterbi). The only place a self-implemented CRF belongs in A1a.

Both Part A and Part B build on the same course-provided skeletons.

In `models/crf.py` / `models/bilstm.py`, you implement `extract_features()` (CRF) or `forward()` (BiLSTM).

In `train.py`, you implement `train_loop()` for both. In `evaluate.py`, you implement `compute_metrics()` for both.

Everything else — CLI parsing, data loading, checkpointing, model loading — is wired up for you.

### Part C — Data-efficiency curve
Train **both** models on 5%, 10%, 25%, and 100% of training set (fixed
subsamples, provided — the same subsamples for every student, regardless of
your `--seed`). Produce **one plot**: slot F1 vs. training fraction,
both curves on the same axes. This figure is the intellectual payload of the
assignment — look at where the curves cross.

### Part D — Error analysis
Examine **ten** test-set failures. Which errors does CRF make that BiLSTM doesn't, and vice versa?

"The model was wrong" is not an analysis;
**"CRF misses unseen city names because word-identity features can't fire"** is.

---

## Gradescope Deliverables

### **1. Coding** — your whole project directory as `.zip`. At its root, it must contain:

- `models/` — your implemented package: `models/__init__.py`, `models/crf.py`,
  and `models/bilstm.py`.

- `crf_predictions.jsonl` and `bilstm_predictions.jsonl` — test-set predictions
   from both models. Each line is a JSON object with `index`, `tokens`, and `slots` (one
   predicted IOB2 label per token) — see `write_predictions()` in `gradescope/autograder_interface.py`.

- `crf_model.pkl` and `bilstm_model.pt` — your trained checkpoints,
   in the format `evaluate.py`'s `load_model()` already expects.

#### ‼️Reproducibility credit reminders～

Autograder re-runs inference by importing *your own* submitted `models/crf.py` / `models/bilstm.py`,
not a blank copy. Omitting `models/` (or submit only the four files below) will cause
reproducibility to score 0/10 with a clear message explaining why, even if your checkpoint is otherwise fine.

#### 🛘Respect empty files☢️

Missing `__init__.py` alone is enough to break the import and zero out reproducibility,
even if `crf.py` and `bilstm.py` themselves are correct — see the starter kit's
own `models/__init__.py` for what it should contain.

Autograder uses your checkpoints and your own `models/` code together to check reproducibility rubric item.

It re-runs *inference only (never retraining)* via your submitted model code, and compares the result against
your submitted predictions (±0.3 F1). `data_loader.py`, `scorer.py`, `evaluate.py`, and the ATIS data
ship with the autograder itself — submitting your own copies of those is harmless, as they're never read.

Only `models/` and the four files above are actually required.

Hyperparameters, like embedding size, hidden size, etc., are all your choices. Part B
never fixes them, and reproducibility doesn't check for any particular architecture,
only that your checkpoint, your `models/` code, and your predictions all come from the same training run.

If you change hyperparameters and retrain after an earlier submission attempt,
regenerate and resubmit the checkpoint and predictions together with the updated code —
mixing versions will surface as a dimension-mismatch error at grading time.

#### 😉Not just what to zip, but also WHERE to zip😏

How to package your submissions:

```bash
cd your-project-directory/
zip -r ../submission.zip models/ crf_predictions.jsonl bilstm_predictions.jsonl \
    crf_model.pkl bilstm_model.pt
```

Beware of messing up with where......

Right-click → "Compress" on the folder itself — whether macOS Finder,
Windows Explorer, or similar GUI tools — commonly wraps everything in one extra top-level folder.

In this case, autograder then can't find any of your files, so it's:
```
Gonna report them all as missing, even though you submitted everything.
```

Use the command offered above, or if you use a GUI tool, verify results by:

```bash
unzip -l submission.zip | head -20
```

Correct output lists `models/`, `crf_predictions.jsonl`, `crf_model.pkl`, etc. directly.

If you instead see them nested under another folder name, re-zip from inside that folder.

### 2. **Report** (≤3 pages):

Namely, data-efficiency curve, error analysis, and one paragraph answering —
*knowing only what you know from this assignment, would you have switched to neural in 2014? At what data size?*

Also don't forget to add your AI-use disclosure.

---

## Grading Rubrics

| Component                  | Details                                                                                                                                                                          | Points    |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Autograded metrics         | Threshold bands, not rank:<br>CRF ≥ 0.90 test F1, with full credit ≥ 0.92.<br>BiLSTM ≥ 0.93, with full credit ≥ 0.95.<br>Leaderboard posted for simply fun — rank earns nothing. | 35        |
| Data-efficiency experiment | All four fractions, both models, honest plot, crossover discussed.                                                                                                               | 20        |
| Error analysis             | Ten real failures, correctly diagnosed, contrasted across models.                                                                                                                | 30        |
| Reproducibility            | Checkpoint inference reproduces your submitted predictions' F1 within ±0.3 (see Deliverables — not a full retrain).                                                              | 10        |
| AI-use disclosure          | Complete and candid. Using AI to write boilerplate code is fine — disclosed.                                                                                                     | 5         |
| **Total**                  |                                                                                                                                                                                  |   **100** |

**Optional bonus: +5 pts** for a correct hand-written Viterbi decoding layer on top of BiLSTM
(see Part B). Automatically detected and graded — no separate submission step needed; just
implement `viterbi_decode()` in your BiLSTM model.

---

## Policies for Every Assignment

- **AI tools**: use them for anything except writing your analysis for you.
  Every submission includes a half-page **AI-use disclosure** appendix: which
  tools, for what, and how you verified the results. Appendix is graded.
  An empty or evasive disclosure on work that plainly used AI is an
  honor-code matter; a candid disclosure never costs you points on the work itself.

- **Late work**: 5 slip days for the semester, at most 2 per assignment,
  applied automatically in whole days. Beyond slip days: −10 pts/day.

- **Reproducibility**: your repo must run from a clean clone — README with
  exact commands, pinned dependencies, fixed seeds. If we can't reproduce your
  headline number within tolerance in 15 minutes, the metric portion is graded from what reproduces.

- **Quizzes**: 15 minutes, in class, closed book/notes, no devices, the
  morning after this assignment is due. Five quizzes total = 25% of course
  grade; lowest is dropped. No makeup sittings. A slip day moves the
  assignment deadline but not the quiz.

- **Regrades**: written requests within 7 days of grade release, on
  Gradescope. Whole component is regraded, in either direction.

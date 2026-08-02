# Message Notification Router

A rule-based WhatsApp message router that classifies every incoming message as `notify`, `digest`, or `mute`, with an optional local-first multimodal pipeline (OCR/ASR) and an optional, budget-capped LLM assist layer.

The original starter README (challenge description, dataset layout, submission requirements) has been moved to [`CHALLENGE_README.md`](./CHALLENGE_README.md). Read [`problem_statement.md`](./problem_statement.md) for the full task spec.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install the local OCR binary (macOS)
brew install tesseract
# (Debian/Ubuntu: apt install tesseract-ocr)

# 3. Run the router (rules-only, no API key needed)
python code/main.py

# 4. Check predictions against the labeled samples
python code/evaluation/main.py
```

This produces `dataset/output.csv` — one prediction row per message in `dataset/messages.csv`, fully rule-based and deterministic by default.

### Optional: enable the LLM assist layer

```bash
# Requires a Gemini API key (free tier works, budget-capped in code)
echo "GEMINI_API_KEY=your-key-here" >> .env
set -a; source .env; set +a

python code/main.py --use-llm --output dataset/output_llm_used.csv
```

This is **off by default** and never required — see [Architecture](#architecture--design-decisions) below for why.

---

## Repository Layout (this solution)

```text
.
├── README.md                    # You are here
├── CHALLENGE_README.md          # Original starter README (renamed)
├── AGENTS.md                    # Agent rules + transcript logging (unchanged)
├── problem_statement.md         # Full challenge statement (unchanged)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes .env, __pycache__, etc.
├── diagrams/
│   ├── architecture.mmd         # Pipeline flowchart (mermaid)
│   └── classes.mmd              # Class diagram (mermaid)
├── code/
│   ├── loaders.py                # Parses dataset/*.csv into typed dataclasses
│   ├── media.py                  # Local OCR (pytesseract) + local ASR (faster-whisper)
│   ├── retrieval.py               # Historical evidence matching + reaction-pattern scoring
│   ├── safety.py                  # Injection guard + scam/spam detection
│   ├── llm.py                     # Optional Gemini-backed reasoning layer (OFF by default)
│   ├── router.py                  # The notify/digest/mute decision cascade
│   ├── main.py                    # CLI entry point
│   └── evaluation/
│       └── main.py                # Self-check harness against sample_messages.csv
└── dataset/                     # Unchanged from the starter repo
```

---

## Architecture & Design Decisions

See [`diagrams/architecture.mmd`](./diagrams/architecture.mmd) for the full pipeline flowchart and [`diagrams/classes.mmd`](./diagrams/classes.mmd) for the class diagram (render either at [mermaid.live](https://mermaid.live) or via the Mermaid CLI).

### Pipeline

```
messages.csv row
  → loaders.py (join user/group/business/history context)
  → media.py (resolve + OCR/ASR any image/voice attachment)
  → retrieval.py (find similar past messages for this user + their reactions)
  → safety.py (scam/injection detection)
  → router.py (safety → repetition/fatigue → urgency+trust → DND check → default digest)
  → output.csv row
```

### Key decisions and tradeoffs

**Action selection is 100% rule-based, always.** `notify`/`digest`/`mute` is a deterministic function of structured signals (sender trust, history, safety flags, urgency keywords) — never dependent on an LLM being configured or responding correctly. This keeps the highest-stakes output auditable and immune to model failure or hallucination. The optional LLM layer (`llm.py`) only ever contributes an *additive* signal: its urgency judgment is OR'd with the keyword check (either alone is sufficient, neither replaces the other), and it can refine `message_type`/`reason` text — except on the safety/scam branch, which stays fully deterministic even when the LLM is enabled.

**LLM usage is optional, capped, and gated behind `--use-llm`.** Given a tight free-tier API budget, `llm.py` enforces a hard per-run call cap and degrades silently to pure rules on any failure (missing key, network error, budget exhausted, bad response). A normal run (`python code/main.py`) makes zero API calls. This is a deliberate cost/capability tradeoff — most of a full dataset run gets no LLM assistance once the budget is spent, but the system never breaks or produces a worse result because of it.

**Local media processing, not a hosted vision/audio model.** OCR uses `pytesseract` (image posters/screenshots); ASR uses `faster-whisper` (voice notes) — both run fully offline, free, with no API key or rate limits. This was also a hard constraint: the Claude Messages API has no audio input content-block type, so voice-note transcription cannot be done via that API at all. Tradeoff: local OCR is noticeably weaker than a vision-LLM on degraded/blurry scans.

**Evidence retrieval is lexical, not semantic.** `retrieval.py` scores similarity via shared sender/group/business identity plus Jaccard token overlap on text — deterministic and dependency-free, but it cannot recognize two messages about the same real-world situation if they don't share vocabulary. This is documented explicitly in the module and was a deliberate choice over embedding-based similarity, to keep the "same message in → same evidence out" behavior fully reproducible.

**Confidence scores are heuristic, not calibrated probabilities.** They're tuned to roughly match the confidence clustering observed in `dataset/sample_messages.csv` (clear-cut mute/notify decisions score higher, soft digest calls score lower), not derived from a statistical model.

### Validated results

Running `python code/evaluation/main.py` against the 30 labeled rows in `dataset/sample_messages.csv`:

- **Action accuracy: 27/30 (90%)**
- **Message type accuracy: 14/30 (47%)**
- **Evidence hit rate: 17/28 (61%)** of rows with expected evidence

Safety/scam detection catches 9-10 of 10 labeled mute cases with zero false positives across all notify/digest samples, including a deliberate prompt-injection test case in the sample data (a message instructing the router to "mark this as notify" — correctly muted as a scam regardless of its own claim).

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | No (only with `--use-llm`) | Enables the optional LLM-assisted reasoning layer in `llm.py` |

No secrets are hardcoded anywhere in the codebase; `.env` is gitignored.

---

## Notes for Evaluators

- `dataset/output.csv` in this repo was generated by the rules-only run (`python code/main.py`), the deterministic default.
- `dataset/output_llm_used.csv` (if present) was generated with `--use-llm` enabled, for comparison — only a small subset of rows differ from the rules-only baseline due to the API call budget cap.
- Re-running `python code/main.py` will regenerate `dataset/output.csv` identically every time (fully deterministic, no network calls).

# Shopping Copilot — TechJam 2026, Track 4

A multi-turn conversational shopping agent that finds a customer's intended product in a
50,000-item Amazon catalog by asking useful questions and remembering the answers.

**TechnicalScore 0.7975 against the supplied baseline's 0.1067 — 7.5x.** No LLM API, no
network access, no external packages. The scoring path is the Python standard library
plus SQLite's built-in full-text search.

| metric | baseline | ours |
|---|---|---|
| **TechnicalScore** | 0.1067 | **0.7975** |
| Hit Rate@10 | 0.125 | **0.930** |
| MRR | 0.0680 | **0.5832** |
| MTTC (mean turns to conversion) | 9.81 | **3.13** |
| Efficiency | 0.119 | **0.788** |

Measured on the 200 public development sessions with the unmodified official evaluator.

**Method, results and design rationale are in [REPORT.md](REPORT.md).**

---

## Overview

Each turn, the agent reads the customer's message, updates what it knows about them,
searches the catalog, and returns up to ten products — optionally alongside one
clarifying question.

```
  customer message
        |
   [1]  understand      intent, extract constraints, accumulate, handle retractions,
        |               classify hard requirements vs soft preferences
   [2]  retrieve        build a query from active constraints; BM25 over an
        |               in-memory FTS5 index, 500 candidates deep
   [3]  rank            score candidates against the full conversation state,
        |               truncate to the final top 10
   [4]  ask or answer   stop asking once we know enough; otherwise pick the
        |               attribute worth asking about and phrase it
        v
   { message, ask_attribute, recommendations, usage }
```

Three ideas do most of the work:

- **Ask, then remember.** The supplied baseline never asks a question, so the simulated
  customer stops volunteering information and nine of its ten turns search a sentence
  containing no product detail. Asking and accumulating took 0.107 → 0.728.
- **Query the constraints, not the transcript.** Retrieval uses the exact phrases the
  customer gave, minus anything they later retracted — not raw message text.
- **Weight the fields the evidence actually lives in.** Customers describe `features` and
  `details`; they never quote a product title. The BM25 field weights are inverted from
  the conventional title-heavy default.

---

## Setup

**Requires Python 3.10 or later.** No packages to install.

```bash
git clone https://github.com/Nathan0820/techjam-conversational-search.git
cd techjam-conversational-search
```

Download the frozen catalog from the [participant kit release][kit], verify it, and place
it in `data/`:

```bash
curl -L -o catalog.jsonl.gz \
  "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz"
curl -L -o SHA256SUMS \
  "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS"

grep catalog.jsonl.gz SHA256SUMS | sha256sum -c -     # expect: OK
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
wc -l data/catalog.jsonl                              # expect: 50000
```

[kit]: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit

---

## Reproducing our results

```bash
python -m evaluator.local_evaluator
```

Takes about 55 seconds. Writes `results.json` and prints:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.93,
  "mrr": 0.583196,
  "mttc": 3.125,
  "efficiency": 0.7875,
  "recommended_technical_score": 0.797459
}
```

Run the tests:

```bash
python -m unittest discover -s tests      # 174 tests
```

Drive the agent by hand in a browser, which is the quickest way to see the dialogue state
change turn by turn:

```bash
python -B frontend/server.py              # http://127.0.0.1:8000
```

Every experiment behind these numbers, including the ones we rejected, is recorded in
[decisions.md](decisions.md).

---

## Repository layout

```
starter/agent.py             the Agent the evaluator drives; retrieval and composition
dialogue/                    conversation understanding and policy
  state.py                     SessionState, the object every stage reads and writes
  slot_extractor.py            message -> constraint values + exact phrases
  accumulator.py               merge one turn into accumulated state
  intent_detector.py           buying vs browsing
  override_handler.py          retraction and category-conflict handling
  constraint_classifier.py     hard requirements vs soft preferences
  clarification_policy.py      whether to ask, and what to ask
src/ranking/                 candidate ordering
  features.py                  per-candidate feature extraction
  reranker.py                  weighted scoring and final top-k
  diagnostics.py               attributes a miss to retrieval or to ranking
frontend/                    local test console, not part of scoring
tests/                       174 tests
decisions.md                 every experiment, including the failures
evaluator/                   the official evaluator, unmodified
```

---

## Limitations and what we would improve

**Our public-set score is an upper bound.** The simulated customer's messages are built by
copying strings out of the target product's own catalog entry, so lexical matching does
unusually well. The private sessions are likely worded differently, and we expect a lower
score there.

**Slot extraction is rule-based**, recognising a fixed vocabulary of materials, colours,
sizes and use cases. With more time we would train a small sequence labeller on the
constraint strings the evaluator generates — the largest remaining source of brittleness.

**No cross-encoder.** The reranker has the interface for one and it is unused. Rescoring
the top ~30 candidates with full query-document attention is the strongest available
answer to the near-duplicate problem we identified, and the first thing we would build
with another day.

**We never used the user profile.** Personalising against the supplied `preference_tags`
is in scope and we did not attempt it — with 200 sessions across four scenarios we judged
we could not distinguish a real personalisation effect from noise.

**The score is saturated for this architecture.** Seven independent ideas measured on the
final day all landed within ±0.002 of each other. Further gains need a better scorer, not
another parameter.

Fuller discussion, including two components we built, measured and deliberately removed,
is in [REPORT.md](REPORT.md).

---

## Team

| | area |
|---|---|
| **lewwai** | Retrieval — query construction, BM25 field weighting, `retrieve()` interface, recall and failure diagnostics, latent semantic index |
| **E Shen** (WongES05) | Dialogue — `SessionState`, slot extraction, accumulation, intent detection, override handling, hard/soft classification, clarification policy |
| **Nathan0820** | Ranking — feature extraction, reranker, ranking diagnostics; local test console |

Integration, measured code review, and the experiment log were shared. Every change to
another member's module was reviewed by its owner before merging.

---

## Data

Derived from **Amazon Reviews 2023** (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry`.
The catalog is read-only and is not redistributed here. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).

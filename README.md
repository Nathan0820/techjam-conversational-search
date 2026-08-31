# Shopping Copilot — TechJam 2026, Track 4

A multi-turn conversational shopping agent that finds a customer's intended product in a
50,000-item Amazon catalog by asking useful questions and remembering the answers.

**TechnicalScore 0.8153 against the supplied baseline's 0.1067 — 7.6x.** No LLM API, no
network access, no third-party packages. The scoring path is the Python standard library
plus SQLite's built-in full-text search.

| metric | baseline | ours |
|---|---|---|
| **TechnicalScore** | 0.1067 | **0.8153** |
| Hit Rate@10 | 0.125 | **0.955** |
| MRR | 0.0680 | **0.5925** |
| MTTC (mean turns to conversion) | 9.81 | **3.00** |
| Efficiency | 0.119 | **0.801** |

| scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.963 | 0.558 | 2.49 |
| browsing | 80 | 0.963 | 0.562 | 2.90 |
| intent_override | 30 | 0.933 | 0.773 | 4.23 |
| boundary | 10 | 0.900 | 0.571 | 4.10 |

Measured on the 200 public development sessions with the unmodified official evaluator.
182 tests pass.

**Method, results and design rationale are in [REPORT.md](REPORT.md).**

---

## Overview

Each turn, the agent reads the customer's message, updates what it knows about them,
searches the catalog, and returns up to ten products — optionally alongside one
clarifying question.

```
  customer message
        |
   [1]  understand      detect intent, extract constraints, accumulate across turns,
        |               handle retractions, classify hard vs soft requirements
   [2]  retrieve        build a query from active constraints; BM25 over an
        |               in-memory FTS5 index, 500 candidates deep
   [3]  rank            score candidates against the full conversation state,
        |               truncate to the final top 10
   [4]  ask or answer   stop asking once we know enough; otherwise choose the
        |               attribute worth asking about and phrase it
        v
   { message, ask_attribute, recommendations, usage }
```

Four ideas do most of the work:

- **Ask, then remember.** The supplied baseline never asks a question, so the simulated
  customer stops volunteering information and nine of its ten turns search a sentence
  containing no product detail. Asking and accumulating took 0.107 → 0.728.
- **Query the constraints, not the transcript.** Retrieval uses the exact phrases the
  customer gave, minus anything they later retracted — not raw message text.
- **Weight the fields the evidence actually lives in.** Customers describe `features` and
  `details`; they never quote a product title. Our BM25 field weights are inverted from
  the conventional title-heavy default.
- **Treat a correction as a transition.** On the turn a customer retracts something, the
  query still contains the old wording while the ranking state has already dropped it.
  That mismatch helps recall and hurts ranking, so ranking steps aside for that one turn.

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

Takes about 50 seconds. Writes `results.json` and prints:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.955,
  "mrr": 0.59248,
  "mttc": 2.995,
  "efficiency": 0.8005,
  "recommended_technical_score": 0.815344
}
```

Run the tests:

```bash
python -m unittest discover -s tests      # 182 tests
```

Drive the agent by hand in a browser, which is the quickest way to watch the dialogue
state change turn by turn:

```bash
python -B frontend/server.py              # http://127.0.0.1:8000
```

Every experiment behind these numbers, including the ones we rejected, is written up in
[REPORT.md](REPORT.md).

---

## Repository layout

```
starter/agent.py             the Agent the evaluator drives; retrieval and composition
dialogue/                    conversation understanding and policy
  state.py                     SessionState, the object every stage reads and writes
  slot_extractor.py            message -> constraint values, exact phrases, hints
  accumulator.py               merge one turn into accumulated state
  intent_detector.py           buying vs browsing
  override_handler.py          retraction and category-conflict handling
  constraint_classifier.py     hard requirements vs soft preferences
  clarification_policy.py      whether to ask, and what to ask
  types.py                     shared value types
src/ranking/                 candidate ordering
  features.py                  per-candidate feature extraction
  reranker.py                  weighted scoring and final top-k
  diagnostics.py               attributes a miss to retrieval or to ranking
frontend/                    local test console, not part of scoring
tests/                       182 tests
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
the top 30 candidates with full query-document attention is the strongest available answer
to the near-duplicate problem we identified, and the first thing we would build with
another day.

**The user profile is not used at runtime.** The reranker serialises `preference_tags`
into its scoring query, but that path activates only with a cross-encoder. With 200
sessions across four scenarios we judged we could not distinguish a real personalisation
effect from noise, so we left the hook in place rather than shipping something we could
not validate.

**Remaining gains are close to the noise band.** Several independent ideas measured on the
final day landed within ±0.002 of each other. Where configurations differed by less than
about one session out of 200 we chose round numbers and stopped rather than selecting the
winner — fitting that spread would not transfer to the hidden set.

Fuller discussion, including two components we built, measured and deliberately removed,
is in [REPORT.md](REPORT.md).

---

## Team

| | area |
|---|---|
| **Lew Wai Loon** (`lewwai`) | Retrieval — query construction, BM25 field weighting, the `retrieve()` interface, recall and failure diagnostics, latent semantic index |
| **Wong E Shen** (`WongES05`) | Dialogue — `SessionState`, slot extraction, accumulation, intent detection, override handling, hard/soft classification, clarification policy |
| **Nathan Wong Yong Jie** (`Nathan0820`) | Ranking — feature extraction, reranker, override routing, retrieval hints; local test console |
| **Lee Chong Sheng** (`matyehh`) | Report drafting and the demo video |

Integration, measured code review, and the experiment log were shared. Every change to
another member's module was reviewed by its owner, with metrics, before merging.

---

## Data

Derived from **Amazon Reviews 2023** (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry`.
The catalog is read-only and is not redistributed here. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).

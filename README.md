# Findly

> Conversational shopping copilot · TechJam 2026, Track 4

Findly connects what a shopper says across multiple turns — requirements, uncertainty,
and corrections — with the most relevant products in a 50,000-item Amazon catalog. It
asks useful questions, remembers the answers, and adapts when the shopper changes their
mind.

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
The complete test suite passes.

How to read the metrics:

- **Hit Rate@10** is the share of sessions where the target product appears anywhere in
  the ten returned recommendations.
- **MRR** rewards placing the target nearer the top of that list; a target ranked first
  contributes `1.0`, ranked second contributes `0.5`, and so on.
- **MTTC** is the mean number of conversation turns needed to surface the target. Lower
  is better.
- **Efficiency** converts MTTC into a higher-is-better score.
- **TechnicalScore** is the official combined measure of retrieval success, ranking
  quality, and conversational efficiency.

---

## Why Findly

Shopping rarely starts with a perfectly formed search query. People remember details
gradually, answer questions, and change their minds. Findly was inspired by that gap
between how people naturally shop and how one-shot product search expects them to behave.

The main lesson from building it was that better conversation state can matter more than
more complicated retrieval. The strongest gains came from asking useful questions,
keeping only active evidence, and handling corrections deliberately. More elaborate
retrieval routes were retained only when they improved the full evaluator, not merely an
isolated ranking metric.

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

In more detail, one call to `Agent.respond()` performs the following sequence:

1. Detect whether the shopper is buying, browsing, or correcting an earlier preference.
2. Extract exact evidence for category, material, colour, size, style, brand, budget,
   feature, and use case.
3. Merge new evidence into `SessionState`, removing retracted values and classifying
   active values as hard constraints or soft preferences.
4. Build a compact query from active evidence and current-turn retrieval hints.
5. Retrieve 500 candidates with weighted BM25 over SQLite FTS5.
6. Rerank those candidates against the conversation state, except during the explicit
   override transition where preserving BM25 order measured better.
7. Return the top ten recommendations and either ask the next useful attribute or stop
   asking when the intent is sufficiently clear.

Five design decisions do most of the work:

- **Ask, then remember.** The supplied baseline never asks a question, so the simulated
  customer stops volunteering information and nine of its ten turns search a sentence
  containing no product detail. Asking and accumulating took 0.107 → 0.728.
- **Query the constraints, not the transcript.** On normal turns, retrieval uses the
  shopper's active constraint phrases rather than replaying the raw conversation.
- **Preserve the signal inside “no preference.”** Findly removes boilerplate such as
  “I don't have an additional preference for size” from persistent search evidence, but
  keeps `size` as a current-turn retrieval hint. These two parts are intentionally
  atomic: cleaning alone regressed 0.8062 → 0.7997, while cleaning plus hints reached
  0.8153. Hints are suppressed when the shopper delegates the decision to the agent.
- **Weight the fields the evidence actually lives in.** Customers describe `features` and
  `details`; they never quote a product title. Our BM25 field weights are inverted from
  the conventional title-heavy default.
- **Treat a correction as a transition.** On the turn a customer retracts something, the
  query still contains the old wording while the ranking state has already dropped it.
  That mismatch helps recall and hurts ranking, so the reranker preserves BM25 order for
  that one turn. Later turns use only the updated intent.

---

## Quick start

### 1. Check the requirements

Findly requires **Python 3.10 or later**. The agent, evaluator, and frontend use only the
Python standard library, so there is no `pip install` step.

```bash
python3 --version
```

All commands below should be run from the repository root.

### 2. Clone the repository

```bash
git clone https://github.com/Nathan0820/findly.git
cd findly
```

### 3. Download the catalog

The public session set is included in the repository. The 50,000-product catalog is not,
so download it from the [participant kit release][kit], verify its checksum, and extract
it into `data/catalog.jsonl`:

```bash
curl -L -o catalog.jsonl.gz \
  "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz"
curl -L -o SHA256SUMS \
  "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS"

# Linux
grep catalog.jsonl.gz SHA256SUMS | sha256sum -c -

# macOS
grep catalog.jsonl.gz SHA256SUMS | shasum -a 256 -c -

mkdir -p data
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
wc -l data/catalog.jsonl                              # expect: 50000 lines
```

Run only the checksum command for your operating system. When setup is complete, these
two inputs should exist:

```text
data/catalog.jsonl      50,000 products
data/public_set.jsonl      200 development sessions
```

[kit]: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit

### 4. Run the tests

Run the complete unit, regression, integration, and frontend-server suite:

```bash
python3 -m unittest discover -s tests -v
```

A successful run ends with `OK`. To test only the frontend backend and performance-metric
publication behavior:

```bash
python3 -m unittest discover -s tests -p 'test_frontend.py' -v
```

### 5. Reproduce the technical score

```bash
python3 -m evaluator.local_evaluator
```

The evaluator creates a fresh agent, runs all 200 public sessions for up to ten turns
each, validates the recommendations against the catalog, and computes the official
metrics. It takes about 50 seconds, writes the detailed result to `results.json`, and
prints the aggregate result:

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

Use the evaluator after changing dialogue, retrieval, or reranking logic. Comparing the
new `recommended_technical_score`, Hit Rate@10, MRR, and MTTC with the values above shows
whether the change improved the full pipeline rather than one isolated component.

To keep a separate experiment result, choose another output path:

```bash
python3 -m evaluator.local_evaluator --output results-experiment.json
```

### 6. Run the frontend

The local frontend is the quickest way to watch Findly's dialogue state and rankings
change turn by turn. Start it from the repository root:

```bash
python3 -B frontend/server.py
```

Wait for these messages:

```text
Loading the product catalog and building the search index...
Frontend ready at http://127.0.0.1:8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. The server binds
only to localhost and does not make Findly available on the network. Stop it with
`Ctrl+C`.

The page contains three useful views:

- **Performance** shows the most recently evaluated TechnicalScore, Hit Rate@10, MRR,
  MTTC, and Efficiency.
- **What the agent knows** shows the detected intent and the active constraints carried
  across the conversation.
- **Top recommendations** shows the ten products returned for the current turn, including
  their scores and catalog identifiers.

On the initial page load, the performance matrix reads the existing `results.json`; it
does not silently rerun the evaluator. Clicking **New session** clears the conversation,
starts at turn 1, runs the current agent on all 200 public sessions, writes a fresh
`results.json`, and updates the matrix when evaluation finishes. This takes about one
minute. The live conversation remains usable after the refresh.

For a quick manual test, try:

1. Send `I want black Nike boots under SGD 100.`
2. Check that intent, category, colour, brand, and budget appear in **What the agent
   knows**.
3. Answer the clarifying question and confirm that the next recommendation list reflects
   the new preference.
4. Send `Actually, ignore Nike; show me Adidas instead.` and confirm that the active brand
   changes rather than accumulating both brands.
5. Click **New session** and confirm that the conversation state and recommendations are
   cleared before the performance matrix refreshes.

### Troubleshooting

- **`Catalog not found`** — confirm the file is exactly `data/catalog.jsonl` and contains
  50,000 lines.
- **`TypeAlias` import error** — `python3` is pointing to Python 3.9 or older. Run the
  commands with a Python 3.10+ executable.
- **Port 8000 is already in use** — stop the previous frontend process before starting a
  new one.
- **The matrix shows old values** — click **New session** and wait for the 200-session
  evaluation to finish. The matrix changes only after the new `results.json` is
  published.

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
tests/                       unit, regression, and integration tests
evaluator/                   the official evaluator, unmodified
```

---

## Limitations and what we would improve

**The reported score is a development-set result, not a guarantee.** The public simulator
often builds customer messages from strings in the target product's catalog entry, which
favours lexical matching. Different wording or scenario distributions in hidden sessions
may change the result.

**Slot extraction is rule-based**, recognising a fixed vocabulary of materials, colours,
sizes and use cases. A catalog-grounded semantic parser, evaluated on held-out paraphrases,
would be the next step toward handling less predictable wording.

**No cross-encoder.** The reranker has an interface for one, but the measured scoring path
does not use it. Rescoring a small candidate set with full query-document attention is a
promising controlled experiment for the near-duplicate problem, but we would only ship it
after validating its latency and score on held-out sessions.

**The user profile is not used at runtime.** The reranker serialises `preference_tags`
into its scoring query, but that path activates only with a cross-encoder. With 200
sessions across four scenarios we judged we could not distinguish a real personalisation
effect from noise, so we left the hook in place rather than shipping something we could
not validate.

**Remaining gains are close to the noise band.** Several independent ideas measured on the
final day landed within ±0.002 of each other. Where configurations differed by less than
about one session out of 200 we chose round numbers and stopped rather than selecting the
winner — fitting that spread would not transfer to the hidden set.

**Experiments that did not earn their complexity were removed.** Requiring hard
constraints as a retrieval filter added no score because near-duplicates usually satisfy
the same constraints. Price-tier sorting also added 0.0000 and was under-exercised by the
public sessions. Multi-route retrieval with reciprocal-rank fusion peaked at 0.815196,
just below the simpler 0.815344 pipeline. A dense latent-semantic route improved MRR but
cost Hit@10; restricted to tie-breaking, it added only 0.0008. Keeping these experiments
out of the final path makes Findly easier to explain, test, and trust.

---

## Team

| | area |
|---|---|
| **Nathan Wong Yong Jie** (`Nathan0820`) | Ranking — feature extraction, reranker, override routing, retrieval hints; local test console |
| **Lew Wai Loon** (`lewwai`) | Retrieval — query construction, BM25 field weighting, the `retrieve()` interface, recall and failure diagnostics, latent semantic index |
| **Wong E Shen** (`WongES05`) | Dialogue — `SessionState`, slot extraction, accumulation, intent detection, override handling, hard/soft classification, clarification policy |
| **Lee Chong Sheng** (`matyehh`) | Report drafting and the demo video |

Integration, measured code review, and the experiment log were shared. Every change to
another member's module was reviewed by its owner, with metrics, before merging.

---

## Data

Derived from **Amazon Reviews 2023** (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry`.
The catalog is read-only and is not redistributed here. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).

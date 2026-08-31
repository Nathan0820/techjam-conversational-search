# Shopping Copilot — TechJam 2026, Track 4

A multi-turn conversational shopping agent that finds a customer's intended product in a
50,000-item Amazon catalog by asking useful questions and remembering the answers.

**TechnicalScore 0.7975 against the supplied baseline's 0.1067 — 7.5x.** No LLM API, no
network access, no external services. The scoring path is the Python standard library
plus SQLite's built-in full-text search.

| metric | baseline | ours |
|---|---|---|
| **TechnicalScore** | 0.1067 | **0.7975** |
| Hit Rate@10 | 0.125 | **0.930** |
| MRR | 0.0680 | **0.5832** |
| MTTC (mean turns to conversion) | 9.81 | **3.13** |
| Efficiency | 0.119 | **0.788** |

| scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.938 | 0.547 | 2.63 |
| browsing | 80 | 0.938 | 0.554 | 3.00 |
| intent_override | 30 | 0.900 | 0.764 | 4.47 |
| boundary | 10 | 0.900 | 0.571 | 4.10 |

Measured on the 200 public development sessions with the unmodified official evaluator.
174 tests pass. Reported token usage is zero — no model is called.

---

## Contents

- [How it works](#how-it-works)
- [What we found](#what-we-found)
- [Setup](#setup)
- [Reproducing our results](#reproducing-our-results)
- [Try it interactively](#try-it-interactively)
- [Repository layout](#repository-layout)
- [Cost, latency and dependencies](#cost-latency-and-dependencies)
- [Limitations and what we would do next](#limitations-and-what-we-would-do-next)
- [Team](#team)

---

## How it works

Every turn runs one pass of the pipeline below. Expensive work — reading the catalog,
building the search index — happens once when the agent is constructed, because
`respond()` is called up to 2,000 times per evaluation.

```
                    STARTUP (once)
  load 50,000 products  ->  build SQLite FTS5 index

                    EVERY TURN
  customer message
        |
   [2]  intent detection            buying or browsing
   [3]  slot extraction             pull constraints out of the message
   [4]  accumulate state            merge into what we already knew
   [5]  override handling           detect retractions, erase stale constraints
   [6]  hard vs soft                classify which constraints are requirements
        |                                                        SessionState
   [7]  build query                 from active constraint phrases, not raw text
   [8]  retrieve                    BM25 over the FTS5 index, 500 deep
        |                                          [(parent_asin, score)] x 500
   [9]  rerank                      score candidates against the full state
  [10]  truncate                    final top 10
        |                                          ranked ids + score spread
  [11]  ask or answer?              stop asking once we know enough
  [12]  choose ask_attribute        which attribute is worth asking about
  [13]  compose message             the sentence a human reads
        |
   response: { message, ask_attribute, recommendations, usage }
```

### Dialogue state

`SessionState` (`dialogue/state.py`) is the single object every stage reads and writes.
Two design decisions in it are load-bearing:

**Raw wording is preserved separately from parsed values.** `slots` holds normalised
constraint values for reasoning; `revealed_text` holds the customer's exact phrases,
never rewritten. Retrieval uses the latter, because the exact wording is what matches
catalog text. Normalising first would discard the signal.

**Retracted constraints are removed, not flagged.** `active_revealed_text` holds only
phrases still in force. When a customer says *"actually, ignore that — I need running
shoes"*, the belt constraints leave the active list rather than being marked stale, so
nothing downstream has to reason about conversation history.

State is written **only after retrieval succeeds**. If a turn raises, the session is left
untouched rather than half-updated — the evaluator swallows exceptions and continues, so
a partially-written state would corrupt every later turn. This invariant is enforced by
`test_failed_retrieval_does_not_commit_turn_or_history`, and it caught a real regression
during integration.

### Override handling

Intent override is one of the four scored scenarios, and phrase-matching alone does not
survive real customers. Two mechanisms work together:

- **Explicit retraction** — "ignore that", "scratch that", "on second thought", and
  similar phrasings across three verbs and six objects
- **Category conflict** — when a newly stated category contradicts the stored one,
  attributes attached to the old category are dropped regardless of wording. This
  catches *"skip the belt, show me shoes"*, which contains no retraction phrase at all

### Retrieval

BM25 over an in-memory SQLite FTS5 index across `title`, `categories`, `features`,
`details`, `store` and `description`.

The per-field weights are `(title 2.0, categories 4.0, features 8.0, details 8.0,
store 1.5, description 1.0)`, inverted from the conventional title-heavy default. The
customer's stated constraints are drawn from a product's `features` and `details`; they
never quote a title. Weighting title heavily points the scorer at the wrong evidence —
measured, a title-dominant configuration scores *below* weighting every field equally.

### Clarification policy

The agent asks only while asking still pays. It scores what it already knows — weighting
`material`, `size`, `budget`, `feature` and `use_case` at 3, and `color`, `style`,
`brand` at 1, plus a bonus for anything classified as a hard requirement — and stops
once that clears a threshold (4 browsing, 5 buying).

Which attribute to ask is chosen from one of four category-aware priority orders, since
the useful question differs by product type: clothing and accessories lead with
`material`, shoes and the default order lead with `feature`, and shoes deprioritise
`budget` relative to everything else. Attributes already known or already asked are
skipped, and once targeted questions stop yielding, the agent falls back to a broad ask
rather than a narrow one.

### Ranking

The reranker scores every retrieved candidate on constraint satisfaction, category match,
price fit, negative preferences, review quality, and the BM25 score itself, then applies
penalties for violating a hard constraint. Retrieval remains the dominant term at 0.70 —
structured state refines its ordering rather than overriding it, because a single noisy
extraction should not outweigh the lexical evidence.

---

## What we found

Three measurements changed what we built. They are the substance of the project, and each
is reproducible from `decisions.md`.

### 1. The starter wastes nine of its ten turns

The supplied agent never sets `ask_attribute`. The simulated customer therefore replies
with a content-free sentence on every turn after the first, and the agent — which queries
only the newest message — searches that sentence. In a browsing session, nine of ten
turns retrieve against text containing no product information at all.

Asking a question and remembering the answers took the score from **0.107 to 0.728**. An
ablation separates the two:

| | ask | accumulate | Score |
|---|---|---|---|
| baseline | no | no | 0.1067 |
| ask only | yes | no | 0.4856 |
| accumulate only | no | yes | 0.2284 |
| both | yes | yes | **0.7284** |

Asking is worth about three times more than remembering, and the two are superadditive:
separately they are worth +0.379 and +0.122, together +0.622. Asking *generates*
constraint text; accumulating *retains* it. Neither is worth much alone.

### 2. Retrieval was never the bottleneck

We measured how deep you must look before the target appears:

| cutoff | recall |
|---|---|
| recall@10 | 0.850 |
| recall@100 | 0.990 |
| **recall@500** | **1.000** |

Every target is retrievable. Of the sessions we were missing, 21 had the target at rank
11–50 and none were absent from the pool. **The failure mode is ranking, not retrieval** —
the target loses a close fight against near-duplicates.

That redirected the team's remaining effort onto ranking, and it retired two ideas that
looked obviously worthwhile:

- **Hard-constraint filtering.** Requiring must-have terms removes products already
  ranked *below* the target and keeps every product ranked *above* it, because a
  near-duplicate satisfies the same constraints. Measured at zero. You cannot filter your
  way out of a ranking problem when the competitors are valid matches.
- **Dense retrieval as a second candidate source.** A second retriever exists to find
  what the first one missed. Ours misses nothing.

### 3. Dense retrieval helps ranking, but not enough to ship

We built it anyway, as latent semantic analysis over the catalog (TF-IDF, truncated SVD
to 300 dimensions) rather than a pretrained encoder — a transformer downloads ~90MB of
weights when constructed, and the submission rules warn final scoring may run with
network access disabled, which would raise before the first turn.

As a weighted term it produced the best MRR we measured all weekend (0.5832 → 0.5957) but
cost Hit@10 (0.930 → 0.920), which carries more weight. Constrained to break ties only
between candidates already scored as near-equal, it held Hit@10 at 0.930 and still
improved MRR — but the net was +0.0008, against adding scikit-learn and scipy, twenty
seconds of startup, and 60MB of vectors.

**Left disabled and documented rather than shipped.** The code is on
`a/dense-lsa-retrieval`.

Its limitation is measurable and worth stating: LSA relates words that co-occur in *this
catalog* (`leather`/`cowhide` +0.43, `purse`/`handbag` +0.73) but not everyday paraphrase
(`comfy`/`comfortable` −0.07, `waterproof`/`water resistant` 0.00).

---

## Setup

**Requirements:** Python 3.10 or later. Nothing else — the scoring path uses only the
standard library and SQLite's bundled FTS5 extension.

```bash
git clone https://github.com/Nathan0820/techjam-conversational-search.git
cd techjam-conversational-search
```

Download `catalog.jsonl.gz` from the [participant kit release][kit], verify it, and
decompress it into `data/`:

```bash
curl -L -o catalog.jsonl.gz \
  "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz"
curl -L -o SHA256SUMS \
  "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS"

grep catalog.jsonl.gz SHA256SUMS | sha256sum -c -    # expect: OK
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
wc -l data/catalog.jsonl                             # expect: 50000
```

[kit]: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit

---

## Reproducing our results

```bash
python -m evaluator.local_evaluator
```

Roughly 55 seconds. Writes `results.json` and prints:

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

Run the test suite:

```bash
python -m unittest discover -s tests      # 174 tests
```

Every experiment behind those numbers, including the ones we rejected, is recorded in
`decisions.md` with what changed, what the metrics did, and why.

---

## Try it interactively

A local console for driving the pipeline by hand — useful for seeing the dialogue state
evolve turn by turn, which aggregate metrics hide.

```bash
python -B frontend/server.py     # http://127.0.0.1:8000
```

It shows the agent's recommendations, the question it chose, and what it currently knows
about you. "New session" re-runs the full 200-session evaluation against the live agent
and refreshes the metrics.

Worth trying, because it exercises the override path:

```
"I'm looking for a leather belt"
"actually ignore that, I need running shoes"
```

The belts disappear — `material: leather` leaves the active state rather than lingering.

---

## Repository layout

```
starter/agent.py             the Agent the evaluator drives; retrieval and composition
dialogue/                    conversation understanding and policy
  state.py                     SessionState, the shared object
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
evaluator/                   official evaluator, unmodified
```

---

## Cost, latency and dependencies

| | |
|---|---|
| Model API calls | **none** |
| Estimated cost | **$0.00** |
| Reported token usage | 0 prompt, 0 completion |
| Network access required | **none**, at build or run time |
| Third-party packages on the scoring path | **none** |
| Startup (index build, once) | ~20s |
| Full 200-session evaluation | ~55s, about 90ms per turn |

The agent is deterministic. There is no fallback path to describe because there is no
external dependency to fall back from.

The latent semantic index is not on `main`; it lives on the `a/dense-lsa-retrieval`
branch, where it requires scikit-learn and ships weighted to zero. Its builder returns
`None` on any failure rather than raising, so even enabled it degrades to lexical
retrieval instead of crashing.

---

## Limitations and what we would do next

**Our public-set score is an upper bound.** The simulated customer's messages are built
by copying strings out of the target product's own catalog entry, so lexical matching
performs unusually well. The evaluator has a branch for sessions that ship their own
intent cards, and the specification reserves the right to paraphrase, so the private
sessions are likely worded differently. We expect the 800-session score to be lower, and
the gap to be largest for the components that lean hardest on exact wording.

**Slot extraction is rule-based.** It recognises a fixed vocabulary of materials, colours,
sizes, styles and use cases. It generalises to phrasings we did not enumerate only by
accident. With more time we would train a small sequence labeller on the constraint
strings the evaluator generates — allowed by the rules, and it would remove the largest
source of brittleness in the system.

**Intent detection and override handling still match phrases.** We generalised both well
beyond the simulator's fixed wording, and added a semantic category-conflict rule that
needs no phrase list. But *"nah, something else"* still leaves stale constraints, because
it names no replacement. Resolving that needs a model of what the customer is negating,
not a longer pattern list.

**We never used the user profile.** Each session ships an anonymized profile with
`preference_tags` and rating style. Personalising against it is explicitly in scope and we
did not attempt it — the effort went to components we could measure a benefit from, and
with 200 sessions across four scenarios we judged we could not distinguish a real
personalisation effect from noise.

**No cross-encoder.** The reranker has the interface for one and it is unused. A
pretrained cross-encoder would rescore the top ~30 candidates using full query-document
attention — the strongest available answer to the near-duplicate problem we identified,
and the thing we would build first with another day.

**The score is saturated for this architecture.** Seven independent ideas measured on the
final day all landed within ±0.002. With recall at 1.000 and Hit@10 at 0.930, further
gains need a fundamentally better scorer rather than another parameter.

**A note on tuning.** Where several configurations scored within about one session of each
other on 200 samples, we chose round numbers and stopped, rather than selecting the
winner. Those differences are noise, and fitting them would not transfer to the hidden
set. `decisions.md` records where we did this deliberately.

---

## Team

| | area |
|---|---|
| **lewwai** | Retrieval — query construction, BM25 field weighting, `retrieve()` interface, recall and failure diagnostics, latent semantic index |
| **E Shen** (WongES05) | Dialogue — `SessionState`, slot extraction, accumulation, intent detection, override handling, hard/soft classification, clarification policy |
| **Nathan0820** | Ranking — feature extraction, reranker, ranking diagnostics; local test console |

Cross-cutting work — integration, code review with measurements, and the experiment log —
was shared, and every change to another member's module was reviewed by its owner before
merging.

---

## Data

Derived from **Amazon Reviews 2023** (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry`.
The catalog is read-only and is not redistributed in this repository. See
`DATA_ATTRIBUTION.md`.

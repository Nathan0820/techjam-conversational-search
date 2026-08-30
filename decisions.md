# Decisions Log

Every change gets measured on the 200 public dev sessions before it ships.
Run: `python -m evaluator.local_evaluator` (~24s), compare against the entry below it.

Metrics: `TechnicalScore = 0.50 x Hit@10 + 0.30 x MRR + 0.20 x Efficiency`,
`Efficiency = clip((11 - MTTC) / 10, 0, 1)`.

Entry format: what changed -> what the numbers did -> why I think that happened.
The last part is the one that matters. Do not skip it.

---

## E0 — Baseline (weak BM25 starter, unmodified)

**Date:** 2026-08-29
**Changed:** nothing. Reproduced the shipped starter to verify setup.

| metric | value |
|---|---|
| TechnicalScore | 0.10671 |
| Hit@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |

| scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.2375 | 0.126508 | 8.625 |
| browsing | 80 | 0.025 | 0.004514 | 10.75 |
| intent_override | 30 | 0.133333 | 0.104167 | 10.066667 |
| boundary | 10 | 0.0 | 0.0 | 11.0 |

**Matches `docs/baseline_results.json` exactly.** Setup verified.

**Observations:**
- Browsing (40% of the set) returns 2 hits out of 80. Boundary returns 0 out of 10.
- The starter never sets `ask_attribute`, so the simulated customer replies with a
  content-free sentence (`local_evaluator.py:171`) on every turn after the first.
- The starter queries only the current message (`agent.py:86`), so it discards
  everything revealed earlier in the session.
- Consequence: in a browsing session, 9 of 10 turns search a sentence containing no
  product information at all.

---

## E1 — Ask an attribute every turn + query from the whole session

**Date:** 2026-08-29
**Owner:** A (retrieval), with throwaway stubs standing in for B

**Changed** (`starter/agent.py`, ~15 lines):
1. `self._sessions` became `dict[str, list[str]]` — each session stores every
   customer message, oldest first. *(stub; B replaces with `DialogState`)*
2. Query is now built from `" ".join(history)` instead of `user_message`.
3. `ask_attribute` rotates `feature -> material -> color` by turn number instead of
   returning `None`. *(stub; B replaces with a real ask policy)*

| metric | E0 | E1 | delta |
|---|---|---|---|
| TechnicalScore | 0.10671 | **0.728419** | +0.62 |
| Hit@10 | 0.125 | **0.850** | +0.725 |
| MRR | 0.068034 | **0.540063** | +0.472 |
| MTTC | 9.81 | **3.93** | -5.88 |
| Efficiency | 0.119 | **0.707** | +0.588 |

| scenario | n | Hit@10 E0 | Hit@10 E1 | MTTC E1 |
|---|---|---|---|---|
| buying | 80 | 0.2375 | 0.8625 | 3.4 |
| browsing | 80 | 0.025 | 0.8375 | 4.0 |
| intent_override | 30 | 0.133333 | 0.866667 | 4.633333 |
| boundary | 10 | 0.0 | 0.800 | 5.5 |

**Why it happened** (ablation below, not speculation):

The simulated customer's replies are copied verbatim out of the target product's own
catalog entry, so once the agent gets the customer to speak at all, BM25 is matching a
product's exact words against an index containing those exact words. That is why hits
land at rank 1 rather than merely inside the top 10.

Two changes, measured independently:

| config | ask | accumulate | Score | Hit@10 | MTTC | browsing Hit@10 |
|---|---|---|---|---|---|---|
| E0 baseline | no | no | 0.1067 | 0.125 | 9.81 | 0.025 |
| A ask only | yes | no | 0.4856 | 0.555 | 6.48 | 0.575 |
| B accumulate only | no | yes | 0.2284 | 0.270 | 8.60 | 0.087 |
| E1 both | yes | yes | **0.7284** | **0.850** | **3.93** | **0.838** |

Asking is worth roughly 3x more than accumulating (+0.379 vs +0.122 on TechnicalScore).
But the two are superadditive: 0.379 + 0.122 = 0.501, while both together give +0.622.
The extra **+0.121 is interaction**.

The mechanism behind the interaction: asking *generates* constraint text, accumulating
*retains* it. Accumulating without asking just saves up content-free replies — row B
barely moves browsing (0.025 -> 0.087). Asking without accumulating throws away each
turn what the previous turn revealed. Neither is worth much alone.

**Still untested:** whether the specific attribute rotation matters, or only that we ask
*something*. Rotating `brand -> category -> budget` should score far worse (those buckets
are empty on the public set) but this has not been run. B should test it when replacing
the stub.

**Known caveat — measured, not speculative:**
The evaluator builds the customer's sentences by copying strings out of the target
product's own catalog entry (`local_evaluator.py:52-71`). Example, session public_0006:

```
catalog features : "Drawstring closure", "High quality mesh for maximum
                    breathability to keep you cool"
customer says    : "For that, what matters is: Drawstring closure; High quality
                    mesh for maximum breathability to keep you cool."
```

So BM25 is matching a product's own words back against an index containing those
exact words. The improvement mechanism (ask, then remember) is legitimate and would
work in a real system; the *magnitude* is inflated by how the local test data is
generated. The hidden 800 sessions may ship real intent cards
(`local_evaluator.py:204-213` has a branch for exactly that) or paraphrase the
wording, in which case exact-word overlap shrinks. Treat 0.728 as an upper bound.

**Not yet isolated:** how much of the gain came from asking vs. from accumulating.
Needs an ask-only ablation run.

---

## E2 — Recall diagnostic: are the remaining misses retrieval or ranking?

**Date:** 2026-08-30
**Owner:** A
**Changed:** nothing in the agent. Diagnostic only — ran the normal dialogue loop but
retrieved 1000 deep each turn and recorded the best rank the target ever reached.

**Question:** Hit@10 is 0.850. Of the 30 sessions that miss, is the target absent from
the candidate pool (retrieval failure, A's problem) or present but ranked too low
(ranking failure, C's problem)?

**Predicted before running:** recall@500 close to 0.85, i.e. misses are mostly
retrieval failures.

**Result:**

| cutoff | recall | sessions |
|---|---|---|
| recall@10 | 0.850 | 170/200 |
| recall@50 | 0.955 | 191/200 |
| recall@100 | 0.990 | 198/200 |
| recall@500 | **1.000** | 200/200 |
| never found | 0.000 | 0/200 |

Where the 30 top-10 misses actually sit:

| best rank reached | sessions |
|---|---|
| 11-50 | 21 |
| 51-100 | 7 |
| 101-500 | 2 |
| never retrieved | 0 |

Recall@1000 is 1.000 in every scenario bucket (buying, browsing, override, boundary).

**Prediction was wrong, and why:** the assumption was that a miss means the target is
unfindable. It isn't. BM25 on the customer's verbatim wording pulls the target into the
pool every single time. What it cannot do is separate the target from its near-duplicates
— a query like "mesh drawstring breathable shorts" matches hundreds of near-identical
items in a 50,000-product clothing catalog. Finding it is easy; winning the close fight
against lookalikes is hard.

**Consequences:**
1. **Dense retrieval is not a score play.** Recall is already 1.000 at depth 500; a second
   retriever cannot improve a metric that is maxed out. Demoted to "insurance if the
   hidden set paraphrases wording", not a priority.
2. **All remaining headroom is ranking.** Hit@10 0.850 -> 1.000 (+0.075 weighted) and
   MRR 0.540 -> ~0.9 (+0.11 weighted). Up to **+0.19 TechnicalScore** available purely
   by reordering candidates we already retrieve. That is C's box.
3. **A -> C handoff depth settled: 100-500 candidates.** 100 captures 99% of targets,
   500 captures 100%. Fewer throws away winnable sessions; more just slows the reranker.

**Caveat:** measured on the public set, where the customer quotes catalog text verbatim
(see E1). Recall on the hidden 800 could be lower if wording is paraphrased, which is the
only remaining argument for building the dense route.

---

## Constraint distribution (reference, measured on public set)

`classify_constraint` over all 800 constraints in the 200 public sessions:

| bucket | count | share |
|---|---|---|
| feature | 404 | 50.5% |
| material | 302 | 37.8% |
| color | 60 | 7.5% |
| style | 19 | 2.4% |
| size | 11 | 1.4% |
| use_case | 4 | 0.5% |
| brand / category / budget | 0 | 0% |

Every session carries exactly 4 unique constraints (2 hard + 2 soft), max 2 revealed
per turn, each revealed only once. `ask_attribute="other"` is a wildcard that matches
any undisclosed constraint (`local_evaluator.py:180`).

**Do not hardcode against this table** — it is an artifact of `classify_constraint`'s
keyword lists applied to derived intent cards. The private set may differ.

---

## Template

```
## E<n> — <short title>

**Date:**
**Owner:**
**Changed:**

| metric | prev | new | delta |
|---|---|---|---|
| TechnicalScore | | | |
| Hit@10 | | | |
| MRR | | | |
| MTTC | | | |

**Predicted before running:**
**Why I think that happened:**
**Keep or revert:**
```

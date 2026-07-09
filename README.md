# FDA Approval Analyzer

A reproducible, evidence-grounded read on a biotech's odds of FDA approval — for
any ticker, not a curated list. Point it at a company and it pulls real regulatory
filings and trial data, extracts structured evidence, scores that evidence against
a fixed rubric, and blends the result with published historical base rates into
one final probability, with every input inspectable.

## The problem this addresses

Biotech investing is unusually exposed to information asymmetry: a retail investor
reading an InvestorsHub thread the night before a PDUFA date is working from the
same noisy, sentiment-driven inputs as everyone else on that thread, while the
actual determinants of approval — trial design rigor, manufacturing readiness,
prior regulatory history, advisory committee dynamics — sit scattered across SEC
filings and clinicaltrials.gov records that few people actually read end to end.
The risk isn't just "the stock could go down" — it's that decisions get made on
vibes, a single analyst's gut call, or whoever posted most confidently that
morning. This tool exists to replace that with something you can point at any
ticker and get the same structured read every time, built from the same public
sources, scored the same deterministic way.

## Safety & governance design

This isn't "AI trading bot" software, and the architecture reflects that on
purpose:

**The LLM is a feature extractor, never the decision maker.** Gemini reads the
assembled evidence (filing excerpts, trial records, community corpus) and returns,
per rubric criterion, an evidence fraction from 0.0–1.0 plus a one-line
justification — nothing more. It never sees a "should this approve?" question and
never outputs a probability. All of the actual math — multiplying fractions by
point caps, summing sections, applying risk deductions, mapping totals to bands —
happens in plain deterministic Python (`fda_analyzer.py::score()`). Run the same
evidence through it twice and you get the same number twice. That's a real
constraint on how the LLM is used, not a caveat bolted on afterward.

**The blended-odds model refuses false precision when evidence is thin.**
`loa_model.py` produces the one headline number by blending two different
questions: what does this drug's *own* evidence say (the rubric score), and what
does history say about drugs at this stage/therapeutic-area/designation profile
(a base rate built from published BIO/Biomedtracker phase-transition statistics
and FDA CDER data)? The weight given to the drug-specific evidence scales with how
much evidence was actually found — `EVIDENCE_WEIGHT = {'rich': 0.80, 'thin': 0.50,
'sparse': 0.20}`. When EDGAR and ClinicalTrials.gov turn up almost nothing, the
rubric score is close to an uninformative neutral 50 and the model *knows* that,
so it defers to the historical base rate instead of reporting a confident-looking
number built on nothing.

**A transient API failure produces an honest "evidence unavailable" state, never
a silent wrong score.** This is a real bug the fund's own audit caught and fixed:
an earlier version of the EDGAR fetcher had no retry logic, so a transient outage
returned an empty result indistinguishable from "this company genuinely has no
recent filings" — and the scorer then defaulted every criterion to the neutral
0.5 fraction, producing a flat, misleadingly neutral 50/100 score that looked
like a real assessment but wasn't. The fix (`_edgar_search()` /
`fetch_clinicaltrials()`) adds retry-with-backoff on 429/500/503, and logs every
failure distinctly from a genuine empty result, so data starvation is visible in
the logs instead of silently masquerading as a coin-flip score.

**Community-corpus scraping is rate-limited and backs off, not hammers.**
`ihub_corpus_builder.py` pulls from InvestorsHub's public RSS feed and
single-message pages (no login, no bypassing an authwall) with a jittered delay
between requests specifically so traffic doesn't look metronomic, and aborts the
run after four consecutive 403/429/503 responses rather than continuing to push
through what looks like active throttling.

## Autonomy boundary

This tool computes and reports. It does not trade, does not file anything, and
does not act on the odds it produces — every output is read by a human before any
decision gets made on it. The only thing resembling an autonomous action is the
optional `--telegram` flag, and that's a one-way, read-only notification: it posts
a summary to a chat and nothing else. There is no code path anywhere in this
repo that places an order, submits a filing, or otherwise acts on its own
output.

## Ethical / responsible-use notes

This is not investment advice. FDA approval outcomes are inherently uncertain —
even well-designed models built on solid historical base rates get individual
calls wrong, and no rubric score should be read as a guarantee. This tool is
meant to augment due diligence, not replace it: it structures and grounds
evidence gathering so a human can reason about a name faster and with fewer
blind spots, not so the human can skip the reasoning. The historical-base-rate
blend exists specifically to guard against a known failure mode — over-weighting
a single vivid data point, like one AdCom vote or one bullish analyst note, as if
it were more determinative than it really is. That's not a hypothetical risk;
it's the exact trap an earlier, rubric-only version of this tool was prone to,
which is why the blended model exists at all.

## How it works

```
SEC EDGAR full-text search (10-K/10-Q/8-K, biased to last 2yr)  ─┐
ClinicalTrials.gov v2 API (recent trials, design, status)        ├─► Gemini feature
Optional investor-community corpus (curated, keyword-filtered)  ─┘   extraction
                                                                        │
                                                              per-criterion evidence
                                                              fraction (0.0-1.0)
                                                                        │
                                                                        ▼
                                                          deterministic rubric scoring
                                                          (fda_rubric.json: 5 sections,
                                                           100 pts, risk deductions)
                                                                        │
                                                                        ▼
                                                      historical base-rate blend
                                                      (loa_model.py: BIO/Biomedtracker
                                                       phase-transition rates × TA ×
                                                       designation multipliers, weighted
                                                       by evidence quality)
                                                                        │
                                                                        ▼
                                                            ONE final approval-odds %
                                                            (fully inspectable — every
                                                             input stays visible)
```

Components:
- **`fda_analyzer.py`** — the core CLI. Fetches evidence, runs Gemini feature
  extraction, scores deterministically, writes `fda_scores/<TICKER>_<date>.json`.
- **`fda_rubric.json`** — the scoring schema: 5 sections (Clinical Evidence
  Quality, Regulatory Process Signals, Manufacturing & CMC Readiness, Company
  Behavior Signals, External Validation), sub-criteria with point caps, risk
  deductions, and probability bands.
- **`loa_model.py`** — the historical base-rate model and the final blended-odds
  calculation. Fully cited sources in the module docstring.
- **`biotech_catalyst_scanner.py`** — a keyless, industry-wide EDGAR 8-K sweep
  that finds biotechs with live regulatory catalysts (PDUFA dates, CRLs, AdCom
  activity, designations) without a hand-maintained watchlist.
- **`score_catalysts.py`** — batch-runs `fda_analyzer.py` over a scanner
  watchlist, idempotent per day.
- **`ihub_corpus_builder.py`** / **`ihub_probe.py`** — optional, rate-limited
  InvestorsHub corpus builder that gives `fda_analyzer.py` a richer evidence base
  than EDGAR + ClinicalTrials.gov alone for tickers with an active retail
  community.
- **`fda_dashboard.py`** — a Flask UI (port 5007) over the analyzer: paste a
  ticker, get a score ring, stage pipeline, rubric breakdown, and risk flags.

## Usage

Install dependencies and set your API key:

```bash
pip install -r requirements.txt
cp .env.example .env    # then edit .env and set GEMINI_API_KEY
export $(cat .env | xargs)   # or use a tool like direnv / python-dotenv
```

Score a single ticker:

```bash
python3 fda_analyzer.py MNKD --drug Afrezza
python3 fda_analyzer.py SAVA --indication "Alzheimer's disease" --telegram
```

Find biotechs with live regulatory catalysts (no watchlist needed):

```bash
python3 biotech_catalyst_scanner.py --days 45 --max-price 5
```

Batch-score everything the scanner just found:

```bash
python3 score_catalysts.py --limit 20
```

Launch the dashboard:

```bash
python3 fda_dashboard.py
# → http://localhost:5007
```

Optional: build a richer evidence corpus for a specific ticker from
InvestorsHub (rate-limited, no login):

```bash
python3 ihub_corpus_builder.py MNKD
```

## Requirements

- Python 3.9+
- A Gemini API key (`GEMINI_API_KEY`) — required, used only for evidence
  extraction as described above
- Optional: `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` for `--telegram` alerts

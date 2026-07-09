#!/usr/bin/env python3
"""
fda_analyzer.py — score a biotech's FDA approval odds against a structured rubric.

Usage:  fda_analyzer.py TICKER [--drug "Drug Name"] [--indication "..."]
        fda_analyzer.py MNKD --drug Afrezza

Design rule: the LLM is a FEATURE EXTRACTOR, never the decision maker.
Gemini reads the assembled evidence and returns, per rubric criterion, an
awarded fraction (0.0-1.0) + a one-line justification. Deterministic Python
multiplies fractions by the criterion max, sums sections, applies risk
deductions, and maps the total to a probability band. The score is reproducible
from the same evidence regardless of LLM mood.

Sources (free, no key): SEC EDGAR full-text + filings, ClinicalTrials.gov v2 API.
Output: fda_scores/<TICKER>_<date>.json + console table (+ optional --telegram).

Requires the GEMINI_API_KEY environment variable (see .env.example).
Optional TELEGRAM_TOKEN / TELEGRAM_CHAT_ID env vars enable --telegram alerts.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

SCRIPTS   = os.path.dirname(os.path.abspath(__file__))
RUBRIC    = os.path.join(SCRIPTS, 'fda_rubric.json')
OUT_DIR   = os.path.join(SCRIPTS, 'fda_scores')
LOG_FILE  = os.path.join(SCRIPTS, 'fda_analyzer.log')
GEMINI_MODEL    = 'gemini-3.1-flash-lite'
GEMINI_URL      = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
EDGAR_UA  = os.environ.get('EDGAR_USER_AGENT', 'FDA-Approval-Analyzer research@example.com')
SEC_FTS   = 'https://efts.sec.gov/LATEST/search-index?q='  # full-text search
SEC_FTS2  = 'https://efts.sec.gov/LATEST/search-index'
CT_API    = 'https://clinicaltrials.gov/api/v2/studies'

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a') as f:
        f.write(f'[{ts}] {msg}\n')
    print(msg)

def load_rubric():
    return json.load(open(RUBRIC))

def load_gemini_key():
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        print('ERROR: GEMINI_API_KEY environment variable is not set.\n'
              'Get a key at https://aistudio.google.com/apikey and set it, e.g.:\n'
              '  export GEMINI_API_KEY="your-key-here"', file=sys.stderr)
        sys.exit(1)
    return key

# ── evidence fetchers ─────────────────────────────────────────────────────────
_EDGAR_DOC_EXCERPT_CHARS = 1800
_EDGAR_MAX_DOCS = 4

def _edgar_search(params):
    """Shared retry-with-backoff wrapper for EDGAR full-text search. Returns hits[]
    or [] after 3 failed attempts (logged — a failure no longer looks identical to
    a genuinely empty result, which previously masked a transient outage as
    "no evidence found" and produced a hollow, misleadingly neutral rubric score)."""
    r = None
    for attempt in range(3):
        try:
            r = requests.get('https://efts.sec.gov/LATEST/search-index', params=params,
                             headers={'User-Agent': EDGAR_UA}, timeout=20)
            if r.status_code == 200:
                return r.json().get('hits', {}).get('hits', [])
            if r.status_code in (429, 500, 503):
                log(f'EDGAR {r.status_code} for {params.get("q")}, retry {attempt+1}/3')
                time.sleep(1.5 * (attempt + 1)); continue
            log(f'EDGAR {r.status_code} for {params.get("q")} — not retrying'); return []
        except Exception as e:
            log(f'EDGAR fetch exception for {params.get("q")} (attempt {attempt+1}/3): {e}')
            time.sleep(1.5)
    log(f'EDGAR fetch gave up for {params.get("q")} after 3 attempts'); return []


def fetch_edgar(ticker):
    """Real filing TEXT (not just headlines) via SEC full-text search, biased to
    the last 2 years so an old boilerplate exhibit (e.g. a 2020 EX-31.2
    certification) doesn't crowd out the recent 8-K that actually carries the
    PDUFA/CRL/trial-result news. An earlier version fetched only filing headlines,
    which handed the LLM lines like "8-K 2026-06-29: Company Name" with zero
    substantive content — every rubric criterion then defaulted to the neutral 0.5
    fraction, producing a flat 50/100 "coin-flip" score that wasn't a real
    assessment at all, just the fallback for having nothing to read.
    """
    from datetime import date, timedelta
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=730)).isoformat()
    hits = _edgar_search({'q': ticker, 'forms': '10-K,10-Q,8-K',
                          'startdt': start, 'enddt': end})
    if not hits:
        # fall back to no date filter — better an old filing than nothing
        hits = _edgar_search({'q': ticker, 'forms': '10-K,10-Q,8-K'})
    # Newest first, then 8-Ks first (stable sort composes: 8-Ks bubble to the top,
    # each group still newest-first) — 8-Ks carry event-driven regulatory news
    # (PDUFA/CRL/topline), which matters more here than routine 10-K/10-Q filings.
    hits.sort(key=lambda h: h.get('_source', {}).get('file_date', ''), reverse=True)
    hits.sort(key=lambda h: h.get('_source', {}).get('form') != '8-K')

    out = []
    for h in hits[:_EDGAR_MAX_DOCS]:
        src = h.get('_source', {})
        form, fdate = src.get('form', '?'), src.get('file_date', '')
        cik = (src.get('ciks') or [None])[0]
        adsh = h.get('_id', '').split(':')
        excerpt = None
        if cik and len(adsh) > 1:
            url = (f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/'
                   f'{adsh[0].replace("-", "")}/{adsh[1]}')
            try:
                time.sleep(0.6)
                doc = requests.get(url, headers={'User-Agent': EDGAR_UA}, timeout=20)
                if doc.status_code == 200:
                    text = re.sub(r'<[^>]+>', ' ', doc.text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    excerpt = text[:_EDGAR_DOC_EXCERPT_CHARS]
            except Exception as e:
                log(f'EDGAR doc fetch failed ({form} {fdate}): {e}')
        out.append({'form': form, 'date': fdate,
                    'excerpt': excerpt or '(document text unavailable — headline only)'})
    return out

def fetch_clinicaltrials(query):
    """Recent trials matching the drug/company; pull design + status. Same retry
    treatment as fetch_edgar — a transient failure here previously looked
    identical to "genuinely no trials exist"."""
    out = []
    r = None
    for attempt in range(3):
        try:
            r = requests.get(CT_API, params={'query.term': query, 'pageSize': 8,
                                             'sort': 'LastUpdatePostDate:desc'}, timeout=20)
            if r.status_code == 200:
                break
            if r.status_code in (429, 500, 503):
                log(f'ClinicalTrials {r.status_code} for {query!r}, retry {attempt+1}/3')
                time.sleep(1.5 * (attempt + 1)); continue
            log(f'ClinicalTrials {r.status_code} for {query!r} — not retrying'); return out
        except Exception as e:
            log(f'ClinicalTrials fetch exception for {query!r} (attempt {attempt+1}/3): {e}')
            time.sleep(1.5)
    if not r or r.status_code != 200:
        log(f'ClinicalTrials fetch gave up for {query!r}')
        return out
    try:
        for s in r.json().get('studies', []):
            ps = s.get('protocolSection', {})
            idm = ps.get('identificationModule', {})
            dm  = ps.get('designModule', {})
            stm = ps.get('statusModule', {})
            out.append({
                'nct': idm.get('nctId'),
                'title': (idm.get('briefTitle') or '')[:160],
                'phase': dm.get('phases'),
                'status': stm.get('overallStatus'),
                'enroll': dm.get('enrollmentInfo', {}).get('count'),
                'design': dm.get('designInfo', {}).get('allocation'),
            })
    except Exception as e:
        log(f'ClinicalTrials parse failed for {query!r}: {e}')
    return out

# ── community corpus (the real edge — EDGAR+CT alone score "sparse") ─────────
CORPUS_KEYWORDS = [
    'fda', 'pdufa', 'approval', 'approved', 'crl', 'complete response',
    'adcom', 'advisory committee', 'phase 3', 'phase iii', 'endpoint',
    'efficacy', 'safety', 'manufactur', 'cmc', 'inspection', 'form 483',
    'warning letter', 'refuse to file', 'clinical hold', 'nda', 'bla',
    'resubmission', 'label', 'topline', 'readout', 'p-value', 'statistically',
    'cash runway', 'dilution', 'offering', 'partnership', 'script', 'trx',
]

def load_corpus_excerpts(ticker, drug, max_posts=120, max_chars=60000):
    """Curated excerpts from <ticker>_corpus.json (ProBoards/Telegram scrape).

    Deterministic selection, newest-first: keep posts matching >=2 regulatory
    keywords, cap count and total chars so the Gemini prompt stays bounded.
    Returns (excerpts, total_posts_in_corpus).
    """
    path = os.path.join(SCRIPTS, f'{ticker.lower()}_corpus.json')
    if not os.path.exists(path):
        return [], 0
    try:
        posts = json.load(open(path))
    except Exception as e:
        log(f'corpus load failed ({path}): {e}')
        return [], 0
    keys = CORPUS_KEYWORDS + ([drug.lower()] if drug else [])
    scored = []
    for p in posts:
        text = (p.get('text') or '')
        lower = text.lower()
        hits = sum(1 for k in keys if k in lower)
        if hits >= 2:
            scored.append((p.get('date') or '', hits, text))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)  # newest first
    out, chars = [], 0
    for date, hits, text in scored:
        snippet = f'[{date}] {text[:700]}'
        if chars + len(snippet) > max_chars or len(out) >= max_posts:
            break
        out.append(snippet)
        chars += len(snippet)
    return out, len(posts)

# ── Gemini feature extraction (NO approval vote) ──────────────────────────────
def gemini_extract(prompt, key):
    for attempt in range(3):
        try:
            res = requests.post(GEMINI_URL, params={'key': key},
                                json={'contents': [{'parts': [{'text': prompt}]}]}, timeout=45)
            if res.status_code == 200:
                return res.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            if res.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            log(f'Gemini {res.status_code}: {res.text[:150]}'); return None
        except Exception as e:
            log(f'Gemini exception: {e}'); time.sleep(2 ** attempt)
    return None

def build_prompt(ticker, drug, indication, rubric, edgar, trials, corpus_excerpts=None):
    crit_lines, risk_lines = [], []
    for sec in rubric['sections']:
        for c in sec['criteria']:
            crit_lines.append(f"  {c['key']} (sec {sec['id']}, max {c['max']}): {c['desc']}")
    for rd in rubric['risk_deductions']:
        risk_lines.append(f"  {rd['key']} ({rd['points']}): {rd['desc']}")
    return f"""You are a regulatory-evidence EXTRACTOR, not a decision maker. Do NOT predict approval.
For {ticker} (drug: {drug or 'unknown'}, indication: {indication or 'unknown'}), read the evidence
below and, for EACH criterion, return the fraction of its max points the evidence SUPPORTS
(0.0 = no/negative evidence, 0.5 = mixed/unclear, 1.0 = strong positive evidence), plus a
one-line justification citing what you saw. For each risk flag, return true only if the
evidence actively indicates it (default false when unknown).

CRITERIA:
{chr(10).join(crit_lines)}

RISK FLAGS:
{chr(10).join(risk_lines)}

EVIDENCE — EDGAR filings:
{json.dumps(edgar, indent=1)}

EVIDENCE — ClinicalTrials.gov:
{json.dumps(trials, indent=1)}

EVIDENCE — investor community corpus (curated excerpts, newest first; long-time
followers who track script counts, FDA correspondence, manufacturing issues and
trial minutiae. Opinionated: extract only FACTUAL claims — trial results, FDA
letters/meetings, inspection findings, prescription data, cash/runway figures.
Ignore hype, price targets and sentiment):
{chr(10).join(corpus_excerpts) if corpus_excerpts else '(no corpus available for this ticker)'}

Return ONLY valid JSON:
{{"criteria": {{"<key>": {{"fraction": 0.0-1.0, "why": "..."}}, ...}},
  "risks": {{"<key>": true/false, ...}},
  "evidence_quality": "rich|thin|sparse"}}
If evidence is sparse for a criterion, use fraction 0.5 and say so — do not invent facts."""

def parse_json(text):
    if not text:
        return None
    m = re.search(r'\{.*\}', text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None

# ── deterministic scoring ─────────────────────────────────────────────────────
def score(rubric, extracted):
    crit = extracted.get('criteria', {})
    risks = extracted.get('risks', {})
    section_scores, detail = [], []
    total = 0.0
    for sec in rubric['sections']:
        s_pts = 0.0
        for c in sec['criteria']:
            frac = float((crit.get(c['key']) or {}).get('fraction', 0.5))
            frac = max(0.0, min(1.0, frac))
            pts = round(frac * c['max'], 2)
            s_pts += pts
            detail.append({'section': sec['id'], 'key': c['key'], 'awarded': pts, 'max': c['max'],
                           'why': (crit.get(c['key']) or {}).get('why', '')[:140]})
        section_scores.append({'id': sec['id'], 'name': sec['name'],
                               'score': round(s_pts, 1), 'max': sec['max']})
        total += s_pts
    deductions = []
    for rd in rubric['risk_deductions']:
        if risks.get(rd['key']) is True:
            total += rd['points']
            deductions.append({'key': rd['key'], 'points': rd['points']})
    total = round(total, 1)
    band = next((b for b in rubric['probability_bands'] if b['min'] <= total <= b['max']),
                rubric['probability_bands'][-1])
    return {'total': total, 'sections': section_scores, 'deductions': deductions,
            'band': band, 'detail': detail}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ticker')
    ap.add_argument('--drug', default='')
    ap.add_argument('--indication', default='')
    ap.add_argument('--telegram', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rubric = load_rubric()
    key = load_gemini_key()
    query = args.drug or args.ticker
    log(f'Analyzing {args.ticker} (drug={args.drug or "?"})')

    edgar = fetch_edgar(args.ticker)
    trials = fetch_clinicaltrials(query)
    corpus_excerpts, corpus_total = load_corpus_excerpts(args.ticker, args.drug)
    log(f'  evidence: {len(edgar)} EDGAR hits, {len(trials)} trials, '
        f'{len(corpus_excerpts)} corpus excerpts (of {corpus_total} posts)')

    raw = gemini_extract(build_prompt(args.ticker, args.drug, args.indication, rubric,
                                      edgar, trials, corpus_excerpts), key)
    ext = parse_json(raw)
    if not ext:
        log('Gemini extraction failed — cannot score'); return 1

    result = score(rubric, ext)
    out = {'ticker': args.ticker, 'drug': args.drug, 'indication': args.indication,
           'generated': datetime.now(timezone.utc).isoformat(),
           'evidence_quality': ext.get('evidence_quality'),
           'evidence_counts': {'edgar': len(edgar), 'trials': len(trials),
                               'corpus_excerpts': len(corpus_excerpts),
                               'corpus_total_posts': corpus_total},
           **result}
    fn = f"{OUT_DIR}/{args.ticker}_{datetime.now().strftime('%Y%m%d')}.json"
    json.dump(out, open(fn, 'w'), indent=2)

    print(f"\n{'='*54}\n  FDA APPROVAL SCORE — {args.ticker} {('('+args.drug+')') if args.drug else ''}\n{'='*54}")
    for s in result['sections']:
        print(f"  §{s['id']} {s['name']:<32} {s['score']:>5.1f} / {s['max']}")
    for d in result['deductions']:
        print(f"  ! RISK {d['key']:<38} {d['points']:>5}")
    print(f"  {'-'*50}\n  TOTAL {result['total']:>6.1f} / 100   →  {result['band']['approval_prob']}  ({result['band']['label']})")
    print(f"  evidence: {ext.get('evidence_quality')} ({len(edgar)} filings, {len(trials)} trials, "
          f"{len(corpus_excerpts)} corpus excerpts)\n  saved: {fn}")

    if args.telegram:
        tg_token = os.environ.get('TELEGRAM_TOKEN')
        tg_chat = os.environ.get('TELEGRAM_CHAT_ID')
        if not tg_token or not tg_chat:
            print('Set TELEGRAM_TOKEN/TELEGRAM_CHAT_ID env vars to enable alerts — skipping Telegram send.')
        else:
            lines = [f"🧬 *FDA SCORE — {args.ticker}* {args.drug}",
                     f"*{result['total']:.0f}/100* → {result['band']['approval_prob']} ({result['band']['label']})"]
            for s in result['sections']:
                lines.append(f"  §{s['id']} {s['name']}: {s['score']:.0f}/{s['max']}")
            if result['deductions']:
                lines.append("  ⚠️ " + ", ".join(f"{d['key']} {d['points']}" for d in result['deductions']))
            lines.append(f"  _evidence: {ext.get('evidence_quality')}_")
            try:
                requests.post(f'https://api.telegram.org/bot{tg_token}/sendMessage',
                              json={'chat_id': tg_chat, 'text': '\n'.join(lines), 'parse_mode': 'Markdown'}, timeout=10)
            except Exception:
                pass
    return 0

if __name__ == '__main__':
    sys.exit(main())

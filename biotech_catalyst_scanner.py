#!/usr/bin/env python3
"""
biotech_catalyst_scanner.py: find biotechs with LIVE FDA/regulatory catalysts.

Answers "how do we find more biotech to score?" without a hand-maintained list.
Engine: an industry-wide SEC EDGAR full-text sweep of recent 8-Ks for regulatory
event language (PDUFA, CRL, Breakthrough, AdCom, sBLA accepted, topline...). EDGAR
returns the filer with its ticker + CIK inline, so every hit is ticker-resolved with
no fuzzy company-name matching. Free, keyless.

Output: biotech_catalysts.json, ranked [{ticker, company, catalysts[], latest_date,
newest_filing_url}], plus an optional Telegram top-N. This watchlist is what you then
feed to ihub_corpus_builder.py + fda_analyzer.py (the scoring half of the pipeline).

Usage:
  biotech_catalyst_scanner.py                 # last 45 days, all keyword classes
  biotech_catalyst_scanner.py --days 90 --telegram
  biotech_catalyst_scanner.py --min-severity 2   # only PDUFA/CRL/AdCom-tier events

Optional TELEGRAM_TOKEN / TELEGRAM_CHAT_ID env vars enable --telegram alerts.
"""
import argparse, json, logging, os, re, sys, time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import yfinance as yf
import loa_model

SCRIPTS  = Path(__file__).resolve().parent
OUT_JSON = SCRIPTS / 'biotech_catalysts.json'
LOG_FILE = SCRIPTS / 'biotech_catalyst_scanner.log'
UA       = {'User-Agent': os.environ.get('EDGAR_USER_AGENT', 'FDA-Approval-Analyzer research@example.com')}
FTS_URL  = 'https://efts.sec.gov/LATEST/search-index'
TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'

# Catalyst classes: phrase -> (label, severity). Severity 3 = binary/near-term decision,
# 2 = filing/review milestone, 1 = designation/early signal.
CATALYST_QUERIES = [
    ('"PDUFA date"',                 'PDUFA date set',            3),
    ('"Complete Response Letter"',   'CRL (rejection)',           3),
    ('"advisory committee"',         'AdCom scheduled/held',      3),
    ('"accepts for review"',         'FDA accepted filing',       2),
    ('"accepted for review"',        'FDA accepted filing',       2),
    ('"Biologics License Application"', 'BLA/sBLA activity',       2),
    ('"New Drug Application"',        'NDA activity',              2),
    ('"topline results"',            'Topline data readout',      2),
    ('"primary endpoint"',           'Pivotal endpoint reported', 2),
    ('"Breakthrough Therapy"',       'Breakthrough designation',  1),
    ('"Fast Track designation"',     'Fast Track designation',    1),
    ('"orphan drug"',                'Orphan designation',        1),
    ('"Priority Review"',            'Priority Review',           2),
]

# Catalyst labels that only pharma/biotech companies ever file; presence of any one
# confirms the ticker is biotech without a SIC lookup.
PHARMA_EXCLUSIVE = {
    'PDUFA date set', 'CRL (rejection)', 'FDA accepted filing', 'BLA/sBLA activity',
    'NDA activity', 'Breakthrough designation', 'Fast Track designation',
    'Orphan designation', 'Priority Review',
}
# EDGAR SIC codes for pharma/biotech/med-research (used only to vet ambiguous-only hits).
BIOTECH_SIC = {'2834', '2835', '2836', '8731', '3826', '3841'}

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [BIOCAT] %(message)s',
    handlers=[logging.FileHandler(str(LOG_FILE)), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

_sic_cache = {}
def is_biotech_sic(ticker, cik_map_rev):
    """Verify a ticker's SIC is pharma/biotech via the EDGAR submissions API. Cached."""
    if ticker in _sic_cache:
        return _sic_cache[ticker]
    cik = cik_map_rev.get(ticker)
    if not cik:
        _sic_cache[ticker] = True; return True  # unknown → don't over-filter
    try:
        r = requests.get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json',
                         headers=UA, timeout=15)
        sic = str(r.json().get('sic', '')) if r.status_code == 200 else ''
        ok = sic in BIOTECH_SIC
    except Exception:
        ok = True  # network hiccup → keep rather than wrongly drop
    time.sleep(0.5)
    _sic_cache[ticker] = ok
    return ok


def load_cik_ticker():
    """Return (cik->ticker, ticker->cik) maps from SEC."""
    try:
        d = requests.get(TICKERS_URL, headers=UA, timeout=20).json()
        fwd = {int(v['cik_str']): v['ticker'] for v in d.values()}
        rev = {v['ticker']: int(v['cik_str']) for v in d.values()}
        return fwd, rev
    except Exception as e:
        log.warning(f'company_tickers.json failed: {e}')
        return {}, {}


def sweep(query, start, end, cik_map):
    """Run one EDGAR full-text query for 8-Ks in [start,end]; yield hit dicts.

    EDGAR returns transient 500/429 under bursty load, so retry with backoff.
    """
    out = []
    r = None
    for attempt in range(4):
        try:
            r = requests.get(FTS_URL, headers=UA, timeout=25, params={
                'q': query, 'forms': '8-K', 'startdt': start, 'enddt': end})
            if r.status_code == 200:
                break
            if r.status_code in (500, 429, 403):
                time.sleep(1.5 * (attempt + 1)); continue
            log.warning(f'FTS {r.status_code} for {query}'); return out
        except Exception as e:
            log.warning(f'FTS exception for {query}: {e}'); time.sleep(1.5); continue
    if not r or r.status_code != 200:
        log.warning(f'FTS gave up ({r.status_code if r else "no resp"}) for {query}'); return out
    try:
        for h in r.json().get('hits', {}).get('hits', []):
            src = h.get('_source', {})
            names = src.get('display_names', [])
            name0 = names[0] if names else ''
            tk = re.search(r'\(([A-Z]{1,6})\)', name0)
            cik = re.search(r'CIK (\d+)', name0)
            ticker = tk.group(1) if tk else None
            if not ticker and cik:
                ticker = cik_map.get(int(cik.group(1)))
            if not ticker:
                continue
            company = re.sub(r'\s*\(.*', '', name0).strip()
            adsh = h.get('_id', '').split(':')[0]
            cikn = int(cik.group(1)) if cik else None
            url = (f'https://www.sec.gov/Archives/edgar/data/{cikn}/'
                   f'{adsh.replace("-","")}/{adsh}-index.htm') if cikn and adsh else ''
            out.append({'ticker': ticker, 'company': company,
                        'date': src.get('file_date', ''), 'url': url})
    except Exception as e:
        log.warning(f'FTS parse exception for {query}: {e}')
    return out


def fetch_prices(tickers, chunk=40):
    """Latest close per ticker via yfinance batch. Missing/failed -> absent from dict."""
    prices = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            df = yf.download(batch, period='1d', progress=False, threads=True)
            if df is None or df.empty:
                continue
            close = df['Close']
            if hasattr(close, 'columns'):          # multi-ticker -> DataFrame
                for t in batch:
                    try:
                        v = close[t].dropna()
                        if len(v):
                            prices[t] = float(v.iloc[-1])
                    except Exception:
                        pass
            else:                                   # single ticker -> Series
                v = close.dropna()
                if len(v):
                    prices[batch[0]] = float(v.iloc[-1])
        except Exception as e:
            log.warning(f'price batch failed ({batch[0]}..): {e}')
        time.sleep(1.0)  # gentle on Yahoo
    return prices


_MONTHS = ('january february march april may june july august september october '
           'november december').split()
_PDUFA_RE = re.compile(
    r'(?:PDUFA|target\s+action|goal)\s+(?:goal\s+)?(?:action\s+)?date[^.]{0,40}?'
    r'([A-Z][a-z]+)\s+(\d{1,2}),?\s+(20\d\d)', re.I)


def _parse_pdufa_text(txt):
    """Regex a PDUFA/target-action date out of filing text -> 'YYYY-MM-DD' or None.
    Tries the strict PDUFA-anchored form, then a looser 'goal/action date of <date>'."""
    txt = re.sub(r'<[^>]+>', ' ', txt)
    for rx in (_PDUFA_RE,
               re.compile(r'(?:goal|target\s+action)\s+date\s+(?:of\s+|is\s+)?'
                          r'([A-Z][a-z]+)\s+(\d{1,2}),?\s+(20\d\d)', re.I)):
        m = rx.search(txt)
        if m:
            mon = m.group(1).lower()
            if mon in _MONTHS:
                return f'{int(m.group(3)):04d}-{_MONTHS.index(mon)+1:02d}-{int(m.group(2)):02d}'
    return None


def extract_pdufa_date(cik, start, end):
    """Find the actual PDUFA goal date for a CIK by scanning its recent PDUFA 8-Ks.
    Checks the top few hits (the newest filing often just references PDUFA without
    restating the date). Returns 'YYYY-MM-DD' or None."""
    try:
        r = requests.get(FTS_URL, headers=UA, timeout=20, params={
            'q': '"PDUFA"', 'forms': '8-K', 'ciks': f'{cik:010d}',
            'startdt': start, 'enddt': end})
        hits = r.json().get('hits', {}).get('hits', []) if r.status_code == 200 else []
        for h in hits[:3]:
            adsh = h.get('_id', '').split(':')
            if len(adsh) < 2:
                continue
            url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{adsh[0].replace("-","")}/{adsh[1]}'
            time.sleep(0.7)
            got = _parse_pdufa_text(requests.get(url, headers=UA, timeout=20).text)
            if got:
                return got
    except Exception as e:
        log.warning(f'PDUFA extract failed CIK {cik}: {e}')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=45)
    ap.add_argument('--min-severity', type=int, default=1)
    ap.add_argument('--max-price', type=float, default=5.0,
                    help='keep only tickers with share price <= this (micro-cap focus). 0 = no price filter.')
    ap.add_argument('--telegram', action='store_true')
    args = ap.parse_args()

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=args.days)
    log.info(f'EDGAR catalyst sweep {start}..{end}, min_severity={args.min_severity}')

    cik_map, cik_rev = load_cik_ticker()
    log.info(f'loaded {len(cik_map)} CIK->ticker rows')

    # ticker -> {company, catalysts:{label:severity}, latest_date, url}
    agg = {}
    for query, label, sev in CATALYST_QUERIES:
        if sev < args.min_severity:
            continue
        hits = sweep(query, start.isoformat(), end.isoformat(), cik_map)
        time.sleep(2.0)  # polite to SEC + avoid burst 500s (well under the 10 req/s limit)
        log.info(f'  {label:28} {len(hits):3} hits')
        for h in hits:
            a = agg.setdefault(h['ticker'], {
                'ticker': h['ticker'], 'company': h['company'],
                'catalysts': {}, 'latest_date': '', 'url': ''})
            # keep highest severity per label; track newest filing
            a['catalysts'][label] = sev
            if h['date'] > a['latest_date']:
                a['latest_date'] = h['date']; a['url'] = h['url']

    # biotech guard: keep anything with a pharma-exclusive catalyst; SIC-vet the rest
    kept, dropped = {}, []
    for tk, a in agg.items():
        if set(a['catalysts']) & PHARMA_EXCLUSIVE:
            kept[tk] = a
        elif is_biotech_sic(tk, cik_rev):
            kept[tk] = a
        else:
            dropped.append(tk)
    if dropped:
        log.info(f'dropped {len(dropped)} non-biotech false positives: {", ".join(sorted(dropped))}')
    agg = kept

    # micro-cap price filter: keep only share price <= max_price (0 = disabled)
    prices = {}
    if args.max_price and args.max_price > 0:
        prices = fetch_prices(sorted(agg.keys()))
        priced_keep, no_price, too_pricey = {}, [], []
        for tk, a in agg.items():
            p = prices.get(tk)
            if p is None:
                no_price.append(tk)          # can't confirm <= threshold -> exclude (warrants/units/delisted)
            elif p <= args.max_price:
                a['price'] = round(p, 4)
                priced_keep[tk] = a
            else:
                too_pricey.append(tk)
        log.info(f'price filter <= ${args.max_price:g}: kept {len(priced_keep)}, '
                 f'dropped {len(too_pricey)} over-price + {len(no_price)} no-price '
                 f'(no-price sample: {", ".join(sorted(no_price)[:8])})')
        agg = priced_keep

    # rank: max severity, then #distinct catalysts, then recency
    rows = []
    for a in agg.values():
        max_sev = max(a['catalysts'].values())
        rows.append({
            'ticker': a['ticker'], 'company': a['company'], 'price': a.get('price'),
            'max_severity': max_sev, 'n_catalysts': len(a['catalysts']),
            'catalysts': sorted(a['catalysts'], key=lambda k: -a['catalysts'][k]),
            'latest_date': a['latest_date'], 'newest_filing_url': a['url']})
    rows.sort(key=lambda r: (r['max_severity'], r['n_catalysts'], r['latest_date']), reverse=True)

    # ── historical LoA + timeline enrichment (loa_model = published FDA base rates) ──
    pdufa_found = 0
    for r in rows:
        cats = r['catalysts']
        stage = loa_model.stage_from_catalysts(cats)
        has_crl = 'CRL (rejection)' in cats
        pdufa = None
        if 'PDUFA date set' in cats:
            cik = cik_rev.get(r['ticker'])
            if cik:
                # PDUFA dates are announced once (often months before the catalyst window)
                # and merely referenced later; search a full year back to find the source.
                pdufa_start = (end - timedelta(days=365)).isoformat()
                pdufa = extract_pdufa_date(cik, pdufa_start, end.isoformat())
                if pdufa:
                    pdufa_found += 1
                time.sleep(1.0)  # gentle SEC pacing
        loa = loa_model.likelihood_of_approval(stage, designations=cats, has_crl=has_crl)
        tt = loa_model.time_to_decision(stage, pdufa)
        r['stage'] = stage
        r['loa'] = loa['loa']                    # historical base-rate probability [0-1]
        r['loa_pct'] = round(loa['loa'] * 100)
        r['loa_basis'] = loa['basis']
        r['rejected'] = has_crl
        r['pdufa_date'] = pdufa
        r['decision_date'] = tt['decision_date']
        r['timeline_days'] = tt['days']
        r['timeline_basis'] = tt['basis']
    log.info(f'LoA/timeline enrichment done ({pdufa_found} real PDUFA dates extracted). '
             f'loa_model {loa_model.MODEL_VERSION}')

    price_note = f' ≤${args.max_price:g}' if args.max_price and args.max_price > 0 else ''
    out = {'generated': datetime.now(timezone.utc).isoformat(),
           'window_days': args.days, 'min_severity': args.min_severity,
           'max_price': args.max_price, 'loa_model': loa_model.MODEL_VERSION,
           'count': len(rows), 'catalysts': rows}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    log.info(f'{len(rows)} tickers with catalysts{price_note} → {OUT_JSON.name}')

    print(f"\n{'='*84}\n  BIOTECH CATALYST SCAN: last {args.days}d{price_note}, {len(rows)} tickers\n{'='*84}")
    print(f"  {'TKR':6} {'$':>7} {'LoA':>4} {'stage':7} {'decision':10} {'days':>5}  catalysts")
    for r in rows[:25]:
        px = f"${r['price']:.2f}" if r.get('price') is not None else '   -'
        dd = r.get('decision_date', '')[:10]
        star = '⛔' if r.get('rejected') else '  '
        print(f"  {r['ticker']:6} {px:>7} {r.get('loa_pct',0):>3}% {r.get('stage','')[:7]:7} "
              f"{dd:10} {r.get('timeline_days',0):>5}{star} {', '.join(r['catalysts'][:2])}")

    if args.telegram:
        tg_token = os.environ.get('TELEGRAM_TOKEN')
        tg_chat = os.environ.get('TELEGRAM_CHAT_ID')
        if not tg_token or not tg_chat:
            log.info('Set TELEGRAM_TOKEN/TELEGRAM_CHAT_ID env vars to enable alerts; skipping Telegram send.')
        else:
            top = rows[:12]
            lines = [f"🧬 *Biotech Catalyst Scan* (last {args.days}d{price_note}, {len(rows)} names)"]
            for r in top:
                sev_emoji = {3: '🔴', 2: '🟠', 1: '🟡'}.get(r['max_severity'], '⚪')
                px = f" ${r['price']:.2f}" if r.get('price') is not None else ''
                lines.append(f"{sev_emoji} *{r['ticker']}*{px} {r['latest_date']}: {r['catalysts'][0]}")
            try:
                requests.post(f'https://api.telegram.org/bot{tg_token}/sendMessage',
                              json={'chat_id': tg_chat, 'text': '\n'.join(lines),
                                    'parse_mode': 'Markdown'}, timeout=10)
            except Exception as e:
                log.warning(f'telegram failed: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

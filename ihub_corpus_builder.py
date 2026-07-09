#!/usr/bin/env python3
"""
ihub_corpus_builder.py — build/grow a per-ticker investor-community corpus from
InvestorsHub, with NO login (public single-message pages + the board RSS feed).

Why this exists: fda_analyzer.py scores "sparse" on EDGAR+ClinicalTrials alone.
Its load_corpus_excerpts() already auto-loads {ticker.lower()}_corpus.json — this
script produces exactly that file for ANY biotech with an iHub board.

How it works (all no-login):
  1. Board RSS  https://investorshub.advfn.com/boards/rss.aspx?board_id=<bid>
     returns the 50 most-recent message_ids + timestamps (structured XML).
  2. Each post  https://investorshub.advfn.com/boards/read_msg.aspx?message_id=<id>
     is a public page; the body lives in <... class="col-message">.
  (The batch board view read_msgs.aspx and the board landing page are login-gated,
   so we ride the 50-post RSS window and ACCUMULATE it across scheduled runs into a
   growing corpus — dedup by content hash. Deep backfill would need Google-indexed
   seed IDs; the recent window is the high-signal part for a live approval score.)

Rate-limiting / ToS-respectful design: requests are paced with a base delay plus
jitter (see DELAY below) so traffic doesn't look metronomic, and the run aborts
after 4 consecutive 403/429/503 responses rather than continuing to hammer the
board once it looks like we're being throttled.

Usage:
  ihub_corpus_builder.py MNKD                 # uses built-in board map
  ihub_corpus_builder.py SAVA --board-id 9538 # arbitrary ticker (find bid via Google:
                                              #   site:investorshub.advfn.com SAVA  ->
                                              #   URL ends -<board_id>)
  ihub_corpus_builder.py MNKD --limit 20      # cap messages fetched this run
"""
import argparse, json, logging, re, sys, time, hashlib, random
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

SCRIPTS = Path(__file__).resolve().parent
LOG_FILE = SCRIPTS / 'ihub_corpus_builder.log'
BASE = 'https://investorshub.advfn.com'
RSS_URL = BASE + '/boards/rss.aspx?board_id={bid}'
MSG_URL = BASE + '/boards/read_msg.aspx?message_id={mid}'
DELAY = 3.0   # base per-request delay (+ jitter); deliberately gentle on iHub's IP throttle
MIN_LENGTH = 150

# Seed board map (board_id is permanent once known; find new ones via Google dork
# `site:investorshub.advfn.com TICKER` — the board URL ends in -<board_id>).
IHUB_BOARDS = {
    'MNKD': 10856,  # MannKind Corp (Afrezza)
    'NVAX': 1758,   # Novavax
    'SAVA': 9538,   # Cassava Sciences
    'DNDN': 4140,   # Dendreon (historical — Provenge)
}

ANALYTICAL_KEYWORDS = {
    'fda', 'trial', 'approval', 'approved', 'clinical', 'data', 'results',
    'pdufa', 'nda', 'bla', 'phase', 'endpoint', 'efficacy', 'safety',
    'interim', 'topline', 'readout', 'manufactur', 'cmc', 'gmp',
    'complete response', 'crl', 'advisory', 'adcom', 'meeting',
    'analysis', 'evidence', 'study', 'statistic',
    'patients', 'survival', 'median', 'p-value', 'significant',
    'catalyst', 'timeline', 'filing', 'submission', 'label',
    'dilution', 'offering', 'shares', 'cash', 'runway', 'balance sheet',
    'price target', 'valuation', 'risk', 'upside', 'downside',
    'short interest', 'short squeeze', 'naked short',
    'revenue', 'profit', 'milestone', 'partnership', 'deal', 'royalty',
    'trx', 'nrx', 'script', 'prescription',  # commercial-stage signal
    'mhra', 'ema', 'designation', 'breakthrough', 'orphan', 'fast track',
}

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [IHUB-CORPUS] %(message)s',
    handlers=[logging.FileHandler(str(LOG_FILE)), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
    'Referer': BASE + '/',
})


def fetch_rss_ids(bid):
    """Return [(message_id, date_str, title)] from the board RSS feed (newest first)."""
    out = []
    try:
        r = SESSION.get(RSS_URL.format(bid=bid), timeout=20)
        if r.status_code != 200:
            log.warning(f'RSS HTTP {r.status_code} for board {bid}'); return out
        for item in re.findall(r'<item>(.*?)</item>', r.text, re.S):
            mid = re.search(r'message_id=(\d+)', item)
            title = re.search(r'<title><!\[CDATA\[(.*?)\]\]></title>', item, re.S)
            pub = re.search(r'<pubDate>(.*?)</pubDate>', item)
            if not mid:
                continue
            date_str = ''
            if pub:
                try:
                    date_str = datetime.strptime(pub.group(1).split('GMT')[0].strip(),
                                                 '%a, %d %b %Y %H:%M:%S').strftime('%Y-%m-%d')
                except ValueError:
                    pass
            out.append((int(mid.group(1)), date_str, (title.group(1).strip() if title else '')))
    except Exception as e:
        log.warning(f'RSS fetch failed board {bid}: {e}')
    return out


def fetch_message(mid, ticker):
    """Fetch one iHub post. Returns {date,text} on success, 'BLOCKED' on 403/429/503
    (so the caller can back off), or None on 404/login/no-body."""
    try:
        r = SESSION.get(MSG_URL.format(mid=mid), timeout=20)
    except Exception as e:
        log.warning(f'msg {mid} request error: {e}'); return None
    if r.status_code in (403, 429, 503):
        return 'BLOCKED'
    if r.status_code != 200 or 'login' in r.url.lower():
        return None
    soup = BeautifulSoup(r.text, 'html.parser')

    # confirm this post belongs to the ticker we asked for (breadcrumb guard)
    bc = soup.find(class_=re.compile(r'breadcrumb'))
    if bc:
        m = re.search(r'\(([A-Z]{1,6})\)', bc.get_text(' ', strip=True))
        if m and m.group(1) != ticker:
            return None  # message_id belongs to another board

    body_el = soup.find(class_='col-message')
    if not body_el:
        return None
    raw = re.sub(r'\s+', ' ', body_el.get_text(' ', strip=True)).strip()

    date = None
    dm = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', raw)
    if dm:
        try:
            date = datetime.strptime(dm.group(1), '%m/%d/%Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    # strip header chrome: "<Weekday, Month DD, YYYY h:mm:ss AMPM> Post # of NNNN Go"
    body = re.sub(r'^.*?[AP]M\s+Post\s+#\s+of\s+[\d,]+\s+Go\s+', '', raw)
    if body == raw:  # fallback: cut up to first "... AM/PM "
        hm = re.search(r'[AP]M\s+', raw)
        if hm:
            body = raw[hm.end():]
    # trailing action bar / reactions
    body = re.split(r'(👍|Public Reply|Private Reply|Bookmark|Share\s|Report abuse)', body)[0].strip()
    return {'date': date or '', 'text': body[:2000]}


# Cross-board promo/wire spam that iHub injects onto every board — not community DD.
PROMO_MARKERS = (
    'issued on behalf of', 'pr newswire', 'globenewswire', 'equity-insider',
    'accesswire', 'newsfile corp', 'news commentary', 'sponsored', 'paid for by',
    'this communication', 'disclaimer:', 'not financial advice and',
)


def is_analytical(text):
    if len(text) < MIN_LENGTH:
        return False
    low = text.lower()
    if any(mk in low for mk in PROMO_MARKERS):
        return False  # drop cross-promotional wire spam
    return any(kw in low for kw in ANALYTICAL_KEYWORDS)


def load_existing(path):
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            pass
    return []


def content_hash(entry):
    return hashlib.md5(entry['text'][:200].encode('utf-8', 'ignore')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ticker')
    ap.add_argument('--board-id', type=int, default=None)
    ap.add_argument('--limit', type=int, default=50)
    args = ap.parse_args()
    ticker = args.ticker.upper()

    bid = args.board_id or IHUB_BOARDS.get(ticker)
    if not bid:
        log.error(f'No board_id for {ticker}. Find it via Google: '
                  f'`site:investorshub.advfn.com {ticker}` (URL ends in -<board_id>), '
                  f'then pass --board-id N.')
        return 1

    out_json = SCRIPTS / f'{ticker.lower()}_corpus.json'
    out_txt = SCRIPTS / f'{ticker.lower()}_corpus.txt'
    existing = load_existing(out_json)
    seen = {content_hash(e) for e in existing if e.get('text')}
    log.info(f'{ticker}: board {bid}, {len(existing)} existing posts in corpus')

    rss = fetch_rss_ids(bid)
    log.info(f'{ticker}: {len(rss)} posts in RSS window')
    if not rss:
        log.warning('empty RSS — nothing to do'); return 1

    added = 0
    consecutive_blocks = 0
    for mid, rss_date, _title in rss[:args.limit]:
        res = fetch_message(mid, ticker)
        time.sleep(DELAY + random.uniform(0, 1.0))  # jitter — don't look metronomic
        if res == 'BLOCKED':
            consecutive_blocks += 1
            if consecutive_blocks >= 4:
                log.warning(f'{ticker}: {consecutive_blocks} consecutive 403/429 from iHub — '
                            f'backing off and ABORTING run to avoid an IP block. '
                            f'Saving what we have; retry later.')
                break
            time.sleep(15 * consecutive_blocks)  # exponential-ish cooldown before next try
            continue
        consecutive_blocks = 0
        if not res:
            continue
        if not res['date'] and rss_date:
            res['date'] = rss_date
        if not is_analytical(res['text']):
            continue
        h = content_hash(res)
        if h in seen:
            continue
        seen.add(h)
        existing.append(res)
        added += 1

    existing.sort(key=lambda e: e.get('date', ''), reverse=True)
    out_json.write_text(json.dumps(existing, indent=2))

    lines = [f'IHUB {ticker} INVESTOR COMMUNITY — ANALYTICAL POSTS',
             f'Total posts: {len(existing):,}   (source: investorshub.advfn.com board {bid})',
             '=' * 60, '']
    for e in existing:
        lines.append(f'[{e.get("date","")}]'); lines.append(e['text']); lines.append('-' * 40)
    out_txt.write_text('\n'.join(lines), encoding='utf-8')

    log.info(f'{ticker}: +{added} new analytical posts → {len(existing)} total. Saved {out_json.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

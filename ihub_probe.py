#!/usr/bin/env python3
"""
iHub Message Probe
Tests whether InvestorsHub individual message URLs are accessible without login.
If they are, maps the message ID ranges for DNDN, MNKD, and NVAX boards.

Usage:
  python3 ihub_probe.py             # run full probe
  python3 ihub_probe.py --scrape    # if probe passes, scrape all accessible messages

How iHub IDs work:
  - message_id is a GLOBAL sequential integer across all boards
  - To find a board's range, we test known message IDs from Google-indexed posts
  - Then binary-search for the first/last message belonging to each board

Known IDs (from Google):
  DNDN:  12080298, 19676719, 112650176   (board_id ~4140)
  MNKD:  107037385, 132630509, 145198208  (board_id ~10856)
  NVAX:  (unknown — will probe)
  SAVA:  (unknown — will probe)
"""

import time, json, logging, sys, re, argparse
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS   = Path(__file__).resolve().parent
OUT_JSON  = SCRIPTS / 'ihub_probe_results.json'
LOG_FILE  = SCRIPTS / 'ihub_probe.log'

BASE_URL  = 'https://investorshub.advfn.com'
MSG_URL   = BASE_URL + '/boards/read_msg.aspx?message_id={msg_id}'

DELAY     = 2.0   # be very polite to avoid bans

# ── Known anchor message IDs per board ───────────────────────────────────────
KNOWN_IDS = {
    'DNDN': {
        'board_id': 4140,
        'board_url': '/Dendreon-Corp-Fka-Dndnq-4140',
        'sample_ids': [12080298, 19676719, 112650176],
        'description': 'Dendreon Corp fka DNDNQ (2007-2019)',
    },
    'MNKD': {
        'board_id': 10856,
        'board_url': '/MannKind-Corp-MNKD-10856',
        'sample_ids': [107037385, 132630509, 135383714, 145198208],
        'description': 'MannKind Corp (2009-present)',
    },
    'NVAX': {
        'board_id': 1758,
        'board_url': '/Novavax-Inc-NVAX-1758',
        'sample_ids': [],   # unknown — will try to find from board page
        'description': 'Novavax Inc',
    },
    'SAVA': {
        'board_id': 9538,
        'board_url': '/Cassava-Sciences-Inc-SAVA-9538',
        'sample_ids': [],
        'description': 'Cassava Sciences Inc',
    },
}

MIN_LENGTH = 150
ANALYTICAL_KEYWORDS = {
    'fda', 'trial', 'approval', 'approved', 'clinical', 'data', 'results',
    'pdufa', 'nda', 'bla', 'phase', 'endpoint', 'efficacy', 'safety',
    'interim', 'topline', 'readout', 'manufacturing', 'cmc', 'gmp',
    'complete response', 'crl', 'advisory', 'adcom',
    'analysis', 'evidence', 'study', 'statistics',
    'patients', 'survival', 'median', 'p-value', 'significant',
    'catalyst', 'timeline', 'filing', 'submission',
    'dilution', 'offering', 'shares', 'cash', 'runway',
    'price target', 'valuation', 'risk', 'upside', 'downside',
    'short interest', 'short squeeze', 'naked short',
    'revenue', 'profit', 'milestone', 'partnership', 'deal',
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [IHUB] %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://investorshub.advfn.com/',
})


def fetch_message(msg_id: int) -> dict | None:
    """
    Fetch a single iHub message by ID.
    Returns dict with {msg_id, board_id, ticker, date, text} or None if blocked/missing.
    """
    url = MSG_URL.format(msg_id=msg_id)
    try:
        r = SESSION.get(url, timeout=15)
    except Exception as e:
        log.warning(f'Request error for msg {msg_id}: {e}')
        return None

    if r.status_code == 403:
        return None  # blocked
    if r.status_code == 404:
        return None  # message doesn't exist
    if r.status_code != 200:
        log.warning(f'HTTP {r.status_code} for msg {msg_id}')
        return None

    soup = BeautifulSoup(r.text, 'html.parser')

    # Check for login wall
    login_form = soup.find('form', action=re.compile(r'login|signin', re.IGNORECASE))
    login_redirect = 'login' in r.url.lower() or 'signin' in r.url.lower()
    if login_form or login_redirect:
        log.info('Login wall detected — iHub requires authentication for messages')
        return {'blocked': True, 'reason': 'login_required'}

    # Extract board/ticker info from breadcrumb or title
    ticker = ''
    board_id_found = None
    breadcrumb = soup.find('div', class_=re.compile(r'breadcrumb', re.IGNORECASE))
    if breadcrumb:
        links = breadcrumb.find_all('a')
        for link in links:
            href = link.get('href', '')
            m = re.search(r'-(\d+)$', href)
            if m:
                board_id_found = int(m.group(1))
            text = link.get_text(strip=True)
            if re.match(r'^[A-Z]{2,5}$', text):
                ticker = text

    # Extract post date
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_tag = soup.find(attrs={'data-timestamp': True})
    if time_tag:
        ts = int(time_tag['data-timestamp'])
        date_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    else:
        date_el = soup.find(class_=re.compile(r'date|time|posted', re.IGNORECASE))
        if date_el:
            raw = date_el.get_text(strip=True)
            m = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', raw)
            if m:
                try:
                    date_str = datetime.strptime(m.group(1), '%m/%d/%Y').strftime('%Y-%m-%d')
                except ValueError:
                    pass

    # Extract message body
    body = (
        soup.find('div', id=re.compile(r'message|post.?body|msg.?text', re.IGNORECASE))
        or soup.find('td', class_=re.compile(r'message|post.?body', re.IGNORECASE))
        or soup.find('div', class_=re.compile(r'message.?text|post.?body', re.IGNORECASE))
    )
    if not body:
        # Fallback: largest text block on page
        divs = soup.find_all('div')
        body = max(divs, key=lambda d: len(d.get_text()), default=None)

    text = body.get_text(separator=' ', strip=True) if body else ''
    text = re.sub(r'\s+', ' ', text).strip()

    return {
        'msg_id':   msg_id,
        'board_id': board_id_found,
        'ticker':   ticker,
        'date':     date_str,
        'text':     text[:2000],
        'accessible': True,
    }


# ── Probe phase: test known IDs ───────────────────────────────────────────────
def run_probe() -> dict:
    """Test known message IDs to determine accessibility."""
    results = {}

    for ticker, info in KNOWN_IDS.items():
        log.info(f'\n--- Probing {ticker} ---')
        ticker_results = {
            'accessible': False,
            'login_required': False,
            'sample_results': [],
            'board_id': info['board_id'],
        }

        for msg_id in info['sample_ids']:
            log.info(f'  Testing msg_id={msg_id}...')
            result = fetch_message(msg_id)
            time.sleep(DELAY)

            if result is None:
                log.info(f'  → 403/404 blocked')
                ticker_results['sample_results'].append({'msg_id': msg_id, 'status': 'blocked'})
            elif result.get('blocked'):
                log.info(f'  → Login required')
                ticker_results['login_required'] = True
                ticker_results['sample_results'].append({'msg_id': msg_id, 'status': 'login_required'})
                break
            else:
                log.info(f'  → ACCESSIBLE! Date: {result["date"]}, Text: {result["text"][:80]}...')
                ticker_results['accessible'] = True
                ticker_results['sample_results'].append({
                    'msg_id': msg_id,
                    'status': 'ok',
                    'date':   result['date'],
                    'text_preview': result['text'][:120],
                })

        results[ticker] = ticker_results

    return results


# ── Find ID range for a board ─────────────────────────────────────────────────
def find_board_range(ticker: str, known_id: int, board_id: int) -> tuple[int, int]:
    """
    Given one known accessible message ID, binary-search outward to estimate
    the first and last message IDs belonging to this board.

    Returns (min_id, max_id) estimate.
    """
    log.info(f'Finding ID range for {ticker} (anchor={known_id}, board={board_id})')

    def belongs_to_board(msg_id: int) -> bool:
        result = fetch_message(msg_id)
        time.sleep(DELAY)
        if not result or not result.get('accessible'):
            return False
        return result.get('board_id') == board_id

    # Search backward for first message
    step = 1000
    low = known_id
    while low > step and belongs_to_board(low - step):
        low -= step
        log.info(f'  Searching backward: {low}')

    # Search forward for last message
    high = known_id
    while belongs_to_board(high + step):
        high += step
        log.info(f'  Searching forward: {high}')

    log.info(f'{ticker} estimated range: {low:,} → {high:,} ({high-low:,} IDs)')
    return low, high


# ── Scrape phase: enumerate IDs if accessible ─────────────────────────────────
def scrape_board(ticker: str, min_id: int, max_id: int) -> list[dict]:
    """
    Enumerate message IDs in range for this board.
    Only keep analytical posts from this specific board.
    """
    board_id = KNOWN_IDS[ticker]['board_id']
    corpus   = []
    total    = 0
    kept     = 0

    out_txt  = SCRIPTS / f'ihub_{ticker.lower()}_corpus.txt'
    out_json = SCRIPTS / f'ihub_{ticker.lower()}_corpus.json'

    log.info(f'Scraping {ticker}: IDs {min_id:,} → {max_id:,}')

    for msg_id in range(min_id, max_id + 1, 5):  # step=5 to skip noise
        total += 1
        result = fetch_message(msg_id)
        time.sleep(DELAY)

        if not result or not result.get('accessible'):
            continue
        if result.get('board_id') != board_id:
            continue   # different board, skip

        text = result.get('text', '')
        if not text or len(text) < MIN_LENGTH:
            continue
        lower = text.lower()
        if not any(kw in lower for kw in ANALYTICAL_KEYWORDS):
            continue

        corpus.append({'date': result['date'], 'text': text})
        kept += 1

        if kept % 100 == 0:
            log.info(f'  {total:,} checked | {kept:,} kept')
            # Incremental save
            out_json.write_text(json.dumps(corpus, indent=2))

    # Final save
    out_json.write_text(json.dumps(corpus, indent=2))

    lines = [
        f'IHUB {ticker} INVESTOR COMMUNITY — ANALYTICAL POSTS',
        f'Total posts extracted: {len(corpus):,}',
        f'Source: investorshub.advfn.com',
        '=' * 60, '',
    ]
    for entry in sorted(corpus, key=lambda x: x['date'], reverse=True):
        lines.append(f'[{entry["date"]}]')
        lines.append(entry['text'])
        lines.append('-' * 40)
    out_txt.write_text('\n'.join(lines), encoding='utf-8')

    size_mb = out_txt.stat().st_size / 1_000_000
    log.info(f'{ticker} corpus saved: {out_txt} ({size_mb:.1f} MB, {len(corpus):,} posts)')
    return corpus


# ── Entry point ───────────────────────────────────────────────────────────────
def main(scrape: bool = False):
    log.info('=== iHub Accessibility Probe ===')

    probe_results = run_probe()

    # Save probe results
    OUT_JSON.write_text(json.dumps(probe_results, indent=2))
    log.info(f'Probe results saved: {OUT_JSON}')

    # Summary
    log.info('\n=== PROBE SUMMARY ===')
    any_accessible = False
    for ticker, res in probe_results.items():
        status = 'LOGIN REQUIRED' if res['login_required'] else ('ACCESSIBLE' if res['accessible'] else 'BLOCKED (403/404)')
        log.info(f'  {ticker}: {status}')
        if res['accessible']:
            any_accessible = True

    if not any_accessible:
        log.info('\niHub individual messages are not accessible. Will need cookies/session auth.')
        log.info('Next steps:')
        log.info('  1. Log into iHub in a browser')
        log.info('  2. Export cookies (EditThisCookie extension or browser DevTools)')
        log.info('  3. Re-run this script with --cookies path/to/cookies.json')
        return

    if scrape:
        log.info('\nStarting scrape of accessible boards...')
        for ticker, res in probe_results.items():
            if not res['accessible']:
                continue
            known_id = KNOWN_IDS[ticker]['sample_ids'][0]
            min_id, max_id = find_board_range(ticker, known_id, res['board_id'])
            scrape_board(ticker, min_id, max_id)
    else:
        log.info('\nRun with --scrape to enumerate all accessible messages.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probe and optionally scrape iHub boards')
    parser.add_argument('--scrape', action='store_true', help='Enumerate messages after probing')
    args = parser.parse_args()
    main(scrape=args.scrape)

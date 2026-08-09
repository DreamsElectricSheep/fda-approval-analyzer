#!/usr/bin/env python3
"""
score_catalysts.py: batch-score a catalyst list through fda_analyzer, so a
dashboard's "Approval odds" sort has data for every name.

Reads biotech_catalysts.json, runs fda_analyzer.py on each ticker (skipping any
already scored today), paced politely. Writes into fda_scores/ (the same cache the
dashboard reads). Intended to run periodically (e.g. weekly) after a catalyst scan.

Usage:
  score_catalysts.py                 # all names in the current scan
  score_catalysts.py --limit 10      # just the top N (severity-ranked order)
  score_catalysts.py --min-severity 3 --force
"""
import argparse, json, logging, subprocess, sys, time
from pathlib import Path
from datetime import datetime

SCRIPTS   = Path(__file__).resolve().parent
CATALYSTS = SCRIPTS / 'biotech_catalysts.json'
SCORES    = SCRIPTS / 'fda_scores'
ANALYZER  = SCRIPTS / 'fda_analyzer.py'
PYTHON    = sys.executable
LOG_FILE  = SCRIPTS / 'score_catalysts.log'
PACE_SEC  = 5          # gentle gap between analyzer runs

logging.basicConfig(level=logging.INFO, format='%(asctime)s [SCORE-CAT] %(message)s',
                    handlers=[logging.FileHandler(str(LOG_FILE)), logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def scored_today(tk):
    return (SCORES / f"{tk.upper()}_{datetime.now().strftime('%Y%m%d')}.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='score only top N (0 = all)')
    ap.add_argument('--min-severity', type=int, default=1)
    ap.add_argument('--force', action='store_true', help='re-score even if already scored today')
    args = ap.parse_args()

    if not CATALYSTS.exists():
        log.error('no biotech_catalysts.json: run biotech_catalyst_scanner first'); return 1
    rows = json.load(open(CATALYSTS)).get('catalysts', [])
    rows = [r for r in rows if r.get('max_severity', 0) >= args.min_severity]
    if args.limit:
        rows = rows[:args.limit]
    SCORES.mkdir(exist_ok=True)
    log.info(f'batch-scoring {len(rows)} catalyst names (min_sev={args.min_severity}, force={args.force})')

    done = skipped = failed = 0
    for i, r in enumerate(rows, 1):
        tk = r['ticker']
        if scored_today(tk) and not args.force:
            skipped += 1; continue
        try:
            res = subprocess.run([PYTHON, str(ANALYZER), tk], cwd=str(SCRIPTS),
                                 timeout=180, capture_output=True, text=True)
            f = SCORES / f"{tk}_{datetime.now().strftime('%Y%m%d')}.json"
            if f.exists():
                total = json.load(open(f)).get('total')
                log.info(f'  [{i}/{len(rows)}] {tk}: {total}/100')
                done += 1
            else:
                log.warning(f'  [{i}/{len(rows)}] {tk}: no score file produced'); failed += 1
        except subprocess.TimeoutExpired:
            log.warning(f'  [{i}/{len(rows)}] {tk}: timed out'); failed += 1
        except Exception as e:
            log.warning(f'  [{i}/{len(rows)}] {tk}: {e}'); failed += 1
        time.sleep(PACE_SEC)

    log.info(f'done: {done} scored, {skipped} already-today, {failed} failed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

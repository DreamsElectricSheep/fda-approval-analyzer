#!/usr/bin/env python3
"""
fda_dashboard.py — paste a biotech ticker, get its FDA approval analysis + score.

Flask UI (port 5007) over fda_analyzer.py. Dark-themed dashboard with an SVG score
ring and component bars. Tabs:
  • Analyze  — paste ticker (+ optional drug/indication) -> score ring, stage pipeline,
               5 rubric section bars w/ per-criterion justification, risk-deduction
               flags, evidence-quality badge. Cached per ticker/day from fda_scores/.
  • Catalysts — the current micro-cap (<=$5) catalyst list from biotech_catalysts.json;
               each row is clickable -> jumps to Analyze and scores it.

Runs fda_analyzer.py as a subprocess (Gemini feature-extractor + EDGAR + ClinicalTrials);
first run for a ticker takes ~30-60s, then it's cached for the day.
"""
import json, os, glob, subprocess, sys, re
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

import loa_model

SCRIPTS    = Path(__file__).resolve().parent
SCORES_DIR = SCRIPTS / 'fda_scores'
CATALYSTS  = SCRIPTS / 'biotech_catalysts.json'
ANALYZER   = SCRIPTS / 'fda_analyzer.py'
RUBRIC     = SCRIPTS / 'fda_rubric.json'
PYTHON     = sys.executable

app = Flask(__name__)
_RUBRIC = json.load(open(RUBRIC))

# catalyst label -> pipeline stage index (0..7)
STAGES = ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3', 'Filed', 'PDUFA', 'AdCom', 'Decision']
CATALYST_STAGE = {
    'Breakthrough designation': 2, 'Fast Track designation': 2, 'Orphan designation': 2,
    'Pivotal endpoint reported': 3, 'Topline data readout': 3,
    'NDA activity': 4, 'BLA/sBLA activity': 4, 'FDA accepted filing': 4,
    'Priority Review': 5, 'PDUFA date set': 5,
    'AdCom scheduled/held': 6, 'CRL (rejection)': 5,
}


def load_catalyst_map():
    if not CATALYSTS.exists():
        return {}
    try:
        rows = json.load(open(CATALYSTS)).get('catalysts', [])
        return {r['ticker']: r for r in rows}
    except Exception:
        return {}


def stage_for(ticker, cmap):
    """Return {index, label, crl, price, catalysts} from the catalyst scan, or None."""
    r = cmap.get(ticker.upper())
    if not r:
        return None
    cats = r.get('catalysts', [])
    idx = max([CATALYST_STAGE.get(c, 0) for c in cats], default=0)
    return {'index': idx, 'label': STAGES[idx], 'crl': 'CRL (rejection)' in cats,
            'price': r.get('price'), 'catalysts': cats, 'latest_date': r.get('latest_date')}


def latest_score_file(ticker):
    files = sorted(glob.glob(str(SCORES_DIR / f'{ticker.upper()}_*.json')))
    return files[-1] if files else None


def today_score_file(ticker):
    f = SCORES_DIR / f"{ticker.upper()}_{datetime.now().strftime('%Y%m%d')}.json"
    return f if f.exists() else None


def run_analyzer(ticker, drug, indication):
    cmd = [PYTHON, str(ANALYZER), ticker.upper()]
    if drug:
        cmd += ['--drug', drug]
    if indication:
        cmd += ['--indication', indication]
    try:
        subprocess.run(cmd, cwd=str(SCRIPTS), timeout=180,
                       capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None, 'analyzer timed out (>180s)'
    f = today_score_file(ticker)
    if not f:
        return None, 'analyzer produced no score file'
    return json.load(open(f)), None


@app.route('/')
def index():
    return HTML


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    ticker = (data.get('ticker') or '').strip().upper()
    if not re.fullmatch(r'[A-Z]{1,6}', ticker):
        return jsonify({'error': 'enter a valid ticker (1-6 letters)'}), 400
    drug = (data.get('drug') or '').strip()
    indication = (data.get('indication') or '').strip()
    force = bool(data.get('force'))

    cmap = load_catalyst_map()
    cached = None if force else today_score_file(ticker)
    if cached:
        result = json.load(open(cached))
        result['_cached'] = True
    else:
        result, err = run_analyzer(ticker, drug, indication)
        if err:
            return jsonify({'error': err}), 500
        result['_cached'] = False

    result['_stage'] = stage_for(ticker, cmap)
    result['_stages'] = STAGES

    # historically-grounded LoA + timeline (TA refined from the indication when known)
    row = cmap.get(ticker)
    st = result['_stage']
    if st:
        cats = st.get('catalysts', [])
        ta = loa_model.infer_therapeutic_area(indication or drug or '')
        loa = loa_model.likelihood_of_approval(st['label'], designations=cats,
                                               therapeutic_area=ta, has_crl=st['crl'])
        result['_loa'] = {
            'pct': round(loa['loa'] * 100), 'basis': loa['basis'], 'ta': ta,
            'decision_date': (row or {}).get('decision_date'),
            'timeline_days': (row or {}).get('timeline_days'),
            'pdufa_date': (row or {}).get('pdufa_date'),
            'model': loa_model.MODEL_VERSION}

    # ONE final blended number: drug-specific evidence (rubric total) + historical
    # base rate for this stage (LoA), weighted by how much real evidence was found.
    # This is the number to look at — the rest of the page is how it was built.
    loa_pct = result['_loa']['pct'] if result.get('_loa') else None
    result['_final'] = loa_model.final_approval_odds(
        result['total'], _RUBRIC, loa_pct, result.get('evidence_quality'))
    return jsonify(result)


def latest_score(ticker):
    """Latest cached fda_analyzer result for a ticker -> {score, band, date,
    evidence_quality} or None."""
    f = latest_score_file(ticker)
    if not f:
        return None
    try:
        d = json.load(open(f))
        return {'score': d.get('total'), 'band': d.get('band', {}).get('label'),
                'date': d.get('generated', '')[:10],
                'evidence_quality': d.get('evidence_quality')}
    except Exception:
        return None


@app.route('/catalysts')
def catalysts():
    if not CATALYSTS.exists():
        return jsonify({'generated': None, 'catalysts': []})
    d = json.load(open(CATALYSTS))
    rows = d.get('catalysts', [])[:60]
    for r in rows:
        sc = latest_score(r['ticker'])
        r['score'] = sc['score'] if sc else None
        r['score_band'] = sc['band'] if sc else None
        r['score_date'] = sc['date'] if sc else None
        r['rejected'] = 'CRL (rejection)' in r.get('catalysts', [])
        # ONE blended odds number, same definition as the Analyze page — only
        # available once this ticker has actually been analyzed (sc exists);
        # until then the base-rate-only loa_pct is the best we have.
        r['final_pct'] = None
        if sc and sc['score'] is not None and r.get('loa') is not None:
            blend = loa_model.final_approval_odds(
                sc['score'], _RUBRIC, r['loa_pct'], sc['evidence_quality'])
            r['final_pct'] = blend['pct']
    return jsonify({'generated': d.get('generated'), 'max_price': d.get('max_price'),
                    'window_days': d.get('window_days'), 'count': d.get('count'),
                    'catalysts': rows})


@app.route('/recent')
def recent():
    out = []
    for f in sorted(glob.glob(str(SCORES_DIR / '*.json')), reverse=True)[:20]:
        try:
            d = json.load(open(f))
            out.append({'ticker': d.get('ticker'), 'total': d.get('total'),
                        'band': d.get('band', {}).get('label'),
                        'generated': d.get('generated', '')[:10]})
        except Exception:
            pass
    return jsonify(out)


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FDA Approval Analyzer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#080b0f;--surface:#0f1419;--border:#1e2a35;--text:#e2e8f0;--muted:#64748b;
--green:#22c55e;--lime:#84cc16;--amber:#f59e0b;--orange:#f97316;--red:#ef4444;
--blue:#3b82f6;--purple:#a855f7;--cyan:#06b6d4;
--font:'Inter',-apple-system,sans-serif;--mono:'JetBrains Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:14px;min-height:100vh;padding:28px 20px 60px}
.header{text-align:center;margin-bottom:22px}
.header h1{font-size:20px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
.header p{color:var(--muted);font-size:12px;margin-top:7px}
.wrap{max-width:920px;margin:0 auto}
.tab-bar{display:flex;gap:4px;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:5px;margin-bottom:22px}
.tab-btn{flex:1;background:transparent;border:none;border-radius:7px;color:var(--muted);font-size:13px;font-weight:600;padding:9px 16px;cursor:pointer;transition:.15s}
.tab-btn.active{background:var(--blue);color:#fff}
.tab-btn:not(.active):hover{background:rgba(59,130,246,.12);color:var(--text)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:22px 26px;margin-bottom:16px}
.card-title{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:16px}
.input-card{border-left:3px solid var(--blue)}
.input-row{display:flex;gap:10px;flex-wrap:wrap}
.inp{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px 16px;color:var(--text);font-size:13px;outline:none;transition:.2s}
.inp:focus{border-color:var(--blue)}
.inp::placeholder{color:var(--muted)}
#tk{flex:0 0 150px;font-family:var(--mono);text-transform:uppercase;font-weight:700}
#drug,#ind{flex:1;min-width:150px}
.btn{background:var(--blue);color:#fff;border:none;border-radius:8px;padding:12px 26px;font-size:14px;font-weight:600;cursor:pointer;transition:.2s;white-space:nowrap}
.btn:hover{opacity:.88}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-sm{background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:6px 13px;font-size:12px;cursor:pointer}
.btn-sm:hover{border-color:var(--blue);color:var(--blue)}
.hint{color:var(--muted);font-size:11px;margin-top:10px}
#loading{display:none;text-align:center;padding:40px}
.spinner{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 14px}
@keyframes spin{to{transform:rotate(360deg)}}
.err{display:none;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);color:#fca5a5;border-radius:8px;padding:12px 16px;margin-bottom:16px}
#results{display:none}
.score-head{display:flex;gap:26px;align-items:center;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px 26px;margin-bottom:16px}
.ring-wrap{position:relative;width:120px;height:120px;flex:0 0 120px}
.score-ring{transform:rotate(-90deg);width:120px;height:120px}
.ring-bg{fill:none;stroke:var(--border);stroke-width:9}
.ring-fill{fill:none;stroke-width:9;stroke-linecap:round;transition:stroke-dashoffset 1.3s cubic-bezier(.4,0,.2,1),stroke .4s}
.ring-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring-num{font-size:30px;font-weight:800;font-family:var(--mono);line-height:1}
.ring-denom{font-size:12px;color:var(--muted)}
.score-info{flex:1;min-width:220px}
.vpill{display:inline-block;font-size:12px;font-weight:700;padding:5px 13px;border-radius:20px;letter-spacing:.03em;margin-bottom:9px}
.tname{font-size:22px;font-weight:800;font-family:var(--mono)}
.tmeta{color:var(--muted);font-size:12px;margin-top:4px}
.badges{margin-top:11px;display:flex;gap:8px;flex-wrap:wrap}
.badge{font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;border:1px solid var(--border);color:var(--muted)}
.badge.rich{color:var(--green);border-color:rgba(34,197,94,.4)}
.badge.thin{color:var(--amber);border-color:rgba(245,158,11,.4)}
.badge.sparse{color:var(--red);border-color:rgba(239,68,68,.4)}
/* stage pipeline */
.pipe{display:flex;gap:0;margin-top:6px}
.pstep{flex:1;text-align:center;position:relative;padding-top:20px;font-size:10px;color:var(--muted);font-weight:600}
.pstep::before{content:'';position:absolute;top:6px;left:0;right:0;height:2px;background:var(--border)}
.pstep .dot{position:absolute;top:1px;left:50%;transform:translateX(-50%);width:12px;height:12px;border-radius:50%;background:var(--border);border:2px solid var(--surface);z-index:2}
.pstep.done .dot{background:var(--blue)}.pstep.done::before{background:var(--blue)}
.pstep.cur .dot{background:var(--amber);box-shadow:0 0 8px var(--amber)}.pstep.cur{color:var(--amber)}
.pstep.crl .dot{background:var(--red);box-shadow:0 0 8px var(--red)}
/* section bars */
.crow{display:flex;align-items:center;gap:14px;margin-bottom:6px}
.cname{flex:0 0 240px;font-size:13px;font-weight:600}
.cname small{color:var(--muted);font-weight:400;font-size:11px}
.track{flex:1;background:var(--border);border-radius:4px;height:10px;overflow:hidden}
.fill{height:100%;width:0;border-radius:4px;transition:width 1.2s cubic-bezier(.4,0,.2,1)}
.cval{flex:0 0 58px;text-align:right;font-family:var(--mono);font-size:13px;font-weight:700}
.detail{margin:2px 0 16px 254px;font-size:11px;color:var(--muted);line-height:1.5}
.detail div{padding:2px 0;border-bottom:1px solid rgba(30,42,53,.5)}
.detail b{color:var(--text);font-weight:600}
.flag{background:rgba(239,68,68,.08);border-left:3px solid var(--red);border-radius:6px;padding:9px 14px;margin-bottom:8px;font-size:13px}
.flag b{color:#fca5a5}
.loa-big{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
.loa-pct{font-family:var(--mono);font-weight:700;font-size:20px;color:var(--text)}
.loa-side{font-size:12px;color:var(--muted);line-height:1.7}
.loa-side b{color:var(--text);font-weight:600}
.loa-basis{margin-top:12px;font-size:11px;color:var(--muted)}
.loa-basis span{display:inline-block;background:var(--bg);border:1px solid var(--border);border-radius:5px;padding:3px 9px;margin:3px 5px 0 0;font-family:var(--mono)}
.loa-note{margin-top:10px;font-size:10px;color:var(--muted);font-style:italic}
.not-this{font-size:10px;font-weight:400;text-transform:none;letter-spacing:0;color:var(--muted)}
.composition{margin-top:10px;font-size:12px;color:var(--muted);line-height:1.6;max-width:520px}
.composition b{color:var(--text)}
.composition .why{margin-top:4px;color:var(--muted)}
/* catalysts table */
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding:8px 10px;border-bottom:1px solid var(--border)}
td{padding:9px 10px;border-bottom:1px solid rgba(30,42,53,.5)}
tr.clk{cursor:pointer;transition:.12s}tr.clk:hover{background:rgba(59,130,246,.06)}
.sev{font-weight:700}.sev3{color:var(--red)}.sev2{color:var(--orange)}.sev1{color:var(--amber)}
.tkr{font-family:var(--mono);font-weight:700;color:var(--blue)}
.px{font-family:var(--mono)}
.scorecell{font-family:var(--mono);font-weight:700}
.rejrow{box-shadow:inset 3px 0 0 var(--red);background:rgba(239,68,68,.05)}
.rejrow:hover{background:rgba(239,68,68,.1)!important}
.rejbadge{display:inline-block;font-size:9px;font-weight:800;color:#fff;background:var(--red);border-radius:4px;padding:2px 6px;margin-left:7px;letter-spacing:.04em;vertical-align:middle}
.sort-ctl{display:flex;gap:6px;align-items:center;margin-bottom:14px;font-size:12px;color:var(--muted)}
.sort-btn{background:transparent;border:1px solid var(--border);color:var(--muted);border-radius:6px;padding:5px 13px;font-size:12px;cursor:pointer;font-weight:600}
.sort-btn.active{background:var(--blue);border-color:var(--blue);color:#fff}
.foot{text-align:center;color:var(--muted);font-size:11px;margin-top:30px}
</style></head><body>
<div class="header"><h1>🧬 FDA Approval Analyzer</h1><p>Paste a biotech ticker → rubric-scored approval odds</p></div>
<div class="wrap">
  <div class="tab-bar">
    <button class="tab-btn active" id="t-an" onclick="showTab('an')">Analyze</button>
    <button class="tab-btn" id="t-cat" onclick="showTab('cat')">Micro-Cap Catalysts</button>
  </div>

  <div id="tab-an">
    <div class="card input-card">
      <div class="input-row">
        <input class="inp" id="tk" placeholder="TICKER" maxlength="6" onkeydown="if(event.key==='Enter')go()">
        <input class="inp" id="drug" placeholder="drug (optional, improves accuracy)">
        <input class="inp" id="ind" placeholder="indication (optional)">
        <button class="btn" id="gobtn" onclick="go()">Analyze</button>
      </div>
      <div class="hint">First run for a ticker takes ~30–60s (live EDGAR + ClinicalTrials + Gemini feature extraction). Same-day re-scores are cached.</div>
    </div>
    <div id="loading"><div class="spinner"></div><div id="lmsg">Pulling filings, trials, and scoring…</div></div>
    <div class="err" id="err"></div>
    <div id="results">
      <div class="score-head">
        <div class="ring-wrap"><svg class="score-ring" viewBox="0 0 120 120">
          <circle class="ring-bg" cx="60" cy="60" r="50"/>
          <circle class="ring-fill" id="ring" cx="60" cy="60" r="50"/></svg>
          <div class="ring-label"><div class="ring-num" id="rnum">0</div><div class="ring-denom">%</div></div>
        </div>
        <div class="score-info">
          <div class="vpill" id="vpill">—</div>
          <div class="tname" id="tname">—</div>
          <div class="tmeta" id="tmeta">—</div>
          <div class="badges" id="badges"></div>
          <div class="composition" id="composition"></div>
        </div>
      </div>
      <div class="card"><div class="card-title">Regulatory Stage</div><div class="pipe" id="pipe"></div></div>
      <div class="card"><div class="card-title">Rubric Breakdown — drug-specific evidence (0–100)</div><div id="sections"></div></div>
      <div class="card" id="flagcard" style="display:none"><div class="card-title">Risk Deductions</div><div id="flags"></div></div>
      <div class="card" id="loacard" style="display:none">
        <div class="card-title">Reference — Historical Base Rate &amp; Timeline <span class="not-this">(input to the number above, not a separate score)</span></div>
        <div id="loabody"></div>
      </div>
      <div style="text-align:right"><button class="btn-sm" onclick="go(true)">↻ Re-run (bypass cache)</button></div>
    </div>
  </div>

  <div id="tab-cat" style="display:none">
    <div class="card">
      <div class="card-title" id="cat-title">Micro-Cap Catalysts</div>
      <div id="cat-body">Loading…</div>
    </div>
  </div>
  <div class="foot">fda_analyzer + rubric · Gemini as feature-extractor only · not investment advice</div>
</div>
<script>
const CIRC = 2*Math.PI*50;
function scoreColor(s){return s>=80?'var(--green)':s>=65?'var(--lime)':s>=50?'var(--amber)':s>=35?'var(--orange)':'var(--red)';}
function showTab(t){
  document.getElementById('tab-an').style.display=t==='an'?'':'none';
  document.getElementById('tab-cat').style.display=t==='cat'?'':'none';
  document.getElementById('t-an').classList.toggle('active',t==='an');
  document.getElementById('t-cat').classList.toggle('active',t==='cat');
  if(t==='cat')loadCatalysts();
}
async function go(force){
  const tk=document.getElementById('tk').value.trim().toUpperCase();
  if(!/^[A-Z]{1,6}$/.test(tk)){showErr('Enter a valid ticker (1–6 letters).');return;}
  document.getElementById('err').style.display='none';
  document.getElementById('results').style.display='none';
  document.getElementById('loading').style.display='block';
  document.getElementById('gobtn').disabled=true;
  try{
    const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker:tk,drug:document.getElementById('drug').value,
        indication:document.getElementById('ind').value,force:!!force})});
    const d=await r.json();
    if(!r.ok||d.error){showErr(d.error||'analysis failed');return;}
    render(d);
  }catch(e){showErr('request failed: '+e);}
  finally{document.getElementById('loading').style.display='none';document.getElementById('gobtn').disabled=false;}
}
function showErr(m){const e=document.getElementById('err');e.textContent=m;e.style.display='block';}
function render(d){
  document.getElementById('results').style.display='block';
  const fin=d._final||{pct:Math.round(d.total),band_label:(d.band||{}).label};
  const s=fin.pct;
  document.getElementById('rnum').textContent=Math.round(s);
  const ring=document.getElementById('ring');
  ring.style.stroke=scoreColor(s);ring.style.strokeDasharray=CIRC;ring.style.strokeDashoffset=CIRC;
  setTimeout(()=>{ring.style.strokeDashoffset=CIRC*(1-s/100);},60);
  const vp=document.getElementById('vpill');
  vp.textContent=s+'% · '+(fin.band_label||'—');
  vp.style.background=scoreColor(s);vp.style.color=(s>=50&&s<65)?'#000':'#fff';
  document.getElementById('tname').textContent=d.ticker+(d.drug?'  ·  '+d.drug:'');
  document.getElementById('tmeta').textContent=(d.indication||'indication n/a')+'  ·  scored '+(d.generated||'').slice(0,10)+(d._cached?' (cached)':'');
  // badges
  const eq=(d.evidence_quality||'—'), ec=d.evidence_counts||{};
  document.getElementById('badges').innerHTML=
    `<span class="badge ${eq}">evidence: ${eq}</span>`+
    `<span class="badge">${ec.edgar||0} filings</span>`+
    `<span class="badge">${ec.trials||0} trials</span>`+
    `<span class="badge">${ec.corpus_excerpts||0} community posts</span>`;
  renderComposition(d);
  renderLoa(d);
  renderPipe(d);
  renderSections(d);
  // flags
  const fc=document.getElementById('flagcard');
  if(d.deductions&&d.deductions.length){
    fc.style.display='';
    document.getElementById('flags').innerHTML=d.deductions.map(x=>
      `<div class="flag"><b>${x.key.replace(/_/g,' ')}</b> &nbsp;${x.points} pts</div>`).join('');
  }else fc.style.display='none';
}
function probColor(p){if(p==null)return 'var(--muted)';return p>=80?'var(--green)':p>=60?'var(--lime)':p>=45?'var(--amber)':p>=30?'var(--orange)':'var(--red)';}
function renderComposition(d){
  const el=document.getElementById('composition'), fin=d._final;
  if(!fin){el.innerHTML='';return;}
  if(fin.loa_pct==null){
    el.innerHTML=`<div>This ticker isn't in the current catalyst scan, so there's no stage-based
      historical rate to blend in — the number above is this drug's evidence-based read only.</div>`;
    return;
  }
  const wPct=Math.round(fin.weight_evidence*100);
  let html=`<div><b>${fin.pct}%</b> = this drug's evidence (${fin.rubric_prob}%, weighted ${wPct}%
    for "${d.evidence_quality}" evidence) blended with the historical base rate for its stage/designations
    (${fin.loa_pct}%, weighted ${100-wPct}%).</div>`;
  const gap=fin.loa_pct-fin.rubric_prob;
  if(Math.abs(gap)>=20){
    const det=(d.detail||[]).map(x=>({...x,ratio:x.max?x.awarded/x.max:1}));
    det.sort((a,b)=>a.ratio-b.ratio);
    const weak=det.filter(x=>x.ratio<0.4).slice(0,3).map(x=>`${x.key.replace(/_/g,' ')} ${x.awarded}/${x.max}`);
    const ded=(d.deductions||[]).map(x=>`${x.key.replace(/_/g,' ')} (${x.points} pts)`);
    const reasons=[...ded,...weak].slice(0,4);
    if(reasons.length) html+=`<div class="why">Its own evidence is ${gap>0?'weaker':'stronger'} than a typical
      drug at this stage because: ${reasons.join(' · ')}</div>`;
  }
  el.innerHTML=html;
}
function renderLoa(d){
  const c=document.getElementById('loacard'), b=document.getElementById('loabody'), lo=d._loa;
  if(!lo){c.style.display='none';return;}
  c.style.display='';
  const dd=lo.decision_date?lo.decision_date.slice(0,10):'—';
  const days=lo.timeline_days!=null?lo.timeline_days:null;
  const dsrc=lo.pdufa_date?('disclosed PDUFA date '+lo.pdufa_date):'stage-median estimate (no disclosed PDUFA date)';
  b.innerHTML=
    `<div class="loa-big">
       <div class="loa-pct">${lo.pct}% of peers approved historically</div>
     </div>
     <div class="loa-side" style="margin-top:6px">
       <div><b>Expected decision:</b> ${dd}${days!=null?' ('+days+' days)':''} — <i>${dsrc}</i></div>
     </div>
     <div class="loa-basis">${(lo.basis||[]).map(x=>`<span>${x}</span>`).join('')}</div>
     <div class="loa-note">Base rate anchored to published FDA approval statistics (BIO/Biomedtracker clinical success rates, FDA CDER) — one of the two inputs blended into the number at the top of the page. Model ${lo.model||''}.</div>`;
}
function renderPipe(d){
  const stages=d._stages||[];const st=d._stage;
  const cur=st?st.index:-1;
  document.getElementById('pipe').innerHTML=stages.map((name,i)=>{
    let cls='pstep';
    if(st&&st.crl&&i===cur)cls+=' crl';
    else if(i===cur)cls+=' cur';
    else if(i<cur)cls+=' done';
    return `<div class="${cls}"><span class="dot"></span>${name}</div>`;
  }).join('') + (st?'':`<div style="color:var(--muted);font-size:11px;padding-top:14px">not in current catalyst scan — stage unknown</div>`);
}
function renderSections(d){
  const byId={};(d.detail||[]).forEach(x=>{(byId[x.section]=byId[x.section]||[]).push(x);});
  const cols=['var(--purple)','var(--blue)','var(--cyan)','var(--amber)','var(--green)'];
  let html='';
  (d.sections||[]).forEach((sec,i)=>{
    const pct=sec.max?100*sec.score/sec.max:0;
    html+=`<div class="crow"><div class="cname">§${sec.id} ${sec.name} <small>/${sec.max}</small></div>
      <div class="track"><div class="fill" style="background:${cols[i%5]}" data-w="${pct}"></div></div>
      <div class="cval">${sec.score.toFixed(1)}</div></div>`;
    const det=byId[sec.id]||[];
    if(det.length)html+=`<div class="detail">`+det.map(x=>
      `<div><b>${x.key.replace(/_/g,' ')}</b> ${x.awarded}/${x.max} — ${x.why||''}</div>`).join('')+`</div>`;
  });
  document.getElementById('sections').innerHTML=html;
  setTimeout(()=>document.querySelectorAll('.fill').forEach(f=>f.style.width=f.dataset.w+'%'),60);
}
let CATDATA=null, catSort='catalyst', hideRejected=false;
async function loadCatalysts(){
  const b=document.getElementById('cat-body');b.innerHTML='Loading…';
  try{
    CATDATA=await(await fetch('/catalysts')).json();
    if(!CATDATA.catalysts.length){b.innerHTML='No catalyst scan yet. Run biotech_catalyst_scanner.py to generate one.';return;}
    document.getElementById('cat-title').textContent=
      `Micro-Cap Catalysts ≤$${CATDATA.max_price} · ${CATDATA.count} names · scan ${(CATDATA.generated||'').slice(0,10)}`;
    renderCat();
  }catch(e){b.innerHTML='failed to load catalysts: '+e;}
}
function setCatSort(s){catSort=s;renderCat();}
function toggleRejected(cb){hideRejected=cb.checked;renderCat();}
function renderCat(){
  let rows=CATDATA.catalysts.slice();
  if(hideRejected)rows=rows.filter(r=>!r.rejected);
  const oddsOf=r=>r.final_pct!=null?r.final_pct:r.loa_pct;
  if(catSort==='odds'){
    rows.sort((a,b)=>(oddsOf(b)==null?-1:oddsOf(b))-(oddsOf(a)==null?-1:oddsOf(a)));
  }else if(catSort==='timeline'){
    rows.sort((a,b)=>(a.timeline_days==null?1e9:a.timeline_days)-(b.timeline_days==null?1e9:b.timeline_days));
  }else{
    rows.sort((a,b)=> b.max_severity-a.max_severity || b.n_catalysts-a.n_catalysts || (b.latest_date>a.latest_date?1:-1));
  }
  let h=`<div class="sort-ctl">Sort by:
    <button class="sort-btn ${catSort==='catalyst'?'active':''}" onclick="setCatSort('catalyst')">Catalyst</button>
    <button class="sort-btn ${catSort==='odds'?'active':''}" onclick="setCatSort('odds')">Approval odds</button>
    <button class="sort-btn ${catSort==='timeline'?'active':''}" onclick="setCatSort('timeline')">Timeline</button>
    <label style="margin-left:auto;cursor:pointer;color:var(--text)"><input type="checkbox" ${hideRejected?'checked':''} onclick="toggleRejected(this)"> Hide rejected</label></div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:10px">Odds = blended (evidence + historical base rate) once a ticker's been analyzed · <span style="font-style:italic">italic</span> = base rate only, click to get the real read</div>`;
  h+=`<table><thead><tr><th>Ticker</th><th>$</th><th>Odds</th><th>Stage</th><th>Decision</th><th>Catalysts</th></tr></thead><tbody>`;
  rows.forEach(r=>{
    const px=r.price!=null?'$'+r.price.toFixed(2):'—';
    const odds=oddsOf(r);
    const confirmed=r.final_pct!=null;
    const oddsTxt=odds!=null?(confirmed?odds+'%':'<i>'+odds+'%*</i>'):'—';
    const rej=r.rejected?'<span class="rejbadge">REJECTED</span>':'';
    const cls='clk'+(r.rejected?' rejrow':'');
    const dd=r.decision_date?r.decision_date.slice(0,10):'—';
    const dtit=r.pdufa_date?`title="PDUFA date ${r.pdufa_date}"`:(r.timeline_basis?`title="${r.timeline_basis}"`:'');
    h+=`<tr class="${cls}" onclick="pick('${r.ticker}')">
      <td><span class="tkr">${r.ticker}</span>${rej}</td>
      <td class="px">${px}</td>
      <td class="scorecell" style="color:${probColor(odds)}" title="${confirmed?'blended: evidence + historical base rate':'not yet analyzed — historical base rate only, click to analyze'}">${oddsTxt}</td>
      <td>${r.stage||'—'}</td>
      <td class="px" ${dtit}>${dd}</td>
      <td>${(r.catalysts||[]).slice(0,2).join(', ')}</td></tr>`;
  });
  document.getElementById('cat-body').innerHTML=h+'</tbody></table>';
}
function pick(tk){document.getElementById('tk').value=tk;showTab('an');go();}
</script></body></html>"""


if __name__ == '__main__':
    SCORES_DIR.mkdir(exist_ok=True)
    app.run(host='0.0.0.0', port=5007, debug=False)

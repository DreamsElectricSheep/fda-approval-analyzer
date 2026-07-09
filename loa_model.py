#!/usr/bin/env python3
"""
loa_model.py — historically-grounded Likelihood of Approval (LoA) + time-to-decision.

The "approval odds" and "timeline" metrics are anchored to PUBLISHED historical FDA
drug-approval statistics, not to hand-waved heuristics. This is base-rate modelling —
the same method professional LoA models use: start from the empirical approval rate
for a drug's development stage, then adjust for the factors history shows matter most
(therapeutic area, special designations, prior CRL). It is NOT a claim of proprietary
per-drug prediction; every number here is traceable to the sources below.

SOURCES (all real, public):
  • BIO / Informa (Biomedtracker) / QLS Advisors, "Clinical Development Success Rates
    and Contributing Factors 2011–2020" (2021). Phase-transition & LoA-by-phase and
    LoA-by-therapeutic-area figures.
  • BIO, "Clinical Development Success Rates 2006–2015" (2016). Corroborating LoA/TA.
  • FDA CDER, NDA/BLA first-cycle & overall approval statistics; PDUFA review-clock
    norms (standard ~10 months from filing acceptance, priority ~6 months).
  • Peer-reviewed analyses of Breakthrough Therapy / Fast Track / Orphan approval
    outcomes (e.g. Hwang et al.; FDA designation program reports).

All rates are AGGREGATE historical base rates. Individual drugs vary; the rubric
evidence score (fda_analyzer) is the drug-specific layer that sits on top of this.

CONFIDENCE NOTE (be honest about what's grounded vs approximate):
  • STRONG: the per-phase base rates (STAGE_LOA) are the published BIO transition rates,
    correctly compounded — these are the load-bearing numbers.
  • APPROXIMATE: the TA / designation / CRL multipliers are directionally supported by
    the literature but the specific coefficients are hand-set, not fitted. They are
    applied with a STAGE TAPER (see STAGE_MODIFIER_WEIGHT): the TA/designation effect is
    trial-stage attrition, so it is strong at Phase 1-2 and fades near-zero by Filed/PDUFA
    (a *filed* drug approves ~90% regardless of area or designation). Without this taper
    v1.0.0 understated filed-oncology at 63% vs the real ~85-90%.
  • VOTE-BLIND: the AdCom base is a flat average; the model cannot see whether the vote
    was positive or negative (a negative vote historically approves <25%).
"""
import re
from datetime import datetime, date

MODEL_VERSION = '1.1.0 (2026-07-06)'

# ── LoA FROM a given development stage (probability a program at this stage is ever
#    approved). Compounded phase-transition rates, BIO 2011–2020 all-disease. ────────
#    P1→P2 52.0% · P2→P3 28.9% · P3→Filing 57.8% · Filing→Approval 90.6%
#    => LoA P1 7.9% · P2 15.1% · P3 52.3% · Filed ~90%.
STAGE_LOA = {
    'Preclinical': 0.055,   # pre-IND, below Phase 1
    'Phase 1':     0.079,   # BIO all-disease LoA from Phase 1
    'Phase 2':     0.151,   # from Phase 2
    'Phase 3':     0.523,   # from Phase 3
    'Filed':       0.900,   # NDA/BLA submitted & accepted -> approval (~90.6%)
    'PDUFA':       0.900,   # PDUFA date set == filed, under review
    'AdCom':       0.750,   # AdCom convened: tougher/borderline cases; positive vote much higher
    'Decision':    0.500,   # decision imminent/unknown outcome
}

# ── Therapeutic-area multiplier vs all-disease baseline (from BIO TA LoA / 7.9%).
#    Applied to the stage base; clamped so late-stage stays realistic. ───────────────
#    TA LoA-from-P1: Heme 23.9% · Oncology 5.3% · Infectious ~19% · Ophtho ~17% ·
#    Metabolic ~15% · CV ~10% · Neuro ~8-9% · Psych ~7% · Respiratory ~5-12%.
TA_MULT = {
    'hematology':      1.55,
    'rare_disease':    1.35,
    'infectious':      1.25,
    'ophthalmology':   1.20,
    'metabolic':       1.05,
    'endocrine':       1.05,
    'autoimmune':      1.00,
    'default':         1.00,
    'cardiovascular':  0.90,
    'respiratory':     0.90,
    'neurology':       0.82,
    'psychiatry':      0.78,
    'oncology':        0.70,
}

# ── Designation multipliers. Programs with these designations approve at materially
#    higher rates historically (selection + FDA engagement). Product is capped. ──────
DESIGNATION_MULT = {
    'Breakthrough designation': 1.35,
    'Fast Track designation':   1.15,
    'Orphan designation':       1.20,
    'Priority Review':          1.10,
}
MAX_DESIGNATION_BOOST = 1.7   # cap the combined designation effect

# ── Prior CRL: a Complete Response Letter is a real setback. Many programs resubmit
#    and eventually approve, but with lower odds + delay. ────────────────────────────
CRL_PENALTY = 0.60

# ── Stage taper: how much of the TA/designation deviation-from-1.0 to apply at each
#    stage. TA-driven attrition is front-loaded in trials, so the effect is full early
#    and near-zero once a drug is filed/under review (fixes v1.0.0 filed-oncology 63%).
STAGE_MODIFIER_WEIGHT = {
    'Preclinical': 1.0, 'Phase 1': 1.0, 'Phase 2': 0.85, 'Phase 3': 0.55,
    'Filed': 0.20, 'PDUFA': 0.20, 'AdCom': 0.30, 'Decision': 0.15,
}

LOA_FLOOR, LOA_CEIL = 0.03, 0.95

# ── Median days from stage to the FDA decision (for the timeline metric). ───────────
#    Late stages from PDUFA review-clock norms; earlier from typical phase durations.
STAGE_DAYS = {
    'Preclinical': 3650, 'Phase 1': 2400, 'Phase 2': 1600, 'Phase 3': 900,
    'Filed': 300, 'PDUFA': 150, 'AdCom': 75, 'Decision': 21,
}

# ── Therapeutic-area inference from indication / drug text (keyword -> area). ────────
TA_KEYWORDS = {
    'oncology': ['cancer', 'tumor', 'tumour', 'carcinoma', 'lymphoma', 'leukemia',
                 'melanoma', 'glioblastoma', 'glioma', 'sarcoma', 'oncolog', 'metasta',
                 'myeloma', 'solid tumor', 'nsclc'],
    'hematology': ['hemophilia', 'haemophilia', 'anemia', 'anaemia', 'thrombocytopenia',
                   'sickle cell', 'thalassemia', 'hematolog', 'bleeding disorder'],
    'infectious': ['infection', 'antibiotic', 'antiviral', 'hiv', 'hepatitis', 'covid',
                   'sepsis', 'bacterial', 'viral', 'fungal', 'vaccine'],
    'ophthalmology': ['ophthalm', 'retina', 'macular', 'glaucoma', 'uveitis', 'ocular', 'eye'],
    'metabolic': ['diabetes', 'diabetic', 'obesity', 'nash', 'metabolic', 'insulin',
                  'hyperlipidemia', 'cholesterol'],
    'neurology': ['alzheimer', 'parkinson', 'epilepsy', 'seizure', 'multiple sclerosis',
                  'als', 'neuro', 'migraine', 'huntington', 'ataxia', 'spinal muscular',
                  'stroke', 'hemorrhage'],
    'psychiatry': ['depression', 'schizophrenia', 'bipolar', 'anxiety', 'ptsd',
                   'psychiatr', 'ppd', 'adhd'],
    'cardiovascular': ['cardiac', 'heart', 'cardiovascular', 'hypertension', 'atrial',
                       'cardiomyopath', 'arrhythmia', 'thrombo'],
    'respiratory': ['asthma', 'copd', 'pulmonary', 'respiratory', 'cystic fibrosis',
                    'lung', 'ipf', 'fibrosis'],
    'autoimmune': ['lupus', 'rheumatoid', 'psoriasis', 'crohn', 'colitis', 'autoimmun',
                   'ibd', 'atopic', 'immunolog'],
    'rare_disease': ['rare disease', 'orphan', 'ultra-rare', 'lysosomal', 'duchenne',
                     'dmd', 'friedreich', 'rett', 'genetic disorder'],
}


# ── Pipeline stage mapping (single source of truth for scanner + dashboard). ────────
STAGES = ['Preclinical', 'Phase 1', 'Phase 2', 'Phase 3', 'Filed', 'PDUFA', 'AdCom', 'Decision']
CATALYST_STAGE = {
    'Breakthrough designation': 2, 'Fast Track designation': 2, 'Orphan designation': 2,
    'Pivotal endpoint reported': 3, 'Topline data readout': 3,
    'NDA activity': 4, 'BLA/sBLA activity': 4, 'FDA accepted filing': 4,
    'Priority Review': 5, 'PDUFA date set': 5,
    'AdCom scheduled/held': 6, 'CRL (rejection)': 5,
}


def stage_from_catalysts(catalysts):
    """Furthest-along pipeline stage implied by a ticker's catalyst labels."""
    idx = max([CATALYST_STAGE.get(c, 0) for c in (catalysts or [])], default=0)
    return STAGES[idx]


def _kw_hit(kw, text):
    """Word-boundary keyword match. Short acronym-like tokens (<=4 letters, e.g. 'als')
    must match as WHOLE words so 'als' no longer fires on 'also'/'metals'; longer tokens
    match as word-start stems so 'oncolog' still catches 'oncology'."""
    if len(kw) <= 4 and kw.isalpha():
        return re.search(r'\b' + kw + r'\b', text) is not None
    return re.search(r'\b' + re.escape(kw), text) is not None


def infer_therapeutic_area(text):
    """Best-guess therapeutic area from an indication/drug string. 'default' if unclear."""
    if not text:
        return 'default'
    low = text.lower()
    best, best_hits = 'default', 0
    for area, kws in TA_KEYWORDS.items():
        hits = sum(1 for k in kws if _kw_hit(k, low))
        if hits > best_hits:
            best, best_hits = area, hits
    return best


def likelihood_of_approval(stage, designations=None, therapeutic_area='default', has_crl=False):
    """Historical base-rate LoA as a probability [0.03, 0.95], with a breakdown.

    stage: one of STAGE_LOA keys. designations: list of catalyst labels (may include the
    designation names). therapeutic_area: a TA_MULT key (or 'default'). has_crl: bool.
    Returns {loa, base, ta_mult, desig_mult, crl_mult, basis[]}.
    """
    base = STAGE_LOA.get(stage, 0.10)
    ta_mult = TA_MULT.get(therapeutic_area, 1.0)
    designations = designations or []
    desig_mult = 1.0
    applied = []
    for d in designations:
        if d in DESIGNATION_MULT:
            desig_mult *= DESIGNATION_MULT[d]
            applied.append(d)
    desig_mult = min(desig_mult, MAX_DESIGNATION_BOOST)
    crl_mult = CRL_PENALTY if has_crl else 1.0

    # Stage taper: TA/designation effects are front-loaded (trial-stage attrition), so
    # shrink their deviation-from-1.0 toward 0 as the drug advances to review.
    w = STAGE_MODIFIER_WEIGHT.get(stage, 0.5)
    ta_eff = 1.0 + (ta_mult - 1.0) * w
    desig_eff = 1.0 + (desig_mult - 1.0) * w

    loa = base * ta_eff * desig_eff * crl_mult
    loa = max(LOA_FLOOR, min(LOA_CEIL, loa))

    basis = [f'stage {stage}: base {base:.0%}']
    if therapeutic_area != 'default':
        basis.append(f'{therapeutic_area} ×{ta_eff:.2f}')
    if applied:
        basis.append(f'{"+".join(a.split()[0] for a in applied)} ×{desig_eff:.2f}')
    if has_crl:
        basis.append(f'prior CRL ×{crl_mult:.2f}')
    if w < 1.0 and (therapeutic_area != 'default' or applied):
        basis.append(f'(late-stage taper w={w:.2f})')
    if stage == 'AdCom':
        basis.append('⚠ vote-blind (flat AdCom base; a negative vote is much lower)')
    return {'loa': round(loa, 3), 'base': base, 'ta_mult': round(ta_eff, 3),
            'desig_mult': round(desig_eff, 3), 'crl_mult': crl_mult, 'stage_weight': w,
            'basis': basis}


# ── ONE final blended approval-odds number ────────────────────────────────────
# The rubric score (drug-specific evidence, 0-100) and the LoA (historical base
# rate for this stage/profile) answer different questions, and showing both as
# separate headline percentages was confusing even with an explanation attached.
# This blends them into a single probability: weight on the drug-specific evidence
# scales with how much real evidence was actually found (evidence_quality). Sparse
# evidence -> the rubric score is close to a neutral 50 placeholder and isn't
# informative, so defer heavily to the historical base rate. Rich evidence -> the
# rubric score reflects a real, specific read on this drug and should dominate.
EVIDENCE_WEIGHT = {'rich': 0.80, 'thin': 0.50, 'sparse': 0.20}


def _parse_prob_range(text):
    """'85-95%' -> (85.0, 95.0); '<25%' -> (LOA_FLOOR*100, 25.0)."""
    nums = [float(x) for x in re.findall(r'\d+', text or '')]
    if not nums:
        return (50.0, 50.0)
    if text.strip().startswith('<'):
        return (LOA_FLOOR * 100, nums[0])
    if len(nums) >= 2:
        return (nums[0], nums[1])
    return (nums[0], nums[0])


def rubric_total_to_prob(total, rubric):
    """Continuous, monotonic score(0-100) -> probability(0-100) mapping, built by
    linearly interpolating across the rubric's OWN probability_bands boundaries —
    avoids the coarse step-function jump a band lookup gives right at e.g. 49 vs 50."""
    bands = rubric.get('probability_bands', [])
    if not bands:
        return max(0.0, min(100.0, total))
    anchors = []
    for b in bands:
        lo_p, hi_p = _parse_prob_range(b.get('approval_prob', ''))
        anchors.append((b['min'], lo_p))
        anchors.append((b['max'], hi_p))
    anchors = sorted(set(anchors))
    if total <= anchors[0][0]:
        return anchors[0][1]
    if total >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= total <= x1:
            if x1 == x0:
                return y0
            return y0 + (total - x0) / (x1 - x0) * (y1 - y0)
    return 50.0


def band_label_for_pct(pct):
    if pct >= 80: return 'Very likely approval'
    if pct >= 65: return 'Likely approval'
    if pct >= 45: return 'Coin-flip / lean approve'
    if pct >= 25: return 'Lean reject'
    return 'Likely reject'


def final_approval_odds(rubric_total, rubric, loa_pct, evidence_quality):
    """The ONE headline number: blends this drug's own evidence (rubric_total,
    mapped to a probability) with the historical base rate for its stage/profile
    (loa_pct), weighted by evidence_quality. Returns pct, band_label, and the two
    inputs + weight so the breakdown stays inspectable, not a black box.
    """
    rubric_prob = rubric_total_to_prob(rubric_total, rubric)
    if loa_pct is None:
        pct = round(rubric_prob)
        return {'pct': pct, 'band_label': band_label_for_pct(pct),
                'rubric_prob': round(rubric_prob, 1), 'loa_pct': None,
                'weight_evidence': 1.0, 'evidence_quality': evidence_quality}
    w = EVIDENCE_WEIGHT.get(evidence_quality, 0.5)
    blended = w * rubric_prob + (1 - w) * loa_pct
    blended = max(LOA_FLOOR * 100, min(LOA_CEIL * 100, blended))
    pct = round(blended)
    return {'pct': pct, 'band_label': band_label_for_pct(pct),
            'rubric_prob': round(rubric_prob, 1), 'loa_pct': loa_pct,
            'weight_evidence': w, 'evidence_quality': evidence_quality}


def time_to_decision(stage, pdufa_date=None, today=None):
    """Days until the FDA decision. Uses a real PDUFA date when known, else the stage
    median. Returns {days, decision_date, basis}."""
    today = today or date.today()
    if pdufa_date:
        try:
            d = datetime.strptime(pdufa_date, '%Y-%m-%d').date()
            return {'days': (d - today).days, 'decision_date': pdufa_date, 'basis': 'PDUFA date (disclosed)'}
        except ValueError:
            pass
    days = STAGE_DAYS.get(stage, 1000)
    est = today.toordinal() + days
    return {'days': days, 'decision_date': date.fromordinal(est).isoformat(),
            'basis': f'{stage} median (no disclosed PDUFA date)'}


if __name__ == '__main__':
    # quick self-check / illustration
    for stg, des, ta, crl in [
        ('PDUFA', ['PDUFA date set', 'Breakthrough designation', 'Orphan designation'], 'rare_disease', False),
        ('PDUFA', ['PDUFA date set', 'CRL (rejection)'], 'ophthalmology', True),
        ('Phase 3', ['Pivotal endpoint reported'], 'oncology', False),
        ('Filed', ['BLA/sBLA activity'], 'default', False),
    ]:
        r = likelihood_of_approval(stg, des, ta, crl)
        print(f'{stg:8} {ta:13} crl={crl!s:5} -> LoA {r["loa"]:.0%}  [{", ".join(r["basis"])}]')

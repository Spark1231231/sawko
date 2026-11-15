"""Pokusná slučka pre vylepšenia gólového modelu: ľubovoľná sada vstupov λ -> stacking + rekalibrácia
(učené rolling po rokoch na populácii „uefa“) -> metriky na ladení a teste. Výber VŽDY podľa ladenia."""
from __future__ import annotations

import hashlib
import json
import sys
from multiprocessing import Pool

import numpy as np

import calib as C
import engine as E
import walk as W
from final_goals import boot_diff, calpop

CACHE = E.ROOT / "data" / "raw" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
w = E.load_world()
CFG = json.loads((E.ROOT / "results" / "goals_tuning.json").read_text())
BASE = dict(CFG["strength_cfg"])
BASE["imp"] = tuple(BASE["imp"])
RHO = CFG["rho"]


def market_lambdas():
    """λ trhu pre zápasy so záverečnými kurzmi (1X2 + over/under) – známe pred zápasom."""
    f = CACHE / "market_lam.npz"
    if f.exists():
        z = np.load(f)
        return z["mlh"], z["mla"]
    mlh, mla = np.full(len(w), np.nan), np.full(len(w), np.nan)
    ok = np.flatnonzero((w.oh.notna() & w.od.notna() & w.oa.notna()).values)
    for i in ok:
        p = E.devig(w.oh.values[i], w.od.values[i], w.oa.values[i])
        po, line = None, 2.5
        if np.isfinite(w.o_over.values[i]) and np.isfinite(w.o_under.values[i]) and np.isfinite(w.ou_line.values[i]):
            qo, qu = 1 / w.o_over.values[i], 1 / w.o_under.values[i]
            po, line = qo / (qo + qu), float(w.ou_line.values[i])
        try:
            lh, la = E.lambdas_from_probs(float(p[0]), float(p[2]), po, RHO, line=line)
            if 0.05 < lh < 8 and 0.05 < la < 8:
                mlh[i], mla[i] = lh, la
        except Exception:
            pass
    np.savez(f, mlh=mlh, mla=mla)
    return mlh, mla


w["mlh"], w["mla"] = market_lambdas()
POP = calpop(w)
ROWS = np.flatnonzero(POP & (w.date >= "2010-01-01").values & w.hg.notna().values)
DATE = w.date.values[ROWS]
YEAR = w.dt.dt.year.values[ROWS]
HG, AG = w.hg.values[ROWS], w.ag.values[ROWS]
HF = (~w.neutral.values[ROWS]).astype(float)
EV = np.isin(w.comp.values[ROWS], E.EVAL_COMPS) & (DATE >= "2018-08-01")
TUNE, TEST = EV & (DATE < "2024-01-01"), EV & (DATE >= "2024-01-01")
O = C.outcome(HG, AG)


def lam(cfg: dict):
    c = {**BASE, **cfg}
    key = hashlib.md5(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]
    f = CACHE / f"exp_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["lh"], z["la"]
    lh, la = W.walk_strength(w, ROWS, c)
    np.savez(f, lh=lh, la=la)
    return lh, la


_ELO = None


def elo():
    global _ELO
    if _ELO is None:
        pre = E.run_elo(w, **CFG["elo_cfg"])
        _ELO = W.walk_elo(w, ROWS, pre=pre)
    return _ELO


def design(feats, side):
    """feats: zoznam (λ_dom, λ_host). side 0 = pohľad domácich, 1 = hostí."""
    cols = [np.ones(len(ROWS))]
    for lh, la in feats:
        own, opp = (lh, la) if side == 0 else (la, lh)
        cols += [np.log(own), np.log(opp)]
    cols.append(HF if side == 0 else np.zeros(len(ROWS)))
    return np.column_stack(cols)


def evaluate(feats, label="", verbose=True):
    Xh, Xa = design(feats, 0), design(feats, 1)
    n = len(ROWS)
    M = np.full((n, E.MAXG, E.MAXG), np.nan)
    P = np.full((n, 3), np.nan)
    for y in range(2018, 2027):
        tr = YEAR < y
        beta = E.glm_poisson(np.vstack([Xh[tr], Xa[tr]]), np.concatenate([np.minimum(HG[tr], 8), np.minimum(AG[tr], 8)]), ridge=1e-3)
        ch, ca = np.exp(Xh @ beta), np.exp(Xa @ beta)
        theta = C.fit_recal(E.outcome_probs(E.score_matrix(ch[tr], ca[tr], RHO)), O[tr])
        sel = EV & (YEAR == y)
        if sel.any():
            Ms = E.score_matrix(ch[sel], ca[sel], RHO)
            Ps = C.apply_recal(theta, E.outcome_probs(Ms))
            M[sel], P[sel] = C.reweight(Ms, Ps), Ps
    out = {}
    for nm, mk in (("ladenie", TUNE), ("test", TEST)):
        out[nm] = C.matrix_metrics(M[mk], P[mk], HG[mk], AG[mk])
    ll = -np.log(np.maximum(P[np.arange(n), O], 1e-12))
    if verbose:
        t, s = out["ladenie"], out["test"]
        print(f"{label:46s} ladenie ll {t['logloss']:.4f} rps {t['rps']:.4f} skóre {t['score_nll']:.4f} ou {t['ou25']:.4f} | "
              f"test ll {s['logloss']:.4f} rps {s['rps']:.4f} skóre {s['score_nll']:.4f} ou {s['ou25']:.4f}", flush=True)
    return out, ll, beta


def _job(cfg):
    lam(cfg)
    return cfg


if __name__ == "__main__":
    cfgs = [dict(), dict(halflife=250), dict(halflife=2000), dict(target="sot", halflife=650, sot_mix=1.0),
            dict(mkt_mix=0.5), dict(mkt_mix=0.25), dict(mkt_mix=0.0), dict(mkt_mix=0.25, halflife=400), dict(mkt_mix=0.25, halflife=1000)]
    with Pool(4) as p:
        p.map(_job, cfgs)
    e = elo()
    base, ll0, _ = evaluate([lam({}), e], "základ (aktuálny prototyp)")
    res = {}
    for label, feats in (
        ("+ polčas 250 d", [lam({}), lam(dict(halflife=250)), e]),
        ("+ polčas 2000 d", [lam({}), lam(dict(halflife=2000)), e]),
        ("+ polčasy 250 a 2000 d", [lam({}), lam(dict(halflife=250)), lam(dict(halflife=2000)), e]),
        ("+ sila zo striel na bránu", [lam({}), lam(dict(target="sot", halflife=650, sot_mix=1.0)), e]),
        ("cieľ: 50 % kurzy", [lam(dict(mkt_mix=0.5)), e]),
        ("cieľ: 75 % kurzy", [lam(dict(mkt_mix=0.25)), e]),
        ("cieľ: 100 % kurzy", [lam(dict(mkt_mix=0.0)), e]),
        ("cieľ: 75 % kurzy, polčas 400", [lam(dict(mkt_mix=0.25, halflife=400)), e]),
        ("cieľ: 75 % kurzy, polčas 1000", [lam(dict(mkt_mix=0.25, halflife=1000)), e]),
        ("základ + cieľ 75 % kurzy (oba vstupy)", [lam({}), lam(dict(mkt_mix=0.25)), e]),
    ):
        out, ll, _ = evaluate(feats, label)
        d_t, d_s = boot_diff(ll[TUNE], ll0[TUNE]), boot_diff(ll[TEST], ll0[TEST])
        print(f"{'':46s} rozdiel vs základ: ladenie {d_t[0]:+.4f} [{d_t[1]:+.4f},{d_t[2]:+.4f}] | test {d_s[0]:+.4f} [{d_s[1]:+.4f},{d_s[2]:+.4f}]", flush=True)

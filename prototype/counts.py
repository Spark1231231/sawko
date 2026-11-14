"""Modely počtov: strely, strely na bránu, rohy, žlté karty, fauly.

Pre každý tím v zápase:   E[počet] = exp( x·β  +  A[tím] − D[súper] )

* x·β – „čo sa dá vyčítať z papierovej sily“: domáce prostredie, rozdiel síl z gólového modelu
  (favorit strieľa viac a fauluje menej), typ súťaže. Poissonova regresia, učená po rokoch len z minulosti.
* A, D – odchýlka tímu od toho, čo predpovedá x·β (štýl: strelecky aktívny tím, tvrdo hrajúci tím…),
  MAP odhad s normálnym priorom a časovým útlmom – rovnaký fit ako pri góloch.
* rozptyl: záporné binomické rozdelenie (gamma-Poissonova zmes), r odhadnuté z minulých odchýlok.

Všetko walk-forward. Porovnáva sa s (a) konštantným priemerom súťaže a (b) fitom útok/obrana bez x·β
(tak počty počíta web).
"""
from __future__ import annotations

from math import lgamma

import numpy as np
import pandas as pd

import engine as E

KINDS = {            # stĺpce (domáci, hostia), maska platnosti
    "shots": ("hsh", "ash", "stats_ok"),
    "sot": ("hsot", "asot", "stats_ok"),
    "corners": ("hcor", "acor", "stats_ok"),
    "fouls": ("hfouls", "afouls", "stats_ok"),
    "yellow": ("hy", "ay", "events_ok"),
}
DEFAULT = dict(halflife=1100.0, window=9.0, sigma=0.18, friendly_w=0.7)


def nb_logpmf(k, mean, r):
    """log P(K=k) záporného binomického rozdelenia so strednou hodnotou `mean` a disperziou r (Var = μ + μ²/r)."""
    k = np.asarray(k, float)
    mean = np.maximum(np.asarray(mean, float), 1e-9)
    if not np.isfinite(r) or r > 1e5:
        return k * np.log(mean) - mean - np.vectorize(lgamma)(k + 1)
    lg = np.vectorize(lgamma)
    return lg(k + r) - lg(r) - lg(k + 1) + r * np.log(r / (r + mean)) + k * np.log(mean / (r + mean))


def nb_pmf(mean, r, n):
    return np.exp(nb_logpmf(np.arange(n), mean, r))


def total_pmf(mh, ma, r, n=80):
    """Rozdelenie súčtu dvoch nezávislých NB (konvolúcia)."""
    return np.convolve(nb_pmf(mh, r, n), nb_pmf(ma, r, n))[:n]


def frame(w: pd.DataFrame, rows: np.ndarray, lh: np.ndarray, la: np.ndarray) -> pd.DataFrame:
    """Jedna tabuľka zápasov s λ z gólového modelu (walk-forward) a počtami."""
    d = w.iloc[rows].copy()
    d["lh"], d["la"] = lh, la
    d["s"] = np.log(lh / la)
    d["hf"] = (~d.neutral.values).astype(float)
    d["friendly"] = (d.comp == "fifa.friendly").astype(float)
    d["major"] = d.comp.isin(["fifa.world", "uefa.euro", "conmebol.america", "concacaf.gold", "caf.nations"]).astype(float)
    d["year"] = d.dt.dt.year
    return d.reset_index(drop=True)


def _X(s_own, home, away, friendly, major):
    return np.column_stack([np.ones_like(s_own), home, away, s_own, s_own ** 2, friendly, major])


def features(d: pd.DataFrame, extra=None):
    """extra = (Eh, Ea): voliteľné ďalšie stĺpce príznakov pre domácich / hostí (napr. predpovedané fauly)."""
    z = np.zeros(len(d))
    Xh = _X(d.s.values, d.hf.values, z, d.friendly.values, d.major.values)
    Xa = _X(-d.s.values, z, d.hf.values, d.friendly.values, d.major.values)
    if extra is not None:
        Xh, Xa = np.column_stack([Xh, extra[0]]), np.column_stack([Xa, extra[1]])
    return Xh, Xa


def walk_counts(d: pd.DataFrame, kind: str, cfg: dict | None = None, start: str = "2019-01-01", use_features: bool = True,
                extra=None):
    """Vracia stredné hodnoty (dom, hos), disperziu r a základný priemer pre zápasy od `start`."""
    cfg = {**DEFAULT, **(cfg or {})}
    ch, ca, okc = KINDS[kind]
    ok = d[okc].fillna(False).astype(bool).values & d[ch].notna().values & d[ca].notna().values
    yh, ya = d[ch].fillna(0).values.astype(float), d[ca].fillna(0).values.astype(float)
    Xh, Xa = features(d, extra)
    day, year = d.day.values, d.year.values
    n_t = int(max(d.h.max(), d.a.max())) + 1
    h, a = d.h.values, d.a.values
    n = len(d)
    mh, ma = np.full(n, np.nan), np.full(n, np.nan)
    bh, ba = np.full(n, np.nan), np.full(n, np.nan)       # baseline: priemer z minulosti
    rr = np.full(n, np.nan)
    off_h, off_a = np.zeros(n), np.zeros(n)
    wbase = np.where(d.friendly.values > 0, cfg["friendly_w"], 1.0)
    target = np.flatnonzero(ok & (d.date.values >= start))
    win = int(cfg["window"] * 365.25)
    sig = cfg["sigma"] / np.sqrt(2)
    last_year, init = None, None
    resid = []   # (rok, y, mu) minulých out-of-sample predpovedí – na odhad disperzie
    i = 0
    tday = day[target]
    while i < len(target):
        D = tday[i]
        j = i
        while j < len(target) and tday[j] == D:
            j += 1
        y = year[target[i]]
        past = ok & (day < D)
        if y != last_year:   # raz za rok: regresia na papierovú silu + disperzia
            tr = ok & (year < y)
            if use_features and tr.sum() > 300:
                beta = E.glm_poisson(np.vstack([Xh[tr], Xa[tr]]), np.concatenate([yh[tr], ya[tr]]))
            else:
                beta = np.zeros(Xh.shape[1])
                beta[0] = np.log(max((yh[tr].sum() + ya[tr].sum()) / max(2 * tr.sum(), 1), 1e-3)) if tr.any() else 0.0
            off_h, off_a = Xh @ beta, Xa @ beta
            if len(resid) > 400:
                R = np.array(resid)
                mu, yy = R[:, 1], R[:, 0]
                exc = np.mean((yy - mu) ** 2) - np.mean(mu)
                r_now = float(np.mean(mu ** 2) / exc) if exc > 1e-6 else np.inf
            else:
                r_now = np.inf
            last_year, init = y, None
        sl = past & (day >= D - win)
        wt = wbase[sl] * 0.5 ** ((D - day[sl]) / cfg["halflife"])
        att, dfn, mu0, gam = E.fit_strength(h[sl], a[sl], yh[sl], ya[sl], wt, np.zeros(sl.sum()), n_t, sig, sig,
                                            off_h=off_h[sl], off_a=off_a[sl], init=init, sweeps=25)
        init = (att, dfn, mu0, 0.0)
        r_ = target[i:j]
        mh[r_] = np.exp(mu0 + att[h[r_]] - dfn[a[r_]] + off_h[r_])
        ma[r_] = np.exp(mu0 + att[a[r_]] - dfn[h[r_]] + off_a[r_])
        base = (yh[past].sum() + ya[past].sum()) / max(2 * past.sum(), 1)
        bh[r_], ba[r_] = base, base
        rr[r_] = r_now
        for k in r_:
            resid.append((yh[k], mh[k]))
            resid.append((ya[k], ma[k]))
        i = j
    return dict(mh=mh, ma=ma, bh=bh, ba=ba, r=rr, yh=yh, ya=ya, target=target)


def mom_r(y, mu) -> float:
    """Disperzia NB metódou momentov: Var = μ + μ²/r."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    exc = np.mean((y - mu) ** 2) - np.mean(mu)
    return float(np.mean(mu ** 2) / exc) if exc > 1e-6 else np.inf


def p_over(mean, r, line):
    k = int(np.floor(line)) + 1
    return float(1 - nb_pmf(mean, r, k).sum())


def evaluate(res: dict, fit_mask: np.ndarray, eval_mask: np.ndarray, lines, factor=None) -> dict:
    """Porovnanie so „zlatým“ konštantným tipom (najlepšia možná konštanta na tom istom období).

    * tím: NLL počtu jedného tímu (NB), konštanta = priemer domácich / hostí v období,
    * súčet: NLL a MAE súčtu oboch tímov, over/under log-loss na čiarach; rozptyl súčtu r_tot sa
      odhaduje z out-of-sample odchýlok na období `fit_mask` (ladenie) a používa aj na teste.
    """
    T = res["target"]
    fit = T[fit_mask[T]]
    t = T[eval_mask[T]]
    f = 1.0 if factor is None else factor
    g = lambda k, idx: res[k][idx]
    mh, ma = g("mh", t) * (f[t] if factor is not None else 1.0), g("ma", t) * (f[t] if factor is not None else 1.0)
    yh, ya = g("yh", t), g("ya", t)
    fm_h, fm_a = g("mh", fit) * (f[fit] if factor is not None else 1.0), g("ma", fit) * (f[fit] if factor is not None else 1.0)
    r_team = mom_r(np.concatenate([g("yh", fit), g("ya", fit)]), np.concatenate([fm_h, fm_a]))
    r_tot = mom_r(g("yh", fit) + g("ya", fit), fm_h + fm_a)
    tot, pred = yh + ya, mh + ma
    out = dict(n=int(len(t)), mean=float(tot.mean()), r_team=r_team, r_total=r_tot)
    out["team_nll_model"] = float(-np.mean(np.concatenate([nb_logpmf(yh, mh, r_team), nb_logpmf(ya, ma, r_team)])))
    ch, ca = yh.mean(), ya.mean()
    rc = mom_r(np.concatenate([yh, ya]), np.concatenate([np.full(len(t), ch), np.full(len(t), ca)]))
    out["team_nll_const"] = float(-np.mean(np.concatenate([nb_logpmf(yh, ch, rc), nb_logpmf(ya, ca, rc)])))
    out["total_nll_model"] = float(-np.mean(nb_logpmf(tot, pred, r_tot)))
    rct = mom_r(tot, np.full(len(t), tot.mean()))
    out["total_nll_const"] = float(-np.mean(nb_logpmf(tot, tot.mean(), rct)))
    out["mae_model"], out["mae_const"] = float(np.mean(np.abs(tot - pred))), float(np.mean(np.abs(tot - tot.mean())))
    out["corr"] = float(np.corrcoef(tot, pred)[0, 1])
    out["lines"] = {}
    ll = lambda p, o: float(-np.mean(np.where(o, np.log(np.clip(p, 1e-9, 1)), np.log(np.clip(1 - p, 1e-9, 1)))))
    for line in lines:
        over = tot > line
        pm = np.array([p_over(m_, r_tot, line) for m_ in pred])
        out["lines"][str(line)] = dict(hit=float(over.mean()), model=ll(pm, over), const=ll(np.full(len(t), over.mean()), over))
    return out


def fit_live(d: pd.DataFrame, kind: str, cfg: dict | None, D: int, eval_mask: np.ndarray | None = None) -> dict:
    """Model počtu `kind` zo všetkých zápasov pred dňom D + disperzie z walk-forward odchýlok."""
    cfg = {**DEFAULT, **(cfg or {})}
    ch, ca, okc = KINDS[kind]
    ok = d[okc].fillna(False).astype(bool).values & d[ch].notna().values & d[ca].notna().values & (d.day.values < D)
    yh, ya = d[ch].fillna(0).values.astype(float), d[ca].fillna(0).values.astype(float)
    Xh, Xa = features(d)
    beta = E.glm_poisson(np.vstack([Xh[ok], Xa[ok]]), np.concatenate([yh[ok], ya[ok]]))
    off_h, off_a = Xh @ beta, Xa @ beta
    day = d.day.values
    sl = ok & (day >= D - int(cfg["window"] * 365.25))
    wt = np.where(d.friendly.values[sl] > 0, cfg["friendly_w"], 1.0) * 0.5 ** ((D - day[sl]) / cfg["halflife"])
    n_t = int(max(d.h.max(), d.a.max())) + 1
    sig = cfg["sigma"] / np.sqrt(2)
    att, dfn, mu0, _ = E.fit_strength(d.h.values[sl], d.a.values[sl], yh[sl], ya[sl], wt, np.zeros(sl.sum()), n_t, sig, sig,
                                      off_h=off_h[sl], off_a=off_a[sl], sweeps=200)
    res = walk_counts(d[d.day < D].reset_index(drop=True), kind, cfg, start="2021-01-01")
    T = res["target"]
    if eval_mask is not None:
        T = T[eval_mask[: len(d[d.day < D])][T]]
    r_team = mom_r(np.concatenate([res["yh"][T], res["ya"][T]]), np.concatenate([res["mh"][T], res["ma"][T]]))
    r_total = mom_r(res["yh"][T] + res["ya"][T], res["mh"][T] + res["ma"][T])
    return dict(kind=kind, beta=beta, att=att, dfn=dfn, mu0=mu0, r_team=r_team, r_total=r_total, walk=res)


def predict_live(fit: dict, h: int, a: int, s: float, home_field: float = 1.0, friendly: float = 0.0, major: float = 0.0):
    """Stredné hodnoty počtu pre domácich a hostí v konkrétnom zápase (s = log λ_dom/λ_host z gólového modelu)."""
    Xh = _X(np.array([s]), np.array([home_field]), np.array([0.0]), np.array([friendly]), np.array([major]))
    Xa = _X(np.array([-s]), np.array([0.0]), np.array([home_field]), np.array([friendly]), np.array([major]))
    mh = float(np.exp(fit["mu0"] + fit["att"][h] - fit["dfn"][a] + Xh @ fit["beta"])[0])
    ma = float(np.exp(fit["mu0"] + fit["att"][a] - fit["dfn"][h] + Xa @ fit["beta"])[0])
    return mh, ma

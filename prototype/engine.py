"""Jadro prototypu: dáta, Elo, časovo vážený Poissonov model sily, matica skóre, metriky.

Všetko je čistý numpy – jeden fit trvá milisekundy, takže walk-forward backtest
(model vidí len zápasy pred dňom predpovede) sa dá prepočítať pre stovky nastavení.
"""
from __future__ import annotations

from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

MAJOR = {"FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
         "Confederations Cup", "CONCACAF Championship", "Oceania Nations Cup"}
ESPN_CLS = {"fifa.friendly": 0, "fifa.world": 3, "uefa.euro": 3, "conmebol.america": 3, "concacaf.gold": 3, "caf.nations": 3}
EVAL_COMPS = ["uefa.nations", "fifa.worldq.uefa", "uefa.euroq", "uefa.euro", "fifa.world"]


def tournament_class(name: str) -> int:
    """0 priateľský, 1 menší turnaj, 2 kvalifikácia / Liga národov, 3 záverečný turnaj."""
    if name == "Friendly":
        return 0
    if name in MAJOR:
        return 3
    if "qualification" in name or "Nations League" in name:
        return 2
    return 1


# --------------------------------------------------------------------------- dáta
def load_world() -> pd.DataFrame:
    """Svetová tabuľka výsledkov (martj42) + zápasy z ESPN, ktoré v nej ešte nie sú.

    Každý riadok má kanonický dátum (podľa martj42, ak existuje) – walk-forward
    rez „deň < D“ tak nikdy neuvidí ten istý zápas pod iným dátumom.
    """
    intl = pd.read_csv(PROC / "intl.csv.gz")
    intl["cls"] = intl["tournament"].map(tournament_class)
    intl["espn_id"] = None
    intl["comp"] = None
    m = pd.read_csv(PROC / "matches.csv.gz", dtype={"id": str})
    md = m[m.done & m.hg.notna()].copy()
    md["d0"] = md.date.str[:10]
    key = {(r.date, r.home, r.away): i for i, r in enumerate(intl.itertuples(index=False))}
    espn_id = [None] * len(intl)
    comp = [None] * len(intl)
    extra = []
    for r in md.itertuples(index=False):
        d = pd.Timestamp(r.d0)
        hit = None
        for dd in (0, -1, 1):
            ds = (d + pd.Timedelta(days=dd)).strftime("%Y-%m-%d")
            if (ds, r.home, r.away) in key:
                hit = key[(ds, r.home, r.away)]
                break
        if hit is not None:
            espn_id[hit] = r.id
            comp[hit] = r.comp
            continue
        swapped = any(((d + pd.Timedelta(days=dd)).strftime("%Y-%m-%d"), r.away, r.home) in key for dd in (0, -1, 1))
        if swapped:
            continue  # ten istý zápas s opačným „domácim“ – v tabuľke už je
        extra.append(dict(date=r.d0, home=r.home, away=r.away, hg=r.hg, ag=r.ag, tournament=r.comp, city=r.city,
                          country=r.country, neutral=bool(r.neutral), cls=ESPN_CLS.get(r.comp, 2), espn_id=r.id, comp=r.comp))
    intl["espn_id"] = espn_id
    intl["comp"] = comp
    w = pd.concat([intl, pd.DataFrame(extra)], ignore_index=True)
    w["dt"] = pd.to_datetime(w["date"])
    w = w.sort_values(["dt"], kind="stable").reset_index(drop=True)
    w["day"] = (w["dt"] - pd.Timestamp("1970-01-01")).dt.days.astype(int)
    teams = sorted(set(w.home) | set(w.away))
    tid = {t: i for i, t in enumerate(teams)}
    w["h"] = w.home.map(tid).astype(int)
    w["a"] = w.away.map(tid).astype(int)
    w["hg"] = w.hg.astype(float)
    w["ag"] = w.ag.astype(float)
    w["neutral"] = w.neutral.astype(bool)
    st = m.set_index("id")[["hsh", "ash", "hsot", "asot", "hcor", "acor", "hfouls", "afouls", "hy", "ay", "hr", "ar",
                            "oh", "od", "oa", "ou_line", "o_over", "o_under", "referee", "stats_ok", "events_ok"]]
    w = w.join(st, on="espn_id")
    w.attrs["teams"] = teams
    return w


# --------------------------------------------------------------------------- Elo
def gd_mult(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def run_elo(w: pd.DataFrame, k=(20.0, 30.0, 40.0, 55.0), hfa: float = 80.0, start: float = 1500.0,
            k_scale: float = 1.0):
    """World-football Elo. Vracia rating oboch tímov PRED každým zápasom a konečné ratingy."""
    n_t = len(w.attrs["teams"])
    R = np.full(n_t, start)
    h, a = w.h.values, w.a.values
    hg, ag = w.hg.values, w.ag.values
    neu, cls = w.neutral.values, w.cls.values
    pre_h = np.empty(len(w))
    pre_a = np.empty(len(w))
    for i in range(len(w)):
        rh, ra = R[h[i]], R[a[i]]
        pre_h[i], pre_a[i] = rh, ra
        dr = rh - ra + (0.0 if neu[i] else hfa)
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        gd = hg[i] - ag[i]
        res = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        delta = k_scale * k[cls[i]] * gd_mult(int(gd)) * (res - we)
        R[h[i]] = rh + delta
        R[a[i]] = ra - delta
    return pre_h, pre_a, R


def glm_poisson(X: np.ndarray, y: np.ndarray, wt: np.ndarray | None = None, iters: int = 30, ridge: float = 1e-6):
    """Poissonova regresia (log link) cez IRLS."""
    wt = np.ones(len(y)) if wt is None else wt
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(np.average(y, weights=wt), 1e-6))
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta, -20, 20))
        g = X.T @ (wt * (y - mu))
        H = (X * (wt * mu)[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def elo_features(dz: np.ndarray, home: np.ndarray, away: np.ndarray) -> np.ndarray:
    """dz = (vlastný − súperov rating)/400; home/away = 1, ak tím hrá doma / vonku (nie na neutrále)."""
    return np.column_stack([np.ones_like(dz), dz, dz * dz, home, away])


def fit_elo_goals(pre_h, pre_a, hg, ag, neutral, wt=None):
    dz = (pre_h - pre_a) / 400.0
    hf = (~neutral).astype(float)
    X = np.vstack([elo_features(dz, hf, np.zeros_like(hf)), elo_features(-dz, np.zeros_like(hf), hf)])
    y = np.concatenate([hg, ag])
    ww = None if wt is None else np.concatenate([wt, wt])
    return glm_poisson(X, y, ww)


def elo_lambdas(beta, rh, ra, neutral):
    dz = (np.asarray(rh, float) - np.asarray(ra, float)) / 400.0
    hf = (~np.asarray(neutral, bool)).astype(float)
    z = np.zeros_like(hf)
    return np.exp(elo_features(dz, hf, z) @ beta), np.exp(elo_features(-dz, z, hf) @ beta)


# --------------------------------------------------------------------------- Poissonov model sily
def fit_strength(h, a, yh, ya, wt, hf, n_teams, sigma_s, sigma_t, off_h=None, off_a=None,
                 init=None, sweeps: int = 60, tol: float = 1e-6):
    """Penalizovaná vážená Poissonova vierohodnosť (MAP odhad):

        log λ_dom  = μ + γ·[doma] + att[dom]  − def[hos]  (+ offset)
        log λ_host = μ            + att[hos]  − def[dom]  (+ offset)

    Prior nie je na att a def zvlášť, ale na ich kombinácie:
        s = (att + def)/2  … celková SILA tímu          ~ N(0, σ_s²)  (široký prior)
        t = (att − def)/2  … gólový ŠTÝL (otvorený/zatvorený futbal) ~ N(0, σ_t²)  (úzky prior)
    Výsledok 1/X/2 určuje hlavne rozdiel síl, počet gólov hlavne štýl – každý potrebuje iné zmrštenie.

    Blokový Newton: v bloku „att“ (resp. „def“) je Hessián diagonálny (v jednom λ vystupuje vždy
    len jeden att a jeden def), takže krok je presný.
    """
    if init is None:
        att = np.zeros(n_teams)
        dfn = np.zeros(n_teams)
        tot = max(float(np.sum(wt)), 1e-9)
        mu = np.log(max((np.sum(wt * yh) + np.sum(wt * ya)) / (2 * tot), 1e-3))
        gam = 0.2
    else:
        att, dfn, mu, gam = init[0].copy(), init[1].copy(), init[2], init[3]
    off_h = 0.0 if off_h is None else off_h
    off_a = 0.0 if off_a is None else off_a
    # log-prior = −s²/(2σ_s²) − t²/(2σ_t²) = −(att+def)²/(8σ_s²) − (att−def)²/(8σ_t²)
    cs, ct = 1.0 / (4 * sigma_s ** 2), 1.0 / (4 * sigma_t ** 2)
    for _ in range(sweeps):
        lh = np.exp(mu + gam * hf + att[h] - dfn[a] + off_h)
        la = np.exp(mu + att[a] - dfn[h] + off_a)
        g = np.bincount(h, wt * (yh - lh), n_teams) + np.bincount(a, wt * (ya - la), n_teams) \
            - cs * (att + dfn) - ct * (att - dfn)
        H = np.bincount(h, wt * lh, n_teams) + np.bincount(a, wt * la, n_teams) + cs + ct
        s_att = np.clip(g / H, -1, 1)
        att += s_att
        lh = np.exp(mu + gam * hf + att[h] - dfn[a] + off_h)
        la = np.exp(mu + att[a] - dfn[h] + off_a)
        g = -np.bincount(a, wt * (yh - lh), n_teams) - np.bincount(h, wt * (ya - la), n_teams) \
            - cs * (att + dfn) + ct * (att - dfn)
        H = np.bincount(a, wt * lh, n_teams) + np.bincount(h, wt * la, n_teams) + cs + ct
        s_def = np.clip(g / H, -1, 1)
        dfn += s_def
        lh = np.exp(mu + gam * hf + att[h] - dfn[a] + off_h)
        la = np.exp(mu + att[a] - dfn[h] + off_a)
        s_mu = np.sum(wt * (yh - lh)) + np.sum(wt * (ya - la))
        s_mu /= max(np.sum(wt * (lh + la)), 1e-9)
        mu += float(np.clip(s_mu, -1, 1))
        lh = np.exp(mu + gam * hf + att[h] - dfn[a] + off_h)
        s_g = np.sum(wt * hf * (yh - lh)) / max(np.sum(wt * hf * lh), 1e-9)
        gam += float(np.clip(s_g, -1, 1))
        if max(np.max(np.abs(s_att)), np.max(np.abs(s_def)), abs(s_mu), abs(s_g)) < tol:
            break
    return att, dfn, mu, gam


# --------------------------------------------------------------------------- rozdelenie skóre
MAXG = 11
_FACT = np.array([lgamma(k + 1) for k in range(MAXG)])


def pois_pmf(lam, n: int = MAXG):
    lam = np.asarray(lam, float)[..., None]
    k = np.arange(n)
    return np.exp(k * np.log(np.maximum(lam, 1e-12)) - lam - _FACT[:n])


def score_matrix(lh, la, rho: float = 0.0, n: int = MAXG):
    """P(dom = i, host = j): nezávislé Poissony + Dixon–Colesova korekcia nízkych skóre, renormované."""
    ph, pa = pois_pmf(lh, n), pois_pmf(la, n)
    M = ph[..., :, None] * pa[..., None, :]
    if rho != 0.0:
        lh_ = np.asarray(lh, float)
        la_ = np.asarray(la, float)
        M = M.copy()
        M[..., 0, 0] *= np.maximum(1 - lh_ * la_ * rho, 1e-6)
        M[..., 0, 1] *= np.maximum(1 + lh_ * rho, 1e-6)
        M[..., 1, 0] *= np.maximum(1 + la_ * rho, 1e-6)
        M[..., 1, 1] *= np.maximum(1 - rho, 1e-6)
    return M / M.sum(axis=(-2, -1), keepdims=True)


_I = np.arange(MAXG)[:, None]
_J = np.arange(MAXG)[None, :]
MASK_H, MASK_D, MASK_A = _I > _J, _I == _J, _I < _J
TOT = _I + _J


def outcome_probs(M):
    return np.stack([(M * MASK_H).sum((-2, -1)), (M * MASK_D).sum((-2, -1)), (M * MASK_A).sum((-2, -1))], -1)


def stretch(lh, la, k: float = 1.0, g: float = 1.0):
    """Roztiahne rozdiel síl (log pomer λ) koeficientom k. g=1 drží geometrický priemer λ, g=0 súčet λ."""
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    s, t = np.log(lh) - np.log(la), np.log(lh) + np.log(la)
    nh, na = np.exp((t + k * s) / 2), np.exp((t - k * s) / 2)
    if g < 1:
        scale = (lh + la) / (nh + na)
        nh, na = nh * scale ** (1 - g), na * scale ** (1 - g)
    return nh, na


# --------------------------------------------------------------------------- metriky
def metrics(lh, la, hg, ag, rho: float = 0.0) -> dict:
    """Out-of-sample metriky pre vektor zápasov (nižšie = lepšie, okrem acc)."""
    M = score_matrix(lh, la, rho)
    P = outcome_probs(M)
    hg = np.asarray(hg, int)
    ag = np.asarray(ag, int)
    o = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    n = len(o)
    p_o = P[np.arange(n), o]
    onehot = np.eye(3)[o]
    cum_p, cum_o = np.cumsum(P, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    hc, ac = np.minimum(hg, MAXG - 1), np.minimum(ag, MAXG - 1)
    p_score = M[np.arange(n), hc, ac]
    p_over = (M * (TOT > 2.5)).sum((-2, -1))
    over = (hg + ag) > 2.5
    return dict(
        n=n,
        logloss=float(-np.mean(np.log(np.maximum(p_o, 1e-12)))),
        rps=float(np.mean(np.sum((cum_p - cum_o) ** 2, 1) / 2)),
        brier=float(np.mean(np.sum((P - onehot) ** 2, 1))),
        acc=float(np.mean(np.argmax(P, 1) == o)),
        score_nll=float(-np.mean(np.log(np.maximum(p_score, 1e-12)))),
        ou25=float(-np.mean(np.where(over, np.log(np.maximum(p_over, 1e-12)), np.log(np.maximum(1 - p_over, 1e-12))))),
    )


def per_match_logloss(lh, la, hg, ag, rho: float = 0.0):
    P = outcome_probs(score_matrix(lh, la, rho))
    hg = np.asarray(hg, int)
    ag = np.asarray(ag, int)
    o = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    return -np.log(np.maximum(P[np.arange(len(o)), o], 1e-12)), P, o


# --------------------------------------------------------------------------- trh
def devig(oh, od, oa):
    """Férové pravdepodobnosti z kurzov 1X2 (mocninová metóda: p_i = q_i^k, Σ p_i = 1)."""
    q = np.stack([1 / np.asarray(oh, float), 1 / np.asarray(od, float), 1 / np.asarray(oa, float)], -1)
    lo, hi = np.full(q.shape[:-1], 0.5), np.full(q.shape[:-1], 3.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        s = np.sum(q ** mid[..., None], -1)
        hi = np.where(s < 1, mid, hi)
        lo = np.where(s >= 1, mid, lo)
    p = q ** ((lo + hi) / 2)[..., None]
    return p / p.sum(-1, keepdims=True)


def lambdas_from_probs(p1, p2, p_over=None, rho: float = 0.0, total_hint: float = 2.7, line: float = 2.5):
    """Spätne nájde (λ_dom, λ_host), ktoré dajú P(1), P(2) a (ak je) P(over `line`). Newton na 2 rovniciach."""
    s = np.log(max(p1, 1e-6) / max(p2, 1e-6)) * 0.6
    t = np.log(total_hint / 2) * 2
    for _ in range(60):
        def f(s_, t_):
            lh, la = np.exp((t_ + s_) / 2), np.exp((t_ - s_) / 2)
            M = score_matrix(lh, la, rho)
            P = outcome_probs(M)
            r1 = np.log(P[0] / P[2]) - np.log(p1 / p2)
            if p_over is None:
                r2 = (lh + la) - total_hint
            else:
                po = float((M * (TOT > line)).sum())
                r2 = np.log(po / (1 - po)) - np.log(p_over / (1 - p_over))
            return np.array([r1, r2])
        r = f(s, t)
        if np.max(np.abs(r)) < 1e-9:
            break
        e = 1e-5
        J = np.column_stack([(f(s + e, t) - r) / e, (f(s, t + e) - r) / e])
        try:
            ds, dt = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            break
        s, t = s + float(np.clip(ds, -1, 1)), t + float(np.clip(dt, -1, 1))
    return float(np.exp((t + s) / 2)), float(np.exp((t - s) / 2))

"""Walk-forward (rolling-origin) predpovede: pre každý deň D sa model odhadne len zo zápasov
pred D a predpovie zápasy dňa D. Žiadny zápas nikdy nevidí vlastný výsledok."""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as E

DEFAULT = dict(
    halflife=900.0,          # polčas útlmu váh v dňoch
    window=12.0,             # dĺžka okna v rokoch
    imp=(0.5, 0.8, 1.0, 1.0),  # váha podľa triedy súťaže: priateľský, menší turnaj, kvalifikácia/LN, turnaj
    sigma_s=1.0,             # prior na celkovú silu tímu (široký)
    sigma_t=0.35,            # prior na gólový štýl tímu (úzky)
    cap=8,                   # orezanie debaklov (10:0 nenesie 10× viac informácie)
    sot_mix=1.0,             # váha gólov v cieli; zvyšok = strely na bránu × konverzia (tlmí šum gólov)
)


def eval_rows(w: pd.DataFrame, start: str, end: str, comps=E.EVAL_COMPS, friendlies: bool = False) -> np.ndarray:
    ok = w.comp.isin(comps) if not friendlies else (w.comp == "fifa.friendly")
    ok &= (w.date >= start) & (w.date <= end) & w.hg.notna()
    return np.flatnonzero(ok.values)


def walk_strength(w: pd.DataFrame, rows: np.ndarray, cfg: dict | None = None, return_fits: bool = False):
    """λ_dom, λ_host pre riadky `rows` z Poissonovho modelu sily (útok/obrana), walk-forward."""
    cfg = {**DEFAULT, **(cfg or {})}
    n_t = len(w.attrs["teams"])
    day = w.day.values
    h, a = w.h.values, w.a.values
    cap = cfg["cap"]
    yh, ya = np.minimum(w.hg.values, cap), np.minimum(w.ag.values, cap)
    g = cfg["sot_mix"]
    if g < 1 and "hsot" in w:
        has = np.isfinite(w.hsot.values) & np.isfinite(w.asot.values)
        conv = (w.hg.values[has].sum() + w.ag.values[has].sum()) / max(w.hsot.values[has].sum() + w.asot.values[has].sum(), 1)
        yh = np.where(has, g * yh + (1 - g) * conv * np.nan_to_num(w.hsot.values), yh)
        ya = np.where(has, g * ya + (1 - g) * conv * np.nan_to_num(w.asot.values), ya)
    hf = (~w.neutral.values).astype(float)
    imp = np.asarray(cfg["imp"])[w.cls.values]
    lh, la = np.empty(len(rows)), np.empty(len(rows))
    order = np.argsort(day[rows], kind="stable")
    rows_s = rows[order]
    days_s = day[rows_s]
    init = None
    fits = {}
    win = int(cfg["window"] * 365.25)
    i = 0
    while i < len(rows_s):
        D = days_s[i]
        j = i
        while j < len(rows_s) and days_s[j] == D:
            j += 1
        lo = np.searchsorted(day, D - win, "left")
        hi = np.searchsorted(day, D, "left")       # striktne pred dňom D
        sl = slice(lo, hi)
        wt = imp[sl] * 0.5 ** ((D - day[sl]) / cfg["halflife"])
        att, dfn, mu, gam = E.fit_strength(h[sl], a[sl], yh[sl], ya[sl], wt, hf[sl], n_t,
                                           cfg["sigma_s"], cfg["sigma_t"], init=init)
        init = (att, dfn, mu, gam)
        r = rows_s[i:j]
        lh[order[i:j]] = np.exp(mu + gam * hf[r] + att[h[r]] - dfn[a[r]])
        la[order[i:j]] = np.exp(mu + att[a[r]] - dfn[h[r]])
        if return_fits:
            fits[int(D)] = (att.copy(), dfn.copy(), mu, gam)
        i = j
    return (lh, la, fits) if return_fits else (lh, la)


def walk_elo(w: pd.DataFrame, rows: np.ndarray, elo_cfg: dict | None = None, train_years: float = 20.0,
             pre=None):
    """λ z Elo: ratingy pred zápasom (sekvenčne, bez úniku) + Poissonova regresia gólov na rozdiel
    ratingov, prepočítaná pre každý kalendárny rok len z minulých zápasov."""
    elo_cfg = elo_cfg or {}
    pre_h, pre_a, _ = pre if pre is not None else E.run_elo(w, **elo_cfg)
    day = w.day.values
    year = w.dt.dt.year.values
    lh, la = np.empty(len(rows)), np.empty(len(rows))
    hg, ag, neu = w.hg.values, w.ag.values, w.neutral.values
    for y in np.unique(year[rows]):
        start_day = (pd.Timestamp(f"{y}-01-01") - pd.Timestamp("1970-01-01")).days
        tr = (day < start_day) & (day >= start_day - int(train_years * 365.25))
        beta = E.fit_elo_goals(pre_h[tr], pre_a[tr], np.minimum(hg[tr], 8), np.minimum(ag[tr], 8), neu[tr])
        sel = year[rows] == y
        r = rows[sel]
        lh[sel], la[sel] = E.elo_lambdas(beta, pre_h[r], pre_a[r], neu[r])
    return lh, la


def blend(l1, l2, wgt: float):
    """Geometrický mix dvoch odhadov λ (wgt = váha druhého)."""
    return np.exp((1 - wgt) * np.log(l1) + wgt * np.log(l2))

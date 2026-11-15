"""Kalibračná vrstva: opravuje systematické chyby gólového modelu, naučená VŽDY len z minulých
out-of-sample predpovedí (rolling origin po kalendárnych rokoch).

1. stack  – Poissonova regresia skutočných gólov na log λ (vlastné, súperove) + doma:
            opraví úroveň a „rozťah“ očakávaných gólov.
2. recal  – multinomická rekalibrácia 1/X/2: log(p1/px) a log(p2/px) dostanú vlastný posun a sklon
            (zovšeobecnené Plattovo škálovanie). Opraví to, že nezávislé Poissony pri nerovných
            súperoch nadhodnocujú remízu a podhodnocujú favorita.
3. reweight – matica skóre sa prenásobí po oblastiach (výhra domácich / remíza / výhra hostí), aby jej
            súčty sedeli s rekalibrovaným 1/X/2 (riešenie s minimálnou KL-divergenciou).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import engine as E


def _design(feats, hf, side):
    """feats: zoznam dvojíc (λ_dom, λ_host) z jednotlivých modelov. side 0 = pohľad domácich, 1 = hostí."""
    hf = np.atleast_1d(np.asarray(hf, float))
    cols = [np.ones(len(hf))]
    for lh, la in feats:
        lh, la = np.atleast_1d(np.asarray(lh, float)), np.atleast_1d(np.asarray(la, float))
        own, opp = (lh, la) if side == 0 else (la, lh)
        cols += [np.log(own), np.log(opp)]
    cols.append(hf if side == 0 else np.zeros(len(hf)))
    return np.column_stack(cols)


def fit_stack(feats, hf, hg, ag, wt=None):
    """Stacking: skutočné góly ~ Poisson(exp(a + Σ_k [b_k·log λ_k,vlastné + c_k·log λ_k,súper] + f·doma))."""
    X = np.vstack([_design(feats, hf, 0), _design(feats, hf, 1)])
    y = np.concatenate([np.minimum(hg, 8), np.minimum(ag, 8)])
    ww = None if wt is None else np.concatenate([wt, wt])
    return E.glm_poisson(X, y, ww, ridge=1e-3)


def apply_stack(beta, feats, hf):
    return np.exp(_design(feats, hf, 0) @ beta), np.exp(_design(feats, hf, 1) @ beta)


def _recal_probs(theta, x1, x2):
    eh = theta[0] + theta[1] * x1
    ea = theta[2] + theta[3] * x2
    mx = np.maximum(np.maximum(eh, ea), 0.0)
    ph, pd_, pa = np.exp(eh - mx), np.exp(-mx), np.exp(ea - mx)
    s = ph + pd_ + pa
    return np.stack([ph / s, pd_ / s, pa / s], -1)


def fit_recal(P, o, wt=None):
    """P: (n,3) pravdepodobnosti modelu, o: skutočný výsledok 0/1/2."""
    P = np.clip(P, 1e-9, 1)
    x1, x2 = np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / P[:, 1])
    wt = np.ones(len(o)) if wt is None else wt
    idx = np.arange(len(o))

    def nll(theta):
        Q = _recal_probs(theta, x1, x2)
        return -np.sum(wt * np.log(np.maximum(Q[idx, o], 1e-12))) / wt.sum()

    r = minimize(nll, np.array([0.0, 1.0, 0.0, 1.0]), method="BFGS")
    return r.x


def apply_recal(theta, P):
    P = np.clip(P, 1e-9, 1)
    return _recal_probs(theta, np.log(P[..., 0] / P[..., 1]), np.log(P[..., 2] / P[..., 1]))


def reweight(M, P_new):
    """Prenásobí oblasti matice skóre tak, aby 1/X/2 = P_new (zachová tvar rozdelenia vnútri oblasti)."""
    P_old = E.outcome_probs(M)
    f = np.asarray(P_new) / np.maximum(P_old, 1e-12)
    return M * (f[..., 0, None, None] * E.MASK_H + f[..., 1, None, None] * E.MASK_D + f[..., 2, None, None] * E.MASK_A)


def outcome(hg, ag):
    hg, ag = np.asarray(hg), np.asarray(ag)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def fit_calibration(feats, hf, hg, ag, rho=-0.06, wt=None):
    """Naučí (beta, theta) z out-of-sample predpovedí."""
    beta = fit_stack(feats, hf, hg, ag, wt)
    ch, ca = apply_stack(beta, feats, hf)
    P = E.outcome_probs(E.score_matrix(ch, ca, rho))
    theta = fit_recal(P, outcome(hg, ag), wt)
    return beta, theta


def predict_calibrated(cal, feats, hf, rho=-0.06):
    """(beta, theta) -> kalibrované λ, matica skóre (po oblastiach preškálovaná) a 1/X/2."""
    beta, theta = cal
    ch, ca = apply_stack(beta, feats, hf)
    M = E.score_matrix(ch, ca, rho)
    P = apply_recal(theta, E.outcome_probs(M))
    return ch, ca, reweight(M, P), P


def matrix_metrics(M, P, hg, ag) -> dict:
    """Metriky priamo z matice skóre a 1/X/2 (po kalibrácii už nie sú čistý Poisson)."""
    hg, ag = np.asarray(hg, int), np.asarray(ag, int)
    o = outcome(hg, ag)
    n = len(o)
    onehot = np.eye(3)[o]
    cp, co = np.cumsum(P, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    hc, ac = np.minimum(hg, E.MAXG - 1), np.minimum(ag, E.MAXG - 1)
    p_over = (M * (E.TOT > 2.5)).sum((-2, -1))
    over = (hg + ag) > 2.5
    return dict(
        n=n,
        logloss=float(-np.mean(np.log(np.maximum(P[np.arange(n), o], 1e-12)))),
        rps=float(np.mean(np.sum((cp - co) ** 2, 1) / 2)),
        brier=float(np.mean(np.sum((P - onehot) ** 2, 1))),
        acc=float(np.mean(np.argmax(P, 1) == o)),
        score_nll=float(-np.mean(np.log(np.maximum(M[np.arange(n), hc, ac], 1e-12)))),
        ou25=float(-np.mean(np.where(over, np.log(np.maximum(p_over, 1e-12)), np.log(np.maximum(1 - p_over, 1e-12))))),
    )

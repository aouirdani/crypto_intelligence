import numpy as np
import pandas as pd
from scipy.optimize import minimize

def mean_variance(weights, mu, cov, l2=0.0):
    w = np.array(weights)
    return -w.dot(mu) + 0.5*w.dot(cov).dot(w) + l2*np.sum(w*w)

def mpt_optimize(returns: pd.DataFrame, target_vol=None, bounds=(0,1), l2=0.0):
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n = len(mu)
    x0 = np.ones(n)/n

    cons = [{"type":"eq","fun": lambda w: np.sum(w)-1}]
    if target_vol is not None:
        cons.append({"type":"ineq","fun": lambda w: target_vol - np.sqrt(w@cov@w)})
    bnds = [bounds]*n

    res = minimize(mean_variance, x0, args=(mu, cov, l2), constraints=cons, bounds=bnds)
    w = res.x
    port_mu = w.dot(mu)
    port_vol = np.sqrt(w@cov@w)
    sharpe = port_mu / (port_vol + 1e-9)
    return {"weights": pd.Series(w, index=returns.columns, name="weight"),
            "mu": port_mu, "vol": port_vol, "sharpe": sharpe}

def risk_parity(returns: pd.DataFrame, bounds=(0,1)):
    cov = (returns.cov().values) * 252
    n = cov.shape[0]
    x0 = np.ones(n)/n
    def risk_contrib(w):
        port_var = w@cov@w
        mrc = cov@w
        rc = w*mrc/port_var
        return rc
    def obj(w):
        rc = risk_contrib(w)
        return np.sum((rc - 1.0/n)**2)

    cons = [{"type":"eq","fun": lambda w: np.sum(w)-1}]
    bnds = [bounds]*n
    res = minimize(obj, x0, bounds=bnds, constraints=cons)
    return pd.Series(res.x, index=returns.columns, name="weight")

def monte_carlo_var(returns: pd.DataFrame, weights: pd.Series, horizon_days=10, n_paths=5000, alpha=0.05):
    mu = returns.mean().values
    cov = returns.cov().values
    w = weights.values.reshape(-1,1)
    # simulate multi-variate normal
    L = np.linalg.cholesky(cov + 1e-12*np.eye(cov.shape[0]))
    pnl = []
    for _ in range(n_paths):
        # rough multi-day portfolio PnL draw
        path = np.random.multivariate_normal(mu, cov, horizon_days)
        pnl.append((path @ w).sum())
    pnl = np.array(pnl).flatten()
    var = np.quantile(pnl, alpha)
    es = pnl[pnl<=var].mean() if np.any(pnl<=var) else var
    return float(var), float(es)

import streamlit as st
import pandas as pd
from utils.data_fetcher import binance_klines
from utils.portfolio_opt import mpt_optimize, risk_parity, monte_carlo_var

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("📈 Portfolio Optimizer")

assets = st.multiselect("Assets", ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"], default=["BTCUSDT","ETHUSDT","SOLUSDT"])
interval = st.selectbox("Interval", ["4h","1d"], index=1)
limit = 1000 if interval=="4h" else 700

dfs = {}
for a in assets:
    df = binance_klines(a, interval=interval, limit=limit)
    df = df.set_index("open_time")[["close"]].rename(columns={"close":a})
    dfs[a] = df
prices = pd.concat(dfs.values(), axis=1).dropna()
rets = prices.pct_change().dropna()

st.subheader("Prices")
st.line_chart(prices)

st.subheader("Mean-Variance (MPT)")
target_vol = st.slider("Target vol (annualized, optional)", 0.0, 1.0, 0.0, step=0.05)
target_vol = target_vol if target_vol > 0 else None
mpt = mpt_optimize(rets, target_vol=target_vol, bounds=(0,1), l2=1e-4)
st.write(mpt["weights"].to_frame())
st.caption(f"μ={mpt['mu']:.2f}  σ={mpt['vol']:.2f}  Sharpe={mpt['sharpe']:.2f}")

st.subheader("Risk Parity")
rp = risk_parity(rets, bounds=(0,1))
st.write(rp.to_frame())

st.subheader("Monte Carlo Risk (portfolio VaR/ES)")
horizon = st.slider("Horizon (days)", 1, 30, 10)
var, es = monte_carlo_var(rets, mpt["weights"], horizon_days=horizon, n_paths=2000, alpha=0.05)
st.metric("VaR (5%)", f"{var:.4f}")
st.metric("Expected Shortfall", f"{es:.4f}")

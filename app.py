import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Crypto Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Custom CSS --------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-success { 
        background-color: #d4edda; 
        border-left-color: #28a745; 
        color: #155724;
    }
    .alert-warning { 
        background-color: #fff3cd; 
        border-left-color: #ffc107; 
        color: #856404;
    }
    .alert-danger { 
        background-color: #f8d7da; 
        border-left-color: #dc3545; 
        color: #721c24;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Data fetch & features --------------------
@st.cache_data(ttl=300)
def get_crypto_data(symbol: str = "bitcoin", days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLC proxy series (price, volume, market_cap) from CoinGecko.
    Falls back to mock data if API fails."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(
            {
                "timestamp": [datetime.fromtimestamp(x[0] / 1000) for x in data["prices"]],
                "price": [x[1] for x in data["prices"]],
                "volume": [x[1] for x in data["total_volumes"]],
                "market_cap": [x[1] for x in data["market_caps"]],
            }
        ).set_index("timestamp")
        # Ensure monotonic index
        df = df[~df.index.duplicated()].sort_index()
        return df
    except Exception:
        # --- Fallback: geometric random walk ---
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        np.random.seed(42)
        base_prices = {
            "bitcoin": 45000,
            "ethereum": 2500,
            "binancecoin": 250,
            "cardano": 0.45,
            "solana": 95,
            "matic-network": 0.85,
        }
        base_price = base_prices.get(symbol, 100.0)
        returns = np.random.normal(0.001, 0.03, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        return pd.DataFrame(
            {
                "price": prices,
                "volume": np.random.uniform(1e9, 5e9, len(dates)),
                "market_cap": np.array(prices) * 19000000,
            },
            index=dates,
        )


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append standard indicators."""
    out = df.copy()
    out["sma_20"] = ta.trend.sma_indicator(out["price"], window=20)
    out["sma_50"] = ta.trend.sma_indicator(out["price"], window=50)
    out["rsi"] = ta.momentum.rsi(out["price"], window=14)
    out["macd"] = ta.trend.macd_diff(out["price"])
    out["bb_upper"] = ta.volatility.bollinger_hband(out["price"])
    out["bb_lower"] = ta.volatility.bollinger_lband(out["price"])
    return out


def simple_prediction(df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
    """Simple ML-ish prediction.
    Trains a small RandomForest on lag features; then combines its drift with a smooth trend + noise for a 30-day path.
    Provides a 95% band as ±5%.
    """
    try:
        feat = pd.DataFrame(
            {
                "price_lag1": df["price"].shift(1),
                "price_lag7": df["price"].shift(7),
                "ret_1d": df["price"].pct_change(),
                "ret_7d": df["price"].pct_change(7),
                "volume": df["volume"],
            }
        ).dropna()
        y = df["price"].shift(-1).dropna()
        feat = feat.iloc[: len(y)]
        if len(feat) < 150:
            raise ValueError("Not enough data")

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(feat, y)

        # Estimate recent drift from model
        last_row = pd.DataFrame(
            {
                "price_lag1": [df["price"].iloc[-1]],
                "price_lag7": [df["price"].iloc[-7] if len(df) >= 7 else df["price"].iloc[-1]],
                "ret_1d": [df["price"].pct_change().iloc[-1]],
                "ret_7d": [df["price"].pct_change(7).iloc[-1] if len(df) >= 7 else 0.0],
                "volume": [df["volume"].iloc[-1]],
            }
        )
        model_next = float(model.predict(last_row)[0])
        last_price = df["price"].iloc[-1]
        base_drift = (model_next / last_price) - 1.0

        preds = []
        p = last_price
        for i in range(days_ahead):
            # decay drift over horizon + small noise
            drift = base_drift * (1 - i / max(1, days_ahead))
            noise = np.random.normal(0, 0.01)
            p = p * (1 + drift + noise)
            preds.append(p)

        future_idx = pd.date_range(df.index[-1] + timedelta(days=1), periods=days_ahead, freq="D")
        preds = np.array(preds)
        upper = preds * 1.05
        lower = preds * 0.95

        return pd.DataFrame(
            {"predicted_price": preds, "confidence_upper": upper, "confidence_lower": lower},
            index=future_idx,
        )
    except Exception:
        last_price = df["price"].iloc[-1]
        future_idx = pd.date_range(df.index[-1] + timedelta(days=1), periods=days_ahead, freq="D")
        baseline = np.array([last_price * (1.01 ** (i / 5)) for i in range(days_ahead)])
        return pd.DataFrame(
            {
                "predicted_price": baseline,
                "confidence_upper": baseline * 1.08,
                "confidence_lower": baseline * 0.92,
            },
            index=future_idx,
        )


# -------------------- UI --------------------
def main():
    st.markdown('<h1 class="main-header">🚀 Crypto Intelligence Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    crypto_options = {
        "Bitcoin": "bitcoin",
        "Ethereum": "ethereum",
        "Binance Coin": "binancecoin",
        "Cardano": "cardano",
        "Solana": "solana",
        "Polygon": "matic-network",
    }

    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency:", list(crypto_options.keys()))
    time_range = st.sidebar.selectbox("Time Range:", ["30 days", "90 days", "1 year"], index=1)
    days = {"30 days": 30, "90 days": 90, "1 year": 365}[time_range]

    # Fetch + features
    with st.spinner("Loading data..."):
        df = get_crypto_data(crypto_options[selected_crypto], days)
        df = add_technical_indicators(df)

    # ---- Key metrics ----
    col1, col2, col3, col4 = st.columns(4)
    current_price = float(df["price"].iloc[-1])
    price_change = ((df["price"].iloc[-1] / df["price"].iloc[-2]) - 1) * 100 if len(df) > 1 else 0.0
    volume_24h = float(df["volume"].iloc[-1])
    market_cap = float(df["market_cap"].iloc[-1])

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>${current_price:,.2f}</h3>
            <p>Current Price</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        color = "#28a745" if price_change > 0 else "#dc3545"
        st.markdown(
            f"""
        <div class="metric-container">
            <h3 style="color: {color};">{price_change:+.2f}%</h3>
            <p>24h Change</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>${volume_24h/1e9:.2f}B</h3>
            <p>24h Volume</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>${market_cap/1e9:.2f}B</h3>
            <p>Market Cap</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ---- Technical chart ----
    st.subheader("📈 Technical Analysis")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & Moving Averages", "RSI"),
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", line=dict(color="blue", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_50"], name="SMA 50", line=dict(color="red")), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", line=dict(color="gray", dash="dash")), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", line=dict(color="gray", dash="dash")), row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=700, title=f"{selected_crypto} Technical Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Predictions ----
    st.subheader("🤖 AI Price Predictions")
    preds = simple_prediction(df, days_ahead=30)

    fig_pred = go.Figure()
    recent = df.tail(60)
    fig_pred.add_trace(
        go.Scatter(x=recent.index, y=recent["price"], name="Historical Price", line=dict(color="blue"))
    )
    fig_pred.add_trace(
        go.Scatter(x=preds.index, y=preds["predicted_price"], name="AI Prediction", line=dict(color="red", dash="dash"))
    )
    # Confidence band
    fig_pred.add_trace(
        go.Scatter(
            x=preds.index,
            y=preds["confidence_upper"],
            mode="lines",
            line=dict(color="rgba(255,0,0,0)"),
            showlegend=False,
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=preds.index,
            y=preds["confidence_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(255,0,0,0)"),
            name="Confidence Band",
            fillcolor="rgba(255,0,0,0.2)",
        )
    )
    fig_pred.update_layout(
        title=f"{selected_crypto} - 30 Day Price Prediction", xaxis_title="Date", yaxis_title="Price (USD)", height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ---- Alerts ----
    st.subheader("⚠️ Trading Signals")
    colA, colB = st.columns(2)

    with colA:
        rsi_now = float(df["rsi"].iloc[-1])
        if rsi_now > 70:
            st.markdown(
                f"""
            <div class="alert-box alert-warning">
                <strong>⚠️ Overbought Signal</strong><br>
                RSI is {rsi_now:.1f} — consider trimming risk
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif rsi_now < 30:
            st.markdown(
                f"""
            <div class="alert-box alert-success">
                <strong>✅ Oversold Signal</strong><br>
                RSI is {rsi_now:.1f} — potential bounce setup
            </div>
            """,
                unsafe_allow_html=True,
            )

        sma_20, sma_50 = float(df["sma_20"].iloc[-1]), float(df["sma_50"].iloc[-1])
        if np.isfinite(sma_20) and np.isfinite(sma_50) and sma_20 > sma_50:
            st.markdown(
                """
            <div class="alert-box alert-success">
                <strong>📈 Bullish Signal</strong><br>
                SMA 20 &gt; SMA 50 (Golden Cross)
            </div>
            """,
                unsafe_allow_html=True,
            )

    with colB:
        current_price = float(df["price"].iloc[-1])
        pred_return = (preds["predicted_price"].iloc[-1] / current_price - 1) * 100
        if pred_return > 5:
            st.markdown(
                f"""
            <div class="alert-box alert-success">
                <strong>🚀 AI Bullish</strong><br>
                Model predicts +{pred_return:.1f}% in 30 days
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif pred_return < -5:
            st.markdown(
                f"""
            <div class="alert-box alert-danger">
                <strong>📉 AI Bearish</strong><br>
                Model predicts {pred_return:.1f}% in 30 days
            </div>
            """,
                unsafe_allow_html=True,
            )

        ann_vol = df["price"].pct_change().std() * np.sqrt(365) * 100
        if ann_vol > 80:
            st.markdown(
                f"""
            <div class="alert-box alert-warning">
                <strong>⚡ High Volatility</strong><br>
                Annual volatility: {ann_vol:.0f}%
            </div>
            """,
                unsafe_allow_html=True,
            )

    # ---- Performance ----
    st.subheader("📊 Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    returns = df["price"].pct_change().dropna()

    with c1:
        tot_ret = (df["price"].iloc[-1] / df["price"].iloc[0] - 1) * 100 if len(df) > 1 else 0.0
        st.metric("Total Return", f"{tot_ret:.1f}%")

    with c2:
        vol = returns.std() * np.sqrt(365) * 100 if len(returns) else 0.0
        st.metric("Volatility (Annual)", f"{vol:.1f}%")

    with c3:
        max_dd = (df["price"] / df["price"].cummax() - 1).min() * 100
        st.metric("Max Drawdown", f"{max_dd:.1f}%")

    with c4:
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0.0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")


if __name__ == "__main__":
    main()

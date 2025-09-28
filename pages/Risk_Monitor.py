import streamlit as st
import pandas as pd
from utils.data_fetcher import binance_klines, news_from_rss
from utils.sentiment import score_texts

st.set_page_config(page_title="Risk Monitor", page_icon="⚠️", layout="wide")
st.title("⚠️ Risk Monitor")

symbol = st.selectbox("Symbol", ["BTCUSDT","ETHUSDT","SOLUSDT"], index=0)
interval = st.selectbox("Interval", ["1h","4h","1d"], index=1)
df = binance_klines(symbol, interval=interval, limit=500)

price = df['close'].iloc[-1]
ret_24 = df['close'].pct_change(24).iloc[-1] if len(df) > 24 else 0.0

alerts = []
if abs(ret_24) > 0.05:
    alerts.append(f"Price 24h move >5%: {ret_24:.2%}")
if (df['high'].iloc[-1] - df['low'].iloc[-1]) / price > 0.03:
    alerts.append("Large intraperiod range (>3%)")
if df['close'].iloc[-1] < df['close'].rolling(50).mean().iloc[-1]:
    alerts.append("Close below 50-period MA")

news = news_from_rss().head(20)
if not news.empty:
    news['sentiment'] = score_texts(news['title'])
    if news['sentiment'].mean() < -0.1:
        alerts.append("News sentiment negative (mean < -0.1)")

st.subheader("Alerts")
if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No active alerts by rules.")

st.subheader("Latest prices")
st.write(df.tail(10))

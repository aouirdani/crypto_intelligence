import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import coingecko_simple_prices, binance_klines
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

st.set_page_config(page_title="Market Overview", page_icon="📊", layout="wide")
st.title("📊 Market Overview")

symbols = st.multiselect("Assets (Binance symbols)", ["BTCUSDT","ETHUSDT","SOLUSDT"], default=["BTCUSDT"])
interval = st.selectbox("Interval", ["15m","1h","4h","1d"], index=2)

prices = coingecko_simple_prices(ids=("bitcoin","ethereum","solana"))
st.dataframe(prices, use_container_width=True)

tabs = st.tabs(symbols)
for tab, sym in zip(tabs, symbols):
    with tab:
        df = binance_klines(sym, interval=interval, limit=600)
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(df['close'], window=200).sma_indicator()
        rsi = RSIIndicator(df['close'], window=14).rsi()
        macd = MACD(df['close'])

        fig = go.Figure()
        fig.add_candlestick(x=df['open_time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="OHLC")
        fig.add_trace(go.Scatter(x=df['open_time'], y=df['sma_50'], mode='lines', name='SMA 50'))
        fig.add_trace(go.Scatter(x=df['open_time'], y=df['sma_200'], mode='lines', name='SMA 200'))
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RSI")
            st.line_chart(rsi.rename("RSI"))
        with col2:
            st.subheader("MACD")
            macd_df = pd.DataFrame({"macd": macd.macd(), "signal": macd.macd_signal()})
            st.line_chart(macd_df)

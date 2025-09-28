import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import binance_klines
from utils.ml_models import prepare_features, random_forest_volatility, market_regime_kmeans, anomaly_detection, lstm_predict

st.set_page_config(page_title="AI Predictions", page_icon="🤖", layout="wide")
st.title("🤖 AI Predictions")

symbol = st.selectbox("Symbol", ["BTCUSDT","ETHUSDT","SOLUSDT"], index=0)
interval = st.selectbox("Interval", ["1h","4h","1d"], index=1)
horizon = st.slider("LSTM forecast steps", 6, 72, 24)

df = binance_klines(symbol, interval=interval, limit=800)
feat = prepare_features(df)

rf, rf_preds = random_forest_volatility(feat)
km, regimes = market_regime_kmeans(feat)
anoms = anomaly_detection(feat)

lstm_out = lstm_predict(df[['open_time','close']].rename(columns={'open_time':'time'}), steps_ahead=horizon)
note = lstm_out.get("note","")
st.caption(f"LSTM backend: {note}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=feat['time'], y=feat['close'], name='Close', mode='lines'))
fig.add_trace(go.Scatter(x=feat['time'], y=regimes*0 + feat['close'].mean(), mode='markers',
                         marker=dict(size=6, color=regimes, showscale=True), name='Regime label'))
anom_points = feat.loc[anoms==1]
fig.add_trace(go.Scatter(x=anom_points['time'], y=anom_points['close'], mode='markers', name='Anomalies', marker_symbol='x', marker_size=10))
fig.update_layout(height=500, margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(fig, use_container_width=True)

last_time = feat['time'].iloc[-1]
future_index = pd.date_range(last_time, periods=horizon+1, freq='H' if interval=='1h' else ('4H' if interval=='4h' else '1D'))[1:]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=feat['time'], y=feat['close'], name='Close'))
fig2.add_trace(go.Scatter(x=future_index, y=lstm_out['yhat'].values, name='Forecast'))
fig2.add_trace(go.Scatter(x=future_index, y=lstm_out['lower'].values, name='Lower CI', line=dict(dash='dash')))
fig2.add_trace(go.Scatter(x=future_index, y=lstm_out['upper'].values, name='Upper CI', line=dict(dash='dash')))
fig2.update_layout(height=450, margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Random Forest Volatility (in-sample)")
st.line_chart(rf_preds.rename("pred_vol"))

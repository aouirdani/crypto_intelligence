import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def prepare_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    feat = df_ohlcv.copy().rename(columns={"open_time":"time"})
    feat["ret"] = feat["close"].pct_change()
    feat["vol"] = feat["volume"].rolling(12).mean()
    feat["hl_spread"] = (feat["high"] - feat["low"]) / feat["close"]
    feat["ma_12"] = feat["close"].rolling(12).mean()
    feat["ma_48"] = feat["close"].rolling(48).mean()
    feat["mom_12"] = feat["close"].pct_change(12)
    feat["volatility_12"] = feat["ret"].rolling(12).std() * np.sqrt(12)
    feat = feat.dropna().reset_index(drop=True)
    return feat

def random_forest_volatility(df_feat: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.Series]:
    y = df_feat["volatility_12"].shift(-1).dropna()
    X = df_feat.loc[y.index, ["ret","vol","hl_spread","ma_12","ma_48","mom_12","volatility_12"]]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    preds = pd.Series(model.predict(X), index=y.index, name="rf_vol_pred")
    return model, preds

def market_regime_kmeans(df_feat: pd.DataFrame, n_clusters=3) -> Tuple[KMeans, pd.Series]:
    X = df_feat[["ret","mom_12","volatility_12","hl_spread"]].fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    return km, pd.Series(labels, index=df_feat.index, name="regime")

def anomaly_detection(df_feat: pd.DataFrame) -> pd.Series:
    X = df_feat[["ret","hl_spread","volatility_12"]].fillna(0.0)
    model = IsolationForest(contamination=0.02, random_state=42)
    scores = -model.fit_predict(X)  # 2 anomaly, 1 normal -> invert to 1/0
    return pd.Series((scores==2).astype(int), index=df_feat.index, name="anomaly_flag")

def lstm_predict(df_close: pd.DataFrame, steps_ahead=12, epochs=5, batch_size=32):
    """TensorFlow LSTM if available; naive fallback otherwise."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception:
        series = df_close["close"].astype(float).values
        last = series[-1]
        std = pd.Series(series).pct_change().rolling(24).std().iloc[-1] or 0.02
        yhat = np.array([last]*(steps_ahead))
        lower = yhat * (1-2*std)
        upper = yhat * (1+2*std)
        idx = pd.RangeIndex(len(df_close), len(df_close)+steps_ahead)
        return {"yhat": pd.Series(yhat, index=idx), "lower": pd.Series(lower, index=idx), "upper": pd.Series(upper, index=idx), "note":"Naive fallback (TensorFlow not available)"}

    values = df_close["close"].astype(float).values.reshape(-1,1)
    window = 24
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)

    model = keras.Sequential([
        layers.Input(shape=(window,1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    last_window = values[-window:].copy()
    preds = []
    cur = last_window
    for _ in range(steps_ahead):
        p = model.predict(cur.reshape(1,window,1), verbose=0)[0,0]
        preds.append(p)
        cur = np.vstack([cur[1:], [[p]]])
    preds = np.array(preds)

    in_pred = model.predict(X, verbose=0).flatten()
    resid = (y.flatten() - in_pred)
    sigma = np.std(resid) if len(resid)>1 else 0.02*np.mean(values[-window:])
    lower = preds - 1.96*sigma
    upper = preds + 1.96*sigma
    idx = pd.RangeIndex(len(df_close), len(df_close)+steps_ahead)
    return {"yhat": pd.Series(preds, index=idx), "lower": pd.Series(lower, index=idx), "upper": pd.Series(upper, index=idx), "note":"TensorFlow LSTM"}

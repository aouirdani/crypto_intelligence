# utils/data_fetcher.py
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com"

def _ts_to_dt(ts): return datetime.fromtimestamp(ts, tz=timezone.utc)
def _now_utc(): return datetime.now(timezone.utc)

# ---------- CoinGecko snapshot ----------
def coingecko_simple_prices(ids=("bitcoin","ethereum","solana"), vs_currencies=("usd","eur")) -> pd.DataFrame:
    ids_param = ",".join(ids)
    vs_param = ",".join(vs_currencies)
    url = f"{COINGECKO}/simple/price"
    r = requests.get(
        url,
        params={
            "ids": ids_param,
            "vs_currencies": vs_param,
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        },
        timeout=10,
        headers={"User-Agent":"crypto-intel/1.0"},
    )
    r.raise_for_status()
    data = r.json()
    rows = []
    for cid, vals in data.items():
        row = {"id": cid, "last_updated": _ts_to_dt(vals.get("last_updated_at", int(_now_utc().timestamp())))}
        for vs in vs_currencies:
            row[f"price_{vs}"] = vals.get(vs)
            row[f"change24_{vs}"] = vals.get(f"{vs}_24h_change")
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- OHLCV with robust fallbacks ----------
_YF_MAP = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD",
    "XRPUSDT": "XRP-USD",
}

def _yf_interval_for(interval: str) -> str:
    if interval in ("1h", "4h"): return "1h"
    if interval == "15m": return "15m"
    if interval == "1d": return "1d"
    return "1h"

def _placeholder_klines(symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    base_map = {"BTCUSDT": 45000, "ETHUSDT": 2500, "SOLUSDT": 100, "BNBUSDT": 300, "XRPUSDT": 0.6}
    base = float(base_map.get(symbol, 100.0))
    freq = "4H" if interval == "4h" else ("15min" if interval == "15m" else ("H" if interval == "1h" else "D"))
    n = int(limit)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq=freq)
    rng = np.random.default_rng(42)
    vol_scale = 0.002 if freq in ("H","4H","15min") else 0.01
    rets = rng.normal(0, vol_scale, n)
    close = base * (1 + pd.Series(rets, index=idx)).add(1).cumprod().values
    high = close * (1 + rng.uniform(0.0, 0.01, n))
    low  = close * (1 - rng.uniform(0.0, 0.01, n))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(1e3, 1e5, n)
    df = pd.DataFrame({"open_time": idx, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return df[["open_time","open","high","low","close","volume"]]

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    out = pd.concat([o.rename("open"), h.rename("high"), l.rename("low"), c.rename("close"), v.rename("volume")], axis=1)
    return out.dropna(how="any")

def _yf_klines(symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    ticker = _YF_MAP.get(symbol, "BTC-USD")
    yf_int = _yf_interval_for(interval)
    period = "7d" if yf_int=="15m" else ("60d" if yf_int=="1h" else "2y")

    df = yf.download(ticker, period=period, interval=yf_int, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return _placeholder_klines(symbol, interval, limit)

    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df.index = pd.to_datetime(df.index, utc=True)
    if {"open","high","low","close","volume"} - set(df.columns):
        return _placeholder_klines(symbol, interval, limit)

    if interval == "4h":
        df = _resample_ohlcv(df, "4H")
        if df is None or df.empty:
            return _placeholder_klines(symbol, interval, limit)

    df = df[["open","high","low","close","volume"]].copy()
    df.index.name = "open_time"
    df = df.reset_index()
    return df.tail(limit)[["open_time","open","high","low","close","volume"]]

def binance_klines(symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    """Try Binance; on error/empty → yfinance; if still empty → synthetic series."""
    url = f"{BINANCE}/api/v3/klines"
    try:
        r = requests.get(
            url,
            params={"symbol":symbol, "interval":interval, "limit":limit},
            timeout=10,
            headers={"User-Agent":"crypto-intel/1.0"},
        )
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} from Binance", response=r)
        arr = r.json()
        if not isinstance(arr, list) or len(arr) == 0:
            raise ValueError("Binance returned empty/invalid klines.")
        cols = [
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades","taker_base","taker_quote","ignore"
        ]
        df = pd.DataFrame(arr, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["open_time","open","high","low","close","volume"]].dropna()
        if df.empty:
            raise ValueError("Binance klines parsed to empty frame.")
        return df.tail(limit)
    except Exception:
        try:
            return _yf_klines(symbol=symbol, interval=interval, limit=limit)
        except Exception:
            return _placeholder_klines(symbol=symbol, interval=interval, limit=limit)

# ---------- News / Reddit ----------
def news_from_rss(feeds=None, max_items=50) -> pd.DataFrame:
    import feedparser
    if feeds is None:
        feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
        ]
    rows = []
    for url in feeds:
        fp = feedparser.parse(url)
        for e in fp.entries[:max_items]:
            rows.append({
                "source": fp.feed.get("title","rss"),
                "title": e.get("title",""),
                "link": e.get("link",""),
                "published": e.get("published",""),
                "summary": e.get("summary",""),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df

def reddit_search(query="bitcoin", limit=25) -> pd.DataFrame:
    url = "https://www.reddit.com/search.json"
    try:
        r = requests.get(
            url,
            params={"q":query,"limit":limit,"sort":"new","t":"day"},
            headers={"User-Agent":"streamlit-app"},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        rows = []
        for c in data.get("data",{}).get("children",[]):
            d = c.get("data",{})
            rows.append({
                "subreddit": d.get("subreddit"),
                "title": d.get("title"),
                "score": d.get("score"),
                "num_comments": d.get("num_comments"),
                "created_utc": pd.to_datetime(d.get("created_utc"), unit="s", utc=True),
                "url": f"https://www.reddit.com{d.get('permalink','')}",
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["subreddit","title","score","num_comments","created_utc","url"])

import requests
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com"

def _ts_to_dt(ts): return datetime.fromtimestamp(ts, tz=timezone.utc)
def _now_utc(): return datetime.now(timezone.utc)

def coingecko_simple_prices(ids=("bitcoin","ethereum","solana"), vs_currencies=("usd","eur")) -> pd.DataFrame:
    ids_param = ",".join(ids)
    vs_param = ",".join(vs_currencies)
    url = f"{COINGECKO}/simple/price"
    r = requests.get(url, params={
        "ids": ids_param,
        "vs_currencies": vs_param,
        "include_24hr_change": "true",
        "include_last_updated_at": "true",
    }, timeout=10, headers={"User-Agent":"crypto-intel/1.0"})
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

_YF_MAP = {"BTCUSDT":"BTC-USD","ETHUSDT":"ETH-USD","SOLUSDT":"SOL-USD","BNBUSDT":"BNB-USD","XRPUSDT":"XRP-USD"}
def _yf_interval_for(i:str)->str:
    if i=="1h": return "1h"
    if i=="4h": return "1h"   # resample below
    if i=="1d": return "1d"
    if i=="15m": return "15m"
    return "1h"

def _yf_klines(symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    ticker = _YF_MAP.get(symbol, "BTC-USD")
    yf_int = _yf_interval_for(interval)
    period = "7d" if yf_int=="15m" else ("60d" if yf_int=="1h" else "2y")
    df = yf.download(ticker, period=period, interval=yf_int, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume"])
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df.index = pd.to_datetime(df.index, utc=True)
    if interval=="4h" and yf_int=="1h":
        o = df["open"].resample("4H").first()
        h = df["high"].resample("4H").max()
        l = df["low"].resample("4H").min()
        c = df["close"].resample("4H").last()
        v = df["volume"].resample("4H").sum()
        df = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    df = df.reset_index().rename(columns={"index":"open_time"})
    return df[["open_time","open","high","low","close","volume"]].tail(limit)

def binance_klines(symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    url = f"{BINANCE}/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit},
                         timeout=10, headers={"User-Agent":"crypto-intel/1.0"})
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} from Binance", response=r)
        arr = r.json()
        cols = ["open_time","open","high","low","close","volume","close_time","quote_asset_volume","trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(arr, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df[["open_time","open","high","low","close","volume"]]
    except Exception:
        return _yf_klines(symbol=symbol, interval=interval, limit=limit)

def news_from_rss(feeds=None, max_items=50) -> pd.DataFrame:
    import feedparser
    if feeds is None:
        feeds = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss"]
    rows = []
    for url in feeds:
        fp = feedparser.parse(url)
        for e in fp.entries[:max_items]:
            rows.append({"source": fp.feed.get("title","rss"), "title": e.get("title",""),
                         "link": e.get("link",""), "published": e.get("published",""),
                         "summary": e.get("summary","")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df

def reddit_search(query="bitcoin", limit=25) -> pd.DataFrame:
    url = "https://www.reddit.com/search.json"
    try:
        r = requests.get(url, params={"q":query,"limit":limit,"sort":"new","t":"day"},
                         headers={"User-Agent":"streamlit-app"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        for c in data.get("data",{}).get("children",[]):
            d = c.get("data",{})
            rows.append({"subreddit": d.get("subreddit"), "title": d.get("title"),
                         "score": d.get("score"), "num_comments": d.get("num_comments"),
                         "created_utc": pd.to_datetime(d.get("created_utc"), unit="s", utc=True),
                         "url": f"https://www.reddit.com{d.get('permalink','')}"})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["subreddit","title","score","num_comments","created_utc","url"])

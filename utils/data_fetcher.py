import requests
import pandas as pd
from datetime import datetime, timezone

COINGECKO = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com"

def _ts_to_dt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def _now_utc():
    return datetime.now(timezone.utc)

def coingecko_simple_prices(ids=("bitcoin","ethereum","solana"), vs_currencies=("usd","eur")) -> pd.DataFrame:
    ids_param = ",".join(ids)
    vs_param = ",".join(vs_currencies)
    url = f"{COINGECKO}/simple/price?ids={ids_param}&vs_currencies={vs_param}&include_24hr_change=true&include_last_updated_at=true"
    r = requests.get(url, timeout=10)
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

def binance_klines(symbol="BTCUSDT", interval="1h", limit=500) -> pd.DataFrame:
    url = f"{BINANCE}/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    arr = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","quote_asset_volume","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(arr, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open_time","open","high","low","close","volume"]]

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
        for entry in fp.entries[:max_items]:
            rows.append({
                "source": fp.feed.get("title", "rss"),
                "title": entry.get("title",""),
                "link": entry.get("link",""),
                "published": entry.get("published",""),
                "summary": entry.get("summary",""),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
        except Exception:
            pass
    return df

def reddit_search(query="bitcoin", limit=25) -> pd.DataFrame:
    url = "https://www.reddit.com/search.json"
    try:
        r = requests.get(url, params={"q":query, "limit":limit, "sort":"new", "t":"day"}, headers={"User-Agent":"streamlit-app"}, timeout=10)
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
                "url": f"https://www.reddit.com{d.get('permalink','')}"
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["subreddit","title","score","num_comments","created_utc","url"])

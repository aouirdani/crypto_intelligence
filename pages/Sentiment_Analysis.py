import streamlit as st
import pandas as pd
from utils.data_fetcher import news_from_rss, reddit_search
from utils.sentiment import score_texts, label_score

st.set_page_config(page_title="Sentiment Analysis", page_icon="📰", layout="wide")
st.title("📰 Sentiment Analysis")

query = st.text_input("Keyword for Reddit search", "bitcoin")
news = news_from_rss()
if not news.empty:
    news["sentiment"] = score_texts(news["title"])
    news["label"] = news["sentiment"].apply(label_score)
st.subheader("News sentiment")
st.dataframe(news.sort_values("published", ascending=False), use_container_width=True)

reddit = reddit_search(query=query, limit=50)
if not reddit.empty:
    reddit["sentiment"] = score_texts(reddit["title"])
    reddit["label"] = reddit["sentiment"].apply(label_score)
st.subheader("Reddit sentiment")
st.dataframe(reddit.sort_values("created_utc", ascending=False), use_container_width=True)

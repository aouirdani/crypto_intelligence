import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def _ensure_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

def score_texts(texts: pd.Series) -> pd.Series:
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    return texts.fillna("").apply(lambda t: sia.polarity_scores(t)['compound'])

def label_score(s: float) -> str:
    if s >= 0.2: return "positive"
    if s <= -0.2: return "negative"
    return "neutral"

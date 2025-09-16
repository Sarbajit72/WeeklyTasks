import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, ngrams
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def preprocess_text(df, column):
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    stop_words = set(stopwords.words('english'))
    df[column] = df[column].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
    return df

def compute_freq_dist(text):
    tokens = word_tokenize(text)
    return FreqDist(tokens)

def compute_ngrams(text, n=2):
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

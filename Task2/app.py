import streamlit as st
import pandas as pd
from nlp_pipeline import preprocess_text, compute_freq_dist, compute_ngrams, sentiment_analysis

st.title("NLTK Text Analytics Web App")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data", df.head())

    column = st.selectbox("Select text column", df.columns)
    df = preprocess_text(df, column)
    st.write("Preprocessed Text", df.head())

    all_text = ' '.join(df[column].tolist())
    freq = compute_freq_dist(all_text)
    st.write("Top 10 Words:", freq.most_common(10))

    bigrams = compute_ngrams(all_text, 2)
    st.write("Top 10 Bigrams:", pd.Series(bigrams).value_counts().head(10))

    df['sentiment'] = df[column].apply(sentiment_analysis)
    st.write("Sentiment Analysis", df[['sentiment']].head())

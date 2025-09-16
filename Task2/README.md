# Task2: NLTK Text Analytics Web App

## Description
This task builds a Streamlit web app to perform text analytics:
- Preprocessing (lowercase, remove punctuation, stopwords)
- Frequency distribution and n-grams
- Sentiment analysis

## Files
- `app.py` : Main Streamlit app
- `nlp_pipeline.py` : Text preprocessing and NLP functions
- `example_data.csv` : Sample dataset for testing

## How to Run
1. Install requirements:
   pip install streamlit pandas nltk
2. Run the app:
   streamlit run app.py
3. Upload a CSV file with a text column to analyze.

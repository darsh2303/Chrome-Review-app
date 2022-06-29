# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re, string, unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emoji
import streamlit as st
import pandas as pd
import numpy as np

def strip_html(txt):
    soup = BeautifulSoup(txt, "html.parser")
    return soup.get_text()

def repalce_contractions(txt):
    return contractions.fix(txt)

def remove_numbers(txt):
    txt = re.sub(r'\d+','',txt)
    return txt

def e_demojize(txt):
    txt = emoji.demojize(txt)
    return txt

#Convert all characters to lowercase from list of tokenized words
def to_lowercase(words):
    new_words = []
    for word in words:
        new = word.lower()
        new_words.append(new)
    return new_words

#Remove Punctuations from the list of tokenized words
def remove_punctuation(words):
    new_words = []
    for word in words:
        new = re.sub(r'[^\w\s]','',word)
        if new != '' :
            new_words.append(new)
    return new_words

# Remove stopwords from the list of tokenized words
def remove_stopwords(words):
    new_words = []
    stop_words = set(stopwords.words('english'))
    stop_words.difference_update(['no','not'])
    for word in words:
        if word not in stop_words :
            new_words.append(word)
    return new_words

# Lemmatize the words, i.e. bring them to their root word
def lemmatize_list(words):
    lemmatizer = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(lemmatizer.lemmatize(word, 'v'))
    return new_words

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_list(words)
    return ' '.join(words)

def sentiment_analysis(rev):
    senti_IA = SentimentIntensityAnalyzer()
    match_list = []
    score = senti_IA.polarity_scores(rev)
    if score['pos'] > 0.7:
        return 1
    else:
        return 0

#creating funtion
def cleaning_text(df):
    #drop null reviews if any
    if df.Text.isnull().any() == True:
        df.drop(df[df.Text.isnull()].index, inplace=True)
        df = df[['Text','Star']]
    else :
        df = df[['Text','Star']]
        

    df['Text'] = df['Text'].apply(lambda x: strip_html(x))
    df['Text'] = df['Text'].apply(lambda x: repalce_contractions(x))
    df['Text'] = df['Text'].apply(lambda x: remove_numbers(x))
    df['Text'] = df['Text'].apply(lambda x: e_demojize(x))
    df['Text'] = df.apply(lambda row: nltk.word_tokenize(row['Text']), axis=1)
    df['Text']= df.apply(lambda row: normalize(row['Text']), axis=1)
    
    return df[df.Star.isin([1,2])]

def main():
    st.title('Chrome review prediction')
    
    data_file = st.file_uploader("Upload CSV",type=["csv"])
    
    if data_file is not None:
        st.write('below are the wrong reviews')
        df = pd.read_csv(data_file)
        df = cleaning_text(df)
        df['wrong_rating'] = df['Text'].map(sentiment_analysis)
        st.dataframe(df[df['wrong_rating'] == 1])

if __name__ == '__main__':
    main()
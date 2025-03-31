# src/preprocess.py

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

MAX_SEQUENCE_LENGTH = 100  # Required in predict.py
MAX_VOCAB = 10000

def preprocess_text(text):
    """Lowercase, clean, and remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def load_data(path):
    df = pd.read_csv(path, encoding="latin1", header=None, names=["Sentiment", "Title"])
    df.dropna(inplace=True)
    return df

def prepare_data(df, max_len=MAX_SEQUENCE_LENGTH):
    # Preprocess the headlines
    df['processed_title'] = df['Title'].apply(preprocess_text)

    # Map sentiment labels to numeric values
    label_mapping = {"DOWN": 0, "UP": 1, "PRESERVE": 2, "negative": 0, "positive": 1, "neutral": 2}
    df["label"] = df["Sentiment"].map(label_mapping)
    df.dropna(subset=["label"], inplace=True)

    texts = df['processed_title'].astype(str).values
    labels = df['label'].astype(int).values

    # Tokenization
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding='post')

    word_index = tokenizer.word_index
    return X, labels, tokenizer, word_index

def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

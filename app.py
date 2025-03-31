# app.py (Streamlit Frontend)
import streamlit as st
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.model import HybridAttentionHAN, AttentionLayer
from src.preprocess import preprocess_text, MAX_SEQUENCE_LENGTH

# ✅ Paths
MODEL_WEIGHTS_PATH = os.path.join('saved_model', 'han_stock_model.weights.h5')
TOKENIZER_PATH = os.path.join('saved_model', 'tokenizer.pickle')

# ✅ Load Tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# ✅ Load Model Architecture
VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))  # Placeholder for predict only
model = HybridAttentionHAN(MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, embedding_matrix, num_classes=3)

# ✅ Build model once to load weights
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
model.load_weights(MODEL_WEIGHTS_PATH)


# ✅ Streamlit UI
st.title("📈 Financial News Sentiment Predictor")
user_input = st.text_area("Enter Financial News Headline:")

if st.button("Predict") and user_input:
    processed_text = preprocess_text(user_input)
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(padded)
    label = np.argmax(prediction)

    label_map = {0: 'Negative 📉', 1: 'Neutral 😐', 2: 'Positive 📈'}
    st.write(f"### Sentiment: {label_map[label]}")
    st.write(f"Probabilities: {prediction}")

st.markdown("---")


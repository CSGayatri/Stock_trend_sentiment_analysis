# src/predict.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import HybridAttentionHAN, AttentionLayer
from preprocess import preprocess_text, MAX_SEQUENCE_LENGTH

# ✅ Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(BASE_DIR, 'saved_model', 'tokenizer.pickle')
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'saved_model', 'han_stock_model.weights.h5')

print(f"Current Working Directory: {os.getcwd()}")
print(f"Tokenizer Path: {TOKENIZER_PATH}")
print(f"Model Weights Path: {MODEL_WEIGHTS_PATH}")

# ✅ Load Tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)
print("✅ Tokenizer loaded successfully!")

# ✅ Rebuild Model Architecture
VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 100
NUM_CLASSES = 3

# Dummy embedding matrix (you don’t need GloVe now since weights are loaded)
dummy_embedding_matrix = np.random.rand(VOCAB_SIZE, EMBEDDING_DIM)

model = HybridAttentionHAN(MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, dummy_embedding_matrix, num_classes=NUM_CLASSES)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Build model before loading weights
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))

# ✅ Load model weights
model.load_weights(MODEL_WEIGHTS_PATH)
print("✅ Model weights loaded successfully!")

# ✅ Predict function
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(padded)
    sentiment_label = np.argmax(prediction)
    return sentiment_label, prediction

# ✅ Example prediction
headline = "Stock market hits record high amid positive earnings reports"
label, prediction_prob = predict_sentiment(headline)
print(f"Predicted Label: {label}, Probabilities: {prediction_prob}")

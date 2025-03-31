import os
import numpy as np
import tensorflow as tf
from preprocess import load_data, prepare_data, load_glove_embeddings
from model import HybridAttentionHAN
from spl import SelfPacedLearning
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import pickle

# Paths
DATA_PATH = os.path.join('data', 'financial_news.csv')
GLOVE_PATH = os.path.join('data', 'glove.6B.100d.txt')
TOKENIZER_PATH = os.path.join('saved_model', 'tokenizer.pickle')
MODEL_WEIGHTS_PATH = os.path.join('saved_model', 'han_stock_model.weights.h5')

# Hyperparameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 5

# Load and preprocess data
print("📥 Loading data...")
df = load_data(DATA_PATH)
print("✅ Data loaded!")

print("🧠 Preparing data...")
texts, labels, tokenizer, word_index = prepare_data(df, MAX_SEQUENCE_LENGTH)
VOCAB_SIZE = len(word_index) + 1

print("📦 Loading GloVe embeddings...")
embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_index, EMBEDDING_DIM)
print("✅ Embeddings loaded!")

# Build Model
print("🛠️ Building HAN model...")
model = HybridAttentionHAN(MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, embedding_matrix, num_classes=3)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("✅ Model compiled!")

# Self-Paced Learning setup
spl = SelfPacedLearning(lmbda=0.5, alpha=0.1)

# Training loop with SPL
print("🚀 Training starts...")
for epoch in range(EPOCHS):
    print(f"\n🔥 Epoch {epoch + 1}/{EPOCHS}")
    history = model.fit(texts, labels, batch_size=BATCH_SIZE, epochs=1, validation_split=0.2)

    # Compute per-sample loss for SPL
    loss_fn = SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    sample_losses = loss_fn(labels, model.predict(texts, verbose=0))

    # Compute SPL sample weights
    sample_weights = spl.compute_sample_weights(sample_losses.numpy())
    print(f"Sample weights for epoch {epoch + 1}: {sample_weights}")

# ✅ Save model weights and tokenizer
print("💾 Saving model weights...")
os.makedirs('saved_model', exist_ok=True)
model.save_weights(MODEL_WEIGHTS_PATH)
print(f"✅ Model weights saved at {MODEL_WEIGHTS_PATH}")

with open(TOKENIZER_PATH, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ Tokenizer saved at {TOKENIZER_PATH}")

# src/model.py
import tensorflow as tf
import numpy as np

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        scores = self.score_dense(inputs)
        attention_weights = tf.nn.softmax(scores, axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

class HybridAttentionHAN(tf.keras.Model):
    def __init__(self, max_seq_len, vocab_size, embedding_dim, embedding_matrix, num_classes=3, **kwargs):
        super(HybridAttentionHAN, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False
        )
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
        self.attention = AttentionLayer()
        self.output_dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        x = self.attention(x)
        return self.output_dense(x)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # Dummy parameters to satisfy loading (if required)
        return cls(max_seq_len=100, vocab_size=10000, embedding_dim=100, embedding_matrix=np.zeros((10000, 100)), num_classes=3)

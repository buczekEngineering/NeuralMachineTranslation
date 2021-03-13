import tensorflow as tf
from AttentionLayer import AttentionLuong

class DecoderAttention(tf.keras.Model):
    def __init__(self, decoder_vocab_size, num_timesteps, embedding_dim, decoder_dim, **kwargs):
        super(DecoderAttention, self).__init__()
        self.decoder_dim = decoder_dim
        self.embedding = tf.keras.layers.Embedding(decoder_vocab_size, embedding_dim, input_length=num_timesteps)
        self.attention = AttentionLuong(embedding_dim)
        self.lstm = tf.keras.layers.LSTM(decoder_dim, return_sequences=True, return_state=True, dropout=0.5)
        self.Ws = tf.keras.layers.Dense(decoder_dim, activation="tanh")
        self.Wc = tf.keras.layers.Dense(decoder_vocab_size)

    def call(self, input, state, encoder_out):
        x = self.embedding(input)
        context_vector, attention_weights = self.attention(x, encoder_out)
        x = tf.expand_dims(tf.concat([x, tf.squeeze(context_vector, axis=1)], axis=1), axis=1)
        x, dec_h, dec_c = self.lstm(x, state)
        x = self.Ws(x)
        x = self.Wc(x)
        return x, dec_h, dec_c, attention_weights
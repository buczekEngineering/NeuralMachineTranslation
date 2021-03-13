import tensorflow as tf


class DecoderLSTM(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dim, decoder_dim):
        super(DecoderLSTM, self).__init__()
        self.decoder_dim = decoder_dim
        self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=num_timesteps)
        self.lstm = tf.keras.layers.LSTM(decoder_dim, return_sequences=True,
                                         return_state=True, dropout=0.5)

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, input, state):
        x = self.embeddings(input)
        x, h, c = self.lstm(x, initial_state=state)
        x = self.dense(x)
        return x, h, c

import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, encoder_vocab_size, num_timesteps, embedding_dim, encoder_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.embedding = tf.keras.layers.Embedding(encoder_vocab_size, embedding_dim, input_length=num_timesteps)
        self.gru = tf.keras.layers.GRU(self.encoder_dim, return_sequences=False, return_state=True, dropout=0.5)

    def call(self, x, state):
        x = self.embedding(x)
        x, state = self.gru(x, initial_state=state)
        return x, state

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.encoder_dim))


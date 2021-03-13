import tensorflow as tf


# num_timestep - is how long is the padded sequence. len(padded_sequence_output)
class Decoder(tf.keras.Model):
    def __init__(self, decoder_vocab_size, num_timestep, embedding_dim, decoder_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim
        self.embeddings = tf.keras.layers.Embedding(decoder_vocab_size, embedding_dim, input_length=num_timestep)
        self.gru = tf.keras.layers.GRU(decoder_dim, return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(decoder_vocab_size)

    def call(self, input, state):
        x = self.embeddings(input)
        x, state = self.gru(x, state)
        x = self.dense(x)
        return x, state


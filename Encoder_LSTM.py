import tensorflow as tf
class EncoderLSTM(tf.keras.Model):
    def __init__(self,vocab_size, num_timesteps, embedding_dim, encoder_dim):
        super(EncoderLSTM, self).__init__()
        self.encoder_dim = encoder_dim
        self.embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=num_timesteps)
        self.lstm = tf.keras.layers.LSTM(self.encoder_dim, return_sequences=True,
                                         return_state=True, dropout=0.5)

    def call(self, input, state):

        x = self.embeddings(input)
        output, h, c = self.lstm(x, initial_state=state)
        return x, h, c

    def init_state(self, batch_size):
        return [tf.zeros((batch_size, self.encoder_dim)), tf.zeros((batch_size, self.encoder_dim))]
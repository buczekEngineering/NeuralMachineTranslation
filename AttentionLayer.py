import tensorflow as tf

class AttentionLuong(tf.keras.layers.Layer):
    def __init__(self, LATENT_DIM):
        super(AttentionLuong, self).__init__()
        self.LATENT_DIM = LATENT_DIM
        self.W = tf.keras.layers.Dense(LATENT_DIM)

        # values - hidden state of encoder, query - hidden state of decoder at particular timestep
    def call(self, query, values):
        query_at_time = tf.expand_dims(query, axis=1)
        attention_score = tf.linalg.matmul(query_at_time, self.W(values), transpose_b=True)
        attention_weights = tf.nn.softmax(attention_score, axis=2)
        context_vector = tf.matmul(attention_weights, values)
        return context_vector, attention_weights

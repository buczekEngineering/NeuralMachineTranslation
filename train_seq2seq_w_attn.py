import os
import time

from EncoderLSTM import EncoderLSTM
from DecoderAttentionLSTM import DecoderAttention
from data_preprocessing import NMT_Data
import tensorflow as tf
from loss import loss_custom
from prediction import predict_sentence_lstm_attn
from evaluate import compute_bleu_LSTM

data = NMT_Data(data_path="deu.txt")
BATCH_SIZE = 16
LATENT_DIM = 256 #1024
EMBEDDING_DIM = 256
EPOCHS = 20
tf.random.set_seed(42)
optimizer = tf.keras.optimizers.Adam()

test_data, training_data, \
max_len_input, max_len_output, \
encoder_padded_input, decoder_padded_input, decoder_padded_output, \
input_seq, target_sequences, \
word2idx_input, word2idx_output, \
vocab_size_encoder, vocab_size_decoder = data.preprocess()

idx2word_input = {i: w for (w, i) in word2idx_input.items()}
idx2word_output = {i: w for (w, i) in word2idx_output.items()}

encoder = EncoderLSTM(vocab_size_encoder + 1, max_len_input, EMBEDDING_DIM, LATENT_DIM)
decoder = DecoderAttention(vocab_size_decoder + 1, max_len_output, EMBEDDING_DIM, LATENT_DIM)
encoder_state = encoder.init_state(BATCH_SIZE)

checkpoint_dir = "checkpoints_attn"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train_teacher_forcing_w_attn(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:
        loss = 0
        encoder_output, enc_h, enc_c = encoder(encoder_in, encoder_state)
        decoder_state = [enc_h, enc_c]
        # using attention we must go through each timestep in decoder_in
        for i in range(decoder_in.shape[1]):
            decoder_in_t = decoder_in[:, i]
            decoder_pred_t, dec_h, dec_c, alignment = decoder(decoder_in_t, decoder_state, encoder_output)
            decoder_state = [dec_h, dec_c]

            loss += loss_custom(decoder_out[:, i], decoder_pred_t)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return loss / decoder_out.shape[1]


for epoch in range(EPOCHS):
    start = time.time()
    for batch, data in enumerate(training_data):
        encoder_in, decoder_in, decoder_out = data

        loss = train_teacher_forcing_w_attn(encoder_in, decoder_in, decoder_out, encoder_state)
    elapsed_time = (time.time() - start) // 60
    print("*" * 60)
    print("Epoch: {}, Loss: {:.4f}, time: {} min.".format(epoch + 1, loss.numpy(), elapsed_time))
    if epoch % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

while True:
    prediction = predict_sentence_lstm_attn(input_seq, target_sequences,encoder_padded_input, encoder, decoder, word2idx_output, word2idx_input, max_len_output)
    check = input("Continue testing? [yes/no]")
    if check.lower().startswith("n"):
        break
    # bleu = compute_bleu_LSTM(test_data, encoder, decoder, BATCH_SIZE, idx2word_output)

encoder.save_weights("attn_encoder_lstm")
decoder.save_weights("attn_decoder_lstm")
checkpoint.save(file_prefix=checkpoint_prefix)

import os
import time

from Encoder_LSTM import EncoderLSTM
from Decoder_LSTM import DecoderLSTM
from data_preprocessing import NMT_Data
import tensorflow as tf
from loss import loss_custom
from prediction import predict_sentence_LSTM
from evaluate import compute_bleu_LSTM

data = NMT_Data(data_path="deu.txt")
BATCH_SIZE = 16
LATENT_DIM = 1024
#NUM_SAMPLES = 10000  # Number of samples to train on.
#MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 256
EPOCHS = 200
tf.random.set_seed(42)
optimizer = tf.keras.optimizers.Adam()

test_data, training_data, \
max_len_input, max_len_output, \
encoder_padded_input, decoder_padded_input, decoder_padded_output, \
input_seq, target_sequences, \
word2idx_input, word2idx_output, \
vocab_size_encoder, vocab_size_decoder = data.preprocess()

idx2word_input = {i:w for (w,i) in word2idx_input.items()}
idx2word_output = {i:w for (w,i) in word2idx_output.items()}

encoder = EncoderLSTM(vocab_size_encoder +1, max_len_input, EMBEDDING_DIM, LATENT_DIM)
decoder = DecoderLSTM(vocab_size_decoder+1, max_len_output, EMBEDDING_DIM, LATENT_DIM)
encoder_state = encoder.init_state(BATCH_SIZE)

checkpoint_dir = "checkpoints_LSTM"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer= optimizer, encoder=encoder, decoder=decoder)

@tf.function
def train_teacher_forcing_LSTM(encoder_in, decoder_in, decoder_out, encoder_state):
    with tf.GradientTape() as tape:


        encoder_output, enc_h, enc_c = encoder(encoder_in, encoder_state)
        decoder_state = [enc_h, enc_c]
        decoder_prediction, dec_h, dec_c = decoder(decoder_in, decoder_state)

        loss = loss_custom(decoder_out, decoder_prediction)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return loss


for epoch in range(EPOCHS):
    start = time.time()
    for batch, data in enumerate(training_data):
        encoder_in, decoder_in, decoder_out = data

        loss = train_teacher_forcing_LSTM(encoder_in, decoder_in, decoder_out, encoder_state)
    elapsed_time = (time.time() - start)
    print("*"*60)
    print("Epoch: {}, Loss: {:.4f}, time: {} sec.".format(epoch+1, loss.numpy(), elapsed_time))
    if epoch % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    predict_sentence_LSTM(encoder, decoder,
                        input_seq, target_sequences,
                        encoder_padded_input, decoder_padded_output,
                        word2idx_output, word2idx_input)

    bleu = compute_bleu_LSTM(test_data, encoder, decoder, BATCH_SIZE, idx2word_output)

encoder.save_weights("encoder_lstm")
decoder.save_weights("decoder_lstm")
checkpoint.save(file_prefix=checkpoint_prefix)
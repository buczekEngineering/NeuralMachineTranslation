import numpy as np
import tensorflow as tf


def predict_sentence_GRU(encoder, decoder,
                         input_seq, target_sequences,
                         encoder_padded_input, decoder_padded_output,
                         word2idx_output, word2idx_input):
    idx2word_output = {i: w for (w, i) in word2idx_output.items()}
    output_sentence = []
    seq_idx = np.random.choice(len(input_seq))
    print("The sentence to translate: ", input_seq[seq_idx])
    print("Expected translation: ", target_sequences[seq_idx])

    encoder_input = tf.expand_dims(encoder_padded_input[seq_idx], axis=0)

    encoder_state = encoder.init_state(1)
    encoder_output, encoder_state = encoder(encoder_input, encoder_state)
    decoder_state = encoder_state
    # input for decoder in prediction ist "<sos>" token
    decoder_in = tf.expand_dims(tf.constant([word2idx_output["<sos>"]]), axis=0)

    while True:
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state)
        # print(output_pred) # output_pred  is a 3D tensor: batch_size = 1, max_len_sentence = 1, word_num =4980
        idx = tf.argmax(decoder_pred,
                        axis=-1)
        word = idx2word_output[idx.numpy()[0][0]]
        output_sentence.append(word)
        if word == "<eos>":
            break
        decoder_in = idx

    print("Prediction: ", " ".join(output_sentence).capitalize())


def predict_sentence_LSTM(encoder, decoder,
                          input_seq, target_sequences,
                          encoder_padded_input, decoder_padded_output,
                          word2idx_output, word2idx_input):
    idx2word_output = {i: w for (w, i) in word2idx_output.items()}
    random_idx = np.random.choice(len(input_seq))
    encoder_in = tf.expand_dims(encoder_padded_input[random_idx], axis=0)
    print("Sentence to translate: ", input_seq[random_idx])
    print("Expected translation: ", target_sequences[random_idx])

    encoder_state = encoder.init_state(1)
    output_sentence = []
    encoder_out, enc_h, enc_c = encoder(encoder_in, encoder_state)
    decoder_state = [enc_h, enc_c]
    decoder_in = tf.expand_dims(tf.constant([word2idx_output["<sos>"]]), axis=0)
    eos = "<eos>"
    word = ""
    while word != eos:
        decoder_pred, dec_h, dec_c = decoder(decoder_in, decoder_state)
        decoder_state = [dec_h, dec_c]

        idx = tf.argmax(decoder_pred, axis=-1)
        word = idx2word_output[idx.numpy()[0][0]]
        output_sentence.append(word)

        decoder_in = idx
    print("Predicted translation:", " ".join(output_sentence))


def predict_sentence_lstm_attn(input_seq, target_sequences,
                               encoder_padded_input, encoder, decoder, word2idx_output, word2idx_input, max_len_output):
    final_sentence = []
    idx2word_input = {i: w for (w, i) in word2idx_input.items()}
    idx2word_output = {i: w for (w, i) in word2idx_output.items()}
    idx = np.random.choice(len(input_seq))
    print("Sentence to translate: ", input_seq[idx])
    print("Translation: ", target_sequences[idx])

    encoder_in = tf.expand_dims(encoder_padded_input[idx], axis=0)
    decoder_in = tf.expand_dims(tf.constant(word2idx_output["<sos>"]), axis=0)
    encoder_state = encoder.init_state(1)

    encoder_out, enc_h, enc_c = encoder(encoder_in, encoder_state)
    decoder_state = [enc_h, enc_c]

    i = 0
    while (i < max_len_output):
        i += 1

        decoder_pred, dec_h, dec_c, attention_weighst = decoder(decoder_in, decoder_state, encoder_out)

        dec_pred = tf.argmax(decoder_pred, axis=-1)
        word = idx2word_output[dec_pred.numpy()[0][0]]
        print(word)
        final_sentence.append(word)

        if word == "<eos>":
            break
        decoder_in = tf.squeeze(dec_pred, axis=1)

    final_sentence = " ".join(final_sentence)
    print("Translated sequence: ", final_sentence)
    return final_sentence
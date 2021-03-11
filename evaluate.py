import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html

def bleu_eval(test_data, encoder, decoder, batch_size, idx2word_output):
    bleu_scores = []

    for batch, data in enumerate(test_data):
        encoder_in, decoder_in, decoder_out = data

        encoder_state = encoder.init_state(batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, encoder_state = decoder(decoder_in, decoder_state)

        decoder_out = decoder_out.numpy()
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()

        # decoder_out.shape  = 10000, 10 (num_samples, seq_len)
        # for each sentence in decoder_out
        for i in range(decoder_out.shape[0]):
            decoder_output_list = decoder_out[i].tolist()
            correct_sentences = [idx2word_output[word] for word in decoder_output_list if word != 0][:-1]
            decoder_prediction_list = decoder_pred[i].tolist()
            predicted_sentences = [idx2word_output[word] for word in decoder_prediction_list if word != 0]
            predicted_sentences = [word for word in predicted_sentences if word != "<eos>"]

            smoothing = SmoothingFunction()
            bleu = sentence_bleu([correct_sentences], predicted_sentences, smoothing_function=smoothing.method1)
            bleu_scores.append(bleu)

    final_bleu = sum(bleu_scores) / len(bleu_scores)
    print("Bleu score: {:.4f}".format(final_bleu))
    return final_bleu


def compute_bleu_LSTM(test_data, encoder, decoder, batch_size, idx2word_output):
    bleu_scores = []
    smoothing = SmoothingFunction()

    for batch, data in enumerate(test_data):

        encoder_in, decoder_in, decoder_out = data
        encoder_state = encoder.init_state(batch_size)
        encoder_out, enc_h, enc_c = encoder(encoder_in, encoder_state)
        decoder_state = [enc_h, enc_c]
        decoder_pred, dec_h, dec_c = decoder(decoder_in, decoder_state)

        # get the tensor with predictions(decoder_pred) and correct sentences(deocder_out)
        # decoder pred.shape = batch_size, seq_len, words, get the best prediction for axis words : -1
        # convert to numpy
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()
        # decoder_out is (batch_size, seq_len) , just convert it to numpy
        decoder_out = decoder_out.numpy()
        num_sentences = decoder_out.shape[0]
        # convert each tensor to list
        decoder_pred = decoder_pred.tolist()
        decoder_out = decoder_out.tolist()
        # go through all sentences in the batch
        for i in range(num_sentences):

            predicted_sentences = [idx2word_output[word] for word in decoder_pred[i] if word != 0]
            expected_sentences = [idx2word_output[word] for word in decoder_out[i] if word != 0][:-1]
            predicted_sentences = [word for word in predicted_sentences if word != "<eos>"]

            bleu = sentence_bleu([expected_sentences], predicted_sentences, smoothing_function=smoothing.method1)
            bleu_scores.append(bleu)

    final_bleu = sum(bleu_scores) / len(bleu_scores)
    print("Bleu score: {:.4f}".format(final_bleu))
    return final_bleu

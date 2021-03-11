import re
import tensorflow as tf
from data_preprocessing import NMT_Data
from Encoder_LSTM import EncoderLSTM
from Decoder_LSTM import DecoderLSTM

data = NMT_Data("deu.txt")
EMBEDDING_DIM = 256
LATENT_DIM = 1024


test_data, training_data, \
max_len_input, max_len_output, \
encoder_padded_input, decoder_padded_input, decoder_padded_output, \
input_seq, target_sequences, \
word2idx_input, word2idx_output, \
vocab_size_encoder, vocab_size_decoder = data.preprocess()

sentence = ""

encoder = EncoderLSTM(vocab_size_encoder +1, max_len_input, EMBEDDING_DIM, LATENT_DIM)
decoder = DecoderLSTM(vocab_size_decoder+1, max_len_output, EMBEDDING_DIM, LATENT_DIM)

encoder.load_weights("encoder_lstm")
decoder.load_weights("decoder_lstm")
encoder_input = data.input4testing(sentence, word2idx_input, max_len_input)
final_sentence = data.compute_prediction_LSTM(encoder_input, encoder, decoder, word2idx_output
                            , max_len_output)
print("Sentence to translate: ", sentence)
print("Translation: ", final_sentence)
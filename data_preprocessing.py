import re
import random 
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class NMT_Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenize_input = None

    def custom_shuffle(self, data_path, new_data_path):
        with open(data_path, encoding="utf-8") as file1: 
            lines = file1.readlines()
        random.shuffle(lines)
        with open(new_data_path, "w") as file2: 
            file2.writelines(lines)
        
    def preprocess(self):
     
        BATCH_SIZE = 16
        #NUM_SAMPLES = 10000  # Number of samples to train on. # 208486
        
        input_seq = []
        input_target_sequences = []
        target_sequences = []
        #t = 0
        num_samples = 0
        self.custom_shuffle(self.data_path, self.data_path)
        for line in open(self.data_path, encoding="utf8"):
            # t += 1
            # if t > NUM_SAMPLES:
            #     break
            num_samples += 1
            line = line.split("\t")
            input = line[0]
            translation = line[1]
            input_target_seq = "<sos> " + translation
            print(input_target_seq)
            target_seq = translation + " <eos>"

            input_seq.append(input.lower())
            input_target_sequences.append(input_target_seq.lower())
            target_sequences.append(target_seq.lower())

        NUM_SAMPLES = num_samples
        # tokenize
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(input_seq)
        input_int_sequences = tokenizer.texts_to_sequences(input_seq)
        word2idx_input = tokenizer.word_index
        vocab_size_encoder = len(word2idx_input)

        tokenize_target = Tokenizer(filters="")
        tokenize_target.fit_on_texts(target_sequences + input_target_sequences)
        word2idx_output = tokenize_target.word_index
        # assert("<eos>" in word2idx_output)
        vocab_size_decoder = len(word2idx_output)

        input_int_targets = tokenize_target.texts_to_sequences(input_target_sequences)
        targets_int = tokenize_target.texts_to_sequences(target_sequences)

        # padding
        max_len_input = max([len(s) for s in input_int_sequences])
        encoder_padded_input = pad_sequences(input_int_sequences, maxlen=max_len_input)

        max_len_output = max([len(s) for s in input_int_targets])
        decoder_padded_input = pad_sequences(input_int_targets, maxlen=max_len_output, padding="post")

        decoder_padded_output = pad_sequences(targets_int, maxlen=max_len_output, padding="post")

        # convert data to tensorflow dataset format
        dataset = tf.data.Dataset.from_tensor_slices(
            (encoder_padded_input, decoder_padded_input, decoder_padded_output))
        dataset = dataset.shuffle(10000)

        # testing and training split
        test_size = NUM_SAMPLES // 4
        test_data = dataset.take(test_size).batch(BATCH_SIZE, drop_remainder=True)
        training_data = dataset.skip(test_size).batch(BATCH_SIZE, drop_remainder=True)

        return test_data, training_data, max_len_input, max_len_output, encoder_padded_input, decoder_padded_input, decoder_padded_output, input_seq, target_sequences, word2idx_input, word2idx_output, vocab_size_encoder, vocab_size_decoder

    def preprocess4testing(self, sentence):
        sentence = sentence.lower().strip()
        sentence = re.sub(r"[!?.,()]", "", sentence)
        sentence = re.sub("\s+", " ", sentence)
        sentence = sentence
        return sentence

    def input4testing(self, sentence, word2index, max_len_input):
        sentence = self.preprocess4testing(sentence)
        input_idx = [word2index[i] for i in sentence.split()]
        print(input_idx)
        padded_input = pad_sequences([input_idx], maxlen=max_len_input)
        input_tensor = tf.convert_to_tensor(padded_input)
        return input_tensor

    def compute_prediction_LSTM(self, encoder_input, encoder, decoder, word2idx_output, max_len_output):
        idx2word_output = {i:w for (w,i) in word2idx_output.items()}
        encoder_state = encoder.init_state(1)

        encoder_out, enc_h, enc_c = encoder(encoder_input, encoder_state)
        decoder_state = [enc_h, enc_c]
        decoder_in = tf.expand_dims(tf.constant([word2idx_output["<sos>"]]), axis = 0)
        #decoder_in = np.zeros((1, 1))
        #decoder_in[0, 0] = word2idx_output["<sos>"]
        final_sentence = []
        for _ in range(max_len_output):
            decoder_pred, dec_h, dec_c = decoder(decoder_in, decoder_state)
            decoder_state = [dec_h, dec_c]
            #idx = np.argmax(decoder_pred[:, :, 0])
            idx = tf.argmax(decoder_pred, axis=-1)
            eos = word2idx_output["<eos>"]
            if idx == eos:
                break
            word = idx2word_output[idx.numpy()[0][0]]

            final_sentence.append(word)
            decoder_in = idx

        final_sentence = " ".join(final_sentence)
        return final_sentence

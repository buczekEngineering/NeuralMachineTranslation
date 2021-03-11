# Encoder-Decoder Neural-Machine-Translation 
The implementation of neural machine translation using tesnorflow. 

## Run
You can directly run train_gru.py or train_lstm.py, depnds on which neural network you want to use. GRU may be a little bit faster. The script contains data preprocessng, training, prediction on each epoch and evaluation.

## Required packages: 
- numpy 1.19.5
- nltk 3.5
- tensorflow 2.4.1

## Dataset
The dataset "deu.txt" contains 208 486 pairs of English-German translations. It can be download here http://www.manythings.org/anki/ 
The project should work also for other languages. If there is any problem with encoding, this parameters should be individually adjusted.
The encoding type for a specific dataset can be checked using chardet library https://chardet.readthedocs.io/en/latest/usage.html 

## Data preprocessing
Before feeding the data into the Neural Network we must preporcess it: preprocess_data.py -> preprocess()
The preprocess script involved: 
- creating encoder_input, decoder_input and decoder_output inputs
- cleaning 
- tokenization 
- padding
- shuffling
- creating tensorflow's datasets from encoder_input, decoder_input, decoder_output
- splitting into test/training set
- creating batches within sets

## Training the Encoder-Decoder LSTM model
Run train_lstm.py to train the model. Model subclassing: Encoder_LSTM.py, Decoder_LSTM.py.
The prediction is picked using "Greedy Decoding".    

## Training the Encoder-Decoder GRU model
Run train_gru.py to train the model. Model subclassing: Encoder.py, Decoder.py

## Model evaluation 
Model is evaualte after the each epoch during the training, on the test data. For the evaluation was used a common machine translation metrc: BLEU("Bilingual Evaluation Understudy")

## Test 
After training the model is saved and can be tested using test_translation.py. The english input sentence must be specified as a "sequence" string variable inside the script.


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from progress.bar import Bar
import json
import process_text
import numpy as np
import os
from gensim.models import Word2Vec
import generate_inputs
import operator

from keras.layers.recurrent import GRU, LSTM
from keras.models import Sequential
from keras.layers.core import Activation, Dense, RepeatVector, Dropout, Masking
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
import process_text
import numpy as np
import os 
from keras import backend as K
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization

EMBEDDING_DIM = 100;
MAX_NB_WORDS_SOURCE = 20000;
MAX_NB_WORDS_TARGET = 1022;

MAX_SEQUENCE_LENGTH = 20;
MAX_SEQ_LENGTH_WITHOUT_PAD = 18;
WORD_VECT_SIZE = 64;
VALIDATION_SPLIT = 0.1;
START_TOKEN = '100 '
END_TOKEN = ' 101'

def read_file(orig_file_name, trans_file_name, max_sentence_length):
    read_file = open(orig_file_name, 'r');
    read_orig_raw = json.load(read_file);
    read_file.close();
    read_file = open(trans_file_name, 'r');
    read_trans_raw = json.load(read_file);
    read_file.close();
    
    read_orig_list = [];
    read_trans_list = [];
    for i in range(0, len(read_orig_raw)):
        read_orig_list.extend(read_orig_raw[i]);
        read_trans_list.extend(read_trans_raw[i]);
    
    compacted_orig_result = [];
    compacted_trans_result = [];
    bar = Bar('Processing', max=(len(read_orig_list)));
    for i in range(0, len(read_orig_list)):
        compacted_orig_result.extend(read_orig_list[i]);
        compacted_trans_result.extend(read_trans_list[i]);
        bar.next();
    bar.finish();

    final_orig_result = [];
    final_trans_result = [];
    final_orig_text = [];
    final_trans_text = [];
    bar = Bar('Processing', max=(len(compacted_orig_result)));
    for i in range(0, len(compacted_orig_result)):
        if compacted_orig_result[i] == None or compacted_trans_result[i] == None:
            continue;
        orig =  process_text.get_tokens(compacted_trans_result[i]);
        trans = process_text.get_tokens(compacted_orig_result[i]);
        if len(orig) < max_sentence_length and len(trans) < max_sentence_length and len(orig) > 0 and len(trans):
            final_orig_result.append(START_TOKEN + compacted_orig_result[i].encode('ascii', 'ignore') + END_TOKEN);
            final_trans_result.append(START_TOKEN + compacted_trans_result[i].encode('ascii', 'ignore') + END_TOKEN);
        final_orig_text.append(START_TOKEN + compacted_orig_result[i].encode('ascii', 'ignore') + END_TOKEN);
        final_trans_text.append(START_TOKEN + compacted_trans_result[i].encode('ascii', 'ignore') + END_TOKEN);
        bar.next();
    bar.finish();
    return final_orig_result, final_trans_result, final_orig_text, final_trans_text;

def create_input(orig_text, trans_text, full_orig_text, full_trans_text):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS_SOURCE)
    tokenizer.fit_on_texts(full_orig_text)
    sequences_orig = tokenizer.texts_to_sequences(orig_text)

    word_index_orig = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index_orig))

    word_original = sorted(tokenizer.word_counts.items(), key=operator.itemgetter(1), reverse=True)

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS_SOURCE)
    tokenizer.fit_on_texts(full_trans_text)
    sequences_trans = tokenizer.texts_to_sequences(trans_text)

    word_index_trans = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index_trans))

    data_orig = pad_sequences(sequences_orig, maxlen=MAX_SEQUENCE_LENGTH)
    data_trans = pad_sequences(sequences_trans, maxlen=MAX_SEQUENCE_LENGTH)

    print(data_orig.shape);
    print(data_trans.shape);

    # # split the data into a training set and a validation set
    x_train = data_trans[:]
    y_train = data_orig[:]
    return x_train, y_train, word_index_orig, word_index_trans, word_original;

orig_text, trans_text, full_orig_text, full_trans_text = read_file("original_text.txt", "translation_text.txt", MAX_SEQ_LENGTH_WITHOUT_PAD);

trans_text = ["100 you told me you hated him . 101"]
x_train, y_train, word_index_orig, word_index_trans, word_freq_original = create_input(orig_text, trans_text, full_orig_text, full_trans_text);

top_freq_words = {};
for i in range(0, MAX_NB_WORDS_TARGET):
    top_freq_words[word_freq_original[i][0]] = i + 2;

word_embed_orig = {};
word_embed_orig_transfer = {};
for word, i in word_index_orig.items():
    if word in top_freq_words:
        word_embed_orig[top_freq_words[word]] = word;
        word_embed_orig_transfer[i] = top_freq_words[word];
word_embed_orig[1] = 'UNK_TOKEN';
word_embed_orig[0] = 'PADDING';
word_embed_orig_transfer[0] = 0;

print(y_train[0]);

EMBEDDING_DIM = 100;
OUTPUT_DIM = 20

NB_EPOCHS = 200;
BATCH_SIZE = 32;

input_text = ["you told me you hated him."]
output_text = ["thou told'st me thou didst hold him in thy hate."]

#Takes in an array of strings and the word2vec model to transform the words. 
#Utilizes nltk get_sent and get_words
#Returns n x m x v size numpy float32 matrix 

def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)


def vectorize(text, vec):
    vectorized_inputs = [];
    max_len = 512;
    vectorized_inputs = np.zeros((len(text), max_len, EMBEDDING_DIM), dtype=np.float32);
    for j in range(0, len(text)):
        tokens = process_text.get_tokens(text[j]);
        #We process the text in reverse direction ...
        vectorized_inputs[j][:][0] = START_TOKEN;
        for i in range(1, len(tokens) + 1):
            if tokens[i - 1] in vec:
                vectorized_inputs[j][i][:] = vec[tokens[i - 1]][:];
            else:
                vectorized_inputs[j][i][:] = UNK_TOKEN; #We define the UNK word as vector of all zeros. 
        #Time for the padding ... 
        vectorized_inputs[j][len(tokens)][:] = STOP_TOKEN; #We define the UNK word as vector of all zeros. 
        for i in range(len(tokens)+1, max_len):
            vectorized_inputs[j][i][:] = STOP_MASK;
    return vectorized_inputs, max_len;

embedding = np.load('embedding_mat.npy')
#Fixed Vector size
FIXED_SIZE = 256;
DENSE_SIZE = 256;
HIDDEN_SIZE = 1024;
W_INIT = 'he_normal';

INPUT_SHAPE = [FIXED_SIZE, EMBEDDING_DIM];

model = Sequential()
embedding_layer = Embedding(embedding.shape[0],
                            EMBEDDING_DIM,
                            weights=[embedding],
                            input_length=OUTPUT_DIM,
                            trainable=False)
model.add(embedding_layer);
M = Masking(mask_value=0.0);
model.add(M);
model.add(LSTM(FIXED_SIZE, go_backwards=True, return_sequences=True)) # Decoder Level
model.add(GRU(FIXED_SIZE, return_sequences=False))
model.add(Dropout(0.5));
model.add(Dense(DENSE_SIZE, activation='relu'))  # Decoding .... 
model.add(RepeatVector(OUTPUT_DIM))         # Inputs goes to encoder
model.add(LSTM(FIXED_SIZE, return_sequences=True))
model.add(GRU(FIXED_SIZE, return_sequences=True))
model.add(Dropout(0.5));
model.add(TimeDistributed(Dense(HIDDEN_SIZE, activation="softmax")))


opt = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0);
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
print(model.summary())

# define the checkpoint
filepath="models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.load_weights('models/weights-improvement-295-1.0564.hdf5');

value = model.predict(x_train);
print(value.shape);

def convert_value_to_words(value):
    result = [];
    converted_result = [];
    for i in range(0, value.shape[1]):
        result.append(np.argmax(value[0][i]));
        converted_result.append(word_embed_orig[np.argmax(value[0][i])])
    print(converted_result);

convert_value_to_words(value);
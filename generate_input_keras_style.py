from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from progress.bar import Bar
import json
import process_text
import numpy as np
import os
#from gensim.models import Word2Vec
#import generate_inputs
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
VALIDATION_SPLIT = 0.0;
START_TOKEN = '100 '
END_TOKEN = ' 101'

def split_punct(some_string):
    accumulated_word = [];
    word = "";
    for character in some_string:
        if character.isalnum() or character == '\'':
            word += str(character);
        else:
            accumulated_word.append(word);
            accumulated_word.append(str(character));
            word = "";
    if len(word) > 0:
        accumulated_word.append(word);
    return accumulated_word;

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
        orig =  process_text.get_tokens(compacted_orig_result[i].encode('ascii', 'ignore'));
        trans = process_text.get_tokens(compacted_trans_result[i].encode('ascii', 'ignore'));
        temp = [];
        for i in range(0, len(orig)):
            if not (ispunct(orig[i]) and len(orig[i]) > 1):
                temp.extend(split_punct(orig[i]));
        orig = temp;
        temp = [];
        for i in range(0, len(trans)):
            if not (ispunct(trans[i]) and len(trans[i]) > 1):
                temp.extend(split_punct(trans[i]));
        trans = temp;
        for i in range(0, len(orig)):
            if len(orig[i]) > 2:
                orig[i].replace('\"', '');
                orig[i].replace('\'', '');
        for i in range(0, len(trans)):
            if len(trans[i]) > 2:
                trans[i].replace('\"', '');
                trans[i].replace('\'', '');
        if len(orig) < max_sentence_length and len(trans) < max_sentence_length and len(orig) > 0 and len(trans) < max_sentence_length:
            final_orig_result.append(START_TOKEN + ' '.join([str(x) for x in orig]) + END_TOKEN);
            final_trans_result.append(START_TOKEN + ' '.join([str(x) for x in trans]) + END_TOKEN);
        final_orig_text.append(START_TOKEN + ' '.join([str(x) for x in orig]) + END_TOKEN);
        final_trans_text.append(START_TOKEN + ' '.join([str(x) for x in trans]) + END_TOKEN);
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
    indices = np.arange(data_orig.shape[0])
    np.random.shuffle(indices)
    data_orig = data_orig[indices]
    data_trans = data_trans[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data_orig.shape[0])

    x_train = data_trans;
    y_train = data_orig;
    x_val = [];
    y_val = [];
    # x_train = data_trans[:-nb_validation_samples]
    # y_train = data_orig[:-nb_validation_samples]
    # x_val = data_trans[-nb_validation_samples:]
    # y_val = data_orig[-nb_validation_samples:]
    return x_train, y_train, x_val, y_val, word_index_orig, word_index_trans, word_original;

orig_text, trans_text, full_orig_text, full_trans_text = read_file("original_text.txt", "translation_text.txt", MAX_SEQ_LENGTH_WITHOUT_PAD);
x_train, y_train, x_val, y_val, word_index_orig, word_index_trans, word_freq_original = create_input(orig_text, trans_text, full_orig_text, full_trans_text);

embeddings_index = {}
f = open(os.path.join('./data/glove.6B/', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index_trans) + 1, EMBEDDING_DIM))
for word, i in word_index_trans.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector        

np.save('embedding_mat.npy', embedding_matrix);
np.save('x_train.npy', x_train);
np.save('x_val.npy', x_val);

#y_train, y_val = get_y_embedding(word_index_trans, y_train, y_val);
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

y_hot_encoded_train = np.zeros((y_train.shape[0], MAX_SEQUENCE_LENGTH, MAX_NB_WORDS_TARGET + 2), dtype=np.uint8);
#y_hot_encoded_val = np.zeros((y_val.shape[0], MAX_SEQUENCE_LENGTH, MAX_NB_WORDS_TARGET + 2), dtype=np.uint8);

for i in range(0, y_train.shape[0]):
    for j in range(0, y_train.shape[1]):
        if y_train[i][j] in word_embed_orig_transfer:
            y_hot_encoded_train[i][j][word_embed_orig_transfer[y_train[i][j]]] = 1;
        else:
            y_hot_encoded_train[i][j][1] = 1;

# for i in range(0, y_val.shape[0]):
#     for j in range(0, y_val.shape[1]):
#         if y_val[i][j] in word_embed_orig_transfer:
#             y_hot_encoded_val[i][j][word_embed_orig_transfer[y_val[i][j]]] = 1;
#         else:
#             y_hot_encoded_val[i][j][1] = 1;



np.save('y_train.npy', y_hot_encoded_train);
#np.save('y_val.npy', y_hot_encoded_val);

trans_file = open('orig_text_translation_dict.json', 'w');
trans_file.write(json.dumps(word_embed_orig));
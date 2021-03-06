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

from theano import tensor as T

BATCH_SIZE = 1;
EMBEDDING_DIM = 100;
OUTPUT_DIM = 20

NB_EPOCHS = 300;
BATCH_SIZE = 32;
_EPSILON = 10e-8

input_text = ["you told me you hated him."]
output_text = ["thou told'st me thou didst hold him in thy hate."]

x_train = np.load('x_train.npy');
y_train = np.load('y_train.npy');
x_valid = np.load('x_val.npy');
y_valid = np.load('y_val.npy');
embedding = np.load('embedding_mat.npy')
print(x_train.shape);
print(y_train.shape);

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
model.add(LSTM(FIXED_SIZE, return_sequences=False))
#model.add(LSTM(FIXED_SIZE, return_sequences=True))
#model.add(LSTM(FIXED_SIZE, return_sequences=False))
model.add(Dropout(0.5));
model.add(Dense(DENSE_SIZE, activation='relu'))  # Decoding .... 
model.add(RepeatVector(OUTPUT_DIM))         # Inputs goes to encoder
model.add(LSTM(FIXED_SIZE, return_sequences=True))
model.add(LSTM(FIXED_SIZE, return_sequences=True))
#model.add(LSTM(FIXED_SIZE, return_sequences=True))
#model.add(LSTM(FIXED_SIZE, return_sequences=True))
model.add(Dropout(0.5));
model.add(TimeDistributed(Dense(HIDDEN_SIZE, activation="softmax")))


opt = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0);
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
print(model.summary())

# define the checkpoint
filepath="models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x_train, y_train,
          nb_epoch=NB_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)

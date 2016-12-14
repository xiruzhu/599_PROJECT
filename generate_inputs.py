from gensim.models import Word2Vec
import process_text
import numpy as np
import json
import os
from progress.bar import Bar

EMBEDDING_DIM = 64;
STOP_MASK = 1.0;
START_TOKEN = .75;
STOP_TOKEN = .5;
UNK_TOKEN = 0.0;

def vectorize(text, vec):
    vectorized_inputs = [];
    max_len = 512;
    vectorized_inputs = np.zeros((len(text), max_len, EMBEDDING_DIM), dtype=np.float32);
    bar = Bar('Processing', max=(len(text)));
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
        bar.next();
    bar.finish();
    return vectorized_inputs;

def generate(range_index):
    orig_file = open('original_text.txt', 'r');
    orig_raw = json.load(orig_file);
    orig_play = [];
    for i in range_index:
        orig_play.extend(orig_raw[i]);
    orig_file.close();
    trans_file = open('translation_text.txt', 'r');
    trans_raw = json.load(trans_file);
    trans_play = [];
    for i in range_index:
        trans_play.extend(trans_raw[i]);
    trans_file.close();

    print('Raw File Read ...')
    compact_orig_play = [];
    compact_trans_play = [];
    bar = Bar('Processing', max=(len(trans_play)));
    for i in range(len(trans_play)):
        compact_orig_play.extend(orig_play[i]);
        compact_trans_play.extend(trans_play[i]);
        bar.next();
    bar.finish();

    print('\nInitial Processing Finished')
    
    MODELS_DIR = 'models/'
    target_vec = Word2Vec.load_word2vec_format(os.path.join(MODELS_DIR, '{:s}.vec'.format('shakespeare_ft')));
    source_vec = Word2Vec.load_word2vec_format('./data/glove.6B/glove_100.txt');

    print('Word Vectors Loaded');
    result_input = vectorize(compact_trans_play, source_vec);
    result_output = vectorize(compact_orig_play, target_vec);

    print('\nSaving results')
    np.save("x_train.npy", result_input);
    np.save("y_train.npy", result_output);

    print(result_input.shape);
    print(result_output.shape);

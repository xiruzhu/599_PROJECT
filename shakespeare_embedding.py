MODELS_DIR = 'models/'

lr = 0.0075
dim = 64
ws = 50
epoch = 50
minCount = 4
neg = 5
loss = 'ns'
t = 1e-5

from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import os
import process_text
import numpy as np

# Same values as used for fastText training above
params = {
    'alpha': lr,
    'size': dim,
    'window': ws,
    'iter': epoch,
    'min_count': minCount,
    'sample': t,
    'sg': 1,
    'hs': 0,
    'negative': neg
}

def preprocess_text(corpus_file,  output_file):
    orig_file = open(corpus_file, 'r');
    orig_text = orig_file.read();
    orig_file.close();


    orig_text = orig_text.lower();
    orig_text = process_text.convert_unicode_char(orig_text);
    orig_token = process_text.get_tokens(orig_text);
    text = " ".join(orig_token);

    orig_file = open(output_file, 'w');
    orig_file.write(text);
    orig_file.close();


def train_models(corpus_file, output_name):
    output_file = '{:s}_ft'.format(output_name)
    if not os.path.isfile(os.path.join(MODELS_DIR, '{:s}.vec'.format(output_file))):
        print('\nTraining word2vec on {:s} corpus..'.format(corpus_file))
        
        # Text8Corpus class for reading space-separated words file
        gs_model = Word2Vec(Text8Corpus(corpus_file), **params); 
        # Direct local variable lookup doesn't work properly with magic statements (%time)
        locals()['gs_model'].save_word2vec_format(os.path.join(MODELS_DIR, '{:s}.vec'.format(output_file)))
        print('\nSaved gensim model as {:s}.vec'.format(output_file))
    else:
        print('\nUsing existing model file {:s}.vec'.format(output_file))
        gs_model = Word2Vec.load_word2vec_format(os.path.join(MODELS_DIR, '{:s}.vec'.format(output_file)));
    return gs_model;

#preprocess_text("shakespeare.txt", "preprocessed_shakespeare.txt");
model = train_models('preprocessed_shakespeare.txt', 'shakespeare')

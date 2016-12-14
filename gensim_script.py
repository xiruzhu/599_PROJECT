from gensim.models import Word2Vec

# Load Google's pre-trained Word2Vec model.
source_vec = Word2Vec.load_word2vec_format('./data/glove.6B/glove_100.txt');

value = (source_vec['100']);
print(value);

print(source_vec.most_similar(positive=[value], negative=[] , topn=10))

value = (source_vec['101']);
print(value);

print(source_vec.most_similar(positive=[value], negative=[] , topn=10))

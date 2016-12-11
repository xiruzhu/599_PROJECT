import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)  

value = (model['thou']);

print(model.most_similar(positive=[value], negative=[] , topn=10))

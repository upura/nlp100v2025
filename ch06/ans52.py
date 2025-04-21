from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "ch06/GoogleNews-vectors-negative300.bin", binary=True
)
result = model.most_similar(positive=["United_States"], topn=10)
print(result)

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "ch06/GoogleNews-vectors-negative300.bin", binary=True
)
us = model["United_States"]
print(us)

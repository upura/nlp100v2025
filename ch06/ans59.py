import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

df = pd.read_csv("ch06/questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5030]
country = list(set(df["v4"].values))

model = KeyedVectors.load_word2vec_format(
    "ch06/GoogleNews-vectors-negative300.bin", binary=True
)

countryVec = []
for c in country:
    countryVec.append(model[c])

X = np.array(countryVec)
tsne = TSNE(random_state=0, n_iter=15000, metric="cosine")
embs = tsne.fit_transform(X)
plt.scatter(embs[:, 0], embs[:, 1])
plt.show()

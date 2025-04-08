import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

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
km = KMeans(n_clusters=5, random_state=0)
y_km = km.fit_predict(X)
print(y_km)

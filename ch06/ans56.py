import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm


def culcCosSim(row):
    global model
    return model(row["Word 1"], row["Word 2"])


tqdm.pandas()
model = KeyedVectors.load_word2vec_format(
    "ch06/GoogleNews-vectors-negative300.bin", binary=True
)
df = pd.read_csv("ch06/wordsim353/combined.csv")
df["cosSim"] = df.progress_apply(culcCosSim, axis=1)

print(df[["Human (mean)", "cosSim"]].corr(method="spearman"))

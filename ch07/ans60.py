import pandas as pd
from sklearn.model_selection import train_test_split

newsCorpora = pd.read_table("ch07/NewsAggregatorDataset/newsCorpora.csv", header=None)
newsCorpora.columns = [
    "ID",
    "TITLE",
    "URL",
    "PUBLISHER",
    "CATEGORY",
    "STORY",
    "HOSTNAME",
    "TIMESTAMP",
]
newsCorpora = newsCorpora[
    newsCorpora["PUBLISHER"].isin(
        ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
    )
].sample(frac=1, random_state=0)

X = newsCorpora[["TITLE", "CATEGORY"]].copy()
X["CATEGORY"] = X["CATEGORY"].map({"b": 0, "e": 1, "t": 2, "m": 3})
y = newsCorpora["CATEGORY"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_test, y_test, test_size=0.5, stratify=y_test, random_state=0
)

X_train.to_csv("ch07/train.txt", sep="\t", index=False, header=None)
X_valid.to_csv("ch07/valid.txt", sep="\t", index=False, header=None)
X_test.to_csv("ch07/test.txt", sep="\t", index=False, header=None)

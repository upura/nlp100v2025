import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

X_train = pd.read_table("ch07/train.txt", header=None)
X_valid = pd.read_table("ch07/valid.txt", header=None)
X_test = pd.read_table("ch07/test.txt", header=None)
use_cols = ["TITLE", "CATEGORY"]
X_train.columns = use_cols
X_valid.columns = use_cols
X_test.columns = use_cols
X_train["TMP"] = "train"
X_valid["TMP"] = "valid"
X_test["TMP"] = "test"

data = pd.concat([X_train, X_valid, X_test]).reset_index(drop=True)
vectorizer = CountVectorizer(token_pattern="(?u)\\b\\w+\\b")
bag = vectorizer.fit_transform(data["TITLE"])
data = pd.concat([data, pd.DataFrame(bag.toarray())], axis=1)

joblib.dump(vectorizer.vocabulary_, "ch07/vocabulary_.joblib")

X_train = data.query('TMP=="train"').drop(use_cols + ["TMP"], axis=1)
X_valid = data.query('TMP=="valid"').drop(use_cols + ["TMP"], axis=1)
X_test = data.query('TMP=="test"').drop(use_cols + ["TMP"], axis=1)

X_train.to_csv("ch07/train.feature.txt", sep="\t", index=False, header=None)
X_valid.to_csv("ch07/valid.feature.txt", sep="\t", index=False, header=None)
X_test.to_csv("ch07/test.feature.txt", sep="\t", index=False, header=None)

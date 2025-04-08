import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

X_train = pd.read_table("ch07/train.feature.txt", header=None)
X_test = pd.read_table("ch07/test.feature.txt", header=None)
y_train = pd.read_table("ch07/train.txt", header=None)[1]
y_test = pd.read_table("ch07/test.txt", header=None)[1]

clf = joblib.load("ch07/model.joblib")

print(f"train acc: {accuracy_score(y_train, clf.predict(X_train))}")
print(f"test acc: {accuracy_score(y_test, clf.predict(X_test))}")

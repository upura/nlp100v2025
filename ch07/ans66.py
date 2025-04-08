import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

X_train = pd.read_table("ch07/train.feature.txt", header=None)
X_test = pd.read_table("ch07/test.feature.txt", header=None)
y_train = pd.read_table("ch07/train.txt", header=None)[1]
y_test = pd.read_table("ch07/test.txt", header=None)[1]

clf = joblib.load("ch07/model.joblib")
y_pred = clf.predict(X_test)

print(f"test recall of None: {recall_score(y_test, y_pred, average=None)}")
print(f"test recall of micro: {recall_score(y_test, y_pred, average='micro')}")
print(f"test recall of macro: {recall_score(y_test, y_pred, average='macro')}")
print(f"test precision of None: {precision_score(y_test, y_pred, average=None)}")
print(f"test precision of micro: {precision_score(y_test, y_pred, average='micro')}")
print(f"test precision of macro: {precision_score(y_test, y_pred, average='macro')}")
print(f"test f1 of None: {f1_score(y_test, y_pred, average=None)}")
print(f"test f1 of micro: {f1_score(y_test, y_pred, average='micro')}")
print(f"test f1 of macro: {f1_score(y_test, y_pred, average='macro')}")

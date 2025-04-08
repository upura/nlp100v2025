import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = pd.read_table("ch06/train.feature.txt", header=None)
X_valid = pd.read_table("ch06/valid.feature.txt", header=None)
X_test = pd.read_table("ch06/test.feature.txt", header=None)
y_train = pd.read_table("ch06/train.txt", header=None)[1]
y_valid = pd.read_table("ch06/valid.txt", header=None)[1]
y_test = pd.read_table("ch06/test.txt", header=None)[1]

C_candidate = [0.1, 1.0, 10, 100]
train_acc = []
valid_acc = []
test_acc = []

for c in C_candidate:
    clf = LogisticRegression(penalty="l2", solver="sag", random_state=0, C=c)
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    valid_acc.append(accuracy_score(y_valid, clf.predict(X_valid)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

plt.plot(C_candidate, train_acc, label="train acc")
plt.plot(C_candidate, valid_acc, label="valid acc")
plt.plot(C_candidate, test_acc, label="test acc")
plt.legend()
plt.savefig("ch07/ans68.png")

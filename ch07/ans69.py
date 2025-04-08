import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from collections import defaultdict
import japanize_matplotlib


def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data


# データの読み込み
df_train = pd.read_csv("ch07/SST-2/train.tsv", sep="\t")
df_dev = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t")

# 特徴ベクトルの構築
data_train = []
for sentence, label in zip(df_train["sentence"], df_train["label"]):
    data_train.append(add_feature(sentence, label))

data_dev = []
for sentence, label in zip(df_dev["sentence"], df_dev["label"]):
    data_dev.append(add_feature(sentence, label))

# 特徴ベクトルの変換
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform([d["feature"] for d in data_train])
y_train = [d["label"] for d in data_train]
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

# 正則化パラメータの範囲を設定
C_values = np.logspace(-5, 5, 21)  # 10^-5から10^5まで21点

# 各正則化パラメータでの正解率を記録
train_accuracies = []
dev_accuracies = []

# 各正則化パラメータでモデルを学習
for C in C_values:
    # ロジスティック回帰モデルの学習
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    # 訓練データと検証データでの正解率を計算
    train_pred = model.predict(X_train)
    dev_pred = model.predict(X_dev)

    train_acc = accuracy_score(y_train, train_pred)
    dev_acc = accuracy_score(y_dev, dev_pred)

    train_accuracies.append(train_acc)
    dev_accuracies.append(dev_acc)

    print(
        f"C = {C:.2e}, 訓練データの正解率: {train_acc:.4f}, 検証データの正解率: {dev_acc:.4f}"
    )

# 結果の可視化
japanize_matplotlib.japanize()
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_accuracies, "o-", label="訓練データ")
plt.semilogx(C_values, dev_accuracies, "o-", label="検証データ")
plt.grid(True)
plt.xlabel("正則化パラメータ C")
plt.ylabel("正解率")
plt.title("正則化パラメータと正解率の関係")
plt.legend()
plt.tight_layout()
plt.savefig("ch07/regularization_accuracy.png")
plt.close()

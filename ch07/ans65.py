import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import accuracy_score


def add_feature(sentence, label):
    data = {"sentence": sentence, "label": label, "feature": defaultdict(int)}
    for token in sentence.split():
        data["feature"][token] += 1
    return data


# モデルとベクトライザーの読み込み
with open("ch07/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("ch07/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

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
X_train = vec.transform([d["feature"] for d in data_train])
y_train = [d["label"] for d in data_train]
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

# 予測
y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)

# 正解率の計算
train_accuracy = accuracy_score(y_train, y_train_pred)
dev_accuracy = accuracy_score(y_dev, y_dev_pred)

# 結果の表示
print(f"学習データの正解率: {train_accuracy:.4f}")
print(f"検証データの正解率: {dev_accuracy:.4f}")

import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


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

# 検証データの読み込み
df_dev = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t")

# 特徴ベクトルの構築
data_dev = []
for sentence, label in zip(df_dev["sentence"], df_dev["label"]):
    data_dev.append(add_feature(sentence, label))

# 特徴ベクトルの変換
X_dev = vec.transform([d["feature"] for d in data_dev])
y_dev = [d["label"] for d in data_dev]

# 予測
y_dev_pred = model.predict(X_dev)

# 評価指標の計算
precision = precision_score(y_dev, y_dev_pred)
recall = recall_score(y_dev, y_dev_pred)
f1 = f1_score(y_dev, y_dev_pred)

# 結果の表示
print("検証データにおける評価指標:")
print(f"適合率 (Precision): {precision:.4f}")
print(f"再現率 (Recall): {recall:.4f}")
print(f"F1スコア: {f1:.4f}")

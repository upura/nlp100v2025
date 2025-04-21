import pandas as pd
import pickle
from collections import defaultdict


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

# 検証データの先頭の事例を取得
first_sentence = df_dev["sentence"].iloc[0]
first_label = df_dev["label"].iloc[0]

# 特徴ベクトルの構築
data = add_feature(first_sentence, first_label)

# 特徴ベクトルの変換
X = vec.transform([data["feature"]])

# ラベルの予測
predicted_label = model.predict(X)[0]
predicted_prob = model.predict_proba(X)[0]

# 結果の表示
print(f"条件付き確率: {predicted_prob}")

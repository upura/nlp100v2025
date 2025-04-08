import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import japanize_matplotlib


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

# 混同行列の計算
cm = confusion_matrix(y_dev, y_dev_pred)

# 混同行列の表示
print("混同行列:")
print(cm)

# 混同行列の各要素の意味を表示
print("\n混同行列の解釈:")
print(f"真陰性 (True Negative): {cm[0][0]} - ネガティブと正しく予測")
print(f"偽陽性 (False Positive): {cm[0][1]} - ポジティブと誤って予測")
print(f"偽陰性 (False Negative): {cm[1][0]} - ネガティブと誤って予測")
print(f"真陽性 (True Positive): {cm[1][1]} - ポジティブと正しく予測")

# 評価指標の計算
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n評価指標:")
print(f"正解率 (Accuracy): {accuracy:.4f}")
print(f"適合率 (Precision): {precision:.4f}")
print(f"再現率 (Recall): {recall:.4f}")
print(f"F1スコア: {f1:.4f}")

# 混同行列の可視化
japanize_matplotlib.japanize()
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["ネガティブ", "ポジティブ"],
    yticklabels=["ネガティブ", "ポジティブ"],
)
plt.xlabel("予測ラベル")
plt.ylabel("実際のラベル")
plt.title("検証データにおける混同行列")
plt.tight_layout()
plt.savefig("ch07/confusion_matrix.png")
plt.close()

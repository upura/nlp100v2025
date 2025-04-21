import pickle

# モデルとベクトライザーの読み込み
with open("ch07/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("ch07/vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

# 特徴量名の取得
feature_names = vec.get_feature_names_out()

# モデルの重みを取得
weights = model.coef_[0]

# 重みと特徴量名のペアを作成
weight_feature_pairs = list(zip(weights, feature_names))

# 重みの高い特徴量トップ20を取得
top_20_positive = sorted(weight_feature_pairs, key=lambda x: x[0], reverse=True)[:20]

# 重みの低い特徴量トップ20を取得
top_20_negative = sorted(weight_feature_pairs, key=lambda x: x[0])[:20]

# 結果の表示
print("重みの高い特徴量トップ20:")
for i, (weight, feature) in enumerate(top_20_positive, 1):
    print(f"{i}. {feature}: {weight:.4f}")

print("\n重みの低い特徴量トップ20:")
for i, (weight, feature) in enumerate(top_20_negative, 1):
    print(f"{i}. {feature}: {weight:.4f}")

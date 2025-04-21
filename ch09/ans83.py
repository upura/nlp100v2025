import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# モデルとトークナイザーの読み込み
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# 入力文のリスト
sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

# 各文の[CLS]トークンの埋め込みベクトルを取得
embeddings = []
for sentence in sentences:
    # トークン化
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # モデルによる推論
    with torch.no_grad():
        outputs = model(**inputs)

    # [CLS]トークンの埋め込みベクトルを取得
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
    embeddings.append(cls_embedding)

# 埋め込みベクトルをnumpy配列に変換
embeddings = np.array(embeddings)

# コサイン類似度の計算（すべての組み合わせを一度に計算）
similarity_matrix = cosine_similarity(embeddings)

# 結果の表示
print("文の組み合わせに対するコサイン類似度:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(
            f"'{sentences[i]}' と '{sentences[j]}' の類似度: {similarity_matrix[i][j]:.4f}"
        )

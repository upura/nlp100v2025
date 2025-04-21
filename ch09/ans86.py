import pandas as pd
from transformers import AutoTokenizer
import torch

# モデルとトークナイザーの読み込み
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)


# データの読み込み
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()


# テキストをトークン列に変換
def tokenize_texts(texts):
    tokenized_texts = []
    for text in texts:
        # トークン化（特殊トークンを追加）
        tokens = tokenizer.tokenize(text)
        tokenized_texts.append(tokens)
    return tokenized_texts


# データファイルのパス
train_path = "ch07/SST-2/train.tsv"
dev_path = "ch07/SST-2/dev.tsv"

# 訓練データと開発データの読み込み
train_texts, train_labels = load_data(train_path)
dev_texts, dev_labels = load_data(dev_path)

# トークン化の実行
train_tokenized = tokenize_texts(train_texts)
dev_tokenized = tokenize_texts(dev_texts)

# 最初の4事例を選択
sample_texts = train_texts[:4]
sample_labels = train_labels[:4]
sample_tokenized = train_tokenized[:4]

# パディングとトークンIDへの変換
encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

# 結果の表示
print("元のテキスト:")
for text in sample_texts:
    print(f"- {text}")

print("\nトークン列:")
for tokens in sample_tokenized:
    print(f"- {tokens}")

print("\nパディング後のトークンID:")
print(encoded["input_ids"])

print("\nアテンションマスク:")
print(encoded["attention_mask"])

print("\nラベル:")
print(torch.tensor(sample_labels))

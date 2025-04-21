import pandas as pd
from transformers import AutoTokenizer


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


# モデルとトークナイザーの読み込み
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# データファイルのパス
train_path = "ch07/SST-2/train.tsv"
dev_path = "ch07/SST-2/dev.tsv"

# 訓練データと開発データの読み込み
train_texts, train_labels = load_data(train_path)
dev_texts, dev_labels = load_data(dev_path)

# トークン化の実行
train_tokenized = tokenize_texts(train_texts)
dev_tokenized = tokenize_texts(dev_texts)

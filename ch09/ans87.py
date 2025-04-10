import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score


# データの読み込み
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()


# 評価用の関数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


# データセットクラスの定義
class SSTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # テキストのエンコーディング
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# モデルとトークナイザーの読み込み
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,  # 2クラス分類（肯定的/否定的）
)

# データの読み込み
train_path = "ch07/SST-2/train.tsv"
dev_path = "ch07/SST-2/dev.tsv"
train_texts, train_labels = load_data(train_path)
dev_texts, dev_labels = load_data(dev_path)

# データセットの作成
train_dataset = SSTDataset(train_texts, train_labels, tokenizer)
dev_dataset = SSTDataset(dev_texts, dev_labels, tokenizer)

# トレーニング引数の設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
)

# トレーナーの作成と訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# 訓練の実行
trainer.train()

# 最終的な評価
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")

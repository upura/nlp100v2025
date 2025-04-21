import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
import pandas as pd
import evaluate


# カスタムモデルの定義
class MaxPoolingClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # BERTの出力を取得
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 各トークンの最大値プーリング
        pooled_output = outputs.last_hidden_state.max(dim=1)[0]

        # ドロップアウトと分類
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {"loss": loss, "logits": logits}


# データの読み込み
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()


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


# 評価用の関数
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


# メイン処理
def main():
    # モデルとトークナイザーの読み込み
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MaxPoolingClassifier(model_id)

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
        output_dir="./results_maxpool",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
    )

    # トレーナーの作成
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
    print(f"最終的な評価結果: {eval_results}")


if __name__ == "__main__":
    main()

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score


def create_prompt(sentence):
    return f"""以下の文の感情を分析してください。文の後に「positive」または「negative」のいずれかで答えてください。

文: {sentence}
感情: """


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    # 生成されたテキストから感情ラベルを抽出
    predicted_labels = []
    for pred in predictions:
        text = tokenizer.decode(pred, skip_special_tokens=True)
        if "positive" in text.lower():
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return {"accuracy": accuracy_score(labels, predicted_labels)}


def main():
    # モデルとトークナイザーの読み込み
    model_name = "llm-jp/llm-jp-3-150m-instruct3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # モデルのパディングトークンIDも設定
    model.config.pad_token_id = tokenizer.pad_token_id

    # データの読み込み
    train_path = "ch07/SST-2/train.tsv"
    dev_path = "ch07/SST-2/dev.tsv"

    train_df = pd.read_csv(train_path, sep="\t", header=0)
    dev_df = pd.read_csv(dev_path, sep="\t", header=0)

    # データセットの作成
    def tokenize_function(examples):
        prompts = [create_prompt(s) for s in examples["sentence"]]
        labels = [
            "positive" if label == 1 else "negative" for label in examples["label"]
        ]
        # プロンプトとラベルを結合
        texts = [prompt + label for prompt, label in zip(prompts, labels)]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(dev_df)

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir="./logs",
        save_strategy="epoch",
        eval_strategy="epoch",
    )

    # トレーナーの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    # 学習の実行
    trainer.train()

    # 最終評価
    eval_results = trainer.evaluate()
    print(f"最終評価結果: {eval_results}")


if __name__ == "__main__":
    main()

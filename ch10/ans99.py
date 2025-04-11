from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
from trl import DPOTrainer


def create_prompt(sentence):
    return f"""以下の文の感情を分析してください。文の後に「positive」または「negative」のいずれかで答えてください。

文: {sentence}
感情: """


def create_preference_data(df):
    # 正解ラベルに基づいて望ましい応答と望ましくない応答を作成
    chosen_responses = []
    rejected_responses = []
    prompts = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        label = row["label"]

        # プロンプトの作成
        prompt = create_prompt(sentence)
        prompts.append(prompt)

        # 正解ラベルに基づいて応答を作成
        if label == 1:  # positive
            chosen_responses.append(prompt + "positive")
            rejected_responses.append(prompt + "negative")
        else:  # negative
            chosen_responses.append(prompt + "negative")
            rejected_responses.append(prompt + "positive")

    return {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses,
    }


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

    # 選好データの作成
    train_preference_data = create_preference_data(train_df)
    dev_preference_data = create_preference_data(dev_df)

    # データセットの作成
    train_dataset = Dataset.from_dict(train_preference_data)
    dev_dataset = Dataset.from_dict(dev_preference_data)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./results_dpo",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=3,
        logging_steps=1,
        optim="adamw_8bit",
        seed=42,
    )

    # DPOトレーナーの作成
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )

    # 学習の実行
    dpo_trainer.train()

    # 最終評価
    eval_results = dpo_trainer.evaluate()
    print(f"最終評価結果: {eval_results}")


if __name__ == "__main__":
    main()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm


def create_prompt(text):
    """感情分析のためのプロンプトテンプレートを作成する"""
    return f"""以下のテキストの感情を分析してください。ポジティブ（positive）かネガティブ（negative）のどちらかで答えてください。

テキスト: {text}

感情:"""


def predict_sentiment(text, model, tokenizer):
    """テキストの感情を予測する"""
    # プロンプトの作成
    prompt = create_prompt(text)

    # トークン化
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # 応答の生成
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
        )

    # 生成された応答をデコード
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 応答から感情を抽出
    response = response.lower()
    return 1 if "positive" in response else 0


def main():
    # モデルとトークナイザーの読み込み
    model_id = "llm-jp/llm-jp-3-150m-instruct3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # SST-2データセットの読み込み
    dev_path = "ch07/SST-2/dev.tsv"
    dataset = pd.read_csv(dev_path, sep="\t", header=0)

    # 予測と正解率の計算
    correct = 0
    total = 0

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row["sentence"]
        label = row["label"]  # 0: negative, 1: positive

        # 感情予測
        predicted_label = predict_sentiment(text, model, tokenizer)

        # 正解率の計算
        if predicted_label == label:
            correct += 1
        total += 1

    # 結果の表示
    accuracy = correct / total
    print(f"正解率: {accuracy:.4f}")
    print(f"正解数: {correct}")
    print(f"総数: {total}")


if __name__ == "__main__":
    main()

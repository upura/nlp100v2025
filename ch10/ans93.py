from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math


def calculate_perplexity(model, tokenizer, text):
    # テキストのトークン化
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # モデルの出力を取得
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # パープレキシティの計算
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # クロスエントロピー損失の計算
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # 平均損失の計算
    mean_loss = loss.mean()

    # パープレキシティの計算（exp(平均損失)）
    perplexity = math.exp(mean_loss.item())

    return perplexity


# モデルとトークナイザーの読み込み
model_id = "llm-jp/llm-jp-3-150m-instruct3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 評価する文
sentences = [
    "The movie was full of surprises",  # 正しい文
    "The movies were full of surprises",  # 正しい文
    "The movie were full of surprises",  # 間違った文（主語と動詞の不一致）
    "The movies was full of surprises",  # 間違った文（主語と動詞の不一致）
]

# 各文のパープレキシティを計算
print("各文のパープレキシティ:")
for sentence in sentences:
    perplexity = calculate_perplexity(model, tokenizer, sentence)
    print(f"文: {sentence}")
    print(f"パープレキシティ: {perplexity:.2f}")
    print("-" * 50)
"""
各文のパープレキシティ:
文: The movie was full of surprises
パープレキシティ: 78.67
--------------------------------------------------
文: The movies were full of surprises
パープレキシティ: 130.23
--------------------------------------------------
文: The movie were full of surprises
パープレキシティ: 215.27
--------------------------------------------------
文: The movies was full of surprises
パープレキシティ: 336.99
--------------------------------------------------
"""

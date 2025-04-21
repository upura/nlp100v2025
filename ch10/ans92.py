from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルとトークナイザーの読み込み
model_id = "llm-jp/llm-jp-3-150m-instruct3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# プロンプトの設定
prompt = "The movie was full of"

# 入力のトークン化
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 生成パラメータの設定
gen_kwargs = {
    "max_new_tokens": 10,  # 短めに設定
    "do_sample": True,
    "temperature": 1.0,
    "return_dict_in_generate": True,
    "output_scores": True,
}

# テキスト生成
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
    )

# 生成されたトークンとその尤度を取得
generated_ids = outputs.sequences[0]
scores = outputs.scores

# 結果の表示
print("生成されたテキストと各単語の尤度:")
current_text = prompt
for i, (token_id, score) in enumerate(zip(generated_ids[len(input_ids[0]) :], scores)):
    token = tokenizer.decode([token_id])
    # 尤度の計算（softmaxを適用）
    probabilities = torch.softmax(score, dim=-1)
    token_prob = probabilities[0, token_id].item()

    # トークンの間に半角スペースを追加
    if not token.startswith(" ") and current_text[-1] != " ":
        current_text += " "
    current_text += token

    print(f"{current_text}")
    print(f"単語: {token}, 尤度: {token_prob:.4f}")
    print("-" * 50)

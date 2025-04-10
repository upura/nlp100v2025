from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルとトークナイザーの読み込み
model_id = "llm-jp/llm-jp-3-150m-instruct3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# プロンプトの設定
prompt = "The movie was full of"

# デコーディング方法と温度パラメータの設定
decoding_configs = [
    {"method": "greedy", "temperature": 1.0},
    {"method": "greedy", "temperature": 0.5},
    {"method": "greedy", "temperature": 2.0},
    {"method": "beam_search", "num_beams": 5, "temperature": 1.0},
    {"method": "top_k", "top_k": 50, "temperature": 1.0},
    {"method": "top_p", "top_p": 0.9, "temperature": 1.0},
]

# 各設定でテキスト生成を実行
for config in decoding_configs:
    print(f"\nデコーディング方法: {config['method']}")
    if "temperature" in config:
        print(f"温度パラメータ: {config['temperature']}")

    # 入力のトークン化
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 生成パラメータの設定
    gen_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "temperature": config.get("temperature", 1.0),
    }

    # デコーディング方法に応じたパラメータを追加
    if config["method"] == "beam_search":
        gen_kwargs.update(
            {
                "num_beams": config["num_beams"],
                "do_sample": False,
            }
        )
    elif config["method"] == "top_k":
        gen_kwargs.update(
            {
                "top_k": config["top_k"],
            }
        )
    elif config["method"] == "top_p":
        gen_kwargs.update(
            {
                "top_p": config["top_p"],
            }
        )

    # テキスト生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )

    # 結果の表示
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成されたテキスト: {generated_text}")
    print("-" * 50)

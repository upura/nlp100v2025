import google.generativeai as genai
import os
import csv
from dotenv import load_dotenv
from pathlib import Path
import random

# 環境変数からAPIキーを読み込む
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# APIキーを設定
genai.configure(api_key=api_key)

# 利用可能な科目のリスト
AVAILABLE_SUBJECTS = {
    "world_history": "世界史",
    "japanese_history": "日本史",
    "philosophy": "哲学",
    "sociology": "社会学",
    "high_school_physics": "高校物理",
    "high_school_mathematics": "高校数学",
    "high_school_chemistry": "高校化学",
    "high_school_biology": "高校生物",
    "college_physics": "大学物理",
    "college_mathematics": "大学数学",
    "college_chemistry": "大学化学",
    "college_biology": "大学生物学",
    "computer_security": "コンピュータセキュリティ",
    "machine_learning": "機械学習",
    "high_school_computer_science": "高校情報科学",
    "college_computer_science": "大学コンピュータ科学",
    "high_school_psychology": "高校心理学",
    "professional_psychology": "専門心理学",
    "high_school_statistics": "高校統計学",
    "econometrics": "計量経済学",
    "high_school_microeconomics": "高校ミクロ経済学",
    "high_school_macroeconomics": "高校マクロ経済学",
    "management": "経営学",
    "marketing": "マーケティング",
    "business_ethics": "ビジネス倫理",
    "professional_accounting": "専門会計",
    "professional_medicine": "専門医学",
    "college_medicine": "大学医学",
    "clinical_knowledge": "臨床知識",
    "medical_genetics": "医学遺伝学",
    "anatomy": "解剖学",
    "human_aging": "人間の老化",
    "nutrition": "栄養学",
    "high_school_geography": "高校地理",
    "high_school_european_history": "高校ヨーロッパ史",
    "international_law": "国際法",
    "jurisprudence": "法理学",
    "formal_logic": "形式論理",
    "logical_fallacies": "論理学",
    "moral_disputes": "倫理的議論",
    "world_religions": "世界宗教",
    "human_sexuality": "セクシュアリティ",
    "security_studies": "セキュリティ研究",
    "electrical_engineering": "電気工学",
    "conceptual_physics": "概念物理学",
    "astronomy": "天文学",
    "prehistory": "先史学",
    "global_facts": "世界事実",
    "miscellaneous": "雑学",
    "abstract_algebra": "抽象代数",
    "elementary_mathematics": "初等数学",
    "virology": "ウイルス学",
    "public_relations": "公共関係",
}


def load_jmmlu_data(file_path):
    """JMMLUのデータセットを読み込む"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:  # 問題、選択肢A、B、C、D、正解の6列があることを確認
                data.append({"question": row[0], "choices": row[1:5], "answer": row[5]})
    return data


def create_prompt(question, choices, prompt_type="standard", choice_symbols=None):
    """プロンプトを作成する"""
    if choice_symbols is None:
        choice_symbols = ["A", "B", "C", "D"]

    if prompt_type == "standard":
        prompt = f"""
以下の問題に対して、選択肢{choice_symbols[0]}、{choice_symbols[1]}、{choice_symbols[2]}、{choice_symbols[3]}の中から最も適切なものを1つ選んでください。
回答は選択肢の記号（{choice_symbols[0]}、{choice_symbols[1]}、{choice_symbols[2]}、{choice_symbols[3]}）のみを返してください。

問題：{question}

選択肢：
{choice_symbols[0]}. {choices[0]}
{choice_symbols[1]}. {choices[1]}
{choice_symbols[2]}. {choices[2]}
{choice_symbols[3]}. {choices[3]}
"""
    elif prompt_type == "detailed":
        prompt = f"""
以下の問題に対して、選択肢{choice_symbols[0]}、{choice_symbols[1]}、{choice_symbols[2]}、{choice_symbols[3]}の中から最も適切なものを1つ選んでください。
回答は選択肢の記号（{choice_symbols[0]}、{choice_symbols[1]}、{choice_symbols[2]}、{choice_symbols[3]}）のみを返してください。
各選択肢を慎重に検討し、最も正確な回答を選んでください。

問題：{question}

選択肢：
{choice_symbols[0]}. {choices[0]}
{choice_symbols[1]}. {choices[1]}
{choice_symbols[2]}. {choices[2]}
{choice_symbols[3]}. {choices[3]}
"""
    elif prompt_type == "concise":
        prompt = f"""
問題：{question}

選択肢：
{choice_symbols[0]}. {choices[0]}
{choice_symbols[1]}. {choices[1]}
{choice_symbols[2]}. {choices[2]}
{choice_symbols[3]}. {choices[3]}

回答（{choice_symbols[0]}、{choice_symbols[1]}、{choice_symbols[2]}、{choice_symbols[3]}のいずれか）：
"""

    return prompt


def shuffle_choices(choices, answer):
    """選択肢の順番をシャッフルする"""
    # 選択肢とインデックスのペアを作成
    indexed_choices = list(enumerate(choices))
    # シャッフル
    random.shuffle(indexed_choices)
    # 新しい順番の選択肢と、元のインデックスを取得
    new_choices = [c[1] for c in indexed_choices]
    # 正解のインデックスを更新
    old_index = ord(answer) - ord("A")
    new_index = indexed_choices.index((old_index, choices[old_index]))
    new_answer = chr(ord("A") + new_index)

    return new_choices, new_answer


def evaluate_model_with_settings(subject_key, settings, num_questions=10):
    """異なる設定でモデルの評価を行う"""
    # 科目名の取得
    if subject_key not in AVAILABLE_SUBJECTS:
        print(f"エラー: 科目 '{subject_key}' は利用できません。")
        print("利用可能な科目:")
        for key, name in AVAILABLE_SUBJECTS.items():
            print(f"  - {key}: {name}")
        return

    # データセットのパスを設定
    dataset_path = Path(f"JMMLU/JMMLU/{subject_key}.csv")

    # データセットの読み込み
    try:
        questions = load_jmmlu_data(dataset_path)
    except FileNotFoundError:
        print(f"エラー: 科目 '{subject_key}' のデータセットが見つかりません。")
        print("データセットをダウンロードしてください：")
        print("git clone https://github.com/nlp-waseda/JMMLU.git")
        return

    # 評価する問題数を制限
    questions = questions[:num_questions]

    # 各設定での結果を格納する辞書
    results = {}

    # 各設定で評価
    for setting_name, setting in settings.items():
        print(f"\n設定: {setting_name}")
        print(
            f"温度: {setting.get('temperature', 0.7)}, プロンプトタイプ: {setting.get('prompt_type', 'standard')}"
        )

        # モデルの設定
        model = genai.GenerativeModel(
            "gemini-1.5-flash-8b",
            generation_config={"temperature": setting.get("temperature", 0.7)},
        )

        correct_count = 0
        total_questions = len(questions)

        for i, q in enumerate(questions, 1):
            # 選択肢の順番をシャッフルするかどうか
            if setting.get("shuffle_choices", False):
                shuffled_choices, shuffled_answer = shuffle_choices(
                    q["choices"], q["answer"]
                )
                choices = shuffled_choices
                answer = shuffled_answer
            else:
                choices = q["choices"]
                answer = q["answer"]

            # 選択肢の記号を変更するかどうか
            choice_symbols = setting.get("choice_symbols", ["A", "B", "C", "D"])

            # プロンプトの作成
            prompt = create_prompt(
                q["question"],
                choices,
                setting.get("prompt_type", "standard"),
                choice_symbols,
            )

            try:
                # APIリクエストの送信
                response = model.generate_content(prompt)
                model_answer = response.text.strip().upper()

                # 選択肢の記号が変更されている場合、回答を変換
                if choice_symbols != ["A", "B", "C", "D"]:
                    # 元の記号に変換
                    symbol_map = {
                        choice_symbols[i]: chr(ord("A") + i) for i in range(4)
                    }
                    model_answer = symbol_map.get(model_answer, model_answer)

                # 正解判定
                is_correct = model_answer == answer
                correct_count += int(is_correct)

                # 進捗表示
                print(
                    f"\r進捗: {i}/{total_questions} (正解率: {correct_count / i * 100:.1f}%)",
                    end="",
                )

            except Exception as e:
                print(f"\nエラー: {e}")
                continue

        # 最終結果の表示
        final_accuracy = correct_count / total_questions * 100
        print("\n\n評価結果:")
        print(f"正解数: {correct_count}/{total_questions}")
        print(f"正解率: {final_accuracy:.1f}%")

        # 結果を保存
        results[setting_name] = {
            "correct_count": correct_count,
            "total_questions": total_questions,
            "accuracy": final_accuracy,
        }

    return results


def run_experiments(subject_key, num_questions=10):
    """様々な実験設定で評価を実行する"""
    # 実験設定
    settings = {
        "標準設定": {
            "temperature": 0.7,
            "prompt_type": "standard",
            "shuffle_choices": False,
            "choice_symbols": ["A", "B", "C", "D"],
        },
        "低温度": {
            "temperature": 0.1,
            "prompt_type": "standard",
            "shuffle_choices": False,
            "choice_symbols": ["A", "B", "C", "D"],
        },
        "高温度": {
            "temperature": 1.0,
            "prompt_type": "standard",
            "shuffle_choices": False,
            "choice_symbols": ["A", "B", "C", "D"],
        },
        "詳細プロンプト": {
            "temperature": 0.7,
            "prompt_type": "detailed",
            "shuffle_choices": False,
            "choice_symbols": ["A", "B", "C", "D"],
        },
        "簡潔プロンプト": {
            "temperature": 0.7,
            "prompt_type": "concise",
            "shuffle_choices": False,
            "choice_symbols": ["A", "B", "C", "D"],
        },
        "選択肢シャッフル": {
            "temperature": 0.7,
            "prompt_type": "standard",
            "shuffle_choices": True,
            "choice_symbols": ["A", "B", "C", "D"],
        },
        "数字記号": {
            "temperature": 0.7,
            "prompt_type": "standard",
            "shuffle_choices": False,
            "choice_symbols": ["1", "2", "3", "4"],
        },
    }

    # 評価実行
    results = evaluate_model_with_settings(subject_key, settings, num_questions)

    # 結果の比較
    print("\n\n実験結果の比較:")
    print("=" * 50)
    print(f"{'設定名':<15} {'正解数':<10} {'正解率':<10}")
    print("-" * 50)

    for setting_name, result in results.items():
        print(
            f"{setting_name:<15} {result['correct_count']}/{result['total_questions']:<10} {result['accuracy']:.1f}%"
        )

    print("=" * 50)


if __name__ == "__main__":
    # 科目を選択（例: 'world_history'）
    subject = "world_history"
    # 評価する問題数
    num_questions = 10  # 実験用に少ない問題数で実行

    run_experiments(subject, num_questions)

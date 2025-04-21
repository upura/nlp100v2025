import google.generativeai as genai
import os
import csv
from dotenv import load_dotenv
from pathlib import Path

# 環境変数からAPIキーを読み込む
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# APIキーを設定
genai.configure(api_key=api_key)

# モデルの設定
model = genai.GenerativeModel("gemini-1.5-flash-8b")

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


def create_prompt(question, choices):
    """プロンプトを作成する"""
    prompt = f"""
以下の問題に対して、選択肢A、B、C、Dの中から最も適切なものを1つ選んでください。
回答は選択肢のアルファベット（A、B、C、D）のみを返してください。

問題：{question}

選択肢：
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}
"""
    return prompt


def evaluate_model(subject_key, num_questions=1):
    """モデルの評価を行う"""
    # 科目名の取得
    if subject_key not in AVAILABLE_SUBJECTS:
        print(f"エラー: 科目 '{subject_key}' は利用できません。")
        print("利用可能な科目:")
        for key, name in AVAILABLE_SUBJECTS.items():
            print(f"  - {key}: {name}")
        return

    subject_name = AVAILABLE_SUBJECTS[subject_key]

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

    correct_count = 0
    total_questions = len(questions)
    total_available = len(load_jmmlu_data(dataset_path))

    print(f"科目: {subject_name}")
    print(f"問題数: {total_questions} (全{total_available}問中)")
    print("評価を開始します...")

    for i, q in enumerate(questions, 1):
        # プロンプトの作成
        prompt = create_prompt(q["question"], q["choices"])

        try:
            # APIリクエストの送信
            response = model.generate_content(prompt)
            answer = response.text.strip().upper()

            # 正解判定
            is_correct = answer == q["answer"]
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


if __name__ == "__main__":
    # 科目を選択（例: 'world_history'）
    subject = "world_history"
    # 評価する問題数を全問題数に設定
    num_questions = None  # Noneを指定すると全問題を評価

    evaluate_model(subject, num_questions)

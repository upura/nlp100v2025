import google.generativeai as genai
import os
from dotenv import load_dotenv

# 環境変数からAPIキーを読み込む
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# APIキーを設定
genai.configure(api_key=api_key)

# モデルの設定
model = genai.GenerativeModel("gemini-1.5-flash-8b")

# 評価対象の川柳リスト
senryu_list = [
    "春風が吹く　スマホの待ち受け　桜満開",
    "菜の花の黄　春の便り届け　花粉症鼻水",
    "新芽の緑色　春の息吹が　若者に芽生え",
    "春の陽射しで　アイスが溶けてゆく　早く食べたい",
    "ひな人形飾る　春の喜びが　満ちるリビング",
    "カエルの合唱　春の夜空に響く　うるさいけど良い",
    "ソメイヨシノ満開　インスタ映えも　幸せの春",
    "春の嵐が　洗濯物を　叩き起こす",
    "チューリップの花　鮮やかな色彩　春の訪れ",
    "春の七草粥　体に良いと　元気になる春",
]

# 評価プロンプトの作成
evaluation_prompt = """
あなたは川柳の専門家として、以下の川柳を評価してください。
各川柳について、面白さを10段階（1〜10）で評価し、その理由を簡潔に説明してください。

評価対象の川柳：
"""

# 各川柳を評価プロンプトに追加
for i, senryu in enumerate(senryu_list, 1):
    evaluation_prompt += f"{i}. {senryu}\n"

evaluation_prompt += """
出力形式：
1. [川柳]
   評価：[1〜10の数値]
   理由：[評価理由の簡潔な説明]

2. [川柳]
   評価：[1〜10の数値]
   理由：[評価理由の簡潔な説明]

（以下10個分続く）

最後に、全体的な評価と総合的なコメントを追加してください。
"""

# APIリクエストの送信
response = model.generate_content(evaluation_prompt)

# 結果の表示
print("川柳の評価結果：")
print(response.text)

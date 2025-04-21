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

# プロンプトの作成
prompt = """
以下の条件で川柳を10個作成してください：

1. お題：「春の訪れ」
2. 川柳の形式：
   - 5音、7音、5音の17音
   - 季語を含める
   - 現代的な表現やユーモアを交える
3. 各川柳の後に簡単な解説を付ける

出力形式：
1. [川柳]
   解説：[解説文]

2. [川柳]
   解説：[解説文]

（以下10個分続く）
"""

# APIリクエストの送信
response = model.generate_content(prompt)

# 結果の表示
print("川柳の案：")
print(response.text)

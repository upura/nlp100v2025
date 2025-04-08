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
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。

参考情報:
- 東急東横線: 渋谷 → 代官山 → 中目黒 → 祐天寺 → 学芸大学 → 都立大学 → 自由が丘 → 田園調布 → 多摩川 → 新丸子 → 武蔵小杉 → 元住吉 → 日吉 → 綱島 → 大倉山 → 菊名 → 妙蓮寺 → 白楽 → 東白楽 → 反町 → 横浜

- 東急大井町線: 大井町 → 下神明 → 戸越公園 → 中延 → 荏原町 → 旗の台 → 北千束 → 大岡山 → 緑が丘 → 自由が丘 → 九品仏 → 尾山台 → 等々力 → 上野毛 → 二子玉川 → 二子新地 → 高津 → 溝の口 → 梶が谷 → 宮崎台 → 宮前平 → 鷺沼 → たまプラーザ → あざみ野 → 江田 → 市が尾 → 藤が丘 → 青葉台 → 田奈 → 長津田 → つきみ野 → 中央林間

- 東急大井町線の急行停車駅: 大井町、大岡山、自由が丘、二子玉川、溝の口、長津田、中央林間
"""

# APIリクエストの送信
response = model.generate_content(prompt)

# 結果の表示
print("解答:")
print(response.text)

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

# チャット履歴の開始
chat = model.start_chat(history=[])

# 最初の質問
user_question1 = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。

参考情報として、東急大井町線の駅一覧と急行停車駅は以下の通りです：

東急大井町線: 大井町 → 下神明 → 戸越公園 → 中延 → 荏原町 → 旗の台 → 北千束 → 大岡山 → 緑が丘 → 自由が丘 → 九品仏 → 尾山台 → 等々力 → 上野毛 → 二子玉川 → 二子新地 → 高津 → 溝の口 → 梶が谷 → 宮崎台 → 宮前平 → 鷺沼 → たまプラーザ → あざみ野 → 江田 → 市が尾 → 藤が丘 → 青葉台 → 田奈 → 長津田 → つきみ野 → 中央林間

急行停車駅: 大井町、大岡山、自由が丘、二子玉川、溝の口、長津田、中央林間
"""

# 最初の回答を取得
response1 = chat.send_message(user_question1)
print("【質問1】")
print("目的地の駅を教えてください。")
print("\n【システムの回答】")
print(response1.text)

# 追加の質問
user_question2 = """
さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
"""

# 追加の回答を取得
response2 = chat.send_message(user_question2)
print("\n【質問2】")
print("反対方向に乗った場合、何駅先で降りれば良いですか？")
print("\n【システムの回答】")
print(response2.text)

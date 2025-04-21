import spacy
from spacy import displacy


text = """メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。"""

# 日本語のモデルをロード
nlp = spacy.load("ja_ginza")

# テキストを解析
text = "メロスは激怒した。"
doc = nlp(text)

# 係り受け木を可視化
displacy.serve(doc, style="dep", port=8000)

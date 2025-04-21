import spacy


def analyze_dependencies(text):
    # 日本語のモデルをロード
    nlp = spacy.load("ja_ginza")

    # テキストを解析
    doc = nlp(text)

    # 係り受け関係を抽出
    for token in doc:
        if token.dep_ != "ROOT":  # ROOTは係り先がないので除外
            # 係り元と係り先のテキストを取得
            source = token.text
            target = token.head.text
            # タブ区切りで出力
            print(f"{source}\t{target}")


# テスト用のテキスト
text = """メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。"""

# 係り受け解析を実行
analyze_dependencies(text)

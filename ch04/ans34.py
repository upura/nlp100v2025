import spacy


def extract_predicates_for_subject(text, subject="メロス"):
    # 日本語のモデルをロード
    nlp = spacy.load("ja_ginza")

    # テキストを解析
    doc = nlp(text)

    # 述語を抽出
    predicates = []
    for token in doc:
        if token.text == subject and token.dep_ == "nsubj":
            # 主語に対応する述語を取得
            predicate = token.head
            predicates.append(predicate.text)

    return predicates


# テスト用のテキスト
text = """メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。"""

# 「メロス」が主語である場合の述語を抽出
predicates = extract_predicates_for_subject(text)
print(f"「メロス」が主語である場合の述語: {predicates}")

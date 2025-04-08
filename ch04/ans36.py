import json
import gzip
import re
import MeCab
from collections import Counter


def remove_markup(text):
    # 強調マークアップの除去
    text = re.sub(r"\'{2,5}", "", text)
    # 内部リンクの除去
    text = re.sub(r"\[\[(?:[^|\]]*?\|)??([^|\]]+?)\]\]", r"\1", text)
    # 外部リンクの除去
    text = re.sub(r"\[http://[^\]]+\]", "", text)
    # HTMLタグの除去
    text = re.sub(r"<[^>]+>", "", text)
    # テンプレートの除去
    text = re.sub(r"\{\{.*?\}\}", "", text)
    return text


def analyze_word_frequency():
    # MeCabの初期化
    mecab = MeCab.Tagger("-Owakati")

    # 単語の出現頻度をカウントするためのCounter
    word_counter = Counter()

    # gzipファイルを読み込む
    with gzip.open("ch04/jawiki-country.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            text = article["text"]

            # マークアップを除去
            text = remove_markup(text)

            # 形態素解析を行い、単語をカウント
            words = mecab.parse(text).strip().split()
            word_counter.update(words)

    # 出現頻度の高い20語を表示
    for word, count in word_counter.most_common(20):
        print(f"{word}: {count}")


if __name__ == "__main__":
    analyze_word_frequency()

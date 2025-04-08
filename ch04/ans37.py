import json
import gzip
import re
import MeCab
from collections import Counter

def remove_markup(text):
    # 強調マークアップの除去
    text = re.sub(r'\'{2,5}', '', text)
    # 内部リンクの除去
    text = re.sub(r'\[\[(?:[^|\]]*?\|)??([^|\]]+?)\]\]', r'\1', text)
    # 外部リンクの除去
    text = re.sub(r'\[http://[^\]]+\]', '', text)
    # HTMLタグの除去
    text = re.sub(r'<[^>]+>', '', text)
    # テンプレートの除去
    text = re.sub(r'\{\{.*?\}\}', '', text)
    return text

def analyze_noun_frequency():
    # MeCabの初期化（品詞情報を取得するために-Ochasenを使用）
    mecab = MeCab.Tagger("-Ochasen")
    
    # 名詞の出現頻度をカウントするためのCounter
    noun_counter = Counter()
    
    # gzipファイルを読み込む
    with gzip.open('ch04/jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            text = article['text']
            
            # マークアップを除去
            text = remove_markup(text)
            
            # 形態素解析を行い、名詞をカウント
            node = mecab.parseToNode(text)
            while node:
                # 品詞が名詞の場合のみカウント
                if node.feature.split(',')[0] == '名詞':
                    noun_counter[node.surface] += 1
                node = node.next
    
    # 出現頻度の高い20語を表示
    for noun, count in noun_counter.most_common(20):
        print(f"{noun}: {count}")

if __name__ == "__main__":
    analyze_noun_frequency()

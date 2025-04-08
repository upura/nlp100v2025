import json
import gzip
import re
import MeCab
import math
from collections import Counter, defaultdict

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

def calculate_tfidf():
    # MeCabの初期化
    mecab = MeCab.Tagger("-Ochasen")
    
    # 文書数をカウントするための変数
    total_docs = 0
    # 各名詞が出現する文書数をカウント
    doc_freq = defaultdict(int)
    # 日本に関する記事の名詞の出現頻度をカウント
    japan_noun_freq = Counter()
    
    # gzipファイルを読み込む
    with gzip.open('ch04/jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            total_docs += 1
            article = json.loads(line)
            text = article['text']
            
            # マークアップを除去
            text = remove_markup(text)
            
            # 形態素解析を行い、名詞をカウント
            node = mecab.parseToNode(text)
            # この文書で出現した名詞を記録
            doc_nouns = set()
            while node:
                if node.feature.split(',')[0] == '名詞':
                    noun = node.surface
                    doc_nouns.add(noun)
                    # 日本に関する記事の場合、出現頻度をカウント
                    if article['title'] == '日本':
                        japan_noun_freq[noun] += 1
                node = node.next
            # 文書頻度を更新
            for noun in doc_nouns:
                doc_freq[noun] += 1
    
    # TF-IDFスコアを計算
    tfidf_scores = {}
    for noun, tf in japan_noun_freq.items():
        # IDFの計算
        idf = math.log(total_docs / doc_freq[noun])
        # TF-IDFスコアの計算
        tfidf_scores[noun] = {
            'tf': tf,
            'idf': idf,
            'tfidf': tf * idf
        }
    
    # TF-IDFスコアの高い順に20語を表示
    for noun, scores in sorted(tfidf_scores.items(), key=lambda x: x[1]['tfidf'], reverse=True)[:20]:
        print(f"{noun}:")
        print(f"  TF: {scores['tf']}")
        print(f"  IDF: {scores['idf']:.4f}")
        print(f"  TF-IDF: {scores['tfidf']:.4f}")

if __name__ == "__main__":
    calculate_tfidf()

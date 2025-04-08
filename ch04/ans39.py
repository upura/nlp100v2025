import json
import gzip
import re
import MeCab
import matplotlib.pyplot as plt
from collections import Counter
import japanize_matplotlib


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

def plot_word_frequency_rank():
    # MeCabの初期化
    mecab = MeCab.Tagger("-Owakati")
    
    # 単語の出現頻度をカウントするためのCounter
    word_counter = Counter()
    
    # gzipファイルを読み込む
    with gzip.open('ch04/jawiki-country.json.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            text = article['text']
            
            # マークアップを除去
            text = remove_markup(text)
            
            # 形態素解析を行い、単語をカウント
            words = mecab.parse(text).strip().split()
            word_counter.update(words)
    
    # 出現頻度の順位と頻度を取得
    frequencies = list(word_counter.values())
    frequencies.sort(reverse=True)
    ranks = range(1, len(frequencies) + 1)
    
    # グラフの描画
    japanize_matplotlib.japanize()
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, 'b-', label='単語の出現頻度')
    plt.grid(True)
    plt.xlabel('出現頻度順位')
    plt.ylabel('出現頻度')
    plt.title('単語の出現頻度順位と出現頻度の関係（Zipfの法則）')
    plt.legend()
    
    # グラフを保存
    plt.savefig('ch04/word_frequency_rank.png')
    plt.close()

if __name__ == "__main__":
    plot_word_frequency_rank()

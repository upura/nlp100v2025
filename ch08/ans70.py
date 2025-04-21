import numpy as np
from gensim.models import KeyedVectors
from typing import Dict, Tuple


def load_word_embeddings(
    model_path: str = "ch06/GoogleNews-vectors-negative300.bin",
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    事前学習済み単語埋め込みを読み込み、単語埋め込み行列と単語-IDの対応関係を作成する。

    Args:
        model_path (str): 事前学習済み単語ベクトルのパス

    Returns:
        Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
            - 単語埋め込み行列 (語彙数+1 × 次元数)
            - 単語からIDへの辞書
            - IDから単語への辞書
    """
    # 事前学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # 語彙数を取得
    vocab_size = len(model.key_to_index)
    embedding_dim = model.vector_size

    # 単語埋め込み行列を作成（語彙数+1 × 次元数）
    # 先頭行はパディングトークン用のゼロベクトル
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

    # 単語からIDへの辞書とIDから単語への辞書を作成
    word_to_id = {"<PAD>": 0}  # パディングトークンのIDは0
    id_to_word = {0: "<PAD>"}

    # 単語埋め込み行列の2行目以降に事前学習済み単語ベクトルを格納
    for i, word in enumerate(model.key_to_index, start=1):
        embedding_matrix[i] = model[word]
        word_to_id[word] = i
        id_to_word[i] = word

    return embedding_matrix, word_to_id, id_to_word


def main():
    # 単語埋め込み行列と辞書を作成
    embedding_matrix, word_to_id, id_to_word = load_word_embeddings()

    # 結果の確認
    print(f"単語埋め込み行列の形状: {embedding_matrix.shape}")
    print(f"語彙数: {len(word_to_id)}")
    print(f"埋め込み次元数: {embedding_matrix.shape[1]}")

    # サンプルとして最初の5単語を表示
    print("\n最初の5単語:")
    for i in range(1, 6):
        word = id_to_word[i]
        print(f"ID: {i}, 単語: {word}, ベクトル: {embedding_matrix[i][:5]}...")


if __name__ == "__main__":
    main()

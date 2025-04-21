import torch
import pandas as pd
from typing import List, Dict, Set
from gensim.models import KeyedVectors


def load_sst2_data(file_path: str) -> pd.DataFrame:
    """
    SST-2のデータを読み込む

    Args:
        file_path (str): データファイルのパス

    Returns:
        pd.DataFrame: 読み込んだデータ
    """
    return pd.read_csv(file_path, sep="\t", header=0)


def get_vocabulary(df: pd.DataFrame) -> Set[str]:
    """
    データセットに含まれる単語の集合を取得する

    Args:
        df (pd.DataFrame): データセット

    Returns:
        Set[str]: 単語の集合
    """
    vocabulary = set()
    for text in df["sentence"]:
        vocabulary.update(text.lower().split())
    return vocabulary


def load_word_embeddings(model_path: str, vocabulary: Set[str]) -> Dict[str, int]:
    """
    必要な単語の埋め込みのみを読み込む

    Args:
        model_path (str): 事前学習済み単語ベクトルのパス
        vocabulary (Set[str]): 必要な単語の集合

    Returns:
        Dict[str, int]: 単語からIDへの辞書
    """
    # 単語からIDへの辞書を作成
    word_to_id = {"<PAD>": 0}  # パディングトークンのIDは0

    # 事前学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # 必要な単語のみを辞書に追加
    for word in vocabulary:
        if word in model.key_to_index:
            word_to_id[word] = len(word_to_id)

    return word_to_id


def convert_text_to_ids(text: str, word_to_id: Dict[str, int]) -> List[int]:
    """
    テキストをトークンID列に変換する

    Args:
        text (str): 変換するテキスト
        word_to_id (Dict[str, int]): 単語からIDへの辞書

    Returns:
        List[int]: トークンID列
    """
    # テキストを小文字に変換し、トークン化
    tokens = text.lower().split()

    # 語彙に含まれるトークンのIDのみを取得
    ids = [word_to_id[token] for token in tokens if token in word_to_id]

    return ids


def process_sst2_data(file_path: str, word_to_id: Dict[str, int]) -> List[Dict]:
    """
    SST-2のデータを処理し、トークンID列に変換する

    Args:
        file_path (str): データファイルのパス
        word_to_id (Dict[str, int]): 単語からIDへの辞書

    Returns:
        List[Dict]: 処理されたデータ
    """
    # データを読み込む
    df = load_sst2_data(file_path)

    processed_data = []

    for _, row in df.iterrows():
        # テキストをトークンID列に変換
        input_ids = convert_text_to_ids(row["sentence"], word_to_id)

        # 空のトークン列の場合はスキップ
        if not input_ids:
            continue

        # データを辞書形式で保存
        data = {
            "text": row["sentence"],
            "label": torch.tensor([float(row["label"])]),
            "input_ids": torch.tensor(input_ids),
        }
        processed_data.append(data)

    return processed_data


def main():
    # 訓練データと開発データを読み込む
    train_df = load_sst2_data("ch07/SST-2/train.tsv")
    dev_df = load_sst2_data("ch07/SST-2/dev.tsv")

    # 必要な単語の集合を取得
    vocabulary = get_vocabulary(train_df)
    vocabulary.update(get_vocabulary(dev_df))

    # 必要な単語の埋め込みのみを読み込む
    word_to_id = load_word_embeddings(
        "ch06/GoogleNews-vectors-negative300.bin", vocabulary
    )

    # 訓練データを処理
    train_data = process_sst2_data("ch07/SST-2/train.tsv", word_to_id)
    print(f"訓練データ数: {len(train_data)}")

    # 開発データを処理
    dev_data = process_sst2_data("ch07/SST-2/dev.tsv", word_to_id)
    print(f"開発データ数: {len(dev_data)}")

    # サンプルを表示
    print("\n訓練データのサンプル:")
    sample = train_data[0]
    print(f"テキスト: {sample['text']}")
    print(f"ラベル: {sample['label']}")
    print(f"トークンID列: {sample['input_ids']}")


if __name__ == "__main__":
    main()

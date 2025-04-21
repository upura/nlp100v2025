import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Set
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset


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


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        単語埋め込みの平均ベクトルを用いた分類器

        Args:
            embedding_dim (int): 埋め込みの次元数
        """
        super().__init__()
        self.linear = nn.Linear(
            embedding_dim, 1
        )  # 1次元出力（シグモイド関数で0-1に変換）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x (torch.Tensor): 入力テンソル [バッチサイズ, 埋め込み次元]

        Returns:
            torch.Tensor: 出力テンソル [バッチサイズ, 1]
        """
        return self.sigmoid(self.linear(x))


def load_word_embeddings(
    model_path: str, vocabulary: Set[str]
) -> tuple[Dict[str, int], torch.Tensor]:
    """
    必要な単語の埋め込みのみを読み込む

    Args:
        model_path (str): 事前学習済み単語ベクトルのパス
        vocabulary (Set[str]): 必要な単語の集合

    Returns:
        tuple[Dict[str, int], torch.Tensor]:
            - 単語からIDへの辞書
            - 単語埋め込み行列
    """
    # 単語からIDへの辞書を作成
    word_to_id = {"<PAD>": 0}  # パディングトークンのIDは0

    # 事前学習済み単語ベクトルを読み込む
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # 必要な単語のみを辞書に追加し、埋め込みを取得
    embeddings = [torch.zeros(model.vector_size)]  # パディングトークン用のゼロベクトル
    for word in vocabulary:
        if word in model.key_to_index:
            word_to_id[word] = len(word_to_id)
            embeddings.append(torch.tensor(model[word]))

    # 単語埋め込み行列を作成
    embedding_matrix = torch.stack(embeddings)

    return word_to_id, embedding_matrix


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


class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict], embedding_matrix: torch.Tensor):
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        input_ids = item["input_ids"]
        embeddings = self.embedding_matrix[input_ids]
        mean_embedding = torch.mean(embeddings, dim=0)
        return mean_embedding, item["label"]


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


def evaluate_model(model: nn.Module, dev_loader: DataLoader) -> float:
    """
    モデルの開発セットにおける正解率を求める

    Args:
        model (nn.Module): 評価するモデル
        dev_loader (DataLoader): 開発データのローダー

    Returns:
        float: 正解率（%）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dev_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    # データの読み込みと前処理
    train_df = load_sst2_data("ch07/SST-2/train.tsv")
    dev_df = load_sst2_data("ch07/SST-2/dev.tsv")

    # 必要な単語の集合を取得
    vocabulary = get_vocabulary(train_df)
    vocabulary.update(get_vocabulary(dev_df))

    # 単語埋め込みと辞書の読み込み
    word_to_id, embedding_matrix = load_word_embeddings(
        "ch06/GoogleNews-vectors-negative300.bin", vocabulary
    )

    # データセットの作成
    dev_data = process_sst2_data("ch07/SST-2/dev.tsv", word_to_id)

    # データセットとデータローダーの作成
    dev_dataset = SST2Dataset(dev_data, embedding_matrix)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # モデルの作成と学習済みパラメータの読み込み
    model = MeanEmbeddingClassifier(embedding_matrix.size(1))
    model.load_state_dict(torch.load("ch08/model.pth"))

    # 開発セットでの評価
    accuracy = evaluate_model(model, dev_loader)
    print(f"開発セットの正解率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

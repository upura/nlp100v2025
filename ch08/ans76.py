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


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    バッチ内の事例をパディングし、長さでソートする

    Args:
        batch (List[Dict]):
            - 各要素は{'text': str, 'label': torch.Tensor, 'input_ids': torch.Tensor}の辞書
            - input_ids: トークンID列 [トークン数]
            - label: ラベル [1]

    Returns:
        Dict[str, torch.Tensor]:
            - 'input_ids': パディングされた入力テンソル [バッチサイズ, 最大トークン数]
            - 'label': ラベルテンソル [バッチサイズ, 1]
    """
    # バッチ内の最大トークン数を取得
    max_len = max(len(item["input_ids"]) for item in batch)

    # 入力テンソルとラベルテンソルを初期化
    batch_size = len(batch)
    input_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
    label_tensor = torch.zeros((batch_size, 1), dtype=torch.float)

    # 各事例の長さを取得
    lengths = [len(item["input_ids"]) for item in batch]

    # 長さでソートするためのインデックスを取得
    sorted_indices = sorted(range(batch_size), key=lambda i: lengths[i], reverse=True)

    # パディングとソートを実行
    for i, idx in enumerate(sorted_indices):
        item = batch[idx]
        input_tensor[i, : len(item["input_ids"])] = item["input_ids"]
        label_tensor[i] = item["label"]

    return {"input_ids": input_tensor, "label": label_tensor}


class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict], embedding_matrix: torch.Tensor):
        self.data = data
        self.embedding_matrix = embedding_matrix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    embedding_matrix: torch.Tensor,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
) -> None:
    """
    モデルを学習する

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのローダー
        dev_loader (DataLoader): 開発データのローダー
        embedding_matrix (torch.Tensor): 単語埋め込み行列
        num_epochs (int): エポック数
        learning_rate (float): 学習率
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # 訓練モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # バッチ内の各事例の単語埋め込みの平均を計算
            embeddings = embedding_matrix[
                batch["input_ids"]
            ]  # [batch_size, max_len, embedding_dim]
            mean_embeddings = torch.mean(
                embeddings, dim=1
            )  # [batch_size, embedding_dim]

            outputs = model(mean_embeddings)
            loss = criterion(outputs, batch["label"])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch["label"].size(0)
            train_correct += (predicted == batch["label"]).sum().item()

        # 開発データでの評価
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for batch in dev_loader:
                # バッチ内の各事例の単語埋め込みの平均を計算
                embeddings = embedding_matrix[batch["input_ids"]]
                mean_embeddings = torch.mean(embeddings, dim=1)

                outputs = model(mean_embeddings)
                loss = criterion(outputs, batch["label"])

                dev_loss += loss.item()
                predicted = (outputs > 0.5).float()
                dev_total += batch["label"].size(0)
                dev_correct += (predicted == batch["label"]).sum().item()

        # 結果の表示
        train_accuracy = 100 * train_correct / train_total
        dev_accuracy = 100 * dev_correct / dev_total
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Acc: {train_accuracy:.2f}%, "
            f"Dev Loss: {dev_loss / len(dev_loader):.4f}, "
            f"Dev Acc: {dev_accuracy:.2f}%"
        )

    # 学習済みモデルを保存
    torch.save(model.state_dict(), "ch08/model.pth")


def evaluate_model(
    model: nn.Module, dev_loader: DataLoader, embedding_matrix: torch.Tensor
) -> float:
    """
    モデルの開発セットにおける正解率を求める

    Args:
        model (nn.Module): 評価するモデル
        dev_loader (DataLoader): 開発データのローダー
        embedding_matrix (torch.Tensor): 単語埋め込み行列

    Returns:
        float: 正解率（%）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dev_loader:
            # バッチ内の各事例の単語埋め込みの平均を計算
            embeddings = embedding_matrix[batch["input_ids"]]
            mean_embeddings = torch.mean(embeddings, dim=1)

            outputs = model(mean_embeddings)
            predicted = (outputs > 0.5).float()
            total += batch["label"].size(0)
            correct += (predicted == batch["label"]).sum().item()

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
    train_data = process_sst2_data("ch07/SST-2/train.tsv", word_to_id)
    dev_data = process_sst2_data("ch07/SST-2/dev.tsv", word_to_id)

    # データセットとデータローダーの作成
    train_dataset = SST2Dataset(train_data, embedding_matrix)
    dev_dataset = SST2Dataset(dev_data, embedding_matrix)

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=8, shuffle=False, collate_fn=collate
    )

    # モデルの作成
    model = MeanEmbeddingClassifier(embedding_matrix.size(1))

    # モデルの学習
    train_model(model, train_loader, dev_loader, embedding_matrix)

    # 開発セットでの評価
    accuracy = evaluate_model(model, dev_loader, embedding_matrix)
    print(f"\n開発セットの正解率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

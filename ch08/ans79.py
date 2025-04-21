import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Set
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, Dataset


# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class TextClassifier(nn.Module):
    def __init__(
        self, embedding_matrix: torch.Tensor, hidden_dim: int = 256, num_layers: int = 2
    ):
        """
        テキスト分類用のニューラルネットワーク

        Args:
            embedding_matrix (torch.Tensor): 単語埋め込み行列 [語彙数, 埋め込み次元]
            hidden_dim (int): 隠れ層の次元数
            num_layers (int): LSTMの層数
        """
        super().__init__()
        embedding_dim = embedding_matrix.size(1)

        # 単語埋め込み層
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        # 1次元畳み込み層
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # 双方向LSTM層
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0,
        )

        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x (torch.Tensor): 入力テンソル [バッチサイズ, 最大トークン数]

        Returns:
            torch.Tensor: 出力テンソル [バッチサイズ, 1]
        """
        # 単語埋め込みの取得 [batch_size, max_len, embedding_dim]
        x = self.embedding(x)

        # 畳み込み層の適用 [batch_size, hidden_dim, max_len]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, max_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        # LSTM層の適用 [batch_size, max_len, hidden_dim]
        x = x.transpose(1, 2)  # [batch_size, max_len, hidden_dim]
        x, _ = self.lstm(x)

        # 最終時刻の隠れ状態を取得 [batch_size, hidden_dim]
        x = x[:, -1, :]

        # 全結合層の適用 [batch_size, 1]
        x = self.fc(x)

        return x


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

    return {"input_ids": input_tensor.to(device), "label": label_tensor.to(device)}


class SST2Dataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

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
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
    """
    モデルを学習する

    Args:
        model (nn.Module): 学習するモデル
        train_loader (DataLoader): 訓練データのローダー
        dev_loader (DataLoader): 開発データのローダー
        num_epochs (int): エポック数
        learning_rate (float): 学習率
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # 訓練モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(batch["input_ids"])
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
                outputs = model(batch["input_ids"])
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
        for batch in dev_loader:
            outputs = model(batch["input_ids"])
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
    train_dataset = SST2Dataset(train_data)
    dev_dataset = SST2Dataset(dev_data)

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=32, shuffle=False, collate_fn=collate
    )

    # モデルの作成とGPUへの移動
    model = TextClassifier(embedding_matrix).to(device)

    # モデルの学習
    train_model(model, train_loader, dev_loader)

    # 開発セットでの評価
    accuracy = evaluate_model(model, dev_loader)
    print(f"\n開発セットの正解率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

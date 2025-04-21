import torch
from typing import List, Dict


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


def main():
    # テスト用のバッチデータ
    batch = [
        {
            "text": "hide new secretions from the parental units",
            "label": torch.tensor([0.0]),
            "input_ids": torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594]),
        },
        {
            "text": "contains no wit , only labored gags",
            "label": torch.tensor([0.0]),
            "input_ids": torch.tensor([3475, 87, 15888, 90, 27695, 42637]),
        },
        {
            "text": "that loves its characters and communicates something rather beautiful about human nature",
            "label": torch.tensor([1.0]),
            "input_ids": torch.tensor(
                [4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964]
            ),
        },
        {
            "text": "remains utterly satisfied to remain the same throughout",
            "label": torch.tensor([0.0]),
            "input_ids": torch.tensor([987, 14528, 4941, 873, 12, 208, 898]),
        },
    ]

    # collate関数を適用
    result = collate(batch)

    # 結果の表示
    print("パディングされた入力テンソル:")
    print(result["input_ids"])
    print("\nラベルテンソル:")
    print(result["label"])


if __name__ == "__main__":
    main()

"""
既存チェックポイント(全7銘柄で学習済み)から、指定した銘柄を除いた新しい
チェックポイントを作る。銘柄埋め込み層(symbol_embedding.weight)の該当行のみ
取り除き、その他の重み(ResNet50・GRU・fusion・出力ヘッド)はそのまま引き継ぐ
ことで、ゼロからの再学習ではなく継続学習として使えるようにする。

⚠️ model.SYMBOL_LISTの末尾の銘柄のみ安全に除外できる(除外しても残りの銘柄の
IDがズレないため)。現時点ではGOLDが末尾。途中の銘柄を除く場合はこのスクリプト
は使えない(残りの銘柄のIDがズレ、既存の学習結果の意味が壊れる)。

出力されるチェックポイントは重みのみ(optimizer/schedulerの状態は含まない)。
銘柄埋め込み層の形状が変わるためoptimizer状態はそのままでは使えず、train.py側も
この形式の読み込み時はoptimizer等を初期化する従来の挙動に自然と合致する。

使い方:
  python migrate_symbol_embedding.py --input saved_model/model_medium.pth \
      --output saved_model/model_medium_no_gold.pth --exclude GOLD

その後、以下のように継続学習できる(--exclude-symbolsは変換時と同じ指定にする):
  cp saved_model/model_medium_no_gold.pth saved_model/model_medium.pth
  python train.py --mode medium --exclude-symbols GOLD
"""
import argparse

import torch

from model import SYMBOL_LIST, get_active_symbol_map


def migrate_checkpoint(input_path, output_path, exclude_symbols):
    exclude_symbols = set(exclude_symbols)

    # 末尾からの除外のみ安全(詳細はmodel.get_active_symbol_map参照)
    tail_len = len(exclude_symbols)
    tail_symbols = set(SYMBOL_LIST[len(SYMBOL_LIST) - tail_len:]) if tail_len else set()
    if not exclude_symbols or exclude_symbols != tail_symbols:
        raise ValueError(
            f"SYMBOL_LISTの末尾からの除外のみサポートしています。"
            f"SYMBOL_LIST={SYMBOL_LIST} に対し、除外対象({sorted(exclude_symbols)})が"
            f"末尾と一致しません。途中の銘柄を除外すると、残りの銘柄のIDがズレて"
            f"既存の学習結果の意味が壊れるため、この移行方法は使えません"
            f"(ゼロからの再学習が必要です)。"
        )

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if "symbol_embedding.weight" not in state_dict:
        raise ValueError(
            f"{input_path} に symbol_embedding.weight が見つかりません。"
            f"MultimodalFXmodelのチェックポイントではない可能性があります。"
        )

    old_embedding = state_dict["symbol_embedding.weight"]
    old_num_symbols = old_embedding.shape[0]
    if old_num_symbols != len(SYMBOL_LIST):
        raise ValueError(
            f"入力チェックポイントは既に{old_num_symbols}銘柄分の埋め込みしか持っていません"
            f"(全{len(SYMBOL_LIST)}銘柄が前提)。既に銘柄を除外した後のチェックポイントに"
            f"対して、このスクリプトを重ねて使うことは想定していません。"
        )

    _, new_num_symbols = get_active_symbol_map(exclude_symbols)

    new_state_dict = dict(state_dict)
    new_state_dict["symbol_embedding.weight"] = old_embedding[:new_num_symbols].clone()

    torch.save(new_state_dict, output_path)
    print(
        f"✅ {input_path} ({old_num_symbols}銘柄) から {sorted(exclude_symbols)} を除いた"
        f"重みのみのチェックポイントを保存しました: {output_path} ({new_num_symbols}銘柄)"
    )
    print("ℹ️ optimizer/schedulerの状態は含まれていません。train.py側で自動的に初期化されます。")


def main():
    parser = argparse.ArgumentParser(
        description="既存チェックポイントから指定銘柄を除いた重みのみのチェックポイントを作る"
    )
    parser.add_argument("--input", type=str, required=True, help="変換元のチェックポイント(.pth)")
    parser.add_argument("--output", type=str, required=True, help="保存先のパス(.pth)")
    parser.add_argument("--exclude", type=str, required=True,
                         help="除外する銘柄をコンマ区切りで指定(例: GOLD)")
    args = parser.parse_args()

    exclude_symbols = {s.strip() for s in args.exclude.split(",") if s.strip()}
    migrate_checkpoint(args.input, args.output, exclude_symbols)


if __name__ == "__main__":
    main()

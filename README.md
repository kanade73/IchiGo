# IchiGo

論理ゲート型ニューラルネット（Deep Differentiable Logic Gate Network の囲碁向け独自構成）で動く囲碁 AI の実装です。
Apple Silicon Mac で推論・探索・GTP 対局を行い、学習は PyTorch（CPU または CUDA、単一サーバーの複数 GPU）で行います。
当面の目標は **9 路 CGOS への出場**、19 路へ拡張できる設計を保ちます。

**入口: [仕様書と実装順序](docs/spec/00-overview.md)** / **進捗: [docs/implementation-status.md](docs/implementation-status.md)**

## 現状（2026-09-08）

| 段階 | 状態 |
|---|---|
| M0 数値基盤（16 ゲート、32ch 特徴、`.ichigo` 形式、CPU 推論の Python/Swift 一致） | 完了 |
| M1 学習（SGF→特徴、KataGo 教師ラベル、shard、学習/再開/離散化、16 局面過学習検収） | 完了。9 路 96k 局面の本ラン実施中 |
| エンジン（evaluator、pure-tree PUCT、GTP、自己対局） | 実装済み。時計制御・Metal 推論・CGOS 接続は未実装 |
| 棋力 | 未評価。policy は学習しているが value head は設計変更（head v2）を検証中 |

学習済みモデル・データ・run 成果物はリポジトリに含めていません（`.gitignore` 参照）。

## 構成

```text
Sources/IchiGoCore       囲碁ルール一式（RinGo から移植、docs/provenance/ringo-import.json）
Sources/IchiGoFeatures   座標、局面 snapshot、32ch 特徴、D4、SGF リプレイ
Sources/LogicModel       .ichigo loader、UInt8 ゲート推論、FP32 head、後処理
Sources/LogicMetal       Metal デバイス検出（カーネルは未実装）
Sources/IchiGoEngine     evaluator actor、白視点 adapter、PUCT 探索
Sources/IchiGoGTP        GTP 状態機械、自己対局
Sources/ichigo           CLI: doctor / inspect / eval / features / gtp / selfplay
Training/ichigo_train    Python: ゲート oracle、配線、モデル、教師 adapter、dataset、学習、export
Scripts/                 pilot、parity 検査、fixture 生成
configs/                 モデル・学習・教師の設定例
Tests/                   Swift テストと共有 fixture
```

## 必要なもの

- macOS 14 以降、Swift 6.2（SwiftPM）
- [uv](https://docs.astral.sh/uv/)（Python 3.12 と torch を `Training/uv.lock` どおりに解決。Linux では cu128 版 torch）
- 教師ラベル生成には [KataGo](https://github.com/lightvector/KataGo) 1.18.2 と公開ネットワーク（`configs/teacher-katago-analysis.cfg`）

## 使い方

```sh
make check-cpu     # Swift tests + pytest（GPU 不要）
make parity-cpu    # Python hard → Swift scalar の全論理層 bit 一致・head 許容誤差・ichigo eval 比較
swift run -c release ichigo doctor
swift run -c release ichigo features --sgf-dir KIFU_DIR --out data/positions-9.jsonl --size 9
cd Training && uv run python -m ichigo_train label --positions ../data/positions-9.jsonl \
    --teacher-bin katago --teacher-model MODEL.bin.gz --teacher-config ../configs/teacher-katago-analysis.cfg --out ../data/labels-9.jsonl
cd Training && uv run python -m ichigo_train build-data --positions ../data/positions-9.jsonl --labels ../data/labels-9.jsonl --out ../data/dataset-9
uv run --project Training python -m ichigo_train train --config configs/train-9.json      # 設定の相対パスはリポジトリ直下基準
uv run --project Training python -m ichigo_train export --checkpoint runs/small-9/checkpoint-best-hard.pt --out models/small-9.ichigo
swift run -c release ichigo gtp --model-9 models/small-9.ichigo --visits 100
```

pilot 一式は `Scripts/train_pilot.sh`（16 局面過学習検収 → 100 step → 2000 step → export → Swift parity）。

## 文書

| 文書 | 内容 |
|---|---|
| [00-overview](docs/spec/00-overview.md) | 決定事項、構成、範囲、段階別成果 |
| [01-network](docs/spec/01-network.md) | 論理ゲート、入力、ネット構造、CPU/Metal、モデル形式 |
| [02-training](docs/spec/02-training.md) | データ、教師、学習、離散化、複数 GPU、強化学習 |
| [03-engine](docs/spec/03-engine.md) | RinGo の移植、探索、勝率、GTP、CGOS |
| [04-tasks](docs/spec/04-tasks.md) | 実装チケット、依存関係、検収条件 |
| [05-validation](docs/spec/05-validation.md) | 数値検証、棋力評価、性能、リリース条件 |
| [06-evidence](docs/spec/06-evidence.md) | 調査根拠、参照ファイル、未検証事項 |
| [implementation-status](docs/implementation-status.md) | 完了チケット、実行した検証、結果、未解決事項 |

## 出所と謝辞

- `Sources/IchiGoCore` と `SGFReader.swift` は同作者の RinGo（katago-mlx）から移植したもので、元の著作権表示を保持しています。ファイル単位の SHA256 は `docs/provenance/ringo-import.json`。
- ルールエンジンは KataGo の C++ 実装を Swift へ移植した RinGoCore に基づきます。教師ラベルは KataGo analysis engine と公開ネットワークで生成しています。
- 方式の根拠は [DLGN](https://arxiv.org/abs/2210.08277)、[Convolutional DLGN](https://arxiv.org/abs/2411.04732)（詳細は 06-evidence）。

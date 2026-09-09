# IchiGo 進捗総括（2026-09-08 〜 2026-09-10 早朝）

時系列の作業ログは [implementation-status.md](implementation-status.md)。本書は「何ができて、何が分かり、何を決めたか」をまとめたもの。

## 1. 到達点

| 段階 | 状態 | 根拠 |
|---|---|---|
| M0 数値基盤 | 完了 | 16 ゲート、32ch 特徴、`.ichigo` 形式、Python hard ↔ Swift scalar の全層 bit 一致（`make parity-cpu`） |
| M1 学習する 9 路 | 完了 | 16 局面過学習検収（3 seed 合格）、KataGo 教師ラベル 96k 局面、prefix 離散化、export、uniform baseline 100 局全勝 |
| M2a 9 路 CGOS 準備 | 完了 | 時計・watchdog、Metal/packed 推論、ベンチ、DDP 1/2/4 GPU 検収、校正、CGOS ローカル統合（fake server）、対局 runner、release パッケージ（`make release-check` PASS） |
| M2b 公開 CGOS 出場 | 未着手 | アカウント・接続先の設定待ち（`docs/runbook.md` 実 CGOS 節） |
| M2c 19 路 | 未着手 | コードは両サイズ対応、学習は 9 路のみ |

現チャンピオン候補: `models/p4-local-wide512-200k.ichigo`（512ch × 8 層、局所配線、gateLR 0.1、200k step）。hard policy top1 0.390、expected MAE 0.291。旧モデル（small gl10）に 63-6-31（paired 95% CI [0.57, 0.745]）。

## 2. システム構成（実装済み）

- **Swift**: IchiGoCore（RinGo 移植のルール）、IchiGoFeatures（座標、snapshot、32ch 特徴、D4、SGF リプレイ、`ichigo features`）、LogicModel（loader、UInt8 scalar oracle、packed CPU、Accelerate head、後処理・校正温度）、LogicMetal（byte/packed ゲート kernel、Metal heads、backend selector）、IchiGoEngine（evaluator actor、白視点 adapter、pure-tree PUCT、reservation batch、tree reuse、時計、DeadlineController）、IchiGoGTP（必須コマンド、kata-genmove_analyze、selfplay）、CLI `ichigo doctor/inspect/eval/features/gtp/selfplay/benchmark`。
- **Python（uv、Training/）**: ゲート oracle、配線生成（bank1 割合・配線 seed・learned-k）、モデル（head v1/v2、Gumbel-STE、tau/エントロピー knobs）、教師 adapter（KataGo analysis JSONL、idle timeout、視点変換）、shard dataset（D4 対称 splitKey、重複排除、sha 検証）、学習ループ（accumulation、prefix 60/30/10、hard/soft validation、best-hard 選抜、resume、SIGINT）、DDP（torchrun、global 正規化、rank0 I/O）、export/inspect/evaluate/calibrate、match runner（paired bootstrap）、実験ランチャー、CNN 診断 baseline。
- **運用**: `Scripts/cgos/`（client、fake server、37 テスト）、`Scripts/release/`（build/verify/MANIFEST）、`docs/runbook.md`。
- **テスト**: Swift 163（CPU）+ 24（Metal）、pytest 193、cgos 37。全 green。

## 3. 性能（Apple M5）

| 項目 | 値 |
|---|---|
| 推論 B=32（small 9 路） | cpu-packed 15,386 局面/秒、metal-packed 11,599（B=64: 20,536）。golden CPU 341 |
| 探索 overhead | fake evaluator で 9 路 4,509 visits/秒、19 路 1,444 |
| 実対局（wide512、Metal auto、1.2 秒/手） | 約 4,100 visits |
| 学習（A4000 1 枚、small、batch 128） | 0.07 秒/step（データ供給修正前 0.9〜1.4） |
| DDP | 2 GPU 0.061 秒/step（1.79 倍）、4 GPU 0.035 秒/step（3.1 倍、efficiency 0.78）。NCCL P2P は要無効化 |
| 教師ラベル | A4000 1 枚 24 局面/秒（Mac Metal 3.4）。108k 局面を 10 GPU で 8 分 |

## 4. 学習実験の結論（9 路、96k 局面、hard validation。基準: 一様 policy CE 3.94、定数 0.5 の expected MAE 0.347）

| 実験 | policy top1 | expected MAE | 結論 |
|---|---|---|---|
| 初期設定（gateLR 0.01、20k） | 0.289 | 0.330 | |
| gateLR 0.1（100k） | 0.364 | 0.301 | **最も効いた変更** |
| gateLR 0.3 | 0.357〜0.381 | 0.293〜0.309 | 0.1 と同程度 |
| 局所配線（dilation 全 1） | 0.369 | 0.294 | 遠距離接続は不要 |
| 幅 512 × 8（局所、200k） | **0.390** | 0.291 | 最良 |
| base 512 × 12 | 0.371 | 0.300 | 深さは効かない |
| 16 層 | 0.360 | 0.289 | 学習が遅い |
| head v2（zbar・ownMean を global head へ） | 差なし | 差なし | 集約の問題は未解決 |
| Gumbel-STE（small / base） | 0.278 / 0.323 | 0.314 / 0.311 | prefix 方式より劣る |
| tau 0.5 + エントロピー罰則 | 0.368 | 0.295 | soft-hard 差は早期に消えるが最終値は同じ |
| learned-k 配線（途中） | 0.28〜0.32 | — | 固定配線より低い |
| **診断用 CNN（64ch、4 residual、同 head/損失）** | **0.528** | **0.159** | データ・ラベル・head は健全。論理ゲート層側の限界 |

value の切り分け（学習なし）: ownership 総和 → sigmoid で MAE 0.250（NN head 0.234 と同水準、局面集合 359）。z_xy の 3×3 領域平均への線形プローブ 0.274 vs 盤全体平均/最大 0.306。**情報は局所 head まで来ており、global 集約で失われている。**

## 5. 決めたこと・方針

- value は探索の中核（policy なしでも MCTS は成立するが value なしでは成立しない）。PUCT が要求するのは「白視点の一貫した期待得点」だけなので、NN wdl head に限定せず、ownership 由来 value・領域プーリング head・プレイアウトも候補として比較する。
- 実験は「今の常識」に縛らず広く回す。計算資源が空いていれば実験を埋める。
- small モデルでは DDP より 1 GPU × 多構成が効率的。DDP は大型・長時間ランに使う。
- 教師は g170e-b20c256x2（KataGo 1.18.2、128 visits）で固定。CNN baseline は本番モデルにしない（Swift 推論経路がない）。

### value 代替の対局検証（2026-09-10 06:00）

`--value-source ownership`（Σownership+komi の sigmoid）は探索で明確に負け（対 network 0.215、CI [0.14, 0.29]）、50/50 混合でも 0.425（CI [0.34, 0.51]）。局面単位の MAE では同水準でも、探索の葉（対局途中の未整理な局面）では NN wdl head の方が一貫している。ownership 由来 value は不採用（インフラは残す）。head v3 の学習結果待ち。

### value の到達点（2026-09-10 夜）

| 構成 | policy top1 | expected MAE |
|---|---|---|
| head v2 系の最良（wide512 局所 200k） | 0.390 | 0.291 |
| head v3 small 200k | 0.367 | 0.267 |
| head v3 wide512 200k | 0.389 | 0.272 |
| head v3 wide512 + 3 倍データ 200k | 0.385 | **0.259** |
| C=2048 head v3（35k 途中） | 0.392 | — |
| CNN baseline | 0.528 | 0.159 |

分かったこと: value に効くのは「head の空間分解能（3×3 領域プーリング）」と「データ量」で、ゲート数は policy に効く。Gumbel-STE、配線学習、深さ、ownership 由来 value、プレイアウト value は効かない（後者 2 つは対局で大敗）。5×5 プーリングはプローブで過学習し保留。データ量は value にだけ効くので、value 目的の追加ラベルは安価。本命構成「C=2048 + head v3 + 3 倍データ」を学習中。

## 6. 進行中（2026-09-10 早朝）

- headVersion 3（3×3 領域プーリング）の実装 → phase 7 学習。
- `--value-source ownership|blend` の実装と A/B 対局（学習不要）。
- phase 6: 訓練データ半分、500k step、gl0.3+bank30。3 倍データ（5,000 局追加ラベル、約 25 万局面）でのデータ量スケーリング。
- learned-k、16 層 × 512 の最終値。

## 7. 未解決・リスク

- value の根本改善は未達。3 倍データ・head v3・ownership value の結果で「大規模化で伸びるか、方式変更が要るか」を判断する。
- 19 路は未学習・未検収。探索は 19 路で遅い（GameState コピー、合法性計算）。
- 校正温度は実対局結果 100 局未満で未検証（T=1 運用）。
- CGOS 実サーバー未接続。対局中のエンジンクラッシュで client が終了する（無人運用には supervisor が必要）。
- self-play は温度 0 で決定的（T35 未実装）。RL（T35/T36）は value 改善後。

## 8. 運用メモ

- サーバー: A4000 × 10、プロジェクトはローカルディスク配下、`Scripts/launch_experiments.py {launch,status,relaunch} --matrix configs/experiments/...`。
- 手順: `docs/runbook.md`。リリース: `Scripts/release/build.sh MODEL` → `verify.sh`。
- 対局比較は必ず固定開局（`configs/openings/`）＋色交換。開局なしは決定的で同一棋譜になる。

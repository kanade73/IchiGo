# IchiGo 進捗総括（2026-09-08 〜 2026-09-23）

時系列の作業ログは [implementation-status.md](implementation-status.md)。本書は「何ができて、何が分かり、何を決めたか」をまとめたもの。§0 が最新の要約、§1 以降は 09-10 時点の記録に追記している。

## 0. 2026-09-23 時点のまとめ

### 9 路はひとまず動く

- 公開 CGOS（yss-aya.com:6809、5 分切れ負け、komi 7）に `RinGo-LGN` で出場した（2 局、相手は 2,900 前後の KataGo 系で 2 敗）。client・release パッケージ・supervisor（`runs/cgos/<name>/run.sh`、gitignore）・投了（`--resign-threshold`）まで実運用で通っている。
- 外部基準: gnugo 3.8 level 10（`--capture-all-dead`）と 800 visits で 10-0-10。CGOS の Gnugo アンカーは 1800 なので、その近辺。
- 対局用の推奨設定: `ichigo gtp --model-9 <model> --backend cpu-packed-mt --leaf-batch 64 --pipeline --resign-threshold 0.03`（GPU を使わない。理由は下の「読みの量より知識」）。モデルは死活ベンチ最良の `p11-wide512-headv3-x3data-dagger-v2-200k`（featureVersion 2）。チャンピオン `p4-local-wide512-200k` に持ち時間をそろえて 23-3-14（0.61、CI [0.46, 0.75]）で、しかも v2 特徴の計算が 1 スレッドで遅く読みは 6 分の 1 だった（高速化が次の課題）。

### 強さ（品質）について分かったこと

1. **弱点は死活。** CGOS の 2 敗はどちらも「教師（KataGo）は死と見る石を生きと読んだ」（1 局目は policy 1 位の正着 B8 を、value/ownership の誤認で探索が捨てた）。自分の対局 728 局から作った死活ベンチ（`Scripts/death_bench.py`、6,526 局面、対局単位で学習と分離）では、既存モデルは教師上の死石の 29〜35% しか死と見ず、生き石は 97〜99% 正しい＝「生き」側への偏り。幅・head v3・3 倍データは policy/value を改善してきたが、この数字はほとんど動かさなかった。
2. **知識を足すと効いた。** 2 入力ゲートは連に沿って情報を集められない、という仮説に沿って (a) featureVersion 2（シチョウ、Benson の pass-alive、呼吸点 4 以上、眼領域、二眼候補を、履歴 t=3〜7 の 10 チャンネルと置き換え。32ch の形は同じ）、(b) 自分の対局の局面を CUDA 教師でラベルして学習に足す（DAgger 型、1 回目は 4.3 万局面）を入れた。どちらも効き、足し算になる:

   | モデル（head v3） | 死石の検出率 | 丸ごと見逃した局面 | value MAE（ベンチ） |
   |---|---|---|---|
   | small v1（100k） | 0.290 | 922 / 3,382 | 0.264 |
   | small v2 + 自分の対局 | **0.456** | 673 | 0.220 |
   | wide512 v1（200k） | 0.346 | 772 | 0.252 |
   | wide512 v2 | 0.440 | 665 | 0.211 |
   | **wide512 v2 + 自分の対局** | **0.462** | **588** | **0.191** |

   small 同士の対局（各 800 visits、40 局）: v2 + 自分の対局 vs v1 = 29-3-8（0.76、CI [0.66, 0.86]）。wide512 v2 は同じ検証局面で hard top1 0.385 → 0.403、expected MAE 0.259 → 0.238 と通常指標も改善した。学習前の線形プローブ（ownership に v2 平面を足すと死石判別の AUC 0.825 → 0.909）が良い予告になった。
3. **読みの量より知識で頭打ち。** CPU 多コア推論（`cpu-packed-mt`）、大きなバッチ、評価と準備の重ね合わせ（`--pipeline`）で探索は最大 1,100 → 23,700 visits/秒になったが、評価待ち 512 葉では仮想損失で探索が崩れ、同じ visits でバッチ 8 に 3-37。質を落とさないバッチ 64 + pipeline（13,300 visits/秒）で持ち時間をそろえると従来既定に 22-3-15（0.59、CI [0.46, 0.71]）、3.4 倍読んで +60 Elo 程度。今の評価が当てにならない局面では、読みを増やしてもあまり伸びない。
4. **RinGo の補助としての LGN は見込みが薄い。** RinGo champion（b20c256）と同じ Mac で、LGN を CPU 8 コアで回しても RinGo の GPU 評価は 3〜7% しか落ちない（実探索は木の処理が CPU なので約 15% 落ちる）。ただし品質では、RinGo の深い探索の最善手が上位 8 手に入る率が RinGo 生ネット 0.98 に対し LGN 0.85〜0.87、RinGo の誤りの予測に LGN を足しても AUC は増えない（0.741 → 0.738）。「難所の検出」「ルートの篩い」は不採用。強さでは RinGo（CGF 2026 9 路 7 位）との差は大きいまま。

### 次の手

- 自分の対局の教師化を繰り返す（新モデルで対局 → CUDA 教師 → 再学習）。連単位の OR プーリング層（連に沿った集約を正確な演算で与える）、C=2048 + v2 + 自分の対局、恒等初期化の割合を上げる実験。
- 19 路（UEC 杯 2026-11-28/29、19 路・30 分切れ負け・日本ルール・コミ 6.5・400 手）に IchiGo で出る方針。KataGo 公開学習棋譜（katagoarchive.org、1 日約 10 万局の 19 路）から 3 万局を選び、`ichigo features --size 19 --feature-version 2 --sample-per-game 15` で約 45 万局面を抽出中。教師は kata1-b18c384nbt、ラベルのルール（日本ルールにするか）は決定待ち。

## 1. 到達点

| 段階 | 状態 | 根拠 |
|---|---|---|
| M0 数値基盤 | 完了 | 16 ゲート、32ch 特徴、`.ichigo` 形式、Python hard ↔ Swift scalar の全層 bit 一致（`make parity-cpu`） |
| M1 学習する 9 路 | 完了 | 16 局面過学習検収（3 seed 合格）、KataGo 教師ラベル 96k 局面、prefix 離散化、export、uniform baseline 100 局全勝 |
| M2a 9 路 CGOS 準備 | 完了 | 時計・watchdog、Metal/packed 推論、ベンチ、DDP 1/2/4 GPU 検収、校正、CGOS ローカル統合（fake server）、対局 runner、release パッケージ（`make release-check` PASS） |
| M2b 公開 CGOS 出場 | 出場済み（09-23） | `RinGo-LGN` で 2 局。投了対応、supervisor 運用（§0） |
| M2c 19 路 | 着手（09-23） | データ準備中（§0 次の手）。コードは両サイズ対応 |

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

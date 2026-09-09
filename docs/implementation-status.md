# IchiGo 実装状況

更新: 2026-09-08（M0 完了後、M1 の T11〜T17 を追記）。仕様は `docs/spec/`。本書は「実装したもの」「実際に実行した検証」「未解決事項」を分けて記録する。
ランダム初期化モデルの完走は学習成功や棋力を意味しない（M0 は棋力主張なし）。

## 環境

- Mac: Apple M5、10 cores、32GB、macOS 26.3.1(a)、Swift 6.2.3（`swift-tools-version 6.0`、deployment target macOS 14）。
- Python: uv 0.10.2、Python 3.12.13、torch 2.14.0（CPU）、numpy 2.5.3、pytest 9.1.1。`Training/uv.lock` に固定。
- CUDA / A4000: 本セッションでは利用不可。CUDA 関連は未検証。
- Metal: デバイス検出のみ（`ichigo doctor`）。Metal カーネルは T22 以降。

## 完了チケット（M0）

| チケット | 内容 | 主な成果物 |
|---|---|---|
| T01 | 骨格 | `Package.swift`（Core/Features/LogicModel/LogicMetal/Engine/GTP/ichigo）、`Training/pyproject.toml` + `uv.lock`、`Makefile`、`.gitignore` |
| T02 | RinGoCore 移植 | `Sources/IchiGoCore/*`（13 ファイル、無改変コピー）、`Tests/IchiGoCoreTests/*`（import 名のみ置換）、`docs/provenance/ringo-import.json`（39 ファイルの sha256、HEAD `9d07c47c…`、dirty=true） |
| T03 | 座標・snapshot | `IchiGoFeatures/Coordinates.swift`、`PositionSnapshot.swift`（`GameState` + 値型 `PositionSnapshot`、7 手分の配置を自前保持） |
| T04 | 32ch + global | `FeatureEncoder.swift` |
| T05 | D4 と合法手 mask | `FeatureSymmetry.swift`、`ichigo_train/symmetry.py`、共有 fixture `Tests/Fixtures/symmetry/perm-{9,19}.json` |
| T06 | Python ゲート oracle | `ichigo_train/gates.py`、`tests/test_gates.py` |
| T07 | 配線・モデル生成 | `wiring.py`、`model.py`、`configs/model-{tiny,small,base}.json` |
| T08 | Swift 離散ゲートと head | `LogicModel/ScalarBackend.swift`、`Heads.swift`、`Postprocess.swift` |
| T09 | exporter | `model_format.py`、`export.py`、CLI `init-model` / `export` / `inspect` |
| T10 | Swift loader + CLI | `ModelManifest.swift`、`ModelLoader.swift`、`SHA256.swift`、`ichigo doctor/inspect/eval`、`Scripts/check_eval_parity.py` |

## 実行した検証と結果

すべて 2026-09-08 にこの Mac 上で実行。

| コマンド | 結果 |
|---|---|
| `swift test --filter IchiGoCoreTests` | 67 tests、0 failures（RinGo 由来の Board/Area/Ladder/Rules/NNInputs golden 全通過） |
| `make check-cpu` | Swift 107 tests + pytest 38（M0 時点）。M1 追記時点: Swift 117 tests、0 failures + pytest 67 passed |
| `make parity-cpu` | fixture 再生成 → Swift `ParityTests`（tiny-9 B=4、tiny-19 B=2）全論理層 bit 完全一致、head は `1e-4+1e-4*|ref|` 以内、後処理 1e-5 以内 → `ichigo eval` 経由の全サンプル比較 PASS（9 路 4 sample、19 路 2 sample）。exit 0 |
| `ichigo doctor` | arm64、10 cores、Metal "Apple M5" 検出、JSON 出力 |
| 参照リポジトリ | `git status --porcelain` 74 行（セッション開始前からの未コミット変更）、セッション開始以降に更新されたファイルなし。HEAD 不変 |

検収条件の対応:

- T03: A9→index0、J1→80、A19→0、T1→360、I 列拒否、snapshot 後の盤変更が snapshot に影響しない（`CoordinateTests`、`FeatureTests.testSnapshotIsIndependentOfLaterMoves`）。
- T04: 9 路空盤、19 路空盤（白番）、取石（pass 含む 7 手履歴）、単純コウ（ch18 と ch17、呼吸点 1/2/3+）、19 路自殺点、2 pass、履歴不足、positional superko（RinGo の二子送り一子返し参照局面を 9 路右端へ埋め込み、単純コウ点なしで ch17=0）を手計算 fixture で全 32ch × 全点を検証。
- T05: 8 変換の逆写像で完全一致、pass 不変、Python/Swift が同一 JSON permutation を再現。
- T06: 16×4 真理値、a/b 入替、直接 16 和と `[C,4]` 縮約の forward/gradient 一致、float64 gradcheck、argmax tie 最小 ID。
- T07: seed 再現、盤外 0、9/19 両 forward（tiny/small）、head shape、bank/offset/A=B の拒否。
- T08/T10: 上記 parity。truncation、sha 不一致、version/featureVersion/rulesId 不一致、NaN manifest、gate>15、offset 範囲外、tensor 欠落/重複/重なり/未整列、`..` パス、symlink、channels 範囲外をすべて拒否（`LoaderRejectionTests` 14 件、Python 側 `test_model_format.py` 同等 11 件 + 4 件）。
- T09: 書いて再 load した hard 出力一致、`--overwrite` なしで拒否、失敗 export の一時ディレクトリ残存なし。

## 設計上の判断（仕様が明示していない点）

- 配線生成の乱数消費順序を `wiring.py` の docstring に固定した（層→channel の順に A（固定でなければ）→B、その後 theta）。保存された `wiring.i32` が正本であり、Swift は再生成しない。
- head 初期化の乱数は `PCG64(seed+1)`（配線 seed と独立）。
- `BoardHistory.numRecentBoards` は 6 で、7 手分の履歴に足りないため `GameState` が全手の配置を自前で保持する。
- `PositionSnapshot` の fingerprint は posHash・手番・単純コウ点・superko 禁止点集合・連続 pass 数・komi・手数の文字列。探索の重複排除（T20）でそのまま使えるが、cache key（03-4）は別途定義が必要。
- LogicModel は Foundation のみ依存とするため SHA-256 を自前実装（既知ベクトルで検証）。
- `.ichigo` の `trainingProvenance` は Swift 側では文字列 JSON として保持（Sendable のため）。
- `ichigo inspect/eval` はモデル不正を exit 2、推論失敗を exit 3 で返す。`eval` の position JSON は 02-2 の `boardSize/spatial/global/legal` のみ必須。

## 未解決・未検証

- `make check-metal` / `parity-metal` / `check-cuda` / `integration` / `release-check` は未実装で明示的に失敗する（T22+/T17/T26/T12+/T37）。
- CUDA/A4000、DDP、学習ループ（T14〜T17）は未着手・未検証。
- `IchiGoEngine` は `ModelCapabilities` のみ、`IchiGoGTP` は名前定数のみ（T18〜T21 で実装）。両テストターゲットはビルド確認用の smoke のみ。
- macOS 14 実機互換は未検証（deployment target のみ）。
- 教師 KataGo、RinGo 既存 `.nngd` コーパスの在庫調査は未実施（T12/T13）。
- Python `LogicNet` は soft/hard forward と prefix 凍結までを持つが、損失・optimizer・スケジュールは未実装。

---

## M1 前半（T11〜T17）: 2026-09-08 追記

### RinGo 既存データの在庫調査（02-11）

| 対象 | 結果 |
|---|---|
| `.nngd` v2 shard | 参照リポジトリの `.tools/` 配下に smoke 用 shard 5 本のみ（cq-smoke / wp2a / wp2b / deep50-val）。局面索引・gameId・対応棋譜の記録なし。`.tools/` は仕様でコピー禁止 |
| 棋譜 | `katago-mlx/kifu/` に 49,893 局。すべて 9 路、RU[Chinese]、KM[7]、置石なし。投了 35,703 局、点数決着 14,190 局、平均 48.1 手、総手数 2.40M |
| 生成設定・sample 対応 | 存在しない。`ringo makedata` の列挙順・skip・symmetry を再現して照合する根拠資料がなく、11.2 の「全 sample 一致の証明」ができない |

判断: 経路 1（既存 target の再利用）は対応を証明できないため採用せず、経路 2（既存棋譜を固定教師で一度だけラベル化）を採用。再利用できた既存ラベルは 0 件。importer（`import-ringo`）と v2 reader は契約どおり実装し、合成 v2 shard で検証した（mapping ファイルが与えられた場合のみ動作し、推測しない）。

### 教師

- KataGo 1.18.2（Homebrew、Metal backend）。ネットは Homebrew 同梱の公開モデル `g170e-b20c256x2-s5303129600-d1228401921.bin.gz`（sha256 `7c8a84ed9ee737e9c7e741a08bf242d63db37b648e7f64942f3a8b1b5101e7c2`）。`kata1-b18c384nbt-s9996604416`（sha256 `9d7a6afe…`）も計測したが 2.2 局面/秒で、b20c256 の 3.0〜3.4 局面/秒を採用。run 途中で切り替えていない。
- 設定 `configs/teacher-katago-analysis.cfg`（`reportAnalysisWinratesAs = SIDETOMOVE`、8 analysis threads × 4 search threads）、128 visits、rules は明示 JSON（POSITIONAL / AREA / NONE / suicide=false / friendlyPassOk）。
- 視点の検証: 黒石 9 個・白番の probe 局面で winrate 0.0009、scoreLead −57.9、中央 ownership −0.98 を確認し、`Tests/Fixtures/teacher/`（実教師 20 局面 + probe の生応答）を fixture 化。
- adapter のタイムアウトは「応答ごとにリセットする idle 60 秒（初回は queue 深さ倍）」とした。1 クエリ = 1 局（最大 4S² turn）で、固定 60 秒/クエリでは queue 待ちで必ず超過して再送が連鎖したため（実際に停止を観測）。

### 完了チケット

| チケット | 成果物 | 検証 |
|---|---|---|
| T11 | `IchiGoFeatures/SGFReader.swift`（RinGo から移植、provenance 追記）、`PositionExport.swift`（replay、canonical positionId/gameId）、`ichigo features` | `SGFReplayTests` 5 件（Python と同一 hash、取石/pass、サイズ不一致/占有点/自殺/コウ/変化/重複配置の拒否、置石順序）。実コーパス 2,200 局 → 2,199 局 108,676 局面、1 局拒否（superko 違反、手数付き） |
| T12 | `ichigo_train/teacher.py`、`tests/fake_teacher.py`、`test_teacher.py`、`test_teacher_fixture.py` | 順不同・重複・欠落 turn の再送/拒否・timeout・視点変換（black/sidetomove）・違法手 visits・error 応答・NaN/長さ不一致を fake teacher で 9 件、実教師 fixture 2 件 |
| T13 | `dataset.py`（shard v1、D4 最小 gameId の splitKey、重複排除、sha 検証）、`ringo_import.py`（v2 reader/inventory/importer）、CLI `label` / `build-data` / `inventory-ringo` / `import-ringo`、`Scripts/generate_teacher_games.py`（未実行） | `test_dataset.py` 5 件（対称不変 splitKey、split 跨ぎなし、重複排除、shard 検証、holdout 空エラー、mask/wdl、v2 読み書き・v1/切詰め拒否・symmetry 逆変換・noResult 除外） |
| T14 | `losses.py`、`optim.py` | `test_losses.py` 5 件（手計算一致、全 mask 0 で NaN なし、違法手 logit、1 step 更新、schedule 形状） |
| T15 | `config.py`、`data_loader.py`、`checkpoint.py`、`metrics.py`、`train.py`、CLI `train` / `evaluate` | `test_train.py` 5 件（10 step と 5+resume 5 が bit 一致、dataset/wiring 不一致 resume 拒否、成果物一式、config 未知キー） |
| T16 | `discretize.py`、train.py の prefix 凍結・heads-only・best-hard 選抜・凍結前後 hard 検証と 20% 警告 | `test_discretize.py` 2 件、`test_train.py` の凍結テスト（凍結層 theta 不変、未凍結層は更新、最終 export が全層 argmax） |
| T17 | `configs/train-tiny.json`、`Scripts/make_overfit_fixture.py`、`Scripts/check_overfit_gate.py`、`Scripts/check_model_parity.py`、`Scripts/train_pilot.sh`、`configs/train-9*.json`、`reports/pilot/` | 下記 |

### T17 結果（この Mac、CPU）

16 局面過学習（tiny、effective batch 16、2000 step、augmentation なし、`runs/overfit-tiny-9*`）:

| seed | soft 達成 step（top1≥0.9 かつ MAE≤0.1） | 最終 hard top1 | 最終 hard MAE | 判定 |
|---|---|---|---|---|
| 20260908 | 500（top1 1.000、MAE 0.064） | 0.8125 | 0.063 | PASS |
| 1 | 1000 | 0.8125 | 0.093 | PASS |
| 2 | 500 | 0.9375 | 0.043 | PASS |

`reports/pilot/overfit-gate.json`。注意: 全 seed で hard は 13/16〜15/16 とぎりぎりであり、stage 1（tau=1）の間は soft top1 1.0 に対し hard top1 0.1〜0.4 と soft-hard 差が大きい。凍結と heads 再学習（stage 3）で hard が回復する。02-11.3 の Gumbel-STE 対照実験は未実施（`gumbel-ste-90-10` は config で拒否）。

small pilot（CPU、A4000 は利用不可）:

| run | データ | step | 秒/step | 合計 | 結果 |
|---|---|---|---|---|---|
| `small-9-pilot100-cpu` | dataset-9-a（1,599 局面） | 100 | 0.324 | 34 s | 速度計測のみ |
| `small-9-pilot2000-cpu` | dataset-9-b（train 6,706 / val 359 / test 447） | 2000 | 0.282 | 577 s（validation 12.7 s） | 下表 |

pilot2000 の hard（全層 argmax、best-hard = step 2000）:

| 指標 | validation | test | baseline（validation / test） |
|---|---|---|---|
| policy CE | 3.012 | 2.937 | uniform 合法 3.943 / 3.993（`reports/pilot/baseline-*.json`） |
| policy top1 | 0.173 | 0.204 | — |
| expected MAE | 0.320 | 0.374 | 定数 0.5: 0.328 / 0.377 |
| score MAE（目） | 5.47 | — | 定数 0: 5.47 |

判定: policy は uniform に対して CE 23.6% 減で 05-4 の 5% 基準を満たす。expected result は validation 0.320 vs 0.328、test 0.374 vs 0.377 と baseline をわずかに下回るだけで、value head は実質未学習に近い。score/ownership も baseline 同等。これは「学習が動く」ことの確認であり、M1 の学習成立ゲートは **policy のみ達成、value は未達** と記録する。データ 7.5k 局面・2000 step の pilot であり、本 run（10 万局面・20,000 step、GPU）は未実施。

Swift parity（学習済み hard モデルの export → `ichigo eval --dump-layers`）:

- `models/overfit-tiny-9.ichigo`: fixture 16 局面すべてで全 4 層 bit 一致、head 許容内（max policy logit diff 7.6e-6）。`reports/pilot/overfit-tiny-swift-parity.json`
- `models/small-9-pilot2000-cpu.ichigo`: validation 8 局面で全 8 層 bit 一致、head 許容内（1.9e-6）。`reports/pilot/small-9-pilot2000-cpu-swift-parity.json`

### 工程別の所要時間

| 工程 | 実測 |
|---|---|
| 棋譜生成 | 0（既存棋譜を使用） |
| 特徴抽出（`ichigo features`、release） | 2,199 局 108,676 局面で 26.5 秒（659 MB JSONL） |
| 教師ラベル生成 | 3.3〜3.4 局面/秒（b20c256、128 visits、M5 Metal）。8,000 局面まで 2,272 秒。108,676 局面全体は約 9 時間で、セッション終了時点も継続中（`data/labels-9.jsonl`、`data/labels-9.progress.log`） |
| dataset 構築 | 8,012 ラベル → 7,512 局面（重複除去後）数秒 |
| 生徒学習（small、CPU） | 0.28〜0.32 秒/optimizer step（effective batch 128） |
| 離散化・head 再学習 | 学習 step に含まれる（60/30/10） |
| 検証（hard+soft、1,024 局面上限） | 約 0.6 秒/回 |

### CUDA（A4000）で実行する手順（未実行）

```sh
cd Training && uv sync                       # CUDA 版 torch はサーバー側で uv の index を指定して解決する
uv run python -m ichigo_train doctor --out ../reports/cuda-doctor.json
cd .. && Scripts/train_pilot.sh              # 16 局面検収 → 100 step → 2000 step → export → Swift parity
uv run --project Training python -m ichigo_train train --config configs/train-9.json   # 本 run 20,000 step
```

`configs/train-9.json` は `data/dataset-9`（ラベル完了後に `build-data` で作成）を前提とする。CUDA 版 torch の lock、DDP（T26）、4GPU 検収は未実施。

### 追加の設計判断

- `label` は 1 局 1 クエリ（`analyzeTurns` に全 turn）で送り、`(id, turnNumber)` で結合する。
- `build-data` は positions を 3 パスで stream し、labels のみ dict 保持する。
- `fixtureMode` config キー（過学習 fixture 用、split しない）、`maxValidationPositions`、`tauFinal`、`freezeWarningThreshold` を config に追加（02-10 の例にない省略可能キー）。
- 凍結後の hard 悪化 20% 超は run-summary の warnings に記録するのみで、区間 2 倍の再 run は自動化していない。
- 教師選択: b18c384nbt より b20c256x2 を採用（ラベル数を優先）。両者の hash と速度を上記に記録。

### レビュー指摘の修正（同日）

export の置換原子性、fingerprint への全履歴/superko の包含、盤外座標・重複配置の拒否、manifest の overflow/1e300 温度、CLI の 0/1 検証を修正し、それぞれ回帰テストを追加した（`test_failed_rename_keeps_old_model`、`testFingerprintCoversMoveOrderAndHistory`、`testOffBoardCoordinatesRejected`、`testDuplicateInitialStonesRejected`、`testHugeByteLengthRejected`、`testCalibrationTemperatureNotFloatRepresentable`）。

### 未検証・未解決

- A4000/CUDA/DDP: 未実行。CUDA 版 torch の lock も未作成。
- 全 108k 局面のラベル生成はセッション中に完了しない。完了後 `build-data` で `data/dataset-9` を作る。
- value head が pilot で学習していない。原因切り分け（データ量・step 数・損失重み・head 容量）は本 run 前に行う。
- Gumbel-STE 対照実験、通常 CNN baseline（T33）は未着手。
- 過学習検収の hard 側は基準ぎりぎり（13/16）。
- `generate_teacher_games.py` は未実行（既存棋譜があるため）。
- T18 以降（evaluator、探索、GTP）は未着手。

---

## 2026-09-08 夜: A4000 本ランと T18〜T21

### 大学 GPU サーバー（A4000 ×10）

- A4000 ×10（16GB）、48 core、driver 570（CUDA 12.8 まで）。ホーム quota が逼迫していたため、プロジェクトと uv キャッシュを ローカルディスク上の作業ディレクトリ（ホームは symlink）に置いた。
- `Training/pyproject.toml` に Linux 限定の `pytorch-cu128` index を追加し、`uv.lock` を universal に更新（Linux: torch 2.11.0+cu128、macOS: 2.14.0）。`uv sync` で CUDA 10 枚を認識。
- KataGo 1.18.2 Linux cuda12.8 版を `tools/` に配置し、torch wheel 同梱の cudnn/cublas を `LD_LIBRARY_PATH` で使用。教師モデルは Mac と同じ g170e-b20c256x2（scp）。
- 教師ラベル: `Scripts/split_positions.py` で 10 chunk に分割し GPU ごとに並列実行。1 GPU あたり 24 局面/秒、全 108,676 局面を約 8 分で完了（拒否 0）。Mac の Metal 版（3.4 局面/秒、10,622 局面）は停止し、本番データはサーバー版で統一（同一モデル・同一 KataGo 版、backend は CUDA。Mac 版ラベルは本番データに混ぜていない）。
- dataset-9（サーバー）: 96,065 局面（train 85,470 / validation 5,407 / test 5,188、重複 12,611 除去）。
- 本ラン: `configs/train-9-gpu.json`（`train-9.json` と同じ effective batch 128、microBatch のみ 8→128。microBatch 8 では 1.41 秒/step・GPU 6% と CPU 律速だったため）。GPU 0 で 0.90 秒/step、20,000 step で約 5 時間、validation 込みで約 5.5 時間の見込み（21:55 開始）。GPU 使用率 21% で、データ供給（numpy 増強・転送）が律速。結果は `runs/small-9/` に出る。
- DDP（T26）は未実装のまま 1 GPU で実行。

### 完了チケット

| チケット | 成果物 | 検証 |
|---|---|---|
| T18 | `IchiGoEngine/LogicEvaluator.swift`（`PositionEvaluating` actor、snapshot→特徴→backend→mask）、`EvaluationAdapter.swift`（白視点、違法手 −1、raw WDL 保持、未推定分散=0） | `EvaluationAdapterTests` 4 件（黒 e=.8→.2、白→.8、draw→.5、score/ownership 反転、終局/サイズ不一致の拒否、実 tiny モデル） |
| T19/T20 | `IchiGoEngine/Search.swift`: pure-tree PUCT（03 §4 の式そのまま、cpuct 1.5、FPU 0、白視点 backup、root 初回評価 1 visit、terminal は Core の正確な結果）、leaf batch 8 と in-flight reservation（仮の負け値 −1 加重）、node 上限、tree reuse（fingerprint 一致時のみ）、generation ID | `SearchTests` 6 件（手計算の選択/バックアップ、draw terminal=0.5 かつ NN 未呼出、合法手のみ・9/19 完走・batch≤8、例外後の reservation 解放と二重加算なし、reuse と reset/違法手拒否、node 上限） |
| T21 | `IchiGoGTP/GTPEngine.swift`（必須コマンド一式、ID/空行/#/CRLF、genmove と kata-genmove_analyze の共通 commit、終局後 genmove=pass、undo、time_settings は sudden death のみ受理）、`SelfPlay.swift`、CLI `gtp` / `selfplay` | `GTPEngineTests` 4 件（プロトコル/エラー、play/undo/違法、analysis 書式、9/19 ランダム模型で 2 pass 終局）。実モデルで `ichigo gtp`（analysis 応答確認）と `ichigo selfplay` 2 局完走 |

RinGo の `Search.swift`（KataGo 全パラメータ移植、graph search 含む 2,400 行）はそのまま持ち込まず、03 §4 が規定する簡略式を新規実装した。actor 所有のノード、reservation 付き batch 収集、`makeMove` での木昇格という構造は RinGo に倣っている。`SearchSettings` の未対応機能（graph merge、LCB、uncertainty、dynamic score utility、ponder、book）は存在しない＝off。

### 既知の制限

- 探索の性能: 各ノードが `GameState` を deep copy し、snapshot 生成で全点の superko 合法性を計算するため、19 路で数 visit/秒程度。時計制御（T28）と Metal（T22）の前にプロファイルが必要。
- selfplay は温度 0・noise なしで決定的（seed は記録のみ）。T35 で温度/noise を入れるまで同一局が出る。
- resign off、時計は `time_settings`/`time_left` を受理するだけで着手時間に未反映（T28）。
- kata-genmove_analyze は最善候補 1 行 + root summary（仕様 §9 の初期制約どおり）。

---

## 2026-09-08 深夜: value head の切り分け（head v2、CNN baseline）

### 観測

| 実験（dataset-9-b: 7.5k 局面、2000 step、CPU） | policy top1 | expected MAE | Brier | 学習側 expected 損失 | score MAE |
|---|---|---|---|---|---|
| 常に 0.5 / 0 の基準 | — | 0.328 | 0.147（=ラベル分散） | 0.693（ln 2） | 5.47 |
| logic small、head v1 | 0.173 | 0.320 | — | 0.69 で固定 | 5.47 |
| logic small、head v2（zbar・ownMean を global head へ） | 0.212 | 0.322 | — | 0.69 で固定 | 5.5 |
| 診断用 CNN（64ch、4 residual、同じ head/損失）`Scripts/cnn_baseline_pilot.py` | 0.382 | **0.232** | **0.115** | 0.70 → 0.49 | 4.6〜6.0 |

サーバー本ラン（96k 局面、head v1、1 GPU）も 3000 step 時点で expected MAE 0.346 = 基準 0.3467、学習側 expected 損失 0.69 固定で同じ症状。policy top1 は 0.275（soft）まで上昇中。

### 切り分け結果

- value/score head の勾配は流れ、wdl logit は局面ごとに変動している（勾配停止ではない）。
- pooled 特徴（m, v、v2 では zbar, ownMean も）に対する閉形式の線形プローブでも validation MAE 0.306（train 0.261）で、head の入力自体に勝敗情報がほぼない。head v2 は効果なし。
- 同じデータ・同じ head・同じ損失の CNN は value を学習できる（MAE 0.328 → 0.232、Brier 0.147 → 0.115）。ownership MSE も 0.29 → 0.15、policy top1 も 0.21 → 0.38。
- 結論: **データ・ラベル・head 配線の問題ではなく、現在の論理ゲート層（small、固定疎配線、2000 step）が value に必要な大域的特徴を作れていない**。policy でも CNN に大きく劣る。

### 判断と次の手

- head v2 は仕様（01 §4）に headVersion 2 として残す（loader は 1/2 両対応、fixture は v2 と v1 の両方）。効果は未確認のため既定にする根拠はないが、v1 に戻す理由もない。
- 本ラン（20k step、head v1）は policy 確認のため継続。value は改善しない見込み。
- 次は表現力側の実験（02 §8 の順序）: 幅/深さ（base プロファイル）、配線 seed、配線の再探索（bank1 の割合、dilation）、学習 step の増加。比較は同一 holdout・同一 step で CNN baseline を対照にする。
- 探索が value 0.5 固定で動く現状では、対局は「policy の visits 分布で打つ」状態になる。

## 次に実装すべきチケット

本ラン完了後: `runs/small-9/checkpoint-best-hard.pt` を export → `Scripts/check_model_parity.py` → M1 ゲート（holdout の uniform CE 5% 減・MAE < 0.5 定数）→ 学習済み 9 路 20 局完走と uniform baseline 100 局（T31 の対局 runner が必要）。並行して T28（時計）、T22（Metal）、T26（DDP）。

### 2026-09-09 00:30 巡回

- 20k step 実験行列（96k 局面、GPU 1 枚ずつ）が完了: small v2 系は hard top1 0.30〜0.33（gateLR 0.03 が最良 0.331、policy CE 2.53）、expected MAE 0.323〜0.326（定数 0.5 基準 0.347、Brier 0.1375 vs 0.147）。value はわずかに動くが弱い。base（512×12）と wide（512×8）は同 step で小さいモデルより悪く、学習が遅い。
- 学習ループの CNN baseline（top1 0.21、MAE 0.330）は、optimizer が `heads` 以外の畳み込み重みを含めていなかった不具合で無効。`optim.py` を全パラメータ対象に修正（logic モデルには影響なし、26 テスト通過）。
- phase 2 として 100k step のラン 7 本（gateLR 0.03 系、bank30、gateLR 0.1、head LR 0.003、配線 seed 1、修正済み CNN baseline、200k step 版）を `configs/experiments/phase2/` で起動。旧コードの small-9-v1（GPU0、1.03 秒/step）は継続中。

### 2026-09-09 01:25 巡回

- phase 1 完了: base（512×12）と wide（512×8）も最終 hard top1 0.307〜0.308 で small 系（0.30〜0.33）と同水準。expected MAE 0.323〜0.326 で value は全構成とも弱い。
- **修正済み CNN baseline（100k step）: hard top1 0.528、policy CE 1.86、expected MAE 0.159、Brier 0.062、score MAE 4.17、ownership MSE 0.116。** 同じデータ・head・損失で value/score/ownership を明確に学習しており、論理ゲート網の表現力/学習性が劣ることが本番規模でも確定。
- phase 2 の logic 系（48k/100k step、stage 1 途中）は hard 指標が 20k 完了時より悪いが、これは soft 学習中の argmax 評価で離散化前のため。最終値で判断する。gateLR 0.1 は途中でも hard top1 0.326。
- 空いた GPU 2/8 に gateLR 0.1 系を追加（bank30、base プロファイル、各 100k）。クラッシュなし。

### 2026-09-09 02:25 巡回

- phase 2（100k step）完了: gateLR 0.1 が最良で hard top1 0.364、policy CE 改善、expected MAE 0.301（基準 0.347）、score MAE 6.57。gateLR 0.03 系は 0.347〜0.356、head LR 0.003 は悪化（0.330）。bank30/配線 seed の差は小さい。CNN baseline は 0.528 / 0.159 で依然大差。
- phase 3 を起動（各 100k、gateLR 0.1 基準）: gateLR 0.3、head LR 0.003、幅 512、tauFinal 0.1、200k step、dilation 全 1。継続中: gl03 200k、gl10 bank30、base gl10。small-9-v1（旧コード）は 14.5k step。
- 最良モデル `p2-small-gl10` を export し Mac へ取得、Swift parity を確認する。

### 2026-09-09 03:25 巡回

- クラッシュなし。phase 3（100k）は 48k 付近で進行中（gateLR 0.3 は途中 hard top1 0.349）、gl03 200k は 152k、gl10 200k は 48k。旧コードの small-9-v1 は 18k step を超え終盤（hard top1 0.266）。GPU 0/2 は完了して空き。

### 2026-09-09 04:25 巡回

- 完了: gl10-local（dilation 全 1）hard top1 0.369 / expected MAE 0.294 / score MAE 6.53（logic 系の最良）、gl10-bank30 0.370 / 0.299、gl03 200k 0.366 / 0.301、gl30 0.357 / 0.309、tau 0.1 0.360 / 0.302、head LR 0.003 0.345 / 0.308。旧コード small-9-v1（20k）0.289 / 0.330。
- 所見: 遠距離 dilation は現状の規模では寄与せず（局所配線が最良）、step 倍増（200k）の利得は小さい。CNN baseline（0.528 / 0.159）との差は依然大きく、配線・学習率の調整では埋まらない。
- phase 4（各 200k、gl10 + 局所配線基準）を空き GPU に起動: bank30、幅 512、16 層、gateLR 0.3、16 層×512。継続中: base gl10、wide gl10、gl10 200k。

### 2026-09-09 05:25 巡回

- クラッシュなし。完了: wide gl10（512×8、100k）hard top1 0.365 / MAE 0.307。phase 4 の 200k 系は 14k〜51k step で進行中（16 層×512 は 0.25 秒/step で約 13 時間かかる見込み）。base gl10 と gl10 200k は残り 1 時間以内。

### 2026-09-09 06:25 巡回

- クラッシュなし。完了: base gl10（512×12、100k）hard top1 0.371 / MAE 0.300 / score MAE 6.67（logic 系で top1 最良）、gl10 200k 0.364 / 0.289 / 6.46（value・score は最良）。phase 4 は 29k〜102k step で進行中。GPU 4/6/7/8/9 は空き。

### 2026-09-09 午前: phase 4 結果と phase 5 起動

- phase 4 完了分（200k step、gateLR 0.1、局所配線）: gateLR 0.3 版 hard top1 **0.381** / MAE 0.293（policy 最良）、bank30 版 0.377 / **0.290**。幅 512・16 層系は進行中（16 層は同 step で劣る）。
- phase 5（学習方式の実験、Codex 実装、pytest 115 件通過）を GPU 4/6/7/8/9 で起動: Gumbel-STE（small 局所 / base 局所）、tauStart 0.5 + gate エントロピー罰則 0.01、learned-k（K=8 候補からの配線学習）、learned-k + Gumbel。learned-k は 0.27 秒/step（K 倍の gather）。
- 新 config キー: `tauStart`、`gateEntropyWeight`、`wiringMode`（fixed / learned-k）、`wiringCandidates`、`wiringTau`。export は常に固定配線・argmax gate の v1 形式。

---

## T28: 時計・watchdog（2026-09-09 追記）

対象: `docs/spec/03-engine.md` §8。依存 T20/T21 完了済み。Swift のみ（`Sources/LogicModel`、`Sources/IchiGoFeatures` は不変更）。

### 成果物

| ファイル | 内容 |
|---|---|
| `Sources/IchiGoEngine/TimeManager.swift`（新規） | `MonotonicClock` protocol / `SystemMonotonicClock`、`TimeManager.reserve/estimatedMoves/budget/percentile95/stopMargin`（すべて純関数）、`validateSuddenDeath(byo:stones:)` |
| `Sources/IchiGoEngine/DeadlineController.swift`（新規） | `Search.run` を非構造化 `Task` ＋ `CommitGate`（actor）でwatchdog包装。fallback順: `savedRootCandidate`→`rootPolicyBestMove`→`legalMovesAscendingFallback`の先頭→pass |
| `Sources/IchiGoEngine/Search.swift` | `run(visits:deadline:)` 追加（deadline手前でnew batch停止、batch p95を記録）。`evaluateBatch` にgeneration引数を追加し、**mutate（expand/backup）前に**generation一致を確認して不一致なら黙って破棄＋reservation解放（従来はbackup後に判定していたため、actor reentrancy中に古い世代の結果が木へ書き込まれ得た）。`savedRootCandidate`/`rootPolicyBestMove`/`legalMovesAscendingFallback` を追加。`init` に `clock` パラメータ追加（デフォルト `SystemMonotonicClock()`）、既存呼び出しは無変更で動作 |
| `Sources/IchiGoGTP/GTPEngine.swift` | `time_settings` 受理時のみ `TimeManager.budget` からdeadline算出、`DeadlineController.run` 経由でgenmove/kata-genmove_analyzeをcommit。`time_settings`なしは従来通り固定visits・deadlineなし。budget/実測時間/timedOutをstderrへログ |
| `Tests/IchiGoEngineTests/TimeTests.swift`（新規、12 tests） | `FakeClock`（`sleep(until:)`をcontinuationで実装、`advance`/`set`/`waitForWaiters`）、`DelayedEvaluator`（同じclockでparkできるfake evaluator）を使用 |
| `Tests/IchiGoGTPTests/GTPEngineTests.swift`（2 tests追加） | `SlowEvaluator`（実時間delay）、`LogCapture` |

### 設計判断（仕様が明示していない点）

- **generation dropの挙動**: 遅い評価結果が「既存のgeneration guardを拡張」する形で、mutateする前にチェックする設計に変更。従来はbackup後にgenerationを確認していたため、makeMove/resetがreentrancy中に走ると古い世代の訪問数がすでに（再利用された可能性のある）木へ加算されてから初めて例外を投げていた。新設計では`evaluate`のawaitから戻った直後、expand/backupの前に一致確認し、不一致なら該当パスのreservationだけ解放して即returnする（例外を投げない）。`run`側の各`await`直後にも同様のガードを追加（`clock.now()`自体もactorのreentrancyポイントのため）。`run`自体は不一致を検知すると例外を投げる（そのgenerationにはもう属さない`root`を使って結果を返す意味がないため）が、この例外はDeadlineControllerが握りつぶすorphan taskからは誰にも観測されない。
- **watchdogの実装**: `withThrowingTaskGroup`ではなく非構造化`Task`を使用。理由: TaskGroupはクロージャがreturnする前に残っている子taskの完了を暗黙に待つため、キャンセルに応答しない（将来のMetal等の）evaluatorを待ってしまい、watchdogの意味が失われる。`CommitGate`（`tryCommit()`が最初の1呼び出しだけtrueを返すactor）で「search自身の完了」と「deadline watchdog」のどちらか一方だけが`Search.makeMove`を呼ぶことを保証。`makeMove`自体がgenerationを進めるため、勝者が確定した時点で遅延taskの結果は自動的に無効化される（別途invalidateは不要）。
- **fallback判定**: 3段とも同期的にroot状態を読むだけ（NN呼び出しなし）。tier1/2は`root.evaluated`/`edges`のvisits・prior比較（tie-breakは既存`chooseMove`と同じ: 大きい方優先、同点は小さいindex優先）、tier3は`root.state.snapshot().legal`から最小index。
- **GTPEngineでのvisits上限**: `time_settings`が有効な間はdeadlineが実質的な停止条件のため、`config.visits`ではなく`config.searchSettings.maxNodes`をvisits上限として渡す（無関係な小さい固定visitsでdeadline前に終わらないようにする）。
- **time_leftの優先順位**: 既存実装通り、`time_left`コマンドが`timeLeft[color]`を無条件上書きするため「サーバー値が優先」を自動的に満たす。genmove側は使った分だけ実測経過時間でローカル値を減算する。

### 検証

`swift test --filter 'IchiGoEngineTests|IchiGoGTPTests'`: 28 tests、0 failures（3回連続実行して安定を確認）。`swift test`（全target）: 144 tests、0 failures（既存ターゲットへの影響なし）。`swift build`: warning/error なし。

カバー内容: `R=0/0.05/1/300`秒のbudget手計算一致、reserve/estimatedMoves/stopMarginの境界値、`time_settings`のsudden-death以外拒否、fallback 3段の直接検証、実clockでdeadline手前にnew batchを止める（visits target未達で終了）、DeadlineControllerの通常完了commit、budget=0でも即legal move、slow evaluator（fake clockでdelay=1000 vs deadline=1）でfallbackが即返り commit exactly once、**exact-tie race（evaluatorがdeadlineと同時刻に解決するよう`FakeClock`で強制、25回ループ）で常にgeneration=1（commit 1回）**、`makeMove`によるinvalidation後に遅延結果が届いても木のgeneration/visits/moves countが不変（generation-id dropの直接テスト）。GTP層: `time_left=0`で即legal move、`time_settings`下でSlowEvaluator（実3秒delay）でもgenmoveが2秒以内に戻りstderrログに`timedOut=true`が出る。

CPU fallback（Metal未実装のためGPU中断不能ケースの実機検証）は範囲外のまま：watchdogが遅延taskを`cancel()`するのはbest-effort（`Task.sleep`ベースのfake evaluatorはcancellationに応答するが、将来の非協調的backendはこれに依存しない設計）。

### 2026-09-09 昼: M1 棋力ゲート、T28/T30/T31

- **M1 baseline 対局: PASS。** `models/p2-small-gl10.ichigo`（100 visits）vs 合法手 uniform baseline、9 路 komi 7、100 局（50 色交換ペア）: 100 勝 0 敗、paired bootstrap 95% CI [1.0, 1.0]、違法手/クラッシュ/タイムアウト/打切り 0（`reports/matches/p2-small-gl10/`）。極端に弱い相手なので大会棋力の主張ではない（05 §6）。
- T28（時計・watchdog）: sudden death 予算式、deadline 付き探索、CommitGate による「着手 commit は必ず 1 回」、generation ID で遅延結果を破棄。TimeTests 12 件 + GTP 2 件。
- T30（CGOS ローカル統合）: 公開 cgos 実装（revision 4dcff875…）の文法を再実装した client、fake server、切断→setup replay、SIGTERM 対局間停止、log rotation。実エンジン 2 台で 2 局完走を含む 37 テスト。実サーバー未接続。
- T31（対局 runner）: `python -m ichigo_train match`、uniform baseline、paired bootstrap、`Scripts/run_baseline_match.sh`。
- commits: T28 5eab2d9、T31 ee924df、T30 54e84d2。

### 2026-09-09 午後: M2a 部品（T22/T25/T29、探索高速化）

- T22 Metal-byte backend（`LogicMetal/MetalBackend.swift`、`Resources/logic_byte.metal`、実行時コンパイル: SwiftPM 6.0 は .metal を metallib 化しないため）: tiny-9/19/headv1 で全層 bit 一致、B=0〜64 で ScalarBackend と一致。`make check-metal` 14 件。
- T25 `ichigo benchmark` と BackendSelector（`configs/backend-profile.example.json`）。M5 実測（release、small モデル 9 路）: gate は Metal で B=32 時 13.99→2.89 ms（4.8 倍）だが CPU head が 79 ms で支配的。B=1 p95 3.2 ms（目標 20 ms 達成）、B=32 は 389 局面/秒（目標 1000 未達）。→ T23/T24（packed と Metal heads、CPU head の Accelerate 化）を着手。
- T29 校正: `sgf-results`（RE 解析、2,199 局: 投了 1,575 / 点数 624）、`calibrate`（温度 fit、Brier/ECE、game bootstrap、100 局未満は insufficientSamples）、`export --calibration`、Swift 後処理の温度適用。小 pilot では T≈0.82、test 12 局のため未検証扱い。
- 探索: 配置を 8 層に限定、fingerprint の逐次更新、edge index 保持で 267→328 visits/秒（実モデル）。fake evaluator では 9 路 4,509 / 19 路 1,444 visits/秒で、木の overhead は小さく NN が律速。
- 回帰: Swift 147 + Metal 14、pytest 189、commit 211cbd1。

### 2026-09-09: T23/T24 Metal-packed・CPU-packed・Metal heads

対象: `docs/spec/01-network.md` §4-5、`docs/spec/04-tasks.md` T23/T24。依存 T22 完了済み。

#### 成果物

| ファイル | 内容 |
|---|---|
| `Sources/LogicModel/PackBits.swift`（新規） | batch方向32bit pack/unpack（`[B,S,S,C]`→`[G,S,S,C]` uint32、`G=ceil(B/32)`）、`validMask`（最後のgroupのみ、シフト32禁止） |
| `Sources/LogicModel/PackedCPUBackend.swift`（新規） | `cpu-packed`。§5の四項式をuint32語に適用、全layer出力へ末尾maskを適用（true/NOT含む）、bank参照はScalarBackendと同一。`layerOutputs`（unpack済み、parity用）と`packedLayerOutputs`（packed、benchmark用）を公開 |
| `Sources/LogicModel/HeadsAccelerated.swift`（新規） | `Heads.evaluateAccelerated`。`#if canImport(Accelerate)` で `cblas_sgemm` により local/global projectionをsegment別（h/m/v/global）のbatched matmulへ再構成（`Heads.evaluate`の明示ループはScalarBackend専用の golden oracle として不変更）。Accelerateがない場合は`Heads.evaluate`にフォールバック |
| `Sources/LogicMetal/Resources/logic_packed.metal`（新規） | `logic_layer_packed` カーネル。`logic_byte.metal`と同じ構造をuint32語（32レーン/word）に拡張、末尾maskを全gateに適用 |
| `Sources/LogicMetal/Resources/heads.metal`（新規） | `heads_reduce`（packed bitからm/v直接計算）、`heads_local`（点ごとのlocal projection、u_xyを材料化せずh/m/v/globalをsegment別に64次元accumulatorへ直接累積、policy/ownership/zBuf出力）、`heads_global`（headVersion 2のzbar/ownMean reductionを同カーネル内で実施、global projection→pass/wdl/score）。`LOCAL_HIDDEN=64`/`GLOBAL_HIDDEN=128`はManifest定数なのでcompile-time固定、可変長thread-local配列を回避 |
| `Sources/LogicMetal/MetalPackedBackend.swift`（新規） | `metal-packed`。gate層Lディスパッチ+heads 3ディスパッチを同一command bufferへencodeし1回commit/wait（`evaluate`）。fast-math off（`MTLCompileOptions.fastMathEnabled=false`）。`layerOutputs`はgateのみ別command buffer。ベンチマーク専用に`evaluateTimed`（gate/head別command bufferでtiming分離、実運用の`evaluate`とは別経路） |
| `Sources/LogicMetal/MetalBackend.swift` | `evaluate`が`Heads.evaluate`から`Heads.evaluateAccelerated`へ変更（byte gates + 高速化CPU head）。ScalarBackendのみ従来のまま |
| `Sources/ichigo/main.swift` | `--backend cpu\|cpu-packed\|metal\|metal-packed\|auto`、`benchmark --backends`に4種、`metal_allocated_bytes`をMetalPackedBackendでも収集 |
| `Makefile` | `parity-metal`が`MetalParityTests`と`MetalPackedBackendTests`の両方を実行 |

#### テスト（新規）

- `Tests/LogicModelTests/PackBitsTests.swift`（6件）: B=0,1,2,31,32,33,63,64,65のgroup数/validMask/pack-unpack round trip、全0/全1入力、padding未設定の確認。
- `Tests/LogicModelTests/PackedCPUBackendTests.swift`（6件）: tiny-9/19/headv1でB=0,1,2,31,32,33,63,64,65の全layer bit一致（ScalarBackend比較）、`@testable import`で構築したtrue(15)/NOT-a(3)ゲート専用モデルでpadding laneが末尾maskで0になることを直接検証。
- `Tests/LogicModelTests/HeadsAcceleratedTests.swift`（4件）: `Heads.evaluateAccelerated`が複数batchサイズで`Heads.evaluate`の§3許容誤差内、post-process後のpolicyも同様。
- `Tests/LogicMetalTests/MetalPackedBackendTests.swift`（10件）: gate層parity（cpu-packed/metal-packedともScalarBackendと全layer bit一致、B=0〜65）、heads許容誤差（固定fixture＋ScalarBackend比較）、決定性、空batch、対応外board size拒否、そして`models/p2-small-gl10.ichigo`＋`data/positions-9.jsonl`先頭32局面でCPU scalarとmetal-packedのtop手が一致（near-tie以外）かつpolicy全要素が1e-5以内。

#### 検証結果（このMac、2026-09-09）

- `swift test --filter 'LogicModelTests|LogicMetalTests'`: 66 tests、1 skipped（no-Metal-deviceパスの意図的skip）、0 failures。
- `make parity-metal`（`MetalParityTests`+`MetalPackedBackendTests`）: 13 tests、0 failures。
- `swift test --filter 'IchiGoCoreTests|IchiGoFeaturesTests|LogicModelTests|IchiGoEngineTests|IchiGoGTPTests'`（check-cpuのSwift部分）: 163 tests、1 skipped、0 failures（既存機能への影響なし）。

#### ベンチマーク（release、`models/p2-small-gl10.ichigo`、small・C=256・L=8・headVersion2・9路、`data/positions-9.jsonl`、warmup50/iters200）

`reports/benchmark/p2-small-gl10-m5-v2.json`（旧: `p2-small-gl10-m5.json`、byte gates + CPU headのみ）。gate_ms/head_ms/total_p50/p95/positions_per_secをbackend×batchで採取:

| backend | B=1 total p50/p95 (ms) | B=32 total p50/p95 (ms) | B=32 positions/sec | B=64 positions/sec |
|---|---:|---:|---:|---:|
| cpu（golden、explicit-loop head） | 2.94 / 3.00 | 93.8 / 94.2 | 341 | 342 |
| cpu-packed（packed gate + accelerated head） | 0.66 / 0.68 | 2.07 / 2.14 | 15,386 | 15,647 |
| metal（byte gate on GPU + accelerated head） | 0.32 / 0.48 | 2.39 / 3.82 | 11,760 | 11,896 |
| metal-packed（packed gate + Metal head、両方GPU） | 2.35 / 2.48 | 2.74 / 2.86 | 11,599 | 20,536 |

§7目標: B=1 total p95≤20msは4backendすべて達成（最大でもcpuの3.0ms）。B=32≥1000 positions/secはcpu以外の3backendが達成（cpu-packedが最速、15.4k/秒）。cpu単独は未達のまま（headがexplicit loopのため）だが、auto/backend-profile選択で目標達成backendを選べる。旧報告（byte gates + CPU head、gate_ms 2.89ms/head_ms 79msでhead支配）から、cpu-packedのhead_msは1.20ms（32C×64C×3batched matmul）、metal-packedのhead_msは2.41ms（GPU heads、B=1では固定overheadのため相対的に遅い＝仕様の想定通り「B=1では32bitの31bitが遊ぶため高速化は保証しない」）。metal-packedはB=64で最速（20,536 positions/sec）。

`reports/benchmark/p2-small-gl10-m5-v2.backend-profile.json`: B=1→metal、B=8→metal、B=32→cpu-packed、B=64→metal-packedを最速として記録（T25のBackendSelector形式のまま、4backend対応）。

#### 設計上の判断（仕様が明示していない点）

- `Heads.evaluate`（明示ループ）はScalarBackendのみが呼ぶ恒久oracleとして凍結し、他backend（metal/cpu-packed）は`Heads.evaluateAccelerated`（Accelerate cblas_sgemm、なければloopにfallback）を使う。両者の数値は§3許容誤差内であることを`HeadsAcceleratedTests`で保証。
- `heads.metal`は`u_xy`/`u_global`をbufferへ材料化せず、入力segment（h/m/v/global、または m/v/zbar/ownMean/global）ごとに64/128次元accumulatorへ直接累積する設計とし、`LOCAL_HIDDEN`/`GLOBAL_HIDDEN`をcompile-time定数にすることで可変長thread-local配列を回避した。
- `MetalPackedBackend.evaluate`はgate+headsを1 command bufferへfuseする（T24契約）。ベンチマークのgate_ms/head_ms分離のためだけに`evaluateTimed`が2 command bufferへ分ける別経路を持つ（実運用の`evaluate`とは独立）。
- 入力のbatch方向packingはCPU側（`PackBits.pack`、S*S*C回のbit set操作のみで軽量）で行い、GPUへは packed bufferとしてアップロードする。pack自体のtimingは`pack_ms`列へ分離せず`gate_ms`へ含めた（コストが無視できるほど小さいため、T22時点から変更していない`feature_ms`/`pack_ms`=0固定の扱いを踏襲）。

### 2026-09-09 夕方: T23/T24 と実対局速度

- T23（CPU-packed / Metal-packed）、T24（Metal heads）、CPU head の Accelerate 化を実装（commit 3047ad6）。ベンチ（M5、small 9 路）: B=32 で cpu-packed 15,386 局面/秒、metal 11,760、metal-packed 11,599（B=64 で 20,536）。§7 の目標（B=1 p95 ≤ 20 ms、B=32 ≥ 1,000/秒）を golden CPU 以外の全 backend で達成。backend profile は B=1/8→metal、B=32→cpu-packed、B=64→metal-packed。
- 実対局: `ichigo gtp --backend auto` で 1 手 1.2 秒に 8,000〜10,000 visits（従来 500）。
- Swift 163 + Metal 66（skip 1）、pytest 189 通過。
- T26 検収（1 GPU vs 2 GPU、100 step、resume）をサーバーの空き GPU 0/3 で実行中。4 GPU 検収は phase 4/5 完了後に実施。
- phase 4 最良モデル `p4-local-gl30-200k` を export、Swift parity PASS、uniform baseline 100 局を実行中。

### 2026-09-09 18:25 巡回

- 完了: p4 wide512（512ch×8、局所配線、gateLR 0.1、200k）hard top1 **0.390** / MAE 0.291（logic 最良）。phase 5: Gumbel-STE（small 局所、100k）0.278 / 0.314 で prefix 方式より劣る。tauStart 0.5 + エントロピー罰則 0.368 / 0.295 で prefix と同等（soft-hard 差は早期に消えるが最終値は変わらず）。進行中: 16 層系 2 本、Gumbel base、learned-k 2 本。クラッシュなし。

### 2026-09-09 19:25 巡回

- `p4-local-gl30-200k` の uniform baseline 100 局: 100 勝 0 敗、CI [1.0, 1.0]、事故 0（PASS）。16 層局所 200k: hard top1 0.360 / MAE 0.289。
- T26 検収: 1 GPU 100 step = 0.109 秒/step（1,179 samples/秒）。2 GPU torchrun 側は run-summary 未生成のため失敗を調査中。

### 2026-09-09 21:00: T26 検収の障害と対処

- 2 GPU の torchrun がコード外で停止: NCCL の P2P 経路がこのサーバー（GPU 間 PXB 接続）で hang する。2 rank の all_reduce 単体テストで `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` のときだけ完了（0.2 秒）を確認。`Scripts/train_ddp.sh` に既定値として設定。
- コード側は DDP wrap 前に全 rank のパラメータ署名を broadcast で照合する事前検査を追加（不一致は NCCL abort ではなく明示エラー）。gloo 2 process で Trainer 全経路（validation/checkpoint/throughputReference）を通すテストを追加、pytest 192 件。
- 最初に見た「15 params vs 0 params」は NCCL 経路の停止に伴う検証 collective の不整合で、モデル構築の差ではない。

### 2026-09-09 21:40: T26 検収（2 GPU）

- 2 つ目の障害: heads-only 段階で gate エントロピー項が forward 外で theta を参照し、DDP reducer の「mark ready only once」違反を起こしていた。heads-only ではエントロピー項を計算しないよう修正（単一 process の数値は不変。gloo 2 process で stage 1→2→3 を跨ぐテストを追加、pytest 193 件）。
- 2 GPU 検収（NCCL、P2P 無効、microBatch 32 × 2 rank × accumulation 2 = 128）: 100 step 完走、0.061 秒/step（1 GPU 0.109 秒/step → 1.79 倍、効率 0.89）、checkpoint → resume で 100 step まで完走。rank 失敗時の全 process 終了は最初の障害時に torchrun 経由で確認済み。4 GPU 検収を実行中。

### 2026-09-09 22:00: T26 検収（4 GPU）

- 4 GPU（NCCL、P2P 無効、microBatch 32 × 4 rank × accumulation 1 = 128）: 100 step 完走、0.035 秒/step、3,680 samples/秒。1 GPU 比 3.1 倍、scaling efficiency 0.78（1 GPU 0.109 秒/step、2 GPU 0.061 秒/step から算出。run-summary の throughputRatio は参照パスの解決不備で None のため手計算）。peak GPU memory rank0 620 MiB、他 239 MiB。
- 同一 global batch の勾配一致は gloo 2 process の CPU テスト（SGD 代替、1e-5 以内）で確認。GPU 上の bit 一致は別指標（05 §3）で未実施。
- M2a の 4GPU 学習検収は満たした。今後の実験は small モデルでは 1 GPU × 多構成が効率的（efficiency 0.78）で、DDP は base 以上の長時間ランに使う。

### 2026-09-09 23:25 巡回

- `p4-local-wide512-200k`（logic 最良、hard top1 0.390）: uniform baseline 100 局 100 勝、事故 0、Swift parity PASS、1.2 秒/手で約 4,100 visits（Metal auto）。実験は 4 本進行中（16 層×512、Gumbel base、learned-k ×2）、クラッシュなし。

### 2026-09-10 00:30: モデル間比較（T31 の初回運用）

- 開局なしの対局は決定的で 100 局中 2 種類の棋譜しか出ず（黒が全勝）、比較にならない。固定開局 `configs/openings/random4-9x9-seed20260909.jsonl`（ランダム 4 手 × 60、seed 固定）を作り、色交換ペアで使用する。
- `p4-local-wide512-200k` vs `p2-small-gl10`（各 100 visits、9 路 komi 7、50 開局 × 色交換）: 63 勝 6 分 31 敗、平均得点 0.66、paired bootstrap 95% CI [0.57, 0.745]、全 100 局が相異なる棋譜、事故 0。wide512 の方が有意に強い（05 §6 の promotion 条件のうち CI 下限 > 0.5 を visits 固定リーグで満たす。1 秒/5 秒の時間リーグは未実施）。

### 2026-09-10 01:00: T37（運用パッケージ、M2a 完了）

- `Scripts/release/build.sh` / `verify.sh`、`make release-check`（check-cpu → Metal 前提確認 → check-metal → cgos tests → build/verify → uniform 2 局 smoke、事故 0 ゲート）を実装し、この Mac で PASS。dist は `.build` 外へ移しても Metal リソース解決を含め検証通過。
- `docs/runbook.md`（構築、モデル準備、fake server 演習、実 CGOS 設定、ログ/rotation、旧モデル復帰、M2b 記録項目、トラブル対応）、`docs/release-report-template.md`。
- 既知事項: 対局中のエンジンクラッシュは client 自体が終了する（無人運用には再起動 supervisor が必要）。cgos の SIGTERM テストは高負荷時に 180 秒 timeout で不安定。
- **M2a のゲート（T22〜T26、T28〜T30）は完了。M2b は公開 CGOS のアカウント・接続先設定待ち。**

### 2026-09-10 01:30: phase 6 と追加データ

- phase 6（GPU 0-2、wide512 局所 gateLR 0.1 基準）: 訓練データ半分（40,960 局面、データ量依存の切り分け）、500k step、gateLR 0.3 + bank30。
- 追加ラベル生成: 棋譜 2,201〜7,200 局目（5,000 局）を Mac で特徴抽出し、サーバー GPU 3-4 で教師ラベル化中（約 25 万局面、同一教師・同一設定）。完了後に 3 倍規模の dataset を作り、データ量スケーリングを確認する。

### 2026-09-10 02:25 巡回

- クラッシュなし。完了: Gumbel-STE base（512×12 局所、100k）hard top1 0.323 / MAE 0.311 で、prefix 方式の base（0.371 / 0.300）より劣る。Gumbel-STE は small/base とも不採用。
- 進行中: learned-k ×2（68k、top1 0.28〜0.31 で固定配線より低い）、16 層×512（130k）、phase 6 の 3 本（7.5k）。
- 追加ラベル: 5,000 局 247,738 局面の抽出完了（拒否 0）、GPU 3-4 で 54,919 局面ラベル済み（約 1 時間で完了見込み）。

### 2026-09-10 03:30: value の代替指標の診断（学習なし、`p4-local-wide512-200k`、dataset-9-b validation 359 局面）

| 推定器 | expected MAE | 備考 |
|---|---|---|
| 常に 0.5 | 0.328 | |
| NN wdl head | 0.234 | この局面集合では比較的良い（全 validation 5,407 局面では 0.291） |
| ownership 総和 + komi → sigmoid（k=6, b=1） | **0.250**（Brier 0.087） | 学習不要。目差推定 MAE 5.06（零基準 5.47）、相関 0.57 |
| 線形プローブ: z_xy 3×3 領域平均（576 次元） | **0.274** | global 平均/最大のプローブ（0.306）より良い |
| 線形プローブ: z_xy 全体（5,184 次元） | 0.356（train 0.108） | 過学習 |
| 線形プローブ: ownership 地図（81） | 0.371 | |

判断: 情報は局所 head まで来ており、盤全体の平均/最大で捨てている。→ (1) headVersion 3（3×3 領域プーリングを global head へ）、(2) ownership 由来 value を探索に混合する `--value-source`（学習不要、A/B 対局で検証）を並行実装。PUCT が要求するのは白視点の一貫した期待得点だけなので、NN の wdl head に限定せず比較する。

### 2026-09-10 05:00: headVersion 3 と phase 7

- headVersion 3（z_xy の 3×3 領域平均 576 次元を global head へ、u_global = 2C+645）を仕様・Python・Swift（scalar/Accelerate/Metal）・fixture に追加。Swift 175 + Metal 30、pytest 201、parity-cpu/metal 通過。
- phase 7 起動: wide512 局所 head v3（200k、GPU 6）、small 局所 head v3（100k、GPU 7）。追加ラベル 247,738 局面完了、3 倍データセット構築 → wide512 局所 200k（GPU 3）を自動起動予定。

### 2026-09-10 06:00: `--value-source`（ownership 由来 value を探索に混合）実装と A/B 対局

上記 03:30 の診断（学習なしの ownership 由来 value_own が NN wdl head 相当の expected MAE に達する）を受け、`IchiGoEngine.ValueSource`（`.network`/`.ownership(k,b)`/`.blend(weightNetwork,k,b)`）を実装した。`EvaluationAdapter.toWhite` が `score_est = Σ_xy ownership_xy + komiSelf` → `value_own = sigmoid((score_est+b)/k)` を計算し、`.ownership` は `expectedResult` をこれで置き換え、`.blend` は `weightNetwork*e_nn + (1-weightNetwork)*value_own` を使う。score lead（`whiteScoreMean`/`whiteLead`）は `.network` 以外なら常に ownership 由来の `score_est` を採用（blend の重みに関わらず）。raw NN WDL は `rawWinDrawLoss`/`rootRawExpected` に常に保持し、GTP ログは `rawNN=`（e_nn）・`expected(draw=0.5)=`（探索値）・`valueSource=` を出す。`Search`/`GTPEngine.Config`/CLI（`ichigo gtp`・`ichigo selfplay` 双方に `--value-source network|ownership|blend`、`--value-blend`既定0.5、`--value-k`/`--value-b`既定6/1.0）まで配線し、match runner の argv にはそのまま渡る（Python 変更なし）。`swift build`・`swift build -c release --product ichigo`・`swift test --filter 'IchiGoEngineTests|IchiGoGTPTests'`（EvaluationAdapter の視点/blend 境界の手計算、fake evaluator で wdl 一様・ownership 一方favourな局面が `.ownership` でだけ選ばれること、GTP フラグのparse/ログ echo を含め全通過）。

A/B 対局（`p4-local-wide512-200k`、9 路 komi 7、各 400 visits、`configs/openings/random4-9x9-seed20260909.jsonl` 50 開局×色交換、seed 20260910、事故 0 で共通）:

| 対戦 | 勝-分-敗（A視点） | 平均得点(A) | paired bootstrap 95% CI | Elo 差(A-B) |
|---|---|---|---|---|
| `--value-source ownership` (A) vs network既定 (B) | 20-3-77 | 0.215 | [0.14, 0.29] | -225 |
| `--value-source blend --value-blend 0.5` (A) vs network既定 (B) | 41-3-56 | 0.425 | [0.34, 0.51] | -52.5 |

結論: 03:30 の診断（孤立局面での expected MAE がNN head相当）はサーチ内での有用性を保証しなかった。ownership 単独は有意に弱い（CI が 0.5 を大きく下回る）。blend 0.5 も敗越しで、CI 上限が 0.51 と 0.5 をわずかに超えるため「有意に悪い」とは言えないが、優位でもない。実装（`--value-source`自体、視点変換、CLI配線、ログ）は事故ゼロで正しく動作しており、原因は value_own の質そのものと考えられる: (1) k=6,b=1 は完了局面寄りのvalidation集合で得点差にフィットした値で、探索が触れる序盤・中盤・末端未解決の局面では ownership 総和のスケールが安定せず sigmoid が過飽和しやすい、(2) headVersion 3（05:00、盤面局所情報を保った global head）の方が同じ情報をNN自身のwdl/score head経由で使わせる分、探索の勾配としては筋が良い可能性が高い。**当面の既定は `.network` のまま**。`--value-source`はheadVersion 3 再学習後の比較や、k/bを探索時局面分布で再フィットする追試のために実装として残す。

### 2026-09-10 07:25 巡回

- 完了: learned-k（K=8 候補からの配線学習、small 局所、100k）hard top1 0.359 / MAE 0.296、learned-k + Gumbel 0.368 / 0.303。固定配線の同条件（gl10 局所 0.369 / 0.294）と同等で、配線学習の利得なし。16 層×512 は 188k で 0.385 / 0.288（完了間近、wide512 最良 0.390 と同水準）。
- head v3（wide512 局所、32k 時点）0.361 / 0.307 で同段階の head v2 より良好。3 倍データは 31k で 0.327 / 0.313。クラッシュなし。

### 2026-09-10 08:25 巡回: head v3 の初結果

- **small 局所 head v3（100k）: hard top1 0.366 / expected MAE 0.277**。同条件の head v2（gl10 局所 100k: 0.369 / 0.294）に対し value が明確に改善（初めて構造変更で value が動いた）。wide512 head v3 は 60k 時点で 0.355 / 0.292。
- 16 層×512（200k）: 0.383 / 0.288 で完了。クラッシュなし。
- head v3 の展開: gl0.3、base、3 倍データ、200k の各版を空き GPU で起動。small head v3 を export して対局で検証する。

### 2026-09-10 09:30: head v3 の対局検証

- `p7-small-headv3`（MAE 0.277）vs `p3-small-gl10-local`（head v2、MAE 0.294）、各 400 visits、固定開局 50 × 色交換: 44 勝 5 分 51 敗、平均 0.465、CI [0.37, 0.56]、事故 0。**value の MAE 改善 0.017 は対局では有意差にならない。** head v3 の wide512・base・3 倍データ・200k 版（phase 8）の結果で改めて判断する。

### 2026-09-10 11:00: 表現力の壁への次の一手（判断メモ）

- soft（連続緩和）段階でも policy top1 は 0.36〜0.39 で頭打ち → 離散化ではなく緩和モデル自体の容量/最適化の限界。
- 計算量比較: CNN baseline は 1 局面あたり約 30 万 MAC、small は 2,048 ゲート評価、wide512 でも 4,096。2 桁の差があり、公平な比較にはゲート数を 1 桁以上増やす必要がある。→ phase 9a: C=1024（GPU 7、0.25 秒/step、4.1 GB）、C=2048（GPU 空き次第）。
- 2 入力ゲートは 16 関数しか表せず、多入力関数には深い木が要る。FPGA の LUT4 のような 4 入力真理値表ゲート（65,536 関数、推論は厳密なビット演算のまま）を実験拡張として実装中（phase 9b、Codex）。gateEncoding `lut4-msb-first`、gates.u16、wiring [L,C,4,4]。Metal は当面未対応（CPU packed で推論）。
- rollout value の A/B は実行中。

### 2026-09-10 12:30 巡回: phase 6 完了（データ量の切り分け）

- wide512 局所 200k、訓練データ半分（40,960 局面）: hard top1 0.380 / MAE 0.296。全データ（85,470 局面）の 0.390 / 0.291 とほぼ同じ。3 倍データ（130k 時点）0.337 / 0.292 も同水準。**現行方式はデータ量律速ではなく、モデル/学習方式の限界。**
- gl0.3 + bank30（wide512 局所 200k）: 0.391 / 0.296（top1 は最良タイ）。500k step は 164k 時点で 0.355 / 0.339（改善なし）。
- phase 9a: C=1024 局所 head v3（7.9k step）、C=2048 局所 head v3 を GPU 0 で起動。

### 2026-09-10 15:30: プレイアウト value の A/B（等時間 60 秒 sudden death、wide512、固定開局 50 × 色交換）

| A vs network | W-D-L | 平均 | CI95 | visits/手 A / B |
|---|---|---|---|---|
| rollout 8 本、W=0.5 | 13-1-86 | 0.135 | [0.075, 0.20] | 18 / 6,332 |
| rollout 4 本、W=0.7 | 11-1-88 | 0.115 | [0.06, 0.175] | 56 / 6,133 |

等時間では visits が 100〜350 分の 1 になり大敗（事故 0）。固定 400 visits の比較は 1 局 15〜30 分かかるため 10 局に縮小して実行中。**policy 誘導プレイアウトは、この探索・推論速度では value の代替にならない**（プレイアウト自体の品質以前に計算コストで不成立）。機構（`--value-source rollout`）は残す。

### 2026-09-10 17:00: head v3 の 200k 結果、LUT-4 起動

- small 局所 head v3 200k: hard top1 0.367 / **expected MAE 0.267**（最良）。gateLR 0.3 版 0.370 / 0.275。head v3 は value に一貫して効く（v2 の 0.29 台 → 0.27 前後）。wide512 head v3 と 3 倍データ版は完了間近。
- LUT-4（4 入力真理値表ゲート、`gateArity 4`、`lut4-msb-first`、gates.u16）を実装（Python 208、Swift 189、parity-cpu 通過、commit 済み）。C=256 / C=512 局所 head v3 を 100k step で起動（0.38 / 0.74 秒/step。soft 評価が 16 項の積和になるため遅い）。
- rollout value の固定 visits 比較は 10 局に縮小して実行中。

### 2026-09-10 18:25 巡回

- 完了: **wide512 局所 head v3 200k: hard top1 0.389 / MAE 0.272**（policy・value とも最良級）。3 倍データ（head v2、wide512、200k）: 0.381 / 0.281（value は 1 倍の 0.291 より改善、policy は同等 → データは value にのみ効く）。
- 進行中: base head v3（74k）、wide512 head v3 3 倍データ（109k）、C=1024（28k）、C=2048（10k、0.51 秒/step）、LUT-4 C=256（3.3k、top1 0.208 と序盤の立ち上がりが速い）、LUT-4 C=512（1.7k）。クラッシュなし。
- 1 秒/手の時間リーグ（small head v3 200k vs チャンピオン wide512 v2）を Mac で実行中。

### 2026-09-10 20:25 巡回

- 完了: **wide512 局所 head v3 + 3 倍データ 200k: hard top1 0.385 / expected MAE 0.259**（value 最良。head v3 とデータ増の効果が加算）。
- C=2048 局所 head v3 は 35k step で top1 0.392（既存最良と同水準に早期到達）、C=1024 は 80k で 0.373 / 0.296。ゲート数増は policy に効く。
- LUT-4: C=256 37k で 0.220、C=512 19k で 0.247。序盤の立ち上がりの後は 2 入力ゲート網より遅い（soft の 16 項積和の最適化が難しい可能性）。最終値で判断。クラッシュなし。

### 2026-09-10 夜: 本命構成の投入と head 分解能のプローブ

- 起動: C=2048 局所 head v3 3 倍データ 200k（GPU 3）、C=1024 同（GPU 6）。C=2048 単独は 35k で top1 0.392。
- 線形プローブ（wide512 head v3 3 倍データ模型、dataset-9-b）: z_xy の 3×3 領域平均 0.271、5×5 0.302（訓練 0.166、過学習）、3×3+5×5 0.313。この学習量では 3×3 が適正で、head v4（5×5）は保留。
- Mac の電池切れで対局リーグが一時停止していたが再開（game 81/100）。サーバーは影響なし。

### 2026-09-10 深夜: 1 秒/手リーグ（sudden death 60 秒、固定開局 50 × 色交換）

- `p8-small-headv3-200k`（MAE 0.267）vs チャンピオン `p4-local-wide512-200k`（head v2）: 47 勝 5 分 48 敗、平均 0.495、CI [0.405, 0.585]、事故 0。**互角でチャンピオン交代なし。** small の value 改善は wide512 の policy 差を埋める程度で、勝ち越すには「幅 + head v3 + 3 倍データ」（学習中）が必要。
- Mac 側の対局はすべて終了。以降、Mac は不要。

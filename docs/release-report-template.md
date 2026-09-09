# IchiGo release report

docs/spec/05-validation.md §8「最終成果の表示」のテンプレート。release ごとにこのファイルをコピーするか
（例: `docs/releases/<date>-<version>.md`）本ファイルを直接更新して使う。5 欄（実装した機能／実際に実行した
検証／各盤サイズの棋力／未実行の環境／model・data・code hash）を必ず分けて書き、混ぜない。CGOS 公開サーバー
接続の結果はローカル fake server 検証と別記する（同じ理由で本テンプレートも §8 の指示どおり両者を別の箇条書き
にしている）。

このインスタンスは 2026-09-09、この Mac（後述）上でこのチケット（T37）の一部として作成した。「実装した
機能」「実際に実行した検証」「各盤サイズの棋力」「model/data/code hash」は docs/implementation-status.md と
このセッションで実行した `make release-check` の結果に基づく現状の実測値で埋めてある。「M2b」関連の項目
（公開 CGOS への実接続）は実アカウントがなくこのセッションでは実行できないため `TODO (M2b)` と明記した。

---

## 1. 実装した機能

M0（数値基盤）・M1（学習して 9 路を動かす）・M2a（9 路 CGOS 優先の基盤部品）まで実装済み。docs/spec/04-tasks.md
のチケット番号で示す。

| 区分 | チケット | 内容 |
|---|---|---|
| M0 | T01〜T10 | 16 ゲート論理網、座標/snapshot、32ch 特徴＋D4 対称、Python ゲート oracle、配線・モデル生成、Swift 離散ゲート/head、`.ichigo` exporter/loader、CLI `doctor`/`inspect`/`eval` |
| M1 | T11〜T17 | SGF リプレイ→特徴 CLI、KataGo 教師 adapter、教師ラベル・dataset builder、損失/optimizer、train/checkpoint/resume、離散化スケジュール、16 局面過学習検収＋pilot |
| M1 | T18〜T21 | evaluator（白視点 adapter 込み）、pure-tree PUCT 探索、batching・tree reuse・cancel、基本 GTP・自己対局 |
| M2a | T22〜T25 | Metal-byte backend、Metal-packed／CPU-packed backend、Metal heads、`ichigo benchmark`／backend 自動選択 |
| M2a | T26 | 複数 GPU（DDP、1/2/4 GPU） |
| M2a | T28〜T30 | 時計・watchdog（sudden death、deadline、generation ID による遅延結果の破棄）、勝率表示・校正（`kata-genmove_analyze`、温度校正）、CGOS ローカル統合（`Scripts/cgos/ichigo_cgos_client.py`、fake server、切断→setup replay、backoff、SIGTERM 対局間停止） |
| M1 直後 | T31 | 対局評価 runner（`ichigo_train match`、uniform baseline、paired bootstrap、`Scripts/run_baseline_match.sh`） |
| — | **T37（本チケット）** | `Scripts/release/build.sh`／`verify.sh`（release パッケージ＋MANIFEST＋Metal リソース解決の検証）、`make release-check`、本 runbook（docs/runbook.md）、本テンプレート |

未実装（意図的に後回し。docs/spec/04-tasks.md の推奨順序どおり）:

- T27（19 路本学習・選抜、M2c）、T32〜T36（失着再ラベル・容量比較・gate 簡約・自己対局 RL・champion 選抜）
- 公開 9 路 CGOS への実接続そのもの（M2b。本チケットの完了条件のうち「実接続できていなければM2aまでと記録
  する」に該当。実アカウントがないため今回は未実施）

## 2. 実際に実行した検証

すべて 2026-09-09、この Mac（環境は §5 参照）で実行。数量は実行時点のもの。

| コマンド | 結果 |
|---|---|
| `make check-cpu`（Swift Core/Features/LogicModel/Engine/GTP tests + pytest） | `make release-check` の一部として実行。下記「release-check 実行結果」参照 |
| `make check-metal`（LogicMetalTests、実 Metal デバイス） | 同上 |
| `cd Training && uv run pytest -q ../Tests/cgos`（config validation、backoff、analysis parser、log rotation、hashing/ledger、fake server protocol、実エンジン 2 台での E2E 対局） | 同上 |
| `Scripts/release/build.sh models/p4-local-wide512-200k.ichigo` → `Scripts/release/verify.sh <dist-dir>` | 同上。加えて dist ディレクトリを別 cwd へコピーし `verify.sh` を再実行して path 非依存性を個別に確認済み（全チェック PASS） |
| 2 局 hard-model smoke match（`ichigo_train match` vs `uniform`、`--visits-a 50 --games 2`） | 同上。`incidents` が全種別 0 であることを確認 |
| `Scripts/run_baseline_match.sh models/p4-local-wide512-200k.ichigo`（uniform baseline 100 局、50 色交換ペア） | 100 勝 0 敗、paired bootstrap 95% CI `[1.0, 1.0]`、事故（違法手/クラッシュ/timeout/打切り）0。`reports/matches/p4-local-wide512-200k/report.json`。M1 baseline gate PASS |
| `models/p4-local-wide512-200k.ichigo` の Swift parity（`Scripts/check_model_parity.py`、validation 8 局面） | 全 8 サンプルで全層 bit 完全一致、head 許容誤差内。`reports/pilot/p4-local-wide512-200k-swift-parity.json` |
| `p4-local-wide512-200k` vs `p2-small-gl10`（ともに 100 visits、100 局、50 色交換ペア） | wide512 側 66 勝 31 敗 6 分、meanScore 0.66、paired CI95 `[0.57, 0.745]`、事故 0。`reports/matches/wide-vs-small-gl10/report.json` |
| Mac ベンチマーク（`ichigo benchmark`、`models/p2-small-gl10.ichigo`、9 路、warmup 50/iters 200、docs/spec/05-validation.md §7） | `reports/benchmark/p2-small-gl10-m5-v2.json`。B=1 total p95 ≤ 20ms は cpu/cpu-packed/metal/metal-packed 全 backend 達成。B=32 ≥ 1000 positions/sec は cpu 以外の 3 backend で達成（cpu-packed 最速 15,386/秒）。backend-profile: B=1/8→metal、B=32→cpu-packed、B=64→metal-packed |
| DDP 検収（1/2/4 GPU、大学 A4000 サーバー、T26） | 100 step 完走、4 GPU で 0.035 秒/step（1 GPU 比 3.1 倍、scaling efficiency 0.78）、resume・rank 失敗時の全 process 終了・勾配一致（gloo CPU cross-check）を確認 |

### `make release-check` 実行結果（このセッション、2026-09-09、このMac上、foreground実行）

| 段階 | 結果 |
|---|---|
| `make check-cpu` | Swift 163 tests（1 skipped、0 failures、246.3 秒）+ pytest 193 passed（17.6 秒） |
| Metal 前提チェック（`ichigo doctor` の `metal.available`） | `true`（Apple M5 検出）。`false` の場合はこの時点で release-check 自体が明示メッセージ付きで exit 1 する（未検証だがロジックはレビュー済み: Makefile の `release-check` ターゲット、`(echo "...no Metal device...refusing to silently pass." >&2; exit 1)` 節） |
| `make check-metal`（`LogicMetalTests`） | 24 tests、1 skipped（no-device path。この host は device ありのため意図的 skip）、0 failures、62.3 秒 |
| `cd Training && uv run pytest -q ../Tests/cgos` | 37 passed、147.5 秒（実エンジン2台のE2E含む） |
| `Scripts/release/build.sh models/p4-local-wide512-200k.ichigo` | `dist/ichigo-0.1.0-a6c6b1ea7e27-dirty-arm64/` を生成。Metal リソースがパッケージ済みレイアウトから解決することを自己検証済み（"OK -- metal backend resolved logic_byte.metal from the packaged dist/ layout"）。MANIFEST.json 11 ファイル、モデル payloadHash `b763668e199b87dea8a1d8c7373d091ac9551f972f26f9c8b8d43278083dde7d` |
| `Scripts/release/verify.sh <dist-dir>` | MANIFEST 再ハッシュ・`ichigo doctor`・`ichigo inspect --model`（payloadHash 一致）・`--backend cpu-packed`/`--backend auto` それぞれの 3-move GTP smoke、全 PASS |
| 2 局 hard-model smoke match（dist 内の release バイナリ、`--visits-a 50 --games 2`、uniform baseline 相手） | 2 勝 0 敗 0 分、meanScore 1.0、`incidents: {crashes:0, illegalMoves:0, timeouts:0, truncations:0}` |
| 最終行 | `release-check: PASS (/Users/kanade/dev/univ/koubou/IchiGo/dist/ichigo-0.1.0-a6c6b1ea7e27-dirty-arm64)` |

**path 非依存性の追加確認**: 上記 dist ディレクトリをリポジトリ外（`/private/tmp/.../scratchpad/moved-dist-final`）へコピーし、リポジトリと無関係な cwd（`/private/tmp`）から `Scripts/release/verify.sh` を再実行して全チェック PASS を確認した（`--backend auto` の Metal 経路含む）。

**この過程で見つけて修正した不具合**: 初回実行時、`Scripts/release/build.sh` 内の `swift build -c release --product ichigo`（再ビルド確認用の2回目呼び出し）が進捗テキストを標準出力へ書き、`Makefile` 側の `DIST=$(Scripts/release/build.sh ...)` によるコマンド置換に混入して dist パスの捕捉が壊れていた（`verify.sh` が `not a directory: [0/1] Planning build...` で失敗）。`swift build ... >&2` へ修正して解消し、以降の実行で再発していない。また `Tests/cgos/test_cgos_integration.py::test_sigterm_stops_only_after_the_current_game`（T30 で実装済みの既存テスト、本チケットの対象外）が、直前に `check-cpu`/`check-metal` を走らせた直後の1回だけ 180 秒のタイムアウトで failed した（単体では 86 秒で安定して pass。CPU 競合による flake と判断、本チケットの対象ファイルではないため変更していない）。最終的な `make release-check` 実行は上表のとおり全段階 PASS。

## 3. 各盤サイズの棋力

- **9 路**: 学習は成立（uniform baseline に対し policy の cross entropy が有意に改善、value head は弱いが
  定数 0.5 baseline をわずかに下回る程度まで動く。docs/implementation-status.md 参照）。M1 の baseline
  ゲート（uniform 相手 100 局、95% CI 下限 > 0.5）は複数モデルで PASS 済み（`p2-small-gl10`、
  `p4-local-gl30-200k`、`p4-local-wide512-200k` すべて 100 勝 0 敗）。ただしこれは「対局として完走し、
  意味のある手を選んでいる」ことの確認であり、大会棋力の主張ではない（対戦相手が極端に弱いランダム合法手
  baseline のため。docs/spec/05-validation.md §6）。
  - 同系統モデル間の相対比較（`p4-local-wide512-200k` vs `p2-small-gl10`、100 visits）: meanScore 0.66、
    paired CI95 `[0.57, 0.745]` で wide512 が優位。
  - docs/spec/05-validation.md §6 が定める「強い基準」（少 visits に制限した固定 KataGo 教師、現行 RinGo の
    固定 model/settings との対局）は **未実施**。
  - CGOS rating: M2b 未実施のため **TODO (M2b)**。
- **19 路**: **未着手**（T27/M2c）。docs/spec/05-validation.md §6 の方針どおり、9 路 CGOS 出場を 19 路学習の
  前提にしていない。baseline 100 局比較も未実施。

## 4. 未実行の環境

- 公開 CGOS サーバーへの実接続（M2b）。ローカル fake server（`Scripts/cgos/fake_cgos_server.py`）での
  2 局完走・切断→setup replay・SIGTERM 対局間停止・analysis 文字列の記録は確認済みだが、実サーバーの
  `setup`/`genmove`/`gameover` の実メッセージ内容や実ネットワーク条件、サーバー側での analysis 表示/保存は
  未確認（docs/spec/03-engine.md §9 の「実サーバーのアカウントと接続先はユーザーが実運用時に設定する」に該当）。
  **TODO (M2b)**。
- CUDA/A4000 上での GPU bit 完全一致比較（同一 global batch の勾配一致は CPU（gloo）cross-check のみ実施。
  GPU 上の bit 一致は docs/spec/05-validation.md §3 の別指標として未実施）。
- macOS 14（deployment target の下限）実機での動作確認。このセッションのビルド/検証はすべて macOS 26.3.1
  上で実施しており、下限バージョンでの実機検証は未実施。
- 19 路のあらゆる検証（学習、探索、CGOS ローカル完走、baseline 比較）。
- docs/spec/05-validation.md §6 の「強い基準」（少 visits KataGo、現行 RinGo）との対局。
- docs/spec/05-validation.md §5 の勝率校正で必要な 100 局以上の実結果 test（現状 test 12 局のみで
  insufficientSamples 扱い。M2b で実対局数が増えれば校正の再検証が必要）。
- T32〜T36（失着再ラベル、容量比較、gate 簡約、自己対局 RL、champion 選抜）はいずれも未着手。

## 5. model / data / code hash

- **git commit**: `a6c6b1ea7e2782c62aed22b919b3538fc27dd65a`（この Mac、`git rev-parse HEAD`、2026-09-09）。dirty
  = true（本チケット自身の変更が未コミットのため。`dist/ichigo-0.1.0-a6c6b1ea7e27-dirty-arm64/` の版名にも
  現れている）。このリポジトリでは学習/対局を回す並行プロセスが独自に定期コミットしており、この commit 自体も
  本チケットとは無関係な内容（openings ファイル、`wide-vs-small-gl10` 対局結果）を主とする。T37 の変更
  （`Makefile`、`Scripts/release/*.sh`、`Scripts/cgos/README.md`、`README.md`、`.gitignore`、
  `docs/runbook.md`、`docs/release-report-template.md`）はこのセッションの時点でまだ未コミット
  （ユーザーからコミット指示がないため）。
- **release バイナリ・パッケージ**: `Scripts/release/build.sh models/p4-local-wide512-200k.ichigo` が
  生成した `dist/ichigo-<version>-<arch>/MANIFEST.json` に、同梱した全ファイルの sha256・モデル
  payloadHash・toolchain/OS バージョン・git commit が記録されている（`Scripts/release/verify.sh` で
  再ハッシュ照合済み）。
- **モデル payload hash**（`ichigo inspect --model` の `payloadHash`、gates/wiring/heads バイナリの内容
  ハッシュ）:
  - `models/p4-local-wide512-200k.ichigo`: `b763668e199b87dea8a1d8c7373d091ac9551f972f26f9c8b8d43278083dde7d`
    （512ch×8 層、局所配線、gateLR 0.1、200k step、headVersion 2、boardSizes=[9]）
  - `models/p2-small-gl10.ichigo`: ベンチマーク・相対比較の対戦相手として使用（hash は
    `reports/benchmark/p2-small-gl10-m5-v2.backend-profile.json` の `modelHash` を参照）
- **学習データ**: `dataset-9`（大学 GPU サーバー側、96,065 局面: train 85,470 / validation 5,407 /
  test 5,188、重複 12,611 除去後）。元棋譜は `katago-mlx/kifu/` の 49,893 局のうち 2,199 局から抽出した
  108,676 局面（1 局 superko 違反で拒否）。教師ラベルは KataGo 1.18.2、`g170e-b20c256x2-s5303129600-d1228401921.bin.gz`
  （sha256 `7c8a84ed9ee737e9c7e741a08bf242d63db37b648e7f64942f3a8b1b5101e7c2`）、128 visits。
- **M2b 対局時のバイナリ・モデル hash**: 対局ごとに `ichigo_cgos_client.py` がプロセス起動時に 1 回計算し、
  各対局開始のログ行（`engine_binary_sha256=...`、`model_manifest_sha256=...`）に記録する
  （docs/spec/03-engine.md §9「対局途中model差替えなし」）。実接続後、この節に実際に使った
  `dist/.../MANIFEST.json` の値を転記すること。**TODO (M2b)**。

## 6. 環境（このレポート作成時点）

- Mac: Apple Silicon、macOS 26.3.1（build 25D771280a）
- Swift: 6.2.3（swift-driver 1.127.14.1）
- uv 0.10.2、Python 3.12.13
- Metal デバイス: `ichigo doctor` の `metal.available` を参照（release-check はこれが `false` の場合
  そもそも失敗して止まる。§2 の実行結果を参照）

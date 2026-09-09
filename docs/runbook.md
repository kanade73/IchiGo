# 運用手順書（T37）

docs/spec/04-tasks.md T37「実運用パッケージ」の runbook。対象は docs/spec/03-engine.md §9（CGOS 運用）と
docs/spec/05-validation.md §1/§8（`make release-check`、release report）。実行例のコマンドはすべてリポジトリ
直下（`IchiGo/`）から実行する前提で書く。

このセッションでは公開 CGOS への実接続は行っていない（実アカウントがないため）。(d)(e) は
`docs/spec/03-engine.md` §9 と `Scripts/cgos/ichigo_cgos_client.py`/`fake_cgos_server.py` のソースコードを
読んで確認した「実装済みの動作」を記述したものであり、実サーバーでの動作確認そのものではない。M2b（公開 9
路 CGOS 20 局）を実施する担当者は、この文書の (d) に従って接続情報を用意し、完了後に
docs/implementation-status.md へ結果を追記する。

## (a) クリーンな作業ディレクトリからのビルド

前提: macOS 14 以降、Swift 6.2（Xcode または単体ツールチェイン）、[uv](https://docs.astral.sh/uv/)。

```sh
git clone <このリポジトリ> IchiGo && cd IchiGo   # 既にチェックアウト済みならそのまま cd IchiGo

# 1. Python 依存（Training/uv.lock どおりに解決。CUDA は Linux のみ、Mac は CPU torch）
cd Training && uv sync && cd ..

# 2. Swift ビルド（デバッグ。release ビルドは Scripts/release/build.sh がまとめて行う）
swift build

# 3. CPU で完結する検証一式（Swift Core/Features/LogicModel/Engine/GTP tests + pytest）
make check-cpu
```

- `make check-cpu` は GPU を一切使わない。Metal がない環境（CI・Linux 開発機など）でもここまでは通る。
- 失敗した場合はまず `swift build` 単体、`cd Training && uv run pytest -q` 単体で切り分ける。
- Metal を使う検証（`make check-metal`）や `make release-check` は Mac 実機（Apple Silicon 推奨、Metal
  デバイスが `ichigo doctor` の `metal.available` で `true` になる環境）が必要。

## (b) モデルの準備

学習（Training 側の `train`/`export`）そのものは本書の範囲外（docs/spec/02-training.md、
docs/implementation-status.md 参照）。ここでは「学習済み checkpoint がある」状態から、対局に使える
`.ichigo` モデルを用意し、CGOS に出す前に確認する手順を書く。

```sh
# 1. export: checkpoint -> models/<name>.ichigo
cd Training
uv run python -m ichigo_train export --checkpoint ../runs/<run>/checkpoint-best-hard.pt \
    --out ../models/<name>.ichigo --board-sizes 9 \
    --calibration ../reports/pilot/<run>-calibrate.json   # 校正済み temperature があれば付ける。なければ省略（T=1.0、未検証のまま記録される）
cd ..

# 2. inspect: manifest・形状・hash を確認（exit 0 でなければモデルとして壊れている。(e) 参照）
swift run -c release ichigo inspect --model models/<name>.ichigo

# 3. parity: Python hard モデルと Swift scalar backend の層ごと bit 一致・head 許容誤差を確認
cd Training
uv run python ../Scripts/check_model_parity.py --model ../models/<name>.ichigo \
    --data ../data/dataset-9 --split validation --count 8 \
    --out ../reports/pilot/<name>-swift-parity.json
cd ..
# report の "layerBitExact": true / "headsWithinTolerance": true をすべてのサンプルで確認する。

# 4. baseline match: docs/spec/05-validation.md §6 の M1 ゲート（合法手 uniform baseline に 100 局、
#    50 色交換ペア、paired bootstrap 95% CI 下限 > 0.5）
Scripts/run_baseline_match.sh models/<name>.ichigo
# 標準出力末尾の "M1 baseline gate (paired 95% CI lower bound > 0.5): PASS" を確認する。
# FAIL の場合はこのモデルを CGOS に出さない。
```

`models/p4-local-wide512-200k.ichigo`（本リポジトリに同梱、`make release-check` が使うモデル）は上記 2〜4
をこのセッションで実行済み: `reports/pilot/p4-local-wide512-200k-swift-parity.json`（8 サンプル全 bit 一致）、
`reports/matches/p4-local-wide512-200k/report.json`（uniform baseline 100 局: 100 勝 0 敗、CI [1.0, 1.0]、
事故 0）。ただし docs/implementation-status.md が記録するとおり value head は弱く（expected result MAE が
定数 0.5 基準をわずかに下回るのみ）、これは「学習が動く」ことの確認であり大会棋力の主張ではない
（docs/spec/05-validation.md §6）。

## (c) ローカル CGOS リハーサル（fake server）

実サーバーに接続する前に、必ずローカルの `fake_cgos_server.py` で 1 局以上完走させる。手順は
`Scripts/cgos/README.md` の "Quick start" と同じ。要点だけ書く。

```sh
# release バイナリが必要
swift build -c release --product ichigo

# 1. fake server（別ターミナル）
python3 Scripts/cgos/fake_cgos_server.py --port 6867 --account ichigo-a:pw-a --account ichigo-b:pw-b

# 2. 各 client 用の password file / config（state_dir・log_dir は本番用と必ず別ディレクトリにする）
mkdir -p /tmp/ichigo-cgos-a /tmp/ichigo-cgos-b
echo pw-a > /tmp/ichigo-cgos-a/password.txt
echo pw-b > /tmp/ichigo-cgos-b/password.txt
# cgos.json の書き方は Scripts/cgos/README.md の例をそのまま使う（host=127.0.0.1, port=6867 など）。

# 3. 各 client を別ターミナルで起動、1 局で終了
python3 Scripts/cgos/ichigo_cgos_client.py --config /tmp/ichigo-cgos-a/cgos.json --games 1
python3 Scripts/cgos/ichigo_cgos_client.py --config /tmp/ichigo-cgos-b/cgos.json --games 1
```

加えて自動テスト一式（fake server 込み、切断→setup replay、SIGTERM 対局間停止まで検証）を実行する:

```sh
cd Training && uv run pytest -q ../Tests/cgos
```

`make release-check` はこの pytest を毎回実行する（下記参照）。ここが通らない状態で (d) の実サーバー接続に
進まない。

## (d) 実 CGOS への接続

**このリポジトリに実サーバーのホスト名・アカウント名・パスワードを一切コミットしない。**

**重要（`Scripts/cgos/README.md` の記述の訂正）**: `Scripts/cgos/README.md` は「`configs/` 自体がこの
リポジトリで gitignore されている」と書いているが、これは誤り —— 実際に確認すると `.gitignore` に
`configs/` のエントリはなく、`configs/cgos.example.json` を含め `configs/` 配下は通常どおり git 管理下
にある（`git check-ignore -v configs/cgos.local.json` は何も返さない＝無視されない）。
**実際に gitignore されているのは `data/`・`runs/`・`models/`・`teachers/`・`tools/` の各ディレクトリだけ**
（リポジトリ直下の `.gitignore` を参照）。したがって実運用の config・パスワードファイルは
**`configs/` の下には絶対に置かない**。`configs/cgos.example.json` の既定値が指す
`../runs/cgos/ichigo/state`・`../runs/cgos/ichigo/logs`（`runs/` 配下、gitignore 済み）のような場所か、
リポジトリ外のパスにコピーしてから編集する。

1. `configs/cgos.example.json` を `runs/cgos/<name>/cgos.local.json`（`runs/` 配下、gitignore 済み。
   またはリポジトリ外の任意のパス）へコピーし、実運用時に確認した値で埋める。**`configs/cgos.local.json`
   のように `configs/` の下へコピーしない**（上記のとおり `configs/` は無視されないため、うっかり
   `git add`/`git add -A` で実アカウント情報をコミットしてしまう）。

   | フィールド | 埋める値 |
   |---|---|
   | `host` / `port` | 接続直前に確認した公開 CGOS サーバーの実際の値（docs/spec/05-validation.md §8:
   「実サーバーのホスト/ポート/制限/対応解析方式は接続直前に確認する」）。本書では未確定のため空欄のまま |
   | `username` | 発行済みの CGOS アカウント名 |
   | `password_file` | **相対パス**（config ファイル自身のディレクトリからの相対）。絶対パスは起動時に拒否される。パスワード本文だけを書いたファイルを、config と同じ gitignore された場所（例: 同じ `runs/cgos/<name>/` の下）に置く。`configs/` の下には置かない |
   | `board_size` | `9`（初回出場は 9 路。19 路は T27/M2c 以降） |
   | `engine_argv` | release バイナリ＋実際に使うモデルへのフルパス。JSON 配列で書く（シェル文字列にしない。パス中の空白を naive split しないため）。例: `["/abs/path/dist/ichigo-<version>-<arch>/ichigo", "gtp", "--model-9", "/abs/path/dist/ichigo-<version>-<arch>/models/<name>.ichigo", "--backend", "auto"]`。`Scripts/release/build.sh` が作る dist ディレクトリをそのまま指す運用を推奨（(b) の parity/baseline を通したモデルと、MANIFEST.json に hash が残るバイナリが一体で揃う） |
   | `analysis` | `true`（`kata-genmove_analyze` を使い、CGOS の analysis 拡張へ送る） |
   | `state_dir` / `log_dir` | 本番専用の新しいディレクトリ（RinGo の CGOS state と混在させない。他の CGOS client インスタンスとも共有しない） |
   | `max_games` | 通常は `null`（`--games N` を CLI 引数で指定する運用を推奨。config を書き換えずに済む） |

2. `password_file` の指すファイルを作成する（パスワード文字列 1 行のみ）。パーミッションは
   `chmod 600` を推奨。

3. `engine_argv` が指す release バイナリとモデルが実在し、`board_size` と一致していることを確認する
   （`ichigo inspect --model <path>` の `boardSizes` を見る）。`Scripts/release/build.sh MODEL_DIR` で
   作った dist ディレクトリをそのまま使えば、MANIFEST.json に記録された `model.payloadHash` と
   `ichigo inspect` の出力を突き合わせて「今まさに engine_argv が指しているモデルが、検収を通した
   ちょうどそのバイナリ入りの版である」ことを確認できる。

4. 起動:

   ```sh
   python3 Scripts/cgos/ichigo_cgos_client.py --config <your-config>.json --games 20
   ```

   `--games 20` は「M2b は 20 局完了を確認する」ための一つの運用方法（`max_games` を config に書く代わりに
   CLI で指定）。省略すると `config.max_games`（既定 `null` = 無制限）に従う。

5. 停止（通常運用）: `SIGINT`（Ctrl-C）または `SIGTERM` を送る。**対局の途中では止まらない** ——
   進行中のゲームが `gameover` を受け取って処理し終わった直後にだけ、サーバーへ `ready` の代わりに
   `quit` を返して終了する（`Scripts/cgos/ichigo_cgos_client.py` の `CGOSClient.run`/`_handle_signal`）。
   エンジンサブプロセスにも `quit` を送ってから終了する。

6. ログ・state の場所とローテーション:
   - `log_dir/ichigo_cgos_client.log`: `logging.handlers.RotatingFileHandler`、20 MiB ごとに
     ローテーション、バックアップ 5 世代（`ichigo_cgos_client.log.1`〜`.5`。最大で本体+5 世代 = 6
     ファイル）。標準エラーにも同時出力される。
   - `state_dir/games/<gid>.json`: 対局ごとの着手台帳。サーバーから `setup` を受けるたび（新規対局・
     再接続どちらでも）ゼロから書き直される。障害調査には使えるが、client 自身の再開ロジックはこの
     ファイルを読み直さない（サーバーの `setup` の着手列だけを信頼する）。

7. モデルの切り戻し（旧モデルへのロールバック）:
   - `models/` 配下の各 `.ichigo` ディレクトリは削除せずに残す。どのバージョンがどの `MANIFEST.json`
     （`Scripts/release/build.sh` が `dist/ichigo-<version>-<arch>/MANIFEST.json` に書く）に対応するかは
     `MANIFEST.json` の `model.payloadHash`（`ichigo inspect --model` の `payloadHash` と同じ値）で追跡する。
   - 切り戻す時は、**client プロセスを止めてから** `engine_argv` の `--model-9`/`--model-19` を旧モデルの
     パスに書き換え、client を再起動する。docs/spec/03-engine.md §9 の「対局途中model差替えなし」の通り、
     client はプロセス起動時に一度だけ engine_argv とモデルの hash を計算し、プロセスの生存期間中は
     エンジンサブプロセスを再起動も差し替えもしない（`CGOSClient.run` は起動時に 1 回だけ
     `GTPEngineProcess.start()`/`hash_engine_and_models()` を呼ぶ）。**同じパスのファイルをプロセスを
     止めずに書き換えるのは禁止** ——client 側はモデルの再読込を行わないため、実際に使われるバイナリ/モデル
     と client が起動時にログへ記録した hash がずれる。
   - 切り戻し後、最初の対局の `setup` ログ行（後述）で `model_manifest_sha256=...` を確認し、意図した
     旧モデルの hash になっていることを確認する。

8. 最初の対局後に解析文字列（analysis）が受理されたことを確認する（client ログ）:
   - `_on_setup` が対局開始時に 1 行ログを出す:
     `game <gid> start: opponent=... color=... boardsize=... komi=... time_ms=... engine_binary_sha256=... model_manifest_sha256=... resume_moves=...`
   - 着手ごとに `_on_genmove` が
     `game <gid>: played <color> <move> (+analysis)` を出す。**`(+analysis)` が付いていることが
     「エンジンが `kata-genmove_analyze` で info 行を返し、client がそれを CGOS analysis JSON へ変換して
     `genmove` の返答に付けて送った」ことの確認**（`config.analysis: true` かつサーバーが
     `genmove_analyze` を提示した場合のみこの経路になる。サーバーが対応していない場合は毎回プレーンな
     `genmove` になり、`(+analysis)` は付かない）。
   - ただし、これで確認できるのは **client が送った** ところまで。サーバーが実際にその解析値を保存・表示
     しているかどうかは、対局終了後にサーバー側の当該対局ページ（Web UI 等）を人手で確認するしかない
     （docs/spec/05-validation.md §6 M2b:
     「クライアントによる解析値受理とサーバー側の保存または表示を確認する」は接続時の手作業）。この文書は
     その手順の記録場所を用意するだけで、確認そのものは M2b 実施時に行う。

9. M2b 完了後に docs/implementation-status.md へ追記する内容（このセッションでは未実施、TODO）:
   - 完了局数（目標 20 局。未完局はサーバー障害由来なら別記し 20 局に数えない、
     docs/spec/05-validation.md §6）。
   - エンジン起因の incident 内訳（違法手・クラッシュ・タイムアウト）が 0 件であることの根拠（client
     ログの抜粋、または対局ごとの `gameover` 結果一覧）。
   - サーバー側で解析値（winrate/score/pv）の保存または表示を確認したという記録（スクリーンショットや
     URL、確認日時）。
   - 使用した release バイナリの `dist/.../MANIFEST.json` の git commit・モデル `payloadHash`。
   - 対局に使った `host`/大会名など、config の非秘匿情報（`username`/`password_file` の中身は書かない）。
   - サーバー障害があれば、それとエンジン起因の incident を混同せずに別項目で記録する。

## (e) トラブルシューティング

### Metal が使えない

- `swift run -c release ichigo doctor` の `metal.available` を確認する。`false` なら Metal デバイスなし
  （このホストに GPU がない、または Metal 非対応の環境）。
- `--backend metal` / `--backend metal-packed` を明示指定した場合、Metal デバイスがなければ **usage
  error（exit 2）** になる（`metal backend unavailable: ...`）。無言で CPU に切り替わることはない
  （Sources/ichigo/main.swift の `makeBackend`）。
- `--backend auto`（既定）は Metal デバイスがなければ **明示的なエラーなしで cpu backend にフォールバック
  する**（docs/spec/01-network.md §5 の既定動作。仕様上そう決めてあり、実行時に stderr 警告は出ない）。
  運用で「今どの backend が動くか」を確実に知りたい場合は `ichigo doctor` を先に確認するか、
  `--backend cpu-packed` のように明示指定する。
- `Scripts/release/build.sh` は packaging 時に一度だけ、パッケージ済みレイアウトから
  `--backend metal` の動作確認を行う。Metal デバイスがあるのに失敗した場合は build.sh 自体が exit 3 で
  停止する（Sources/LogicMetal のリソース解決が壊れているという意味なので、そのまま出荷しない）。ビルド
  ホストに Metal デバイスがない場合は stderr に明示的な warning を出して続行する（この場合パッケージには
  Metal シェーダのリソースは同梱されるが、そのホストでは検証されていない）。

### モデルが `ichigo inspect`/`ichigo eval`/`ichigo gtp` に拒否される

`.ichigo` ディレクトリの検証（`LogicModel/ModelLoader.swift`）はロード前に schema・バイト長・hash・値域を
すべて確認し、少しでもおかしければロードを拒否する（exit 2）。観測済みの拒否理由の例
（docs/implementation-status.md T08/T10 の検収記録より）:

- ファイルの truncation（宣言バイト長と実バイト長の不一致）
- `manifest.json` に記載の sha256 と実ファイルの sha256 の不一致
- `version`/`featureVersion`/`rulesId` が対応外
- manifest 中の NaN・overflow・巨大すぎる値（例: calibrationTemperature が `1e300` など）
- gate 値が範囲外（0〜15 の外）
- head tensor の byte offset が範囲外・重複・重なり・末尾未整列
- tensor の欠落・二重定義
- `..` を含むパスやシンボリックリンクの使用（パス逸脱対策）
- channels/boardSizes が対応外

対処: 該当する export をやり直す（`uv run --project Training python -m ichigo_train export ...`）。
コピー中の破損が疑われる場合は sha256 を再計算して比較する。`ichigo inspect --model PATH` はエラー内容を
そのまま標準エラーに出す。

### `time_settings` で byo-yomi を送るとエラーになる

v1 は sudden death（`byo=0, stones=0`）のみ対応で、`byo>0` または `stones>0` は
`time_settings` の時点で GTP エラー応答になる（仕様どおりの動作。
docs/spec/03-engine.md §8「byo>0はv1でエラーとして未対応を明示」、実装は
`Sources/IchiGoEngine/TimeManager.swift` の `validateSuddenDeath`）。手元で GTP を手打ちして
`time_settings 300 30 5` のようなコマンドを送ると `?` 応答が返るのは仕様どおりで、バグではない。

`Scripts/cgos/ichigo_cgos_client.py` は実サーバーの `setup` に含まれる主時間だけを見て、常に
`time_settings <秒> 0 0`（sudden death）としてエンジンへ送る（`GTPEngineProcess.notify_time_settings`）ので、
CGOS 経由の通常運用でこのエラーが起きることは想定していない。もし実サーバーが本当に秒読みを要求してきて
それをエンジン側の対局挙動に反映する必要が出た場合は、v1 の制約として scope 外である（本チケットの範囲では
対応しない）。

### エンジンが対局中にクラッシュした場合の client の挙動

`Scripts/cgos/ichigo_cgos_client.py` のソースを読んで確認した、実装済みの挙動（未接続のため実サーバーでは
未検証）:

- エンジンサブプロセス（`ichigo gtp`）が対局中に終了する、または GTP コマンドに `?` エラーを返すと、
  `GTPEngineProcess._send_raw` が `GTPProtocolError` を送出する。
- この例外は **ネットワーク断（`OSError`）やプロトコルエラー（`CGOSProtocolError`）とは別クラス** で、
  `CGOSClient.run` の再接続ループはこれを catch しない。つまり **client プロセス全体がその場で終了する**
  （`finally` 節でサーバー接続のクローズとエンジンへの `quit` 送信は試みるが、`run()` の外まで例外が伝播し、
  未捕捉のまま Python プロセスが非 0 の終了コードでトレースバックを出して終了する）。
- 自動でその対局を諦めて次の対局を待つ、投了を送る、といった救済処理は **ない**。ネットワーク断のような
  1/2/4/8/16/30 秒 backoff での自動再接続の対象にもならない。
- 運用上の対処: 実サーバーで無人運用する場合は、client を **再起動スーパーバイザの下で** 動かすことを
  強く推奨する（例: `while true; do python3 Scripts/cgos/ichigo_cgos_client.py --config C.json; sleep 5; done`
  という単純なシェルループ、または launchd の `KeepAlive`）。プロセスが予期せず終了したら、まず
  `log_dir/ichigo_cgos_client.log` の末尾（Python のトレースバックは標準エラーに出るため、
  `RotatingFileHandler` を通らずログファイルに残らない場合がある点に注意——ログファイルと一緒に、
  client を起動していた端末/supervisor 側の標準エラー出力も必ず確認する）を見て原因（エンジンのクラッシュ
  か、単なる接続断か）を切り分ける。
  - 対局が再開できるかどうかはサーバー側の実装次第: docs/spec/03-engine.md §9 が定める
    「切断再接続はサーバーsetupからclear→size→komi→全着手replay→時計復元」は client が再接続できた場合の
    復帰手順であり、サーバーが対局にタイムアウト等で既に決着を付けていれば、そのまま次の対局として
    `ready` からやり直すことになる。いずれの場合も `state_dir/games/<gid>.json` の台帳ファイル自体は
    client の自動復帰には使われない（診断用）。

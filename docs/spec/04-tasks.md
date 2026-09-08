# 実装チケット

各チケットは単独で差分をレビューできる大きさを想定する。大きい場合は番号にa/bを付けて分割し、契約や検収条件を省略しない。以下のパス・コマンドは実装後の目標である。

## 共通の完了条件

- 実装、必要fixture、対象テスト、help/config、implementation-statusが揃う。
- public APIには形状・単位・視点・失敗条件のコメントがある。
- placeholderの定数出力で学習/推論の完了を偽装しない。fake evaluatorはTests内だけ。
- 前提を変えたら対応する仕様とfixtureを先に更新する。
- チケットを跨ぐ高速化・探索改変を同じ差分に混ぜない。

## A. 基盤（M0）

### T01 プロジェクト骨格
依存: なし。
対象: Package.swift、Sources各target、Training/pyproject.toml、Makefile、.gitignore。
手順: 00の依存関係でSwiftPMを作る。Pythonはtorch/numpy/pytest、解決した正確なversionをlockに記録。Mac runtimeへMLX/CUDA依存を入れない。Make targetを05に合わせて用意する。
検収: 空CLIのhelp、Swift CPU test target、Python importが動く。不要なGPU初期化がない。

### T02 RinGoソース記録とCore移植
依存: T01。
対象: Sources/IchiGoCore、Tests/IchiGoCoreTests、docs/provenance/ringo-import.json。
手順: 03のallowlistをコピーしhash記録。module名と依存補助だけ調整。移植テストのgoldenを保存する。
検収: 元Coreの移植可能テストがすべて通る。元リポジトリが変更されていない。

### T03 座標・履歴snapshot
依存: T02。
対象: IchiGoFeatures/PositionSnapshot.swift、CoordinateTests。
手順: GTP/SGF/index変換、pass、深い履歴copy、初期配置を実装。
検収: A9/index0、J1/index80、A19/index0、T1/index360、I列拒否。snapshot後の元Board変更が結果を変えない。

### T04 特徴32chとglobal
依存: T03。
対象: FeatureEncoder.swift、FeatureTests、Fixtures/features-v1。
手順: 01の表を順に実装。履歴不足、pass、呼吸点、superko合法性を個別fixture化。
検収: 空盤・コウ・取石・自殺・2passの両サイズ出力が手計算fixtureと完全一致。値が0/1。

### T05 D4と合法手mask
依存: T04。
対象: FeatureSymmetry.swift、Python symmetry.py、shared fixtures。
手順: 8写像と逆写像、policy/ownership/legal変換を共通JSON permutationで定義。
検収: 全8変換を逆変換すると完全一致、pass不変、両サイズでPython/Swift一致。

### T06 Pythonゲートoracle
依存: T01。
対象: Training/ichigo_train/gates.py、tests/test_gates.py。
手順: truth table生成、soft多項式、theta縮約、argmax tie処理を実装。
検収: 16×4=64真理値、a/b実数で直接和と縮約の一致、float64 gradcheckが通る。

### T07 配線とモデル生成
依存: T06。
対象: wiring.py、model.py、configs/model-{tiny,small,base}.json。
手順: 01の固定配線生成と初期化、各層のshape、global/local headsを実装。
検収: seed再現、範囲外offset=0、9/19両forward、head shape検証。配線実体を保存。

### T08 Swift離散ゲートとhead
依存: T01,T06,T07。
対象: LogicModel/ScalarBackend.swift、Heads.swift。
手順: UInt8ゲート、bank参照、境界、ping-pong、Float headを実装。
検収: Pythonが輸出した小fixtureで各層bit完全一致、raw head誤差05以内。

### T09 model exporter
依存: T07。
対象: Training/ichigo_train/export.py、model_format.py。
手順: manifest、wiring/gates/heads、sha、tensor offsets、atomic出力を実装。
検収: 書いて再loadしたhard出力一致。overwrite制御と不完全export隔離を確認。

### T10 Swift model loader
依存: T08,T09。
対象: LogicModel/ModelManifest.swift、ModelLoader.swift、CLI inspect/eval。
手順: 全schema/byte長/hash/範囲チェック後だけ配列へ変換する。
検収: Python→Swift eval一致。truncated、unknown version、NaN、範囲外gate、パス逸脱fixtureを拒否。

M0 gate: T01〜T10完了、`make check-cpu` と `make parity-cpu` 成功。

## B. 学習して9路を動かす（M1）

### T11 SGFリプレイとfeature CLI
依存: T04,T05。
対象: SGFReader/Writer移植、CLI features、reject reporter。
手順: 主変化、AB/AW/PL、pass、途中局面、全履歴を読み、02のJSONLをstream出力。
検収: replay後の盤・手番・legalが固定fixtureと一致。不正棋譜を黙って修正しない。

### T12 教師adapter
依存: T11。
対象: Training/ichigo_train/teacher.py、tests/fake_teacher.py。
手順: subprocess JSONL、id結合、明示rule/視点変換、visits policy、reject/retry。
検収: 順不同・重複・timeout・欠落・白黒反転・passをfake teacherで検証。実教師20局面との保存fixtureも作る。

### T13 棋譜生成とdataset builder
依存: T12。
対象: Scripts/generate_teacher_games.py、dataset.py、build_data.py。
手順: 02-11のRinGo v2 inventory→対応局面照合→target importerを先に実装。照合不能なら既存棋譜の再ラベル化。棋譜がない場合だけ教師自己対局。ID・D4 splitKey、重複排除、shard、manifestを書く。
検収: train/val/testにgame family/position重複なし。合法policy合計1、欠損mask、hash、1shardずつ読み込み確認。既存targetの視点/順序/validity、全sample照合、noResult除外、再ラベルfallbackをfixture検証。

### T14 lossesとoptimizer
依存: T07,T13。
対象: losses.py、optim.py。
手順: 5損失のmask/weight正規化、AdamW group、lr schedule、clipを実装。
検収: 小数値例の手計算一致、全欠損maskでもNaNなし、1stepでtheta/headが更新される。

### T15 train/checkpoint/resume
依存: T14。
対象: train.py、checkpoint.py、metrics.py。
手順: optimizer step基準のloop、RNG/サンプラ保存、soft/hard validation両方、CSVを実装。
検収: CPU deterministicの10step連続と5step+resume5stepが同一。dataset不一致resume拒否。

### T16 離散化スケジュール
依存: T15。
対象: discretize.py、train.py、export.py。
手順: 60/30/10区間、prefix凍結、theta更新停止、head-only期間、best-hard選抜を実装。
検収: freeze前後のlayer状態とoptimizer更新範囲がfixture一致。最後は全層0/1でexportされる。収束不良時は02-11のGumbel-STE対照実験を追加し、forward/gradient surrogate/noiseなしexportを別に検証。

### T17 過学習検収と1GPU pilot
依存: T10,T16。
対象: configs/train-tiny.json、Scripts/train_pilot.sh、reports/pilot。
手順: 16局面fixture→CPU tiny検収→1GPU smallを100step計測→既存9路最大10万局面の2000step pilot→方式選択→9路の初期run。データ生成/教師/生徒/離散化/評価の時間を分離記録。
検収: 02の過学習基準達成、hard-modelがSwiftで同じ出力、VRAM/step時間/run設定保存。

### T18 evaluatorと白視点adapter
依存: T10,T04。
対象: IchiGoEngine/LogicEvaluator.swift、EvaluationAdapter.swift。
手順: snapshot→feature→backend→mask→WDL→白視点変換、capabilitiesを実装。
検収: 黒でe=.8ならwhiteExpected=.2、白なら.8、draw=1なら双方.5。score/ownershipも反転一致。

### T19 pure-tree探索の移植
依存: T02,T18。
対象: Search.swift、SearchSettings.swift、SearchTests。
手順: RinGoのplain-tree経路からModelDesc/Precision依存を外す。03の選択式、terminal、visit数、未対応機能無効化を実装。
検収: fake評価の小木で選択/バックアップ手計算一致、合法手のみ、draw terminal=.5、9/19で完走。

### T20 batching・tree reuse・cancel
依存: T19。
対象: SearchBatch.swift、SearchLifecycleTests。
手順: reservation、要求順序、深いsnapshot、木昇格、generation IDを実装。
検収: 成功/例外/cancelでreservationゼロ、同じleaf二重加算なし、komi/model変更で木破棄。

### T21 基本GTP・自己対局
依存: T19,T11。
対象: IchiGoGTP、CLI gtp/selfplay、GTPTests。
手順: 03の必須コマンドからanalysis/時計以外を実装。genmove共通commit、model9/19選択、SGF出力。
検収: ID/空行/不正入力/undo、両盤サイズのランダムモデル完走。9路学習済みモデルで20局完走。

M1 gate: T11〜T21完了。過学習成功、9路教師holdout改善、学習済み9路20局完走、ランダム合法手baselineとの05の比較。

## C. 9路CGOS優先と後続19路（M2a〜M2c）

### T22 Metal-byte
依存: T10。
対象: LogicMetal/MetalBackend.swift、Resources/logic_byte.metal。
手順: device/queue/buffer管理、1layer1dispatch、最後だけ完了待ち、CPU headへ接続。
検収: 全層bit exact、GPU欠如/command buffer errorの失敗動作、9/19双方。

### T23 Metal-packedとCPU-packed
依存: T22。
対象: logic_packed.metal、PackBits.swift、PackedCPUBackend.swift。
手順: batch方向32bit pack/unpack、valid mask、境界とbank参照を実装。
検収: B=1,31,32,33,63,64,65の全gate/全layer一致、NOT/true末尾padding=0。

### T24 Metal heads
依存: T23。
対象: heads.metal、GlobalReduce.metal。
手順: bit→float、mean/max、local/global projection、ReLU/headをMetal化。同一command bufferにまとめる。
検収: CPU FP32 tolerance以内、policy順位と最終prob検証、head時間を別計測。

### T25 benchmark/auto
依存: T24,T20。
対象: CLI benchmark、BackendSelector、reports/benchmark。
手順: 05のwarmup/latency/throughput/memory測定、batch別backend選択ファイル生成。
検収: このM5上の実測レポートを残す。速くない場合も結果を記録し最速実測backendを選ぶ。

### T26 DDP
依存: T17。
対象: distributed.py、train.py、configs/train-ddp.json。
手順: doctor、rank/sampler、欠損maskのglobal正規化、同期freeze、rank0保存を実装。
検収: 同じglobal batchの1/2/4GPU勾配比較、4GPUで100optimizer stepとresume、意図的rank失敗で全processが終了。throughputとscaling efficiencyも保存。

### T27 19路学習と選抜（M2c、9路CGOS出場後）
依存: T17,T26（複数GPU使用時）,T25。
対象: configs/train-19.json、19路dataset/run、champion19。
手順: 9路と同じsmall、19路専用datasetでpilot→初期学習→hard選抜。9路重みの無検証流用をしない。
検収: 19路holdout指標、Swift parity、20局完走、9路championも引き続き動く。

### T28 時計・watchdog
依存: T20,T21。
対象: TimeManager移植、DeadlineController、TimeTests。
手順: sudden-death、time_left、budget、late GPU結果の隔離、fallback候補を実装。
検収: fake clockでR=0/.05/1/300秒、slow evaluator、同時完了/timeoutでも着手1回。

### T29 analysis・勝率校正
依存: T21,T18,T17。
対象: GTPAnalysis、calibrate.py、model manifest、AnalysisTests。
手順: best move summary、PV16手、score、ownership、raw/search識別、温度校正。
検収: 03の応答をCGOS parserで受理、黒白視点・draw・[0,1]単位一致。test Brier/ECE保存。

### T30 CGOSローカル統合
依存: T17,T25,T28,T29。19路の追加検収時だけT27。
対象: Scripts/cgos、configs/cgos.example、fake server integration tests。
手順: 公開clientをrevision固定、IchiGo argv/state分離、setup replay、backoff、対局間停止。
検収: まず9路20局、途中切断replay、analysisが保存される、対局途中model差替えなし、既存RinGoに干渉しない。19路20局はM2cで追加。

M2a gate: T22〜T26/T28〜T30完了、9路学習済みhardモデル・推論parity・時計・CGOSローカル完走・勝率表示。M2b gate: T37の9路実運用工程で公開CGOS20局完了。M2c gate: 後続T27と19路統合検収。19路学習を9路出場の依存にしない。

## D. 強化と運用（M3/M4）

### T31 対局評価runner
依存: T21。
対象: Scripts/match.py、stats.py、reports/matches。
手順: 対応開局の色交換、同時間/同visits別リーグ、SGFとseed保存、paired bootstrap。
検収: 勝敗集計の手計算fixture、timeout別集計、CIの再現性、05のpromotion report生成。

### T32 失着・データ再ラベル
依存: T31,T12。
対象: analyze_losses.py、relabel.py。
手順: holdout以外の敗局で教師との差・終盤pass・シチョウ・コウを抽出し512visits再ラベル。
検収: test局面混入なし、元データ50%以上保持、変更前後の同時間対局レポート。

### T33 容量・接続・通常CNN比較
依存: T31,T17。19路実験のみT27。
対象: configs/experiments、baseline_cnn.py、experiment runner。
手順: C/L/配線seedを1項目ずつ変更。研究用CNNは32→64の3x3 stem、64chの2-conv residual block×4、同じglobal/local head、同じ5損失で比較する。
検収: 学習予算・局面数・hard gap・棋力・推論時間の表。CNNは診断用で本番logicモデルとは明示的に区別。

### T34 gate簡約・学習kernel最適化
依存: T25,T31。
対象: optimizer.py、必要ならCUDA extension、Metal specialized kernel。
手順: 定数伝播/恒等線/未使用gate削除をgraph IRへ実装。学習kernelはforward/backward referenceを維持。
検収: 簡約前後全bit一致、head一致、9/19境界一致、実対局時間短縮。効果がなければoffを維持。

### T35 自己対局targetとreplay buffer
依存: T31,T16。
対象: selfplay output、replay.py、RL dataset builder。
手順: 02-9のvisit target/noise/温度/結果mask、世代ID、bounded bufferを実装。
検収: 正規化・手番視点・打切りmask、再起動して同じbuffer索引、教師50%混合。

### T36 RL iterationとchampion選抜
依存: T35,T32。
対象: iterate.py、promotion report、checkpoint registry。
手順: 生成→学習→hard化→parity→対局→候補登録、各段階を再実行可能にする。
検収: 小規模1iteration完走、失敗candidateはchampionにならない、手動で選抜結果を確認可能。

### T37 実運用パッケージ
依存: T30,T31。
対象: README運用節、Scripts/release、configs/cgos、runbook。
手順: releaseビルド、metallib resources、model hash固定、CGOS起動/通常停止/ログ復元/旧model復帰を文書化。
検収: クリーンな作業ディレクトリからdoctor→inspect→GTP→ローカルCGOSを再現。接続先とアカウントを設定した運用段階で、公開9路CGOSの20完了局と解析受け渡しを確認する（M2b）。実接続できていなければM2aまでと記録する。

推奨順序: A → B → T22〜T26 → T28〜T30 → T37（9路公開CGOS）。T31はM1直後に実装して初期から棋力を測る。T27の19路本学習は9路出場後へ置く。M2b達成後、9路はT32→T33→T35→T36で改善し、計測で必要ならT34を行う。

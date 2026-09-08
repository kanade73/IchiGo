# データ生成・学習・強化計画

## 1. 最初の学習経路

```text
棋譜/教師自己対局
 → Swiftによるリプレイ・合法性・特徴抽出
 → 外部KataGo analysisで同じ局面を評価
 → shard生成（局面IDで結合）
 → Pythonでsoft学習・hard検証
 → 段階的離散化・head再学習
 → .ichigo export
 → Swift CPU/Metal parity
 → 固定条件の対局評価
```

教師 API は [KataGo analysis の公式仕様](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md) を使う。出力順序に依存せず `id,turnNumber` で結合する。teacher executable・モデルSHA256・設定・revisionを記録し、run途中で切り替えない。

初期経路はRinGoの既存9路データの再利用を優先する。T13で `.nngd` v2のラベルimporterを作り、対応棋譜からIchiGo特徴を再生成する。ラベルと局面を照合できた分は教師の再実行を省き、足りない分だけ再ラベル化する。KataGo V7の22chとIchiGoの32chは別物であり、履歴が復元できない既存shardをそのまま入力へ変換しない。詳細は11節。

## 2. 中間局面 JSONL

`ichigo features --sgf-dir DIR --out positions.jsonl --size S --rules cgos-area-psk-v1`。
SGF の主変化のみ、UTF-8、初期配置AB/AW・PL・passを処理する。盤サイズ不一致、未対応ルール、自殺・superko違反はファイル名と手数付きで rejects.jsonl に出力し、その棋譜を除外。SGF の結果だけを盤上終局とみなさない。

各行に `schemaVersion=1, positionId, gameId, boardSize, komi, rulesId, initialStones, initialPlayer, moves, turnNumber, toMove, spatial, global, legal` を持つ。moves はその局面までの `[color,GTPmove]` リスト、color B/W。spatial/global/ legal は01の一次元配列。positionIdはUTF-8の正規化JSON（キー辞書順、空白なし）の `{boardSize,komi,rulesId,initialStones,initialPlayer,moves}` のSHA256。initialStonesは色→y→xの順に整列。gameIdは同様の全棋譜内容のSHA256で、結果・コメント・ファイルパスを含めない。
JSONL はデバッグ・結合用。大規模実験はchunk streamingし、棋譜全体や特徴全体をRAMへ一括loadしない。

## 3. 教師ラベル

教師の analysis request は boardXSize=boardYSize=S、initialStones、initialPlayer、moves、komi、明示的ルールJSON、analyzeTurns=[turnNumber]、maxVisits、includeOwnership=true を指定。rulesを名前から推測しない。教師設定の結果視点を手番に固定できる場合も、adapter はblack/white/to-moveを明示的に受け取る。

- policy: moveInfos の visits を合法手へ写像し正規化。sum=0はそのラベルを拒否。教師が訪問していない合法手のtargetは0。固定教師のmaxVisits=128を初期値とする。
- expectedResult: rootInfo.winrate を手番視点の期待得点 [0,1] として記録。勝ち確率と引分確率をこの1値から分解しない。
- score: rootInfo.scoreLead を手番視点に変換した目差。
- ownership: 教師の配列座標と視点を adapter fixture で検証して手番視点に変換。
- wdl: 棋譜が規定ルールで盤上終局し、信頼できる結果がある場合のみ別ターゲットを付与。投了は勝敗を使えるが、score/ownership の正解を投了時盤面から作らない。

教師の未対応ルール自動変更、NaN、サイズ不一致、合法性不一致は reject。stderrの警告も保存する。タイムアウト60秒/要求を初期値、同一idを最大2回再試行し、それでも失敗したらreject。同一idの重複応答を二重追加しない。キュー深さはGPU実測で調整する。

初期棋譜がなければ、外部KataGoを用いる `Scripts/generate_teacher_games.py` がGTPで自己対局する。序盤20手まで訪問分布温度1、その後0.2、投了なし、seed保存。まず9路100局から開始し、19路はM2c開始時に100局を生成する。手数上限は4*S*S、到達時は打切り扱いで勝敗正解を付けず、教師評価局面としてのみ使う。SGF・着手列・教師設定を保存する。

## 4. shard v1

`dataset/manifest.json`, `train/*.npz`, `validation/*.npz`, `test/*.npz`, `positions-index.jsonl`。
NPZ は pickleを禁止し `allow_pickle=False`。1 shard は最大4096局面、単一盤サイズ。各配列は以下。

| name | dtype / shape | 意味 |
|---|---|---|
| spatial | uint8[N,S,S,32] | 0/1 |
| global | float32[N,4] | 01定義 |
| legal | uint8[N,S*S+1] | 合法手 |
| policy | float32[N,S*S+1] | 合計1、違法手0 |
| expected_result | float32[N] | 教師の期待得点 |
| score | float32[N] | 手番視点目差 |
| ownership | float32[N,S,S] | [-1,1] |
| wdl | float32[N,3] | win,draw,loss、既知なら合計1 |
| target_mask | uint8[N,5] | policy,expected_result,score,ownership,wdl の有無 |
| sample_weight | float32[N] | 初期1、正有限 |
| position_id | uint8[N,32] | SHA256 raw bytes |
| game_id | uint8[N,32] | SHA256 raw bytes |

欠損targetは配列値0かつmask=0。teacher label は最初の4mask=1、wdlは通常0。終局勝敗ターゲットを併用する場合、wdlはその局面の手番視点に変換する。indexはIDに対する棋譜参照・手数・教師IDを持つ。manifestには各shardのSHA256/局面数/盤サイズ/分割、schema・feature・rule version、seed、教師来歴を含める。

分割は game_id先頭8byteをbig-endian整数にした値 mod100: 0..89 train、90..94 validation、95..99 test。学習前に同一正規化棋譜をまとめ、D4同型棋譜も同じsplitへ置くため8変換の最小gameIdをsplitKeyに用いる。開局の共有等で重複するpositionIdは train→validation→test の優先順で一箇所に残す（評価側の重複を除く）。拡張は分割後、trainのみで行う。holdoutが空なら明示エラーでデータを増やす。小さな過学習fixtureはこの分割を使わない。

## 5. 学習式

出力 p=合法手softmax(policy)、q=softmax(wdl)、e=q_win+0.5*q_draw。

```text
Lpolicy = -Σ target_policy * log(p)
Lexpected = -t*log(e) -(1-t)*log(1-e)
Lwdl = -Σ target_wdl * log(q)
Lscore = Huber((pred_score-target_score)/S, delta=1)
Lowner = mean_xy((pred_owner-target_owner)^2)
L = Lpolicy + Lexpected + 0.5*Lwdl + 0.25*Lscore + 0.25*Lowner
```

各項はmask=1のsample_weight総和で割って別々に平均。該当0なら項は0でありNaNにしない。log入力は1e-7以上へclip。policy違法手はlog_softmaxのmask処理で0*−infのNaNを避ける。学習モデルには校正temperatureを適用しない。引分ラベルが不足する段階のwdl内訳は校正未検証と明示する。

AdamW: gate logits lr=0.01/weight_decay=0、heads lr=0.001/weight_decay=1e-4、betas=(0.9,0.999)、eps=1e-8。global norm clip=1。warmupは全stepの5%、その後cosineで初期lrの10%まで。gate logits/softmaxはFP32を維持する。初期実装は全体FP32。

初期学習設定: small、microbatch=8/GPU、effective batch=128、max optimizer steps=20000、validation間隔=500、checkpoint間隔=1000、seed=20260908。これらは採用済み最適値ではない。OOMならmicrobatchを半減しgradient accumulationを増やす。max_stepsはoptimizer更新回数で、microstepではない。

## 6. 離散化を学習経路に組み込む

softが良いだけのモデルをchampionにしない。validationごとにすべてのgateをargmaxに固定したhard forwardも走らせる。hardは常に0/1入力と真理値表評価で、単にtauを下げたsoft値ではない。

1. 最初の60%step: tau=1。全層soft。
2. 次の30%: tauを1→0.2へ線形減少。この区間をL等分し、各区間の終わりに入力側から1層ずつargmax固定。固定層はtheta更新を停止し、出力をhard gateにする。
3. 最後の10%: 全gateと配線を固定し、headsだけ再学習。

段階2ではprefixがhardで残りはsoft。固定層を通じて未固定の前段を学習する必要がない順序を守る。固定前後のhard-validation lossを保存。固定でlossが直前hard値より20%以上悪化したらそのrunを警告状態にし、直前checkpointから区間を2倍に延ばす設定の別runを1回だけ試す。改善しない場合は容量/配線/初期化実験へ進む。都合の良いsoftモデルへ戻して完了扱いにしない。

選抜はhard validation総損失の最小、同値なら早いstep。選抜checkpointが全層固定前でもexportのgateは全層argmax。head再学習後との対局比較を残す。

checkpointはmodel/optimizer/scheduler/step/凍結mask/全RNG状態/sampler位置/config/dataset hash/配線を保持。resume時にfeatureVersion、配線、datasetを照合。不一致は新runとして開始する。

## 7. 小規模検証から複数GPUへ

最初はCPU tinyで、16個の固定局面targetを2000step以内にpolicy top1≥90%、value MAE≤0.1へ過学習させる。教師targetには局面ごとに異なる合法手・値を使う。達成しなければ大規模runは禁止。hardでもtop1≥80%、MAE≤0.15を初期検収値とする。失敗を閾値変更で隠さず方式を修正する。

A4000環境では `doctor` が nvidia-smi、GPU名/VRAM、torch/CUDA/driver、GPU数、ディスク残量をJSON化。1GPUでforward/backward/100step時間/peak memoryを測定してからDDPへ進む。
正式対応は同一サーバー内の1/2/4GPU。4枚では4process、microbatch=8、accumulation=4でeffective batch=128となる。1枚ではaccumulation=16、2枚では8として、比較時のglobal batchを揃える。4枚のVRAMが単一の大きなVRAMになる方式ではなく、各GPUにモデル・optimizerの複製を置く。複数サーバーDDPは将来拡張とし、初期の4枚対応と区別する。

DDPは1GPU=1process、NCCL。データ分割はDistributedSampler、epochごとにset_epoch、同一stepで全rankが同じ盤サイズとfreeze境界を使う。DDPがデータを自動分割すると思い込まない。根拠: [PyTorch DDP公式](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
N GPUs×microbatch×accumulation=effective batchを保存。割り切れない設定は拒否。欠損maskの損失はglobal有効重み総和をall-reduceし、DDPの勾配平均を考慮して各rankの分子にworld_sizeを掛けて正規化する。rankごとの単純平均は不可。validationはrank0単独で全holdoutを実行してbroadcast。他rankは同じ同期点で待機する。保存はrank0のみ。

FP32安定後にheadsのみAMP、activation checkpointing、fused CUDA kernelの順で測る。Torch標準実装を数値oracleとして残す。単体GPUで収まらないモデルがDDPで収まるという前提は禁止。

## 8. 棋力を伸ばす実験順序

| 優先 | 実験 | 固定する比較条件 | 採用条件 |
|---|---|---|---|
| 1 | hard gap改善、head再学習 | 同じ配線・データ・seed集合 | hard lossと対局改善 |
| 2 | データ10万→100万局面/盤サイズ | 同じsmallと探索 | 同時間勝率改善 |
| 3 | 苦手局面を教師512visitsで再ラベル | 元データ50%以上保持 | 戦術/対局両方改善 |
| 4 | small→base、配線seed3種 | 同学習予算・同時間対局 | 推論低下込みで改善 |
| 5 | 学習済みgateの定数伝播・dead gate除去 | 同一モデル入出力 | bit exactかつ実時間短縮 |
| 6 | 9/19共同学習 | 盤サイズ別batchを交互、同局面数 | 片側の劣化なし |
| 7 | 自己対局RL | champion対戦セット固定 | 統計的改善 |

実験1〜4は「初めて動いてから考える」作業にせず、初期からmetrics/manifest/runnerを作って実行できるようにする。深さ・幅・学習率を同時に変更しない。基準となる通常CNN（同じ特徴とheads、小型3x3 residual trunk）をPython上で一度学習し、データ不良か論理回路の表現力不足かを判定する研究用baselineを後続チケットで用意する。

## 9. 自己対局RL v1（M4）

学習済みhard championから開始、ランダム初期化の純自己対局は初期経路にしない。自己対局PUCTのroot priorに0.25 Dirichlet noise、各合法手alpha=10/legalCount、最初の20手temperature=1、その後0.2。学習targetはroot訪問回数分布、勝敗は規定ルール終局の手番視点。投了は初期off。

1 iteration=各盤サイズ1000完了局を初期値。4*S*S手で打ち切った局は結果損失をmaskし、教師再ラベル用に隔離する。replayは直近最大100万局面。batchはteacher50%、selfplay50%から始める。自己対局のscore/ownershipは盤上で終局した正解のみ、教師予測との出所を別メタデータに記録する。

離散championから再学習する際はthetaを採用gate=3/他0で再構築、配線とheadsを引き継ぐ。これを元soft checkpoint再開と混同しない。02-6の離散化工程を毎iteration適用する。candidate生成とchampion昇格を分け、05の固定対局ゲートを通ったものだけを次の生成器に使う。4GPU運用の初期構成は、ラベルを事前生成して4枚すべてでDDP学習する時間分割とする。継続学習では教師/生成2枚＋学習2枚も比較する。同じGPUを教師とDDPで競合させず、GPU割当とeffective batchを記録する。3枚DDPはeffective batchの整除条件を満たす設定を別途指定した場合のみ拡張対象とする。

## 10. 学習CLIと設定ファイルの契約

以下は実装後に提供するコマンド例。CLIは `python -m ichigo_train` に統一し、Trainingをeditable installして利用する。

```sh
python -m ichigo_train doctor --out reports/cuda-doctor.json
python -m ichigo_train label --positions data/positions-9.jsonl --teacher-bin /path/to/katago --teacher-model /path/to/teacher.bin.gz --visits 128 --out data/labels-9.jsonl
python -m ichigo_train build-data --positions data/positions-9.jsonl --labels data/labels-9.jsonl --out data/dataset-9
python -m ichigo_train train --config configs/train-9.json
python -m ichigo_train train --config configs/train-9.json --resume runs/small-9/checkpoint-latest.pt
python -m ichigo_train evaluate --checkpoint runs/small-9/checkpoint-best-hard.pt --data data/dataset-9 --split validation --mode hard --out reports/small-9-val.json
python -m ichigo_train export --checkpoint runs/small-9/checkpoint-best-hard.pt --out models/small-9.ichigo
```

`configs/train-9.json` の最小実用例:

```json
{
  "schemaVersion":1,
  "runId":"small-9-seed20260908",
  "data":"data/dataset-9",
  "out":"runs/small-9",
  "boardSize":9,
  "profile":"small",
  "seed":20260908,
  "device":"cuda",
  "microBatch":8,
  "effectiveBatch":128,
  "maxSteps":20000,
  "validationInterval":500,
  "checkpointInterval":1000,
  "gateLearningRate":0.01,
  "headLearningRate":0.001,
  "headWeightDecay":0.0001,
  "gradClipNorm":1.0,
  "discretization":"prefix-60-30-10",
  "augmentation":"d4",
  "precision":"fp32"
}
```

未知キーはエラー。省略可能なキーはこの例の値を既定とするが、data/out/boardSize/runIdは必須。profileは01の3種、deviceはcpu/cuda、precisionは初期fp32のみ。discretizationはprefix-60-30-10、対照実験実装後のみgumbel-ste-90-10も受理する。コマンドで変更できる項目はconfigとresumeだけとし、学習設定overrideの優先順位を増やさない。設定の相対パスは起動cwdから解決し、解決前の設定と解決済み実行設定をrun内へ両方保存する。
19路はboardSize/data/out/runIdを変更する。DDPは `torchrun --standalone --nproc_per_node=N -m ichigo_train train --config PATH`、Nは実際のGPU数。torchrun環境変数がなければ1process。

終了コード: 0正常、2設定/schema不正、3入力/教師/デバイス失敗、4非有限loss/勾配等の学習異常。SIGINTは現在の安全なoptimizer-step境界でcheckpointを保存して終了。異常なcheckpointでlatest正常版を上書きしない。
1runの成果は `config.json, resolved-config.json, environment.json, metrics.csv, training.log, checkpoint-latest.pt, checkpoint-best-hard.pt, wiring.npz, validation/*.json`。終了時に `run-summary.json` を書き、学習/検証したstep数、未達ゲート、best modelの場所を持たせる。

初回pilotではGPUを予約し続けず100stepで一度終了する。概算学習時間は `実測秒/optimizer step × 残りstep + 実測validation時間 × 回数 + export/evaluation時間` で算出する。ラベル生成時間は別計測。ユーザーの許容時間は未指定なので、何週間かかる巨大runを既定動作にせず、maxStepsのある有限runを積み重ねる。

### 4GPU検収の具体条件

```sh
torchrun --standalone --nproc_per_node=4 -m ichigo_train train --config configs/train-9.json
```

T26で1/2/4GPUそれぞれ同じglobal batchの勾配を比較し、4GPUで100optimizer step、checkpoint/resume、rank障害時の全process終了を検証する。DDP wrapperでrank0だけvalidation forwardするとcollective待ちが起こり得るため、rank0は同期後のunwrapped moduleで評価し、全rankで結果broadcastと次stepの同期を行う。
性能はsamples/sec、step時間、GPU別peak memory、4GPU throughput / 1GPU throughput、scaling efficiency=throughput比/4を保存。4倍速を完了条件にしない。小モデルで通信負荷が勝つ場合は4枚対応を維持し、運用は1枚ずつ4seedの独立実験や2枚学習＋2枚ラベル生成も選べる。


## 11. RinGo既存データ優先と学習時間の短縮

蒸留は当初から採用しているが、初回9路CGOSでは特にデータ生成を省く経路を優先する。総所要時間を `棋譜生成 + 教師ラベル生成 + 特徴変換/I/O + 生徒学習 + hard化/head調整 + 検証` に分けて計測する。既存ラベルがあれば最初の2項を省ける。蒸留による生徒の必要step減少は実験対象であり、逆伝播1stepの時間や離散化問題が自動的に消えるとはしない。

### 11.1 再利用の優先順

1. 対応棋譜・生成設定・局面索引がある `.nngd` v2: 既存targetをimportし、棋譜から32ch特徴を作り直す。
2. 棋譜だけある、またはshardとの対応が証明できない: 棋譜生成を省き、固定教師で一度だけラベル生成してcacheする。
3. 利用可能な棋譜がない: 既定の教師自己対局を実行する。

確認できたのは形式と生成コードであり、実際のコーパス量、教師品質、棋譜/サンプル対応の現存はまだ未調査。実装時に入力パスを指定しinventory reportを作る。稼働中RinGoのstate・private model・アカウントを収集対象にしない。

### 11.2 importerの契約

CLI: `python -m ichigo_train import-ringo --shards DIR --positions FILE --mapping FILE --out DIR`。
mappingはJSONL、1行に `shardSha256, sampleIndex, positionId, symmetryId, sourceRunId` を持つ。symmetryIdはRinGo側の変換番号とし、元の座標へ逆変換してからIchiGo座標へ揃える。索引が存在しない場合は、元コードrevision・SGF列挙順・skip条件・sampling・symmetry設定を再現して候補mappingを生成し、全サンプルのV7 spatial/global一致を確認する。盤面配置だけや先頭数件だけの一致では承認しない。入力特徴の一致だけでも完全履歴は証明できないため、生成順序・出所の裏付けも必須。再現できなければ経路2へ戻す。

RinGoData v2は24byte header、sample byte数は9路7631。readerはmagic/version/22 spatial/19 global/長さ/有限値を検証し、1sampleずつstreamする。v1はこのimporterでは拒否する。

- dense policyは手番視点、pass末尾。合法性と総和を検証し、不一致sampleはreject。one-hotをsoft蒸留ラベルと呼ばず、ラベル種別を来歴に保存する。
- value順序はwin/loss/noResultであり、IchiGoのwin/draw/lossとは異なる。noResultが1e-6を超えるsampleはv1 importerでは除外。残りはexpected_result=win/(win+loss)。既存3成分をwdlへコピーせずwdl mask=0。真正の棋譜結果がある場合だけ別途wdlを構成する。
- scoreはscoreValidに従い、投了時の合成±15を学習しない。ownershipもownershipValidに従う。int8値をfloat32へ変換し、座標と手番視点を確認する。
- rules、komi、teacher/model hash、生成法、元のtrain/val/testを来歴へ保存する。既存splitを跨いで移動させず、IchiGoのgame-family分離も満たす。矛盾するfamilyは全体を除外する。
- importedとfreshラベルはsourceTypeで区別する。既存のfinal outcome targetと教師の途中評価targetを同じ生成法として記録しない。

出力は02-4の正規shardとimport-report.json。reportに照合数、再利用数、reject理由、無効score/owner数、規則別数、ラベル種類、節約した教師呼び出し数を含める。NNGD内部にはgameIdや全棋譜がないため、shard単独から完全履歴が得られると仮定しない。

### 11.3 有限pilotで方式を選ぶ

最初は照合済み9路最大10万局面、small、同一holdoutで比較する。16局面過学習後、まず100stepで速度、その後2000optimizer stepまでのpilotを実施する。prefixスケジュールはpilotの全step数を基準に再計算する。生徒train中はteacher processを起動しない。

baselineは既定prefix法。収束が遅い/soft-hard差が大きい場合、Gumbel noise + straight-through推定の実験をT16内で追加し、同じwall-clock予算とhard評価で比較する。研究根拠は [Mind the Gap](https://arxiv.org/abs/2506.07500)。論文では画像分類で学習時間短縮を報告しているが、囲碁への速度倍率は未検証。

実験モード `gumbel-ste-90-10` のIchiGo試験契約: 学習用p=softmax((theta+Gumbel(0,1))/tau)、h=one_hot(argmax(p))、pST=h-stop_gradient(p)+pを01の16関数混合に使う。tau=1固定、最初の90%で全層を学習、残り10%はnoiseなしargmaxで全gate固定しheadのみ学習。noiseは層/channel/gateごとで空間共有、rank別seedとcheckpoint RNGを保存。validation/exportは常にnoiseなしargmax(theta)。これは論文の完全再現を主張する設定ではなく、共通oracleで試せる対照実験。
検収はforwardが0/1、gradientは固定noise時のsoft surrogateと一致、noiseなしexportのPython/Swift parity、hard holdout loss/棋力/所要時間。STEのbackwardは真の離散関数の微分ではないので、離散forwardへの通常gradcheck合格を要求しない。

4GPUでは既定DDPに加え1GPUずつ配線seed/学習方式を比較する使い方も可能。DDPの効率が低い小モデルでは後者を先に使う。採用方式を決めてから全量学習へ進み、pilotの結果なしに数週間のrunへ拡大しない。

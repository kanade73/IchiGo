# エンジン・勝率・GTP・CGOS

## 1. RinGo の取り込み境界

参照先 `~/dev/univ/koubou/katago-mlx`。調査時 HEAD は `9d07c47cdf97f9bc93c619561ca980354d124ffc`、作業ツリーには未コミット変更がある。HEADだけで現物を再現できないため、移植時は対象ファイルのSHA256・元パス・HEAD・dirtyフラグを `docs/provenance/ringo-import.json` に記録する。自作コードのコピー許可はユーザーから得ている。

| 参照対象 | 方針 |
|---|---|
| Sources/RinGoCore/*.swift | Coreへコピー、module importのみ調整して既存ルールテストを先に通す |
| Board/History/Rules/Area/Ladder/Hash/Symmetry | 一式の依存関係を保つ。書き直さない |
| NNInputs.swift | NNPos等必要な座標補助だけ抽出。V7をIchiGo特徴と混同しない |
| Engine/NNEvaluator.swift | NNRequest/NNEvaluatingという境界を参考に専用actorを作る。MLX本体はコピーしない |
| Engine/NNOutput.swift | plain出力型を抽出。KataGo postprocessorは使わない |
| Engine/Search.swift, SearchSettings.swift, SearchRandom.swift | pure-tree経路を移植。下記の限定設定で先に検証 |
| Engine/TimeManager.swift | 時計ロジックと関連テストを移植 |
| Engine/GTPEngine.swift, GTPAnalysis.swift | 必要なGTPコマンド・応答整形を移植しモデル依存を注入 |
| Engine/SGFReader.swift, SGFWriter.swift | CLI特徴抽出・棋譜記録用に移植 |
| Scripts/cgos/safe_cgos_client.py, tests | 通信・再接続・停止の参考。IchiGo用に設定と状態ディレクトリを分離 |

参照リポジトリの `.tools/`、アカウント、稼働中プロセス、モデル、運用データはコピーしない。元リポジトリは編集しない。CGOSクライアントは公開ソースのrevision固定で取得する。移植ファイルの著作権表示を保持し、必要な第三者通知を更新する。

## 2. API と状態所有

```swift
struct ModelCapabilities: Sendable {
    let boardSizes: Set<Int>
    let rulesID: String
    let hasOwnership: Bool
    let hasScoreUncertainty: Bool // v1=false
}
struct PositionSnapshot: Sendable {
    // deep-copied Board, BoardHistory, last 8 stone layouts,
    // toMove, komi, moveNumber, legal moves, full-state fingerprint
}
struct LogicEvaluation: Sendable {
    let policy: [Float]            // 合法手の合計1、違法手0
    let winDrawLoss: [Float]       // 手番視点
    let expectedResult: Float     // win + draw/2
    let scoreMean: Float          // 手番視点
    let ownership: [Float]        // 手番視点
}
protocol PositionEvaluating: Actor {
    func evaluate(_ positions: [PositionSnapshot]) async throws -> [LogicEvaluation]
    func preWarm(size: Int) async throws
}
```

PositionSnapshotは参照共有のまま `@unchecked Sendable` で済ませない。探索が入力を変更できないようdeep copyする。Featuresはsnapshotから生成し、配列をactor境界で渡す。NN要求の重複排除は同じ完全状態と同じモデルの時だけ。

RinGo Search の `ModelDesc` / `KataGoNetwork.Precision` をコンストラクタから取り除き、ModelCapabilities / evaluator を受け取る。偽のKataGoネットやダミーMLXモデルを作って依存解決しない。`import RinGoModel`を残さない。

## 3. 後処理と視点変換

logitsの安定softmaxは最大値を引く。policy合法手maskを適用して正規化。合法手がない非終局、サイズ不一致、非有限値はエラー。学習時のmaskと推論時のmaskは同じルールと同じ局面から得る。
内部LogicEvaluationは手番視点。RinGoの探索互換層は白視点とする。

```text
e = pWin + 0.5*pDraw
whiteExpected = (toMove == white) ? e : 1-e
whiteWinValue = 2*whiteExpected-1
whiteScore = (toMove == white) ? score : -score
whiteOwnership = (toMove == white) ? ownership : -ownership
```

RinGo NNOutputの互換フィールドは `whiteWinProb=whiteExpected`、`whiteLossProb=1-whiteExpected`、`whiteNoResultProb=0`。このwhiteWinProbは期待得点互換値であり、厳密な白勝利確率ではない。引分をKataGoのno-resultに入れない。raw WDLはLogicEvaluation/解析ログに別途保持する。
`whiteLead=whiteScoreMean=whiteScore`。互換上必須の `whiteScoreMeanSq=whiteScore*whiteScore`、varTimeLeft/shortterm errors=0。これらは未推定であるため、分散・誤差を利用する探索機能はcapabilityで禁止する。値0を「高確信」と解釈させない。
policyの互換層だけはRinGoの違法手sentinel=-1に変換する。LogicModel/学習データは違法手0を維持する。

## 4. 探索 v1

初期はpure-tree PUCT、score utility係数=0、win/loss utility係数=1、no-result utility=0。graph/transposition merge、LCB、uncertaintyによる重み付け、dynamic score utility、ponder、opening bookはoff。tree reuseとleaf batchは早期に有効化する。これらの停止は無学習の補助値を使わないための初期範囲であり、ルール合法性を緩めるものではない。

標準設定として下記の選択式へ対応させ、RinGo固有の追加係数は無効化する。

```text
Qparent(a) = parentが白 ? Qwhite(a) : -Qwhite(a)
U(a) = 1.5 * prior(a) * sqrt(max(1,Nparent)) / (1+Nedge(a))
select = argmax(Qparent(a)+U(a))
```

未訪問Qは親のNN白評価を親視点に変換したもの（初期FPU reduction=0）。同点はpolicy index昇順。バックアップは常に白視点、深さごとの符号反転を重ねない。rootの初回NN評価も1visitと数える。着手は最大edge visits、同点はprior、さらにindex昇順。評価モードのtemperature=0、noise=0。

batched leaf selectionはedgeへin-flight reservationを置く。選択式のNedgeにreservation数を加え、Qparentには仮の負け値-1をreservation分だけ加重して選択する。成功/失敗/cancelの全経路でreservationを必ず解除し、実訪問に二重算入しない。terminal leafはNNを呼ばない。

終局判定・得点計算はCore。自己対局手数上限は終局勝敗を捏造する条件ではない。pass後も勝手な目数推定だけで終了しない。木の再利用は実着手の子にrootを移すが、komi/rules/model/boardsize/clear/undoで無効化する。

cacheは初期off。後続有効化時のkeyにmodel payload hash、featureVersion、rule全フィールド、komi、toMove、盤配置、過去7手、完全superko履歴、pass状態を含める。128bit hash一致だけでなく完全fingerprint一致を確認。特徴だけ一致してもsuperko状態が異なる場合は探索ノードを共有しない。

初期上限: NN cache=0、tree nodes=100000、leaf batch=8、evaluator queue=64。メモリbudgetはプロセスRSS 8GiB目標。上限到達時は追加展開を止めて現在のroot最善手を返す。探索継続のための無制限ノード生成は禁止。速度測定後、batch=1〜64を調整する。

## 5. ルールと得点

v1 rulesId=`cgos-area-psk-v1`:
area scoring、positional superko、suicide=false、tax=none、button=false、handicap bonus=zero、friendlyPassOk=true。komiはGTP入力、初期値9路7.0/19路7.5。実サーバーから送られたkomiで必ず上書き。半整数か整数、範囲[-150,150]のみ受理。日本ルール、territory scoring、任意ルール切り替えは未対応として拒否する。

単純コウだけに簡略化しない。二連続pass、superkoに対するpass例外、seki中立点は移植Coreの対応テストと参照KataGoで一致させる。area scoringでも残っている死に石をownershipから勝手に除去しない。学習・自己対局では盤上で解決して終局する。GTP final_scoreは規定Coreの盤上得点、途中局面は暫定値であることをstderrに残す。CGOSの公式結果はサーバーgameoverを正とし、ローカル推定で接続を終えない。

## 6. CLI

| コマンド | 必須引数/成果 |
|---|---|
| `ichigo doctor` | CPU/OS/Metal能力をJSON出力。個体識別番号は出さない |
| `ichigo inspect --model PATH` | manifest検証と形状・hash表示 |
| `ichigo eval --model PATH --position FILE --backend cpu` | raw出力とLogicEvaluationをJSON、学習fixtureと比較 |
| `ichigo features --sgf-dir DIR --out FILE --size 9` | 02のJSONL |
| `ichigo gtp --model-9 PATH --model-19 PATH --backend auto` | stdin/stdout GTP、片側modelのみも可 |
| `ichigo benchmark --model PATH --positions FILE --batches 1,8,32` | 05のJSON/CSV |
| `ichigo selfplay --model PATH --games N --out DIR --seed N` | SGF+root visit targets |

GTPの通常ログはstderrだけ。終了コード0正常/2引数やモデル不正/3推論実行失敗。CLIのhelpは未実装フラグを列挙しない。すべての計測・対局結果にモデルhashを付ける。

## 7. GTP 状態機械

必須: protocol_version, name, version, known_command, list_commands, boardsize, clear_board, komi, play, genmove, time_settings, time_left, undo, showboard, final_score, quit, kata-genmove_analyze。
name=IchiGo、protocol_version=2。行頭の数値IDを成功/失敗応答にそのまま返す。成功 `=ID payload\n\n`、失敗 `?ID message\n\n`。IDなしなら記号直後は空白。空行・#コメント・CRLFを処理。

- boardsizeは9/19かつ対応modelがある場合だけ受理。受理時盤/履歴/探索をクリアし対応modelをprewarm。失敗時元状態を保持。
- komi変更は履歴のルールと探索cacheを更新する。保存モデルのsupported ruleを確認。
- playは合法性確認後に1回だけ状態更新。相手色の指定などGTP上の明示色はCoreに渡し、履歴手番整合を処理。通常の交互対局以外のセットアップは専用replayで検証する。
- genmove/kata-genmove_analyzeは同じgenerateMove関数を使う。探索→合法性最終確認→着手commit→応答。formatterから再着手しない。
- undoは盤だけでなく履歴/pass/ko/手番を復元し木を破棄する。空履歴はエラー。
- 既に盤上終局してgenmoveされた場合、passを返して状態を二重更新しない。
- resignは初期off。後続では校正済み勝率閾値・連続判定・投了なし検証を追加してから有効化。

## 8. 時計

time_settings main byo stones を受理し、まずCGOSで必要な byo=0,stones=0 のsudden deathを正式対応。byo>0はv1でエラーとして未対応を明示し、list_commandsだけで秒読み対応を主張しない。time_leftは色ごとにサーバー値を保存し、受信値を最優先する。

単調時計を使用。remaining=R秒、moveNumber=mとして:
`reserve=max(0.1,min(2.0,0.01*R))`、`estimatedMoves=max(10,0.6*S*S-m/2)`、`budget=max(0,min((R-reserve)/estimatedMoves,R-reserve))`。
探索開始前の特徴/推論準備時間もbudgetに含める。deadline前に新batchを止める余裕は `max(0.01,2*直近batch p95)` 秒。budgetがそれ以下なら保存済root候補、なければ合法手のpolicy最大、NNを待てないなら合法点index最小（点がなければpass）を即時返す。
GPUの処理中断ができなくても、deadline watchdogで応答経路を解放し、遅延結果を次手に混入させない。requestにgeneration IDを付ける。time_left=0、CPU fallback、極端に遅いfake evaluatorのfixtureを必須にする。watchdogと正常完了の両方が着手をcommitしないことをテストする。

## 9. 勝率表示と CGOS

CGOSが対応するanalysis経路は `kata-genmove_analyze` を採用する。CGOS側がwinrate/score/pvを受け取れる根拠は [CGOS公開実装](https://github.com/zakki/cgos)。対応クライアントrevisionと実際のparserで検証する。

```text
入力: 17 kata-genmove_analyze b
応答:
=17
info move D4 visits 80 winrate 0.620000 scoreLead 2.300000 prior 0.120000 order 0 pv D4 E4
play D4

```

winrateは要求color視点の探索期待得点[0,1]。1万倍の整数にしない。scoreLeadは同視点の目差。PVは実際の子ノードを最大訪問順に辿り最大16手、合法性を確認して打ち切る。子ごとのscoreがない間はroot scoreを全候補へ複製せず、最善候補のroot-summaryのみ出力する。これは将来のchild score出力までの明示的制約。ownershipを出す場合は要求color視点、上段左からS*S値。

人間向けrunログでは「勝率（引分0.5）」と明記し、raw NN期待値と探索値を別フィールドへ保存。WDL確率はroot NN由来、探索のWDL内訳がない場合に探索勝率から逆算しない。表示の見た目だけを優先した0/100%固定は禁止。
校正temperatureはvalidationでq logitsへの単一正温度をfitし、testでBrier/ECEを確認してmanifestへ保存。探索も同じ校正済み評価を使う。学習不足では校正未検証フラグをrunメタデータへ記録する。

CGOSはIchiGo専用config/state/log/passwordを使用する。バイナリ/モデルは対局開始時にhashを固定。次対局までモデル更新しない。GTP起動引数は配列でsubprocessに渡す。パス空白をnaive splitしない。
切断再接続はサーバーsetupからclear→size→komi→全着手replay→時計復元。同じ着手を二重適用しない。通常停止は対局間、backoffは1/2/4/8/16/30秒上限。ログrotateは20MiB×5。サーバー手数上限の処理は接続先protocolに合わせ、RinGo大会用400手固定を全サーバーへコピーしない。
ローカルfake serverで両サイズのゲーム完了・解析保存・再接続を検収してから実接続する。公開CGOSのアカウントと接続先はユーザーが実運用時に設定する。本仕様作成作業には実接続やメッセージ送信は含めない。

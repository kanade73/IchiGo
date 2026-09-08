# 検証・測定・昇格条件

## 1. Make契約

| command | 中身 | GPU要否 |
|---|---|---|
| make check-cpu | Swift Core/Features/LogicModel/Engine/GTP tests、Python pytest（CUDA/Metal除外） | 不要 |
| make parity-cpu | 固定Python hard→Swift scalar/packed各層・head比較 | 不要 |
| make check-metal | Metalカーネル/loader/lifecycle tests | Mac Metal |
| make parity-metal | CPU scalar↔Metal-byte↔packed、両盤サイズ全fixture | Mac Metal |
| make check-training | loss/gradcheck/resume/freeze/過学習 | CPU、小規模 |
| make check-cuda | CUDA forward/backward、1/2/4GPU DDP tests | 大学CUDA |
| make integration | fake教師・GTP・fake CGOS・時計・再接続 | CPU fake modelで可 |
| make release-check | 上記CPU/Metal/integration、対象リリースのhardモデル対局smoke（初期9路、M2c以降両サイズ）、bundle検査 | Mac Metal |

Swift test filterは実際のtarget名で実装する。CPU suiteの起動時にMetal専用テストがGPU初期化しないようtargetを分離。必要GPUがない場合は明示的skip理由と未検証ゲートを出し、release-checkを成功扱いにしない。

## 2. 必須fixture

| 分類 | ケース/検証 |
|---|---|
| ゲート | 全16×4真理値、a/b入替、id12=a、id10=b、argmax tie最小ID |
| 微分 | float64 random a/b/theta、gradcheck、縮約式と直接16和のforward/gradient一致 |
| 盤 | 9/19空盤、単石/複数石捕獲、自殺、単純コウ、長周期superko、seki、中立地、pass例外 |
| 履歴 | 7手より短い履歴、初期配置、pass、undo、同盤面でも異なる過去・合法性 |
| 座標 | GTP I列スキップ、SGF左上、pass末尾、全8D4逆写像 |
| 配線 | 各bank、盤隅、dilation8、盤外0、元入力再注入、A=B拒否 |
| bitpack | B=0,1,2,31,32,33,63,64,65、padding、NOT、定数true、全0/全1 |
| head | 定数h、疎h、mean/max、scoreの単位、mask後softmax、全target欠損 |
| モデル | Python往復、Swift往復、破損sha/truncation/NaN/shape/overflow/version/path |
| 探索 | root visit定義、白視点backup、FPU、draw、terminal、in-flight解除、tree reuse invalidation |
| protocol | GTP ID/CRLF/error/quit、analysisとgenmoveの同一着手・状態、stdoutログ汚染なし |
| 運用 | R=0、遅延NN、切断setup、途中model変更防止、正常停止、ログrotate、別state干渉なし |

囲碁ルールgoldenはRinGo既存fixtureと、固定KataGo referenceで採取した独立結果の両方を用いる。NN出力はKataGoとの一致を求めず、IchiGo Python hardが数値正本。教師をoracleとする対象と、自分のnetworkをoracleとする対象を混同しない。

## 3. 数値合格条件

- 全論理層出力: bit完全一致。平均誤差で許さない。
- Python FP32 hard headsとSwift scalar: 各要素 `abs(a-b) <= 1e-4 + 1e-4*abs(reference)`。
- Metal FP32 heads: 同じ条件。fast-mathを初期offにし、失敗時に無断で許容誤差を緩めない。
- 後処理prob: 絶対誤差≤1e-5、合法手sumは1±1e-5、違法手は0（互換層は-1）。
- scoreMean: 上記relative tolerance、さらにfixtureで手番反転が符号反転すること。
- CPU deterministic resume: 配線とstep同一、乱数/optimizer/state同一。GPU/DDPの完全bit再現は別指標、CPUとの勾配比較は絶対/相対1e-4。
- near-tieのargmax着手差はlogit marginも記録する。bit一致失敗をnear-tieとして隠さない。

## 4. 学習の合否

最初の16局面過学習は02の閾値。次に盤サイズ別holdoutでuniform合法policyに対してcross entropyが5%以上減少し、expected result MAEが定数0.5 baselineを下回ることをM1/M2の学習成立ゲートとする。baselineとcandidateを同じtarget maskとsample weightで測る。

毎validationで保存する項目:
soft/hard total loss、policy CE/top1、expected-result MAE/Brier、WDL CE（有効labelだけ）、score MAE（目数）、ownership MSE、soft-hard CE差、layerごとのtheta grad norm、gate entropy、constant率、恒等率、feature variance、freeze状態。

loss改善だけを棋力改善と呼ばない。対局結果も必要。教師のscoreやwinrateに合ったことと、真の対局結果に校正されたことも区別する。

## 5. 勝率校正

真の対局結果があるvalidationを使い、`T∈[0.25,4]` の正temperatureをNLLまたは期待得点BCEでfit。teacher予測だけしかない場合は校正未検証、T=1を保持する。
独立testで Brier=`mean((expectedResult-outcome)^2)`、outcomeは勝1/引分.5/負0。ECEは予測値を10個の等幅binへ分け、bin内平均予測と平均outcomeの差をサンプル数重み付けして合計。空binは除く。

同一対局内の相関を考慮してgame単位bootstrap CIを出す。初期は100完了局以上を目標、満たない場合はサンプル不足を表示。自己対局の極端な優劣だけで良い校正値に見せない。root rawとsearch別に指標を記録し、NNのTだけを調整する。探索値の追加変換は別実験。

## 6. 機能完走と棋力評価

M1: 9路hardモデルで20局、違法手/クラッシュなし。加えて合法手uniform baselineと100局（50色交換ペア）、勝1/引分.5で勝率の95%下限>0.5。baselineは合法点から一様、点がある間pass確率1%、なければpass、seed固定。極端に弱いbaselineなので、これを大会棋力と呼ばない。
M2a: 9路の学習成立・20局CGOSローカル完走・4GPU学習検収。M2b: 公開9路CGOSで20局完了、エンジン起因の違法手/クラッシュ/時間切れ0、クライアントによる解析値受理とサーバー側の保存または表示を確認する。サーバー障害は別記録し、未完局は20局に数えない。初回出場に特定ratingや勝率の最低値を設けず、M1のbaseline検収を棋力の最低条件とする。M2c: 19路の学習成立・20局CGOSローカル完走。19路もbaseline100局の比較を行い、達成しなければ「19路は動作のみ、棋力検収未達」と報告して改善チケットを継続する。

棋力強化の正式比較:

- 盤サイズ別、同じkomi/rules、投了off、同じ開局セットを色交換して使う。
- 開局は訓練testと別の固定50以上の局面を用意し、seed/開始手数を記録する。
- 主評価は1手1秒（壁時計）と1手5秒の別リーグ。副評価は400visits固定。必ず同じ探索設定で比較する。
- Mac1台なら両モデルをload/prewarmし交互に着手計測。candidateだけ異なる熱状態にならないよう対局順を交替する。同時GPU実行で競合させない。
- 初期スクリーニング100局、その後最終判定は新しい開局/seedで固定400局（200色交換ペア）。スクリーニング結果を最終CIに混ぜない。
- candidate expected scoreをペア単位で平均し、ペアbootstrap10000回、seed固定のpercentile95%CIを出す。400局完了前に勝ち越したから終了しない。
- promotion: 主評価1秒でCI下限>0.5、5秒では点推定≥0.5、違法手/クラッシュ0、タイムアウト増加なし、他盤サイズのchampionを無検証で置き換えない。
- p=0/1時の無限Eloを出さず、Elo換算は参考。CGOS ratingとローカル換算を同一尺度として扱わない。

強い基準: 固定KataGo教師を少visitsに制限した相手、現行RinGoの固定model/settings。勝てることは保証条件にせず、progress reportの長期指標にする。速度だけ高くても同時間の勝率が落ちる最適化は既定にしない。

## 7. Macベンチマーク

対象small/base、9/19別、B=1,2,4,8,16,32,64、backend=scalar/CPU-packed/Metal-byte/Metal-packed。同じ合法局面集合、同じモデルhash。各設定warmup50batch→計測200batch。非同期GPUは完了まで測定。初回compileとmodel load/prewarmを別欄へ記録。

出力:
`model_hash, size, batch, backend, samples, feature_ms, pack_ms, gate_ms, head_ms, post_ms, total_p50_ms, total_p95_ms, positions_per_sec, peak_rss_bytes, metal_allocated_bytes, load_ms, hardware, os, toolchain`。
キャッシュoffの生NN測定と、cache/tree reuseあり実探索を分ける。GPUカーネル時間だけを対局速度と呼ばない。

smallの目安budgetはB=1 total p95≤20ms、B=32で≥1000 positions/sec、プロセスRSS≤8GiB。ただし現時点で未測定の開発目標であり必須の「実現済み性能」ではない。未達時は各成分を報告しCPU優位/batch待ち/head負荷を切り分ける。CGOS実時間ゲートは時計・timeout・完走で評価する。

ゲート自体の規模はsmall19で `8*256*361=739328` gate evaluations/position（head除外）。学習thetaは `8*256*16=32768` 個で、空間共有のためpositionごとに別重みを持たない。学習activationだけでもB=8で1層約2.82MiBのFP32、8層約22.6MiB。これは入力gather・autograd・optimizer・head・CUDA allocatorを除く下限であり、総VRAM推定に使わない。

## 8. 最終成果の表示

release reportは「実装した機能」「実際に実行した検証」「各盤サイズの棋力」「未実行の環境」「model/data/code hash」を別欄で持つ。CGOS公開サーバー接続はローカルfake server検証と別記する。実サーバーのホスト/ポート/制限/対応解析方式は接続直前に確認する。

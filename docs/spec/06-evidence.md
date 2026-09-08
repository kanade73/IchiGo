# 根拠・調査記録・限界

確認日: 2026-09-08。本書は研究成果の保証ではなく、実装判断の根拠と未確認箇所の記録。

## 1. 一次資料

- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277): 微分可能な緩和で論理ゲートを学習する基本方式の根拠。本仕様の囲碁構造、head、学習スケジュールは独自の設計判断。
- [公式 difflogic](https://github.com/Felix-Petersen/difflogic): CUDA学習、batch方向bit packing、CPU生成コードの参考。公式READMEはCUDA依存を明記するためMacランタイムへ直接組み込まない。ライブラリ全体を移植せず小さな正解実装から開始する。
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732): 空間共有、論理木、OR pooling、残差的初期化を扱う。囲碁の位置ごとのpolicy出力を保つため本仕様は空間縮小poolingを採用せず、global headで全体集約する。画像分類の高速化倍率は囲碁やM5へ転用しない。
- [KataGo Analysis Engine](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md): JSONL要求、非同期応答、局面指定とラベル取得の基準。adapter実装時に固定revisionのフィールド意味をfixture化する。
- [CGOS 公開実装](https://github.com/zakki/cgos): GTPクライアントがwinrate/score/PVを解析拡張から送れることを確認。特定大会サーバーの現在設定を保証する資料ではない。
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): 1process/1GPUとデータ分割の責務を確認。stable URLの表示versionをそのまま将来環境の依存pinにしない。実装時にA4000環境で動く実版をlockする。
- [Apple Metal compute guide](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu): Metal compute pipelineの入口。詳細APIとresource bundleの動作は実装時にinstalled SDKでbuild検証する。

## 2. ローカル参照

RinGo root: `~/dev/univ/koubou/katago-mlx`。

| 確認ファイル | 確認事項 |
|---|---|
| AGENTS.md / README.md | Core/Model/Engine/Trainの分離、9路学習と19路推論、運用上の既存制約 |
| Package.swift | SwiftPM、macOS14、MLX依存、各targetの依存方向 |
| Sources/RinGoEngine/NNEvaluator.swift | plain requestとNNEvaluating actorの境界。現物はMLX importを含む |
| Sources/RinGoEngine/NNOutput.swift | 白視点、違法手sentinel=-1、KataGo固有score/postprocess依存 |
| Sources/RinGoEngine/Search.swift | ModelDesc/Precision依存、白視点value、tree/graph双方がある |
| Sources/RinGoCore/Rules.swift | area/ko/suicide/komiなどを明示的に設定可能 |
| Sources/RinGoEngine/GTPAnalysis.swift | winrate浮動小数[0,1]、要求手番視点。現物のPVは1手、scoreはroot値 |
| Sources/RinGoEngine/GTPEngine.swift | genmoveとanalysisの共通着手経路 |
| Scripts/cgos/README.md | 専用state、対局間停止、再起動、時計、analysis接続の参考 |
| Scripts/oss/THIRD-PARTY-NOTICES.md | 第三者コード・依存の通知の参考 |

重要な差分: IchiGoはRinGoの既存1手PVをそのまま多手読み筋として表示せず、実際の木からPVを構築する。RinGoのKataGoネット用postprocessorにlogic raw出力を流し込まない。既存nngdを新しい32chデータと同じものとして扱わない。

この調査はコード読み取りであり、RinGoのREADME上の棋力・数値パリティを本セッションで再測定していない。参照先には多数の未コミット変更があったため、HEADだけのpinでは足りない。実際に読んだ主要ファイルのhashは [reference-snapshot.json](reference-snapshot.json) を参照。

## 3. 未確定だが実装を止めない項目

| 項目 | 当面の処理 | 確定するチケット |
|---|---|---|
| A4000台数/VRAM/driver | 1GPU FP32からdoctorで検出、単一サーバー4GPU DDPを検収 | T17/T26 |
| 学習所要時間 | 100step pilotから局面数/stepsを見積もる | T17 |
| 初期教師model | 利用可能な公開KataGoモデルを明示選択、hash固定 | T12 |
| 初期SGFがない場合 | まず9路教師自己対局100局から増量、19路はM2c | T13 |
| Macでどのbackendが速いか | scalar/packed/Metalを同じ条件で測る | T25 |
| gate容量/離散化の限界 | 過学習→hard holdout→対局→CNN診断 | T17/T33 |
| CGOS接続先/アカウント | config必須、ローカルfake serverで先に検証 | T30/T37 |
| macOS14の実機互換 | deployment targetと実機検証を区別 | T37 |

## 4. この仕様が約束しないこと

論理ゲートが通常CNNより強いこと、M5で必ず速いこと、A4000何時間で大会棋力へ到達することは未検証。9/19両対応のコードと学習経路は設計上可能だが、19路での強さは長距離依存・表現容量・教師データ・探索時間に依存する。
その不確実性を後回しにせず、初期からhardモデル選抜、長距離接続、global head、棋力runner、計測を作る。失敗した場合も、学習/離散化/表現力/探索/運用のどこが原因かを判断できることが本仕様の狙い。

## 5. 学習時間に関する追加調査

[蒸留の原論文](https://arxiv.org/abs/1503.02531)を知識転移の根拠とし、IchiGoでの必要step短縮は実測する。[Mind the Gap](https://arxiv.org/abs/2506.07500)はGumbel noiseとSTEによる収束・離散化改善を扱う。画像分類の速度倍率を囲碁へ保証しない。
ローカルの `Scripts/ringo-data-format.md` と `Sources/RinGoEngine/TrainingData.swift` を確認し、v2にはsoft policy/valueとscore/ownership validityがあり、完全棋譜・gameIdはないことを確認した。実コーパスの存在量や対応索引は未確認。02-11とT13を既存target再利用優先へ変更した。

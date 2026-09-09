# ネットワーク・推論・保存形式

本文で規定する構造は IchiGo 独自の v1 である。設定値の変更とファイル形式の変更を区別する。

## 1. 座標と入力

盤サイズ `S ∈ {9,19}`、点 `p=y*S+x`。`x=0` は左、`y=0` は上。
GTP は `x=0 → A`、I を飛ばし、行番号は `S-y`。SGF `aa` は左上。pass の policy index は `S*S`。resign は policy 要素を持たない。
Python spatial は C-contiguous `[B,S,S,32] uint8`、Swift は同じ一次元順序 `(((b*S+y)*S+x)*32+c)`。値は厳密に0/1。盤外を含む19路への9路パディングは行わない。

| channel | 意味 |
|---|---|
| 0,1 | 現局面の手番側の石、相手の石 |
| 2,3 ... 14,15 | 1〜7手前の手番側の石、相手の石。視点は常に現局面手番に固定 |
| 16 | 現在の空点 |
| 17 | 現在の手番が着手できる点。履歴を含む完全な合法性 |
| 18 | 現在の単純コウ禁止点（なければ全0） |
| 19,20,21 | 手番側の石の連の呼吸点数が1、2、3以上 |
| 22,23,24 | 相手の石の連の呼吸点数が1、2、3以上 |
| 25 | 直前着手点。pass/履歴なしは全0 |
| 26 | 2手前着手点。pass/履歴なしは全0 |
| 27 | 盤の最外周の点 |
| 28 | 盤内定数1（境界を判別可能にする） |
| 29 | 直前がpassなら盤全体1、その他0 |
| 30 | 2手前がpassなら盤全体1、その他0 |
| 31 | 現手番が黒なら盤全体1、白なら0 |

存在しない過去の石配置は全0。pass も1手として履歴を進める。初期配置ありの棋譜では初期配置を t=0 とし、それ以前を0とする。Python でルールを再実装せず Swift feature exporter が正式な学習入力を生成する。

`global[B,4] float32`:
0: `komiSelf/(S*S)`（白手番なら komi、黒なら -komi）、1: `S/19`、2: `min(moveNumber,2*S*S)/(2*S*S)`、3: `min(consecutivePasses,2)/2`。
komi は白への加点、moveNumber は初期配置後の着手数。v1 は03の単一ルールのみ対応し、ルールを global で近似しない。

`legal[B,S*S+1] uint8` を別に持つ。pass の合法性もルールから計算する。終局局面は推論要求せず exact outcome を返す。
D4 は入力空間・policy・ownership・legal を同じ写像で変換し、global と pass を変えない。回転 `R(x,y)=(S-1-y,x)`、反転 `F(x,y)=(S-1-x,y)` とし、id 0..3 は R^id、4..7 は R^(id-4)∘F。逆写像を別テーブルで生成する。

## 2. 16ゲートの定義

ゲート ID `g=0..15` は真理値表そのもの。入力 `(a,b)` の行番号 `i=2*a+b`、出力は `(g >> i)&1`。
従って false=0、AND=8、XOR=6、OR=14、a=12、b=10、NOT a=3、NAND=7、true=15。
外部 difflogic の ID 順序をそのまま流用しない。

連続入力 a,b ∈ [0,1] について:

```text
q00=(1-a)*(1-b); q01=(1-a)*b
q10=a*(1-b);     q11=a*b
f_g(a,b)=Σ[i=0..3] bit(g,i)*q_i
p_g=softmax(theta_g / tau)
y_soft=Σ[g=0..15] p_g*f_g(a,b)
g_hard=最小IDの argmax(theta)
y_hard=bit(g_hard, 2*a+b)  // a,b が離散の場合
```

全16関数を各点ごとに `[B,S,S,C,16]` として保存しない。`t_i=Σ_g p_g*bit(g,i)` を `[C,4]` に先に縮約し `Σ_i t_i*q_i` を計算する。同じ多項式であり、浮動小数点誤差以外は等価。学習は FP32、まず自動微分で実装する。

## 3. 空間共有論理層

各層の出力は `[B,S,S,C]`。各出力 channel c は2個の入力参照 A/B と16 logitsを持ち、位置 (x,y) 間で共有する。参照は `(bank,channel,dx,dy)`。

- bank=0: 直前の層（第0層では元入力32ch）。bank=1: 元入力32ch（第1層以降だけ許可）。
- 入力点は `(x+dx,y+dy)`。盤外は0。ラップアラウンド禁止。
- 第 l 層の offset 候補は `dx,dy ∈ {-d_l,0,d_l}`。
- 配線は学習中固定。保存ファイルに実体を記録し、Swift 側で乱数から再生成しない。
- 第0層: 各 c の A は `(0,c%32,0,0)`、B はランダムな bank=0 参照。
- 第1層以降: c%4=0 の A は `(0,c,0,0)`。残りの A と全 B は、90% bank=0 / 10% bank=1 で channel と offset を一様に選ぶ。
- A=B のとき B を再抽選。モデル生成乱数は NumPy PCG64、初期 seed=20260908。配線ファイルが再現性の正本。
- 全層の c%4=0 は `theta[12]=3, 他=0`、残りは Normal(0,0.05)。恒等ゲートへの初期バイアスとし、固定配線スキップとは呼ばない。

| profile | C | L | d_l |
|---|---:|---:|---|
| tiny | 64 | 4 | 1,1,2,1 |
| small（初期学習） | 256 | 8 | 1,1,2,1,4,1,8,1 |
| base（拡張実験） | 512 | 12 | 1,1,2,1,4,1,8,1,4,1,2,1 |

tiny は実装検証用。small の空間的な最大到達範囲は盤全体を覆い得るが、疎な配線が有効な情報を運べるかは実験する。深さを増やせば強くなるという前提は禁止。層別勾配、定数ゲート率、特徴分散を記録する。

## 4. 浮動小数点ヘッド

最後の論理層 h を FP32 に変換。`m_c=mean_xy(h_c)`、`v_c=max_xy(h_c)`。推論時 m は popcount 等価、v は OR 等価だが初期版は単純縮約でよい。

```text
u_xy = concat(h_xy, m, v, global)         // 3C+4
z_xy = ReLU(u_xy @ Wlocal + blocal)        // 64
policy_xy = z_xy @ Wpolicy + bpolicy       // 1
ownership_xy = tanh(z_xy @ Wowner + bowner)// 1
zbar = mean_xy(z_xy)                        // 64  (headVersion 2, 3)
zreg = concat_{i,j=0..2}(mean_{(x,y) in region(i,j)}(z_xy))  // 9*64=576 (headVersion 3)
ownMean = mean_xy(ownership_xy)             // 1   (headVersion 2, 3)
u_global = concat(m,v,zbar,ownMean,global)       // 2C+69  (headVersion 2)
u_global = concat(m,v,zbar,zreg,ownMean,global)  // 2C+645 (headVersion 3)
// headVersion 1: u_global = concat(m,v,global) = 2C+4
z_global = ReLU(u_global @ Wglobal+bglobal)// 128
passLogit = z_global @ Wpass+bpass         // 1
wdlLogits = z_global @ Wwdl+bwdl           // 3: win,draw,loss
scoreMean = z_global @ Wscore+bscore       // 1: 手番視点、目数
```

`zreg` の region `(i,j)`（`i,j∈{0,1,2}`、順序は行優先 `i*3+j`）は盤を3×3に分割した領域内の `z_xy` の平均で、各領域64要素を連続配置する。分割は整数演算で行い、region `(i,j)` は行 `[floor(i*S/3), floor((i+1)*S/3))`・列 `[floor(j*S/3), floor((j+1)*S/3))` を覆う（9路: 3×3点の領域が3×3個、19路: 6/6/7点ずつの領域が3×3個）。行と列の分割は同じ境界式を独立に使い、領域面積は行数×列数。`S≥3` であれば全領域が非空になる。

各 W は `[in,out]` row-major float32、bias は `[out]`。local は `[3C+4,64]`、global は headVersion 3 で `[2C+645,128]`、headVersion 2 で `[2C+69,128]`、headVersion 1 で `[2C+4,128]`。headVersion 2 は 2026-09-08 の pilot で global head の入力 (m,v) に勝敗情報がほぼ無いことが判明したため追加した。headVersion 3 は 2026-09-10 の診断（局所 hidden map `z_xy` の3×3領域平均を使う線形プローブが global mean/max pooling だけの場合より value を大きく上回って予測できた: MAE 0.274 対 0.306）を受けて、`zreg`（3×3領域ごとの `z_xy` 平均）を global head へ追加で渡す。loader は 1・2・3 のすべてを受理し、manifest の `headVersion` で形状を決める。後続 W は順に `[64,1]`, `[64,1]`, `[128,1]`, `[128,3]`, `[128,1]`。Xavier uniform、bias=0で初期化する。
policy は全点の logit の後に passLogit を連結。softmax 前に非合法手を除外。ownership は手番の領有 +1、相手 -1、中立0。score はコミ込み。目差の分散・KataGo固有補助ヘッドは v1 で予測しない。
浮動小数点ヘッドは性能測定の独立項目。ヘッドが支配的なら local projection を共有成分と点固有成分に分解して同じ関数を高速化する。無断で巨大 CNN に置き換えない。

## 5. CPU と Metal の実装契約

`LogicBackend.evaluate(features: FeatureBatch) async throws -> RawBatch`。
RawBatch は policyLogits[B,S*S+1]、wdlLogits[B,3]、scoreMean[B]、ownership[B,S*S]。順序は入力と同一。B=0 は空結果、S混在は拒否。非有限値はエラー。

CPU scalar: 各層で UInt8 ping-pong 配列を使い、gを参照して1bitを評価する。head は Float の明示ループ。最適化前の golden oracle として永久に保持。

Metal v1 はまず同じ UInt8 配置のゲートカーネルを実装。その後 UInt32 バッチパックを実装する。パック時は `packed[((group*S+y)*S+x)*C+c]` の bit k がサンプル `b=group*32+k`。G=ceil(B/32)。位置や channel をビットに詰めない。

```text
valid = remainder==0 ? 0xffffffff : ((1u<<remainder)-1u)  // 最後のgroupのみ
out = ((g&1)!=0 ? ~a&~b : 0)
    | ((g&2)!=0 ? ~a& b : 0)
    | ((g&4)!=0 ?  a&~b : 0)
    | ((g&8)!=0 ?  a& b : 0)
out &= valid
```

シフト32は禁止。B=32のmaskは全bit1。g=15・NOT にも末尾maskを適用する。各層は別 dispatch、同一 command buffer に順に encode し、各層で CPU 同期待機しない。まず完了時に結果を読み戻しCPUヘッド、次にヘッドもMetal化。前後バッファを同時上書きしない。Swift/Metal間の descriptor は素の構造体レイアウトに頼らず Int32/UInt32 配列で受け渡す。

Metal device がなければ `--backend metal` は明示エラー、`auto` はCPU。GPUエラー時に途中結果を返さず要求全体を失敗させる。CPU fallback は次要求から行う設定とし、時間制限中の無制限再試行は禁止。

B=1では32bitの31bitが遊ぶため高速化は保証しない。CPU、Metal-byte、Metal-packed を B=1,2,4,8,16,32,64 で測定し、auto の閾値を端末別プロファイルに保存。モデルごとの符号化定数化・ゲート融合は parity 後の別チケット。

## 6. `.ichigo` モデルディレクトリ v1

`.ichigo` はディレクトリであり ZIP ではない。

```text
model.ichigo/
  manifest.json
  wiring.i32
  gates.u8
  heads.f32
```

manifest の必須項目:

```json
{
  "format":"ichigo.logic", "version":1,
  "featureVersion":1, "headVersion":2,
  "boardSizes":[9], "rulesId":"cgos-area-psk-v1",
  "channels":256, "layers":8,
  "dilations":[1,1,2,1,4,1,8,1],
  "gateEncoding":"truth-table-lsb-2a-plus-b",
  "layout":"NHWC", "endianness":"little",
  "valuePerspective":"to-move", "wdlOrder":["win","draw","loss"],
  "calibrationTemperature":1.0,
  "files":{}, "headTensors":[],
  "trainingProvenance":{}
}
```

上記はフィールド説明例。実際のfiles等が空なら無効。filesは3ファイルそれぞれ `{byteLength,sha256}`。headTensorsは各テンソルに `{name,shape,byteOffset,byteLength}`。必須14 tensors は Wlocal,blocal,Wpolicy,bpolicy,Wowner,bowner,Wglobal,bglobal,Wpass,bpass,Wwdl,bwdl,Wscore,bscore。重複・欠落・重なり・不整合を拒否。heads末尾の余分なbyteも拒否。

wiring は `[L,C,2,4] int32`、末尾が bank/channel/dx/dy。gates は `[L,C] uint8`。構造チェックは bank範囲、channel範囲、offset候補、g≤15、d数=Lまで。C 16〜4096、L 1〜64、d 1〜19、payload合計512MiB以下を初期上限にする。範囲外は警告付き読み込みではなくエラー。byte計算は overflow 検査を先に行う。

各ファイルサイズ・hash検証後にloadする。未知version・ルール・featureを拒否。JSONの NaN/Infinity、パスの絶対指定・`..`・symlink は拒否。ファイル名は上記3種固定。headTensorsは4byte alignment。

trainingProvenance は run ID、code revision、dataset manifest hash、teacher ID/hash、seed、選抜指標、hard検証結果、生成日時。大学環境の秘密情報・絶対ホームパスは入れない。
export は同じ親の一時ディレクトリに書く→再読込検証→rename。既存出力への上書きは `--overwrite` がなければ拒否。学習再開用 checkpoint は別形式であり、Swift に torch/pickle を読ませない。

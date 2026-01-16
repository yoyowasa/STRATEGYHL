# PROJECT_SPEC.md
**Hyperliquid ETH Perp MMBOT（研究 → 運用）仕様書 / v1**

このドキュメントは、本チャットで確定した **設計・仕様・固定関係・検証手順**を、リポジトリに置ける 1 枚の仕様書として整理したものです。  
（研究で合格したエッジ＝**f15**、mm_shadow/mm_live のログ設計、A/B窓検証、運用固定、現在のブロッカーと次の実装チケットまで含む）

---

## 目次
1. [Overview](#1-overview)  
2. [Scope / Non-Goals](#2-scope--non-goals)  
3. [Hyperliquid前提・制約](#3-hyperliquid前提制約)  
4. [リポジトリのモードとCLI](#4-リポジトリのモードとcli)  
5. [データモデル（blocks / windows / logs）](#5-データモデルblocks--windows--logs)  
6. [研究プロトコル（窓A/B）](#6-研究プロトコル窓ab)  
7. [採用エッジ：f15（boost by abs_mid_ret）仕様](#7-採用エッジf15boost-by-abs_mid_ret仕様)  
8. [ログ仕様（JSONL）](#8-ログ仕様jsonl)  
9. [mm_shadow 合格ゲート（チェック仕様）](#9-mm_shadow-合格ゲートチェック仕様)  
10. [mm_live 実装仕様（Runner / OrderManager）](#10-mm_live-実装仕様runner--ordermanager)  
11. [レート制限・整合性対策（A→B突破の必須仕様）](#11-レート制限整合性対策ab突破の必須仕様)  
12. [運用固定（Git/タグ）](#12-運用固定gittag)  
13. [現状ステータスと次のマイルストーン](#13-現状ステータスと次のマイルストーン)  
14. [チェックリスト](#14-チェックリスト)  
15. [戦略置き場 - MMBOTエッジ探索計画](#15-戦略置き場---mmbotエッジ探索計画)  

---

## 1. Overview
目的：Hyperliquid の **ETH perpetual** を対象に、Market Making Bot（MMBOT）のエッジ探索→検証→戦略化→運用固定→mm_shadow/mm_live で実弾検証までを、再現可能なフレームワークで回す。

本プロジェクトの「再現性の契約」：
- 研究は **同じblocks / 同じ窓**で A/B する
- candidate は **変更点を1つだけ**（YAML差分）で作る
- 合格条件を満たしたものだけを **運用YAML 1本**に統合し、**コミット＋タグ**で固定する
- mm_sim（近似）で勝っても mm_live（実弾）で必ず確認（キュー未観測のため）

---

## 2. Scope / Non-Goals
### Scope（やる）
- mm_sim：窓A/Bでエッジ探索（効果量で選別）
- mm_shadow：市場データで「意思決定が仕様通り」か検証（送信なし）
- mm_live：実弾で約定・手数料・レート制限・post-onlyの挙動を検証
- 研究合格エッジ（f15）を運用固定し、段階的にスケールする

### Non-Goals（当面やらない）
- キュー位置推定の高精度化（queue position を完全再現すること）
- 複雑な ML や大量パラメータの最適化（オーバーフィット回避のため）
- 多銘柄展開（ETH で固めた後に注意点付きで拡張）

---

## 3. Hyperliquid前提・制約
（本チャットで共有された公式仕様ベースの前提）

### 3.1 取得可能データ
- WS subscribe：`l2Book`, `trades`, `bbo`, `activeAssetCtx`（ETH）
- /info：`metaAndAssetCtxs`（szDecimals, funding, premium, oraclePx, markPx等）, `fundingHistory`,
  `l2Book snapshot`（最大20level/side）, `candleSnapshot`（最大5000本）

### 3.2 Funding
- Funding は毎時間支払い
- premium は 5 秒サンプルの 1 時間平均
- 支払い式：`position_size * oracle_price * funding_rate`（oracleで計算）

### 3.3 丸め（重要）
- 価格：最大5有効数字＋MAX_DECIMALS制約
- サイズ：`szDecimals` に丸め

### 3.4 RateLimit / Nonce（重要）
- 0.1 秒バッチ推奨
- ALOバッチをIOC/GTCと分ける（運用設計上の前提）
- nonce は timestamp ベースで一意化（衝突禁止）

---

## 4. リポジトリのモードとCLI
### 4.1 モード
- `mm_sim`：シミュレーション（blocks.parquet replay + fillモデル）
- `mm_shadow`：市場データで意思決定・注文生成のみ（送信しない）
- `mm_live`：市場データで実際に送信する

### 4.2 CLI（想定）
- 研究（窓A/B）：`scripts/run_ab_windows.py`
- shadow/live：`scripts/mm_live.py`

#### 実行例：mm_shadow（15分）
```powershell
$runId = "shadow_f15_" + (Get-Date -Format 'yyyyMMdd_HHmmss')
$env:PYTHONPATH='.'
$env:PYTHONUNBUFFERED='1'
python scripts/mm_live.py --config configs\strategy_prod_f15.yaml --mode shadow --run-id $runId --log-dir outputs_shadow_f15 --coin ETH --duration-sec 900
```

#### 実行例：mm_live（Stage-0, 15分）
```powershell
$runId = "live_f15_stage0_" + (Get-Date -Format 'yyyyMMdd_HHmmss')
$env:PYTHONPATH='.'
$env:PYTHONUNBUFFERED='1'
# 秘密鍵などは runner が参照する環境変数で渡す（例：HL_PRIVATE_KEY）
python scripts/mm_live.py --config configs\strategy_prod_f15.yaml --mode live --run-id $runId --log-dir outputs_live_f15 --coin ETH --duration-sec 900
```

---

## 5. データモデル（blocks / windows / logs）
### 5.1 blocks.parquet（研究入力）
- l2Book / trades などを一定粒度（ブロック）にまとめたイベント集合
- **板は“スナップショット的”**（キュー位置は不明）

### 5.2 windows（研究の再現単位）
- 例：`window_sec=7200（2h）`, `step_sec=1800（30m）`, `n_windows=20`
- 各窓を独立のA/B単位として、`Δpnl` の分布を作る

### 5.3 logs（shadow/liveの運用入力）
- JSONL（追記型）を正とする（再接続・再起動に強い）
- 出力先：
  - `outputs_shadow_f15/<run_id>/...`
  - `outputs_live_f15/<run_id>/...`

---

## 6. 研究プロトコル（窓A/B）
### 6.1 基本ルール（固定）
- baseline から始める
- candidate は **変更点を1つだけ**
- 同じ windows（同じ blocks, 同じ窓）で baseline/candidate を回す
- 合格したら次へ、不合格なら仮説終了

### 6.2 合格条件（固定）
- `median(Δpnl) > 0`
- `win_rate >= 0.55`
  - `win_rate = count(Δpnl > 0)/N`

### 6.3 追加の最低限チェック（固定）
- 非重複窓（2h/2h）で符号の再確認（Nが小さくても可）
- fillモデル感度（lower-alpha 0.3/0.5/0.7）で符号維持
- in/out（トリガー窓内/外）で分解して稼ぎの場所を確認  
  - `realized_spread_5s_in/out`
  - `markout_5s_in/out`
  - `pnl_in/out`

---

## 7. 採用エッジ：f15（boost by abs_mid_ret）仕様
### 7.1 概要（決定済み）
- 「abs_mid_ret（短期変動）」が大きい局面は **停止（halt）ではなく、サイズを増やして spread capture を増やす**
- 研究で f15 が A/B 合格、非重複窓でも合格、fillモデル感度でも符号維持

### 7.2 特徴量定義（固定）
#### mid（固定）
- `mid_t = (best_bid_t + best_ask_t)/2`（l2Book best bid/ask 由来）

#### abs_mid_ret（固定）
- l2Book 更新時のみ更新
- `abs_mid_ret_t = abs(mid_t / mid_{t-1} - 1)`
- reconnect直後は `mid_prev` をリセット（誤発火防止）

> /info の markPx/oraclePx を abs_mid_ret 計算に使わない（研究とズレる）

### 7.3 ルール（固定）
- 通常：baseline quoting（base_spread_bps / base_size / inventory skew）
- 条件：`abs_mid_ret > thr_p90`
  - `quote_size = base_size * boost_size_factor`

### 7.4 固定パラメータ（決定済み）
- `thr_p90(abs_mid_ret) = 6.721220573650064e-05`
- `boost_size_factor = 1.5`
- 研究での boost_trigger_rate（参考）：約6%（中央値）

### 7.5 期待PnLドライバ（研究分解）
- boost窓内で **realized spread が増える**（主要ドライバ）
- markout の悪化は小さく、相殺されない（小幅プラス寄り）

---

## 8. ログ仕様（JSONL）
### 8.1 出力ファイル（固定）
- `market_state.jsonl`
- `decision.jsonl`
- `orders.jsonl`
- `events.jsonl`
- `fills.jsonl`
- `config_resolved.yaml`
- `manifest.json`

### 8.2 最低限のフィールド（推奨）
#### market_state.jsonl
- `ts_ms`
- `best_bid`, `best_ask`（価格）
- （任意）`best_bid_sz`, `best_ask_sz`
- `mid`, `mid_prev`（推奨）
- `abs_mid_ret`
- `top_px_change`（best_bid/askが前回から変わったか）
- `top_px_change_count`（集計側で算出でも可）

#### decision.jsonl
- `ts_ms`
- `boost_active`（bool）
- `abs_mid_ret`
- `base_size`, `boost_size_factor`, `quote_size`
- `target_bid_px`, `target_ask_px`, `target_bid_sz`, `target_ask_sz`
- `skip_reason`（min_quote_lifetime / cooldown / min_send_interval 等）

#### orders.jsonl（live）
- `ts_ms`
- `action`（new/replace/cancel/cancel_all）
- `side`（bid/ask）
- `px`, `sz`, `post_only`
- （new/replaceのみ）`effective_spread_bps`（その注文で実際に使ったスプレッド設定）
- `client_oid`
- （可能なら）`exchange_order_id (oid)`
- `status`（sent/error/ack 等）
- `error_message`

#### events.jsonl
- `ts_ms`
- `event`（rate_limited / cooldown / reconnect / startup_sync / error 等）
- `detail`（backoff_ms, cooldown_until_ms, ws_msg_total, poll_total 等）

#### fills.jsonl（live）
- `ts_ms`
- `side`
- `fill_px`, `fill_sz`
- `fee`, `rebate`（取れるなら）
- `liquidity`（maker/taker 判別できるなら）

---

## 9. mm_shadow 合格ゲート（チェック仕様）
mm_shadow は「戦略が仕様通り動くか」の確認に使う。PnLは目的ではない。

### 9.1 合格条件（固定）
- `ratio_median == 1.5`（boost倍率が丸めで潰れていない）
- `crossed_count == 0`（post-only前提が壊れていない）
- `batch_ok == True`（min_batch_gap_ms など）
- `reconnect_guard_ok == True`（発生時に安全に復帰）
- `boost_trigger_rate` がレンジ内（目安：0.03〜0.10）

### 9.2 inconclusive（判定不能）ルール（固定）
短時間で価格が動かないと trigger_rate が下振れするため：
- `top_px_change_count` が少ない場合は `trigger_rate` を fail にせず **inconclusive** 扱い  
（例：`top_px_change_count < 20` なら trigger_rate gate をスキップ）

---

## 10. mm_live 実装仕様（Runner / OrderManager）
### 10.1 Runnerの責務（固定）
- 市場データ取得（現状：/info polling）
- 特徴量更新（mid, abs_mid_ret）
- 意思決定（boost_active, target quotes）
- 注文の生成（post-only、丸め適用）
- バッチ送信（order_batch_sec）
- 再接続・同期（必要なら cancel_all）
- JSONL ログ出力（market_state/decision/orders/events/fills）

### 10.2 注文生成の必須仕様（固定）
- **丸め後の(px,sz)**で差分判定（丸め差で無限更新を防ぐ）
- post-only を維持（crossは禁止）
- nonce 一意性（timestamp等）
- **1バッチ=1送信**（bid/askまとめて）

---

## 11. レート制限・整合性対策（A→B突破の必須仕様）
現在のmm_liveは (A) 「発注が成立していない」で詰まっている。  
理由：
- `Too many cumulative requests`（累積リクエスト制限）
- `Cannot modify canceled or filled order`（存在しない注文をreplaceしようとしている）

### 11.1 目標（このチケットのDone条件）
- orders の `status: sent` が継続的に出る（最低 bid/ask で2以上）
- `Too many cumulative requests` が連発しない（0〜散発）
- `Cannot modify canceled...` がほぼ0（出ても一過性）
- これを満たして初めて (B)（約定しない原因＝価格/キュー）へ進む

### 11.2 必須仕様：レート制限バックオフ（長期化）
- エラー文字列に `Too many cumulative requests` を含む場合：
  - `cooldown_until_ms = now + backoff_ms`
  - backoff は指数増加（上限まで）
- **推奨値（Stage-0用）**
  - `rate_limit_backoff_ms = 60000`（最初から60秒）
  - `rate_limit_backoff_max_ms = 600000`（最大10分）
- cooldown 中は **一切送信しない**（decisionログのみ）
- **成功（sent）が出た時だけ backoff を初期値に戻す**

### 11.3 必須仕様：グローバル送信間隔
- `min_send_interval_ms = 5000`（5秒）
- どんな理由があっても、前回送信から 5秒未満は送らない
- bid/ask はまとめて 1送信にする

### 11.4 必須仕様：replace の前提を厳格化（整合性）
- replace は **取引所注文ID（oid）を保持している場合のみ**
- `cancel_all` を送ったら **ローカルの oid を即クリア（bid/askとも）**
- `Cannot modify canceled or filled order` を受けたら：
  - 該当sideの oid を即クリア
  - 次回は replace せず new からやり直す（ただし cooldown/interval に従う）
- cancel_all と new/replace を同じバッチに混ぜない（cancel_allは単独送信）

### 11.5 追加（推奨）：min_quote_lifetime_ms
- side別に最後の更新時刻を持つ
- lifetime 未満は更新をスキップ  
（ただし **rate limit 対策の本丸は backoff + min_send_interval + replace前提**）

### 11.6 YAMLでの設定例（Stage-0で一旦強制）
```yaml
# configs/strategy_prod_f15.yaml（例：Stage-0）
mode: mm_live
paths:
  data_dir: data
  output_dir: outputs_live_f15
  log_dir: outputs_live_f15

strategy:
  name: prod_f15
  leverage: 1.0
  risk_limit: 0.1
  max_positions: 10

  extra_params:
    # --- quoting baseline ---
    base_spread_bps: 5.0
    base_size: 0.01
    inventory_skew_bps: 2.0
    inventory_target: 0.0

    # --- f15 boost（固定） ---
    boost_when_abs_mid_ret_gt: 6.721220573650064e-05
    boost_size_factor: 1.5

    # --- kill switch（Stage-0） ---
    stop_max_abs_position: 0.1
    stop_max_intraday_drawdown_usdc: -0.2

    # --- live safety（A→B突破用） ---
    min_quote_lifetime_ms: 30000        # Stage-0は長めでOK
    min_send_interval_ms: 5000          # ★必須
    rate_limit_backoff_ms: 60000        # ★必須（最初から1分）
    rate_limit_backoff_max_ms: 600000   # ★必須（最大10分）
```

---

## 12. 運用固定（Git/タグ）
### 12.1 固定方針（固定）
- 運用YAMLは **1本に統一**：`strategy_prod_f15_live.yaml`（B3.2 確定合格）
- `strategy_prod_f15.yaml` は履歴/Stage-0基準として保持
- 研究成果（f15）と関連コード・テスト・スクリプトは **同一コミット**にまとめ、タグで固定
- 既存：`prod-f15-v1`（commit: 32e4fad）
- 次の固定タグ（B3.2運用固定）：`prod-f15-live-v2`

### 12.2 注意（LF/CRLF）
- Git warning（LF→CRLF）は別問題（整形タスク）
- ただし `hlmm/mm/strategy.py` の import 変更は **循環import解消の必須差分**  
  - `from hlmm.mm import Order` → `from hlmm.mm.sim import Order`

---

## 13. 現状ステータスと次のマイルストーン
### 13.1 現状
- 研究：f15は合格・固定済み
- shadow：合格ゲート通過済み
- live：Stage‑0 合格（maker で約定し、fills/fee がログ化済み）  
  - run: `outputs_live_f15/live_f15_stage0_20260103_014942`
  - maker_fills=8 / taker_fills=0（crossed=false 100%）
  - builder_fee_sum=0.0
  - maker_fee_bps=1.3679（maker 料金帯に整合）
  - fills.jsonl 出力確認済み

### 13.2 次のマイルストーン（最短）
**Milestone B**：fills を使った評価ループに移行
- at_touch_rate
- replace頻度（キュー捨て）
- スプレッド距離（遠すぎ）
- realized_spread / markout の簡易計測

**Milestone C**：Stage‑1〜スケール
- Stage‑0 → ×2 → ×2 → 通常
- 各段階で crossed/taker/fees/rebates/funding を確認

---

## 14. チェックリスト
### 14.1 研究（mm_sim）
- [ ] baseline を決める（最小）
- [ ] candidate は変更点1つ（YAML差分）
- [ ] 窓A/B（N=20）で medianΔpnl>0 & win_rate>=0.55
- [ ] 非重複窓で符号確認
- [ ] fillモデル感度（alpha 0.3/0.5/0.7）で符号維持
- [ ] in/out分解でドライバ確認

### 14.2 shadow（mm_shadow）
- [ ] ratio_median == 1.5
- [ ] crossed_count == 0
- [ ] batch_ok == True
- [ ] trigger_rate_ok（inconclusive除外）
- [ ] reconnect_guard_ok

### 14.3 live（mm_live）
**A→B（まずここ）**
- [ ] sent が継続的に出る
- [ ] Too many cumulative requests が連発しない
- [ ] Cannot modify canceled... がほぼ0
- [ ] cooldown中に再送しない（eventsで確認）
- [ ] cancel_all後にoidがクリアされ、replaceが出ない

**B以降**
- [ ] taker_fills == 0
- [ ] crossed_count == 0
- [ ] fillsが出る（fee/rebateが取れる）
- [ ] realized_spread / markout が計測できる
- [ ] kill switch が正しく止まる（必要時）

---

## 15. 戦略置き場 - MMBOTエッジ探索計画
### 15.1 次にやるべきこと（Bフェーズ最小評価ループ）
- (1) レポートの健全性チェック → (2) 合否判定ルールを固定 → (3) 1ノブずつ動かす
- run_report.json が貼られたら Gate（inconclusive / pass / fail）で判定できるが、判定がブレない見方を先に固定する

### 15.2 run_report の健全性チェック（必須・数分で済む）
#### A. joins の健全性（fills→market_state の突合）
- `fills_joined_count`（fillsのうち、market_state が引けた数）
- `fills_join_miss_count`
- `fills_join_dt_ms_p50` / `fills_join_dt_ms_p95`（fill時刻と採用したmarket_state時刻のズレ）
- `mid_source`（`best_bid_ask` 推奨。研究とズレるため markPx 系は避ける）

#### B. サイン（符号）の健全性
- BUY fill は `fill_px < mid` なら edge が正
- SELL fill は `fill_px > mid` なら edge が正
- 逆符号が出たら side 判定か式の反転を疑う

#### C. takerガードは crossed を一次情報に
- taker 判定は crossed ベースでOK（fee 符号は補助）
- run_report の `taker_guard_trip` はこの方針で固定

### 15.3 run_report 合否ゲート（固定ルール）
Bフェーズ最小として、まずはこの4つだけで機械判定する。

#### Gate A（安全）
- `taker_guard_trip == false`
- `taker_notional_share == 0`（または極小）

#### Gate B（判定可能性）
- `fills_count >= N_min`（例：10〜30）
- `notional_sum >= notional_min`（例：$300〜$1000）
- ここ未満は **inconclusive**

#### Gate C（成績）
- `net_bps.median > 0` を最優先
- `net_bps.p10` が極端に悪くない（例：-5 bps を大きく割らない）

#### Gate D（逆選好チェック）
- `markout_bps["30"].median` が強烈にマイナスなら要注意
  - edge が出ても直後に逆行して吐き出している可能性

### 15.4 「2枚維持」達成後、Bフェーズで最初に触るべきノブ
- 更新頻度を落とす（`order_action_per_min` を下げる）
  - 先頭維持が伸びやすく、markout が改善することが多い
- at_touch を上げすぎない（特に ask 側）
  - at_touch を上げるほど fills は増えるが、逆選好も増えやすい
- `base_spread_bps` を微調整（狭める/広げるは1ノブずつ）
  - edge と fill 率のトレードオフが綺麗に出る

### 15.5 JSONを貼るときのおすすめ（判定が速くなる）
- run_report.json に加えて以下があると原因まで踏み込める
  - `fills_count` / `notional_sum` / `maker_fills` / `taker_fills`
  - `fills_join_dt_ms_p50` / `fills_join_dt_ms_p95`（あれば）

### 15.6 現時点の位置づけ
- Stage-0 の要件を満たし、**Bに入る準備が完了**
- 次は run_report で「net が正か」「逆選好が強すぎないか」を機械的に判定する

### 15.7 次にやること（優先順）
#### 1) “基準run（baseline）” を3本作る
目的は「いまの設定で net が安定してプラスか」を見ること。
- 同じ `strategy_prod_f15.yaml`（Stage-0のまま、サイズも据え置き）
- run を最低3本（時間帯が違うとベター）
- 各runで `run_report.json` を保存
- 重要なのは時間ではなく母数
  - `fills_count` と `notional_sum` が小さすぎるrunは “fail” にせず inconclusive 扱いで捨てる

### 15.8 baseline 判定後の分岐（PASS / FAIL / inconclusive）
#### baseline が PASS（net_bps が安定してプラス）
次にやるべきはサイズを段階的に上げて同じ指標が保てるかを確認する（Stage‑1相当）。
- 例：`base_size` を `0.01 → 0.02 → 0.05`（1段ずつ）
- 各runで `run_report.json` を保存して以下を確認
  - taker混入なし
  - `net_bps` が崩れない
  - `markout_bps["30"].median` が悪化しない
  - rate_limit / cooldown が増えない
- 主眼は「bps が維持できるか」（サイズを上げると約定のされ方が変わるため）

#### baseline が FAIL（`net_bps.median <= 0`）
ここで初めて **“1ノブだけ”** いじって candidate を作り、live A/B に入る。
おすすめの順番（効きやすい順）：
- 更新頻度を下げる（`order_action_per_min` を落とす）
  - キュー位置が観測できない環境では「更新しすぎて順番を捨てる」が頻出の負け筋
  - `min_send_interval_ms` や更新条件を少し緩める（=無駄更新を減らす）
- スプレッドを少し広げる（`edge_bps` を上げる）
  - 目標は「fee_bps（≈1.4〜1.5bps）を上回る edge」
  - 広げすぎると fills が減って inconclusive になりやすいので “小さく” から
- boost局面の挙動（f15）に“spread側の補助”を足す
  - 例：boost時だけわずかに広げる／更新頻度を落とす
  - 研究で勝っている f15 のコア（size boost）を壊さない範囲の1ノブとして扱う

#### baseline が inconclusive（fillsが少ない等）
これは “負け” ではなくデータ不足。
- run を増やす（母数条件を満たすまで）
- あるいは「fills が出る設定」に寄せる
  - 例：スプレッドをほんの少し狭める、更新頻度を下げて隊列維持 など
  - Stage‑0 で守った takerゼロは絶対に崩さない

### 15.9 B1 の読み取り（次の一手が “spread 続行”で良い根拠）
- `fee_bps_per_fill.median` が 3本とも ~1.3678bps で安定（fee 側は固定）
- `edge_bps.median` は
  - r1/r3 で ~0.793bps（改善）
  - r2 で ~0.159bps（ほぼベースライン水準）
- `net_bps.median` は
  - r1/r3 で ~-0.574bps（改善だがまだマイナス）
  - r2 で ~-1.209bps（改善が乗っていない）

ここから言えること：
- 改善が起きるrunは確かにある（-1.21 → -0.57）
- ただし 0 を超えていない
- run間ブレもあるため、+1bps 追加で符号反転を狙うのは合理的

### 15.10 B1.1 の実行プラン（最短で“判定可能”にする）
- 変更は `base_spread_bps: 2.15 → 3.15` のみ（他は一切触らない）
- GateB で inconclusive が増えやすいので、run条件は “時間で吸収”
  - run時間を延ばす（評価条件側で吸収）
  - もしくは 3本 → 5本に増やす
  - 「変更点1つ」原則は維持する

#### B1.1 の一行チェック（最初に見る）
- `edge_bps.median` が上がったか？
  - 上がってないなら、spread を広げても fill時の edge が増えない構造 → B1系列は打ち止め候補
- `net_bps.median` が 0 を跨いだか？
  - 跨いだら、その設定で母数を増やして再現性チェックへ

### 15.11 B3.2 確定合格後の判定ルール（運用固定）
B3.2 を運用固定する際の判定ルールは以下で固定する。

#### run単体（mm_live）
- HARD FAIL：`taker_guard_trip == true` または `taker_notional_share > 0.1%`
- HARD FAIL：`net_bps.p10 < -5`
- INCONCLUSIVE：GateB 未達（`fills_count < 10` または `notional_sum < 300`）、または `fills_joined_count == 0`
- EVAL（GateB達成runのみ）：`net_bps.median > 0` を PASS、それ以外は FAIL

#### candidate（run集合）
- GateB_pass を 10 本集めて判定
- `net_med` は run_report の `net_bps.median`
- `median(net_med) > 0` かつ `win_rate(net_med>0) >= 0.55` で合格

#### WARN（markout30）
- `markout30_med < -5` は WARN（hard gate にしない）
- 10本中2本以上、または連続2回で発生したら要対応（次ノブ検討）

---

## 付録：研究で確定した f15 の固定値（コピペ用）
- `boost_when_abs_mid_ret_gt: 6.721220573650064e-05`
- `boost_size_factor: 1.5`
- 研究での目安 trigger_rate：~0.06（参考値）

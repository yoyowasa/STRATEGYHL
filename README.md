# HLMM

Minimal project skeleton providing the `hlmm` command line entrypoint.

## Usage

```bash
poetry install
poetry run hlmm --help
# 正規化済み設定の出力
poetry run hlmm --config configs/example.yaml --print-config
# features -> dataset 生成
poetry run hlmm dataset --features features.parquet --dataset-out dataset.parquet --splits-out splits.json --horizons 1,5,15,60
# 単変量エッジスクリーニング
poetry run hlmm edge --dataset dataset.parquet --splits splits.json --out-dir edge_output --target y_ret_1s --ic-threshold 0.01
# blocks -> mmシミュレーション
poetry run hlmm mm-sim --blocks blocks.parquet --out-dir mm_sim_out --taker-fee-bps 0.0 --maker-rebate-bps 0.0 --max-abs-position 1.0 --fill-model lower --lower-alpha 0.5
# レポート生成
python - <<'PY'
from hlmm.research import generate_report
generate_report("run1", "mm_sim_out/ledger.parquet", "mm_sim_out/sim_trades.parquet")
PY
# 戦略（例: Strategy1）
# stateを dict で渡し、decide_orders が bid/ask を返す
python - <<'PY'
from hlmm.mm import StrategyParams, decide_orders
state = {"mid": 100.0, "position": 0.5}
print(decide_orders(state, StrategyParams()))
PY
```

## 運用固定（prod-f15-live-v2）
- 運用YAML: `configs/strategy_prod_f15_live.yaml`（B3.2固定）
- 判定ルール/WARN: `PROJECT_SPEC.md` の 15.11
- run_report 生成: `python scripts/run_report.py --run-dir outputs_live_f15/<run_id> --out reports_live_f15/<run_id>/run_report.json`
- 監視集計: `python scripts/monitor_live.py --reports-root reports_live_f15 --window 10`
- 出力: 標準出力 + `reports_live_f15/_monitor/summary.json`（`reports*`/`outputs*` は .gitignore 対象）

## データ収集の前提（最小セット）
- l2Book（ETH perp）: `time`, `bids/asks[[px,sz]]`。クロックとして使用。
- trades: `time`, `side`, `px`, `sz`, `trade_id`。フロー/フィルモデルに必須。
- activeAssetCtx: `time`, `markPx`, `oraclePx`（あれば `funding`/`openInterest`）。basis/funding系特徴。
- funding系列: `fundingHistory` または `userFundings`（`time`, `amount`）。1h fundingを台帳に反映。
- bbo（推奨）: `time`, `bid_px/bid_sz/ask_px/ask_sz`。mid/spread安定化。
- userFills/userFundings（replay用）: 実約定と手数料/リベート、funding支払/受取。
- メタ: `szDecimals`（丸め整合）、欠損フラグ（missing_book/trades/ctx）、`recv_ts_ms`/`event_id` など再現性メタ。

## Hyperliquid 短期OHLCV（candleSnapshot）
- 取得: `python scripts/fetch_hl_candles.py --intervals 1m,5m,15m,30m,1h,4h,1d`
- 出力先: `data/hyperliquid/candles/`（例: `data_eth_perp_1m.json`, `data_ethusdc_1h.json`）
- spot の ETHUSDC は API 上 `@index`（例: `@151`）で取得します（`spotMeta` は `data/hyperliquid/meta/spotMeta.json` に保存）。
- 45m は API 非対応のため、15m から集約して生成: `python scripts/derive_hl_candles.py --input data/hyperliquid/candles/data_eth_perp_15m.json --output data/hyperliquid/candles/data_eth_perp_45m.json --target-interval 45m`

## 本気検証（l2Book/trades → blocks → mm_sim）
PowerShell 例（ETH-USDC PERP）:

```powershell
# 1) l2Book+trades を短時間取得（429が出るなら poll を 2000〜5000ms に上げる）
poetry run python scripts/capture_hl_l2_trades.py --coin ETHUSDCPERP --duration-sec 600 --poll-interval-ms 2000 --book-depth 20 --out raw_data/hl_eth_perp.jsonl

# 2) raw -> blocks.parquet（l2Bookをクロックにtradeをバケット化）
poetry run python scripts/raw_to_blocks.py --input raw_data/hl_eth_perp.jsonl --symbol ETH --out-blocks data/blocks.parquet

# 3) baselineでmm_sim（出力: outputs/mm_sim_baseline/）
poetry run hlmm --config configs/strategy_baseline.yaml mm-sim --fill-model lower --lower-alpha 0.5

# 4) features を作って realized spread も含めてレポート
poetry run python -c "import pyarrow as pa, pyarrow.parquet as pq; from hlmm.features import compute_features; blocks=pq.read_table('data/blocks.parquet').to_pylist(); feats=compute_features(blocks); pq.write_table(pa.Table.from_pylist(feats), 'data/features.parquet')"
poetry run python -c "from hlmm.research import generate_report; generate_report('baseline', 'outputs/mm_sim_baseline/ledger.parquet', 'outputs/mm_sim_baseline/sim_trades.parquet', features_path='data/features.parquet')"
Get-Content reports/baseline/metrics.json
```

## 比較（baseline vs stop/pull）
`reports/<run_id>/metrics.json` を baseline 差分つきで1枚表にする:

```powershell
poetry run python scripts/compare_metrics.py --baseline baseline --runs baseline,stop,pull --format md
```

原因特定用に `realized_spread/markout/inventory` に加えて、`fills_when_*` / `pnl_when_*`（stop/pull区間の約定回数・PnL）も一緒に出す:

```powershell
poetry run python scripts/compare_metrics.py --baseline baseline --runs baseline,stop,pull --format md --metrics pnl,max_drawdown,fill_rate,num_fills,notional_traded,realized_spread_5s,markout_5s,inventory.mean,inventory.p95,inventory.max_abs,stop_trigger_rate,pull_trigger_rate,fills_when_stop,fills_when_pull,pnl_when_stop,pnl_when_pull
```

### stop/pull の接続チェック（強制発動）
「実装に繋がっているか」だけを確認したい場合は、強制版 config を使う。

```powershell
poetry run hlmm --config configs/strategy_stop_force.yaml mm-sim --blocks data/blocks.parquet --out-dir outputs/mm_sim_stop_force --fill-model lower --lower-alpha 0.5
poetry run python -c "from hlmm.research import generate_report; generate_report('stop_force', 'outputs/mm_sim_stop_force/ledger.parquet', 'outputs/mm_sim_stop_force/sim_trades.parquet')"

poetry run hlmm --config configs/strategy_pull_force.yaml mm-sim --blocks data/blocks.parquet --out-dir outputs/mm_sim_pull_force --fill-model lower --lower-alpha 0.5
poetry run python -c "from hlmm.research import generate_report; generate_report('pull_force', 'outputs/mm_sim_pull_force/ledger.parquet', 'outputs/mm_sim_pull_force/sim_trades.parquet')"

poetry run python scripts/compare_metrics.py --baseline baseline --runs baseline,stop_force,pull_force --format md
```

## Testing

```bash
poetry run pytest
```
# STRATEGYHL

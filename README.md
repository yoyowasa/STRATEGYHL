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
- 出力先: `data/hyperliquid/candles/`（例: `data_eth_perp_1m.json`, `data_ueth_usdc_1h.json`）
- spot は UETH/USDC に remap されるため `@index`（例: `@151`）で取得します（`spotMeta` は `data/hyperliquid/meta/spotMeta.json` に保存）。
- 45m は API 非対応のため、15m から集約して生成: `python scripts/derive_hl_candles.py --input data/hyperliquid/candles/data_eth_perp_15m.json --output data/hyperliquid/candles/data_eth_perp_45m.json --target-interval 45m`

## Testing

```bash
poetry run pytest
```
# STRATEGYHL

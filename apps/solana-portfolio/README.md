# Orbit · Solana portfolio

React foundation for replacing the Streamlit app. Combines portfolio equity, Kamino collateral/debt and per-obligation health, candlesticks with configurable Supertrend, and clickable execution markers. Supports stablecoin-funded longs and borrowed-token shorts. Overview, analytics, positions and searchable activity share the selected account.

The page-wide market strip shows ETH, SOL and BTC USD marks on desktop and mobile. Its labeled Refresh button bypasses the live account cache and clears cached candles before the charts reload. Demo and imported marks are labeled; imported history remains a fixed snapshot. Clicking a price opens that asset's analytics chart.

## Run

From the repository root, after the existing workspace setup (`uv sync --all-packages --group dev` and `pnpm install --frozen-lockfile`):

```sh
npm run dev:portfolio
```

Open **http://127.0.0.1:5174**. The launcher starts Vite and the Python API together on ports 5174 and 8788. It uses `.venv/bin/python` when available, otherwise `uv run --package arblab`. Both bind to localhost. The existing Streamlit apps and root default `dev` command remain available during migration.

The initial workspace is visibly labeled **Fictional demonstration**. Select **Load my wallet** to use the address read at runtime from `private/address.txt`, or enter another public Solana address. No address or RPC key is bundled into the frontend. Optional server configuration in the root `.env`:

```dotenv
SOLANA_RPC_URL=https://your-mainnet-rpc-provider
```

For separate processes, run `.venv/bin/python tools/run_solana_portfolio.py` and `npm --prefix apps/solana-portfolio run dev`. Production build: `npm run build:portfolio`. A production deployment must serve the generated `dist` assets and proxy `/api` to this local API; the static bundle alone does not provide account data.

## Accounting and coverage

- Equity = supplied underlying assets + owned native/SPL wallet balances − all token and stablecoin debt. Risk-weighted debt is used for health, rather than equity. Each obligation has its own health; collateral in a different obligation does not provide protection.
- Live health reads the protocol-recorded deposited, debt, allowed-borrow and unhealthy-borrow values, including elevation-group terms. Holdings use the existing on-chain loader. On-chain state can be stale until refreshed by the protocol; account and reserve reads are not an atomic single-slot snapshot. Verify current health against Kamino before acting.
- Wallet amounts include native SOL and wrapped SOL once each. Unknown token mints, unsupported assets and possible receipt tokens are unpriced rather than assigned zero. Full equity becomes unavailable when valuation/provider coverage is incomplete; **Known portfolio equity** displays only the priced/loaded portion. Local observations preserve that portion and its coverage flags in ignored `private/solana-portfolio/mainnet-<wallet>.json` files.
- Live activity paginates **all wallet-referencing signatures to inception** (1,000 per RPC page), with a wallet-specific SQLite cache in ignored `private/solana-portfolio`. A background worker resumes unfinished downloads across restarts; processed/pending/missing counts remain visible and chart markers update every five seconds. Known failed signatures are counted separately and do not appear as successful executions. Public RPC may rate-limit or prune transactions; public-mainnet transaction requests are paced and HTTP 429 is retried once. Explicit Jupiter route, Orca Whirlpool/legacy swap, and Raydium AMM/CLMM/CPMM swap instructions are decoded with two opposing owned asset changes. Stablecoin, SOL-quoted, and token/token swaps preserve both execution legs; their prices remain in actual quote units. Liquidity operations and mere program touches do not establish swaps. Cached raw transactions are reclassified when the decoder version changes, without downloading them again. Native/wrapped SOL changes are normalized for network fees and owned token-account rent. Wrapping by itself is not a trade. For verified direct Orca/Raydium swaps, explicit separate outgoing System SOL transfers are isolated and shown as separate outflows of unknown purpose; incoming transfers and transfers to third-party wrapping accounts remain ambiguous. Other DEX programs, composite Kamino actions and ambiguous transfers remain unclassified. Loans are never inferred from a swap's timing. Live execution prices are in the actual quote, without assuming dollar parity.
- Price charts use **Coinbase USD spot candles**, not the DEX pool's candles. The provider paginates the requested range, including Inception; a 4h chart aggregates 1h bars. Complete candle responses are reused for five minutes. Long hourly histories can take substantially longer to load than daily candles. Only completed candles are returned. Supertrend uses Wilder ATR, explicit warmup and close-confirmed direction changes. Chart hover shows OHLC, zoom/pan controls inspect history, and execution markers use transaction timestamps mapped to candle buckets. Imported charts are anchored to the import's last observation.
- **Liquidations** have a dedicated activity filter and red diamond chart markers. The details group records by transaction and obligation, distinguishing seized collateral from repaid debt using the official liquidation withdrawal flag. Provider token valuations are preserved; they do not establish an exact liquidation penalty or realized trading P&L.
- The chart's **1 Day / 1 Week / 1 Month** controls select the candle interval. Weeks start Monday UTC; months use actual calendar boundaries. Incomplete periods and periods with missing source candles are excluded. The **Equity** checkbox adds the selected equity scope on an independent purple USD axis; missing values, daily gaps and valuation-coverage changes break the line.
- **Combined Supertrend** colors each candle and its background by equally weighted daily, weekly and monthly Supertrend directions: deep red for three bearish, intermediate colors for mixed states, and deep green for three bullish. The status row and legend identify each state. All three use the selected ATR settings and only higher-timeframe candles completed by the displayed candle's close; unavailable directions make the combined color gray. Additional historical daily data provides warm-up independently of the visible range.
- A historical **health heatmap** beneath Supertrend uses liquidation limit divided by borrow-factor-adjusted debt. Select the current obligation, another obligation, or the worst across all obligations. Red is at/below 1.00, green is 1.50 or higher; no debt is distinct from unavailable values. Missing daily snapshots are gray. Intraday candles share their provider daily bucket value, so the strip does not claim exact health at each trade. Hover or focus the strip and use the arrow/Home/End keys to inspect dates and values.
- Historical **Kamino net equity** is available from the official all-market transaction and daily obligation metrics APIs, including historical obligations. It excludes wallet assets. The scope selector offers aggregate Kamino and each obligation separately, alongside portfolio observations. Daily labels are provider buckets rather than exact midnight states. Missing active-obligation buckets are omitted, with no fabricated zero balances or carry-forward; gaps remain visible. These values do not establish complete portfolio equity or flow-adjusted returns.
- Live APY, cumulative interest, loan linkage, realized trade P&L, exact intraday historical health and cash-flow-adjusted returns are unavailable without the necessary index/cost basis. This app exposes those gaps instead of substituting demo values. Importing full equity/cash-flow history enables portfolio returns, without inventing current positions.

## Historical import

**Data & settings → Download example JSON** provides the schema. Upload JSON under 2 MB; at most 10,000 observations and trades. Example:

```json
{
  "source": {
    "name": "My accounting export",
    "account": "My main wallet",
    "equityScope": "wallet-and-kamino",
    "cashFlowCoverage": "complete"
  },
  "complete": true,
  "flowTiming": "period-end",
  "history": [
    {"time": 1790553600, "equity": 1000, "externalFlow": 0, "solPrice": 118},
    {"time": 1790640000, "equity": 1600, "externalFlow": 500, "solPrice": 120}
  ],
  "trades": []
}
```

Times are ascending, unique **UTC Unix seconds**. `equity` is total USD equity across wallet and Kamino. `externalFlow` is USD-valued capital entering the combined portfolio (positive) or leaving it (negative), assumed to occur at each observation's period end. Opening capital is already in the first equity value, so its flow is zero. Kamino deposits/withdrawals within the portfolio, borrow/repay actions and DEX swaps are **internal movements**, not external flows.

The importer checks structure and retains source attribution; completeness remains **declared by you**, rather than independently verified. Set `complete: false` and `cashFlowCoverage: partial` if coverage is incomplete; returns remain unavailable. No source file is uploaded to a third-party service or persisted by the API.

P&L = ending equity − starting equity − subsequent external flows. Time-weighted return chains `(ending equity − period-end flow) / preceding equity`. Drawdown uses this chained index. SOL-equivalent P&L removes each flow at its observed SOL price. These formulas require period-end flows; arbitrary intraperiod flows require finer observations. Zero ending equity produces a −100% return; further periods starting at nonpositive equity cannot establish a valid compounded return.

Optional trade records need unique `id`, `time`, `type` (buy/sell/borrow/repay/deposit/withdraw), `asset` (SOL/ETH/BTC/USDC/USDT), positive `amount` and **USD** `price`. Supported optional fields are `protocol`, `funding`, `signature`, `quote` (USD only), and `feeSol`. Optional fields are validated and arbitrary fields removed. Actual realized P&L requires opening cost basis and remains unavailable.

## Check

```sh
npm run test:portfolio
npm run build:portfolio
npm --prefix apps/solana-portfolio run test:browser
```

Browser checks cover marker selection, asset/timeframe/indicator controls, navigation/search, live failure isolation, import behavior and mobile overflow. Test screenshots are written to `/tmp/solana-portfolio-desktop.png` and `/tmp/solana-portfolio-mobile.png`.

Opt into the actual supplied-wallet browser check with `PORTFOLIO_LIVE_TEST=1 npm --prefix apps/solana-portfolio run test:browser`. It reads mainnet and can take 1–2 minutes on public RPC. Account refreshes are cached for 60 seconds. The live screenshot stays in `/tmp/solana-portfolio-live.png`.

Next migration work: wider DEX/composite action support, dated wallet balances reconciled to protocol history and cost basis, then the Streamlit simulator's scenario/recovery workflows. The current app is read-only and does not execute trades or replace those workflows yet.

Provider references: [Solana JSON structures](https://solana.com/docs/rpc/json-structures), [Solana getTransaction](https://solana.com/docs/rpc/http/gettransaction), [Coinbase candle API](https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles), [Kamino public API](https://api.kamino.finance/), [historical obligation metrics](https://kamino.com/docs/build/api-reference/borrow/user-and-loans-data/obligation-metrics-history.md).

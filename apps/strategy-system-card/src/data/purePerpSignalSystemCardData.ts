import assetDiagnostics from "./purePerpAssetDiagnostics.json";
import type { ChartPoint, MetricRow, TrafficState } from "../types";

export interface AssetDiagnosticPoint {
  timestamp: string;
  strategy: number;
  buyHold: number;
  relativePct: number;
  drawdownPct: number;
}

export interface AssetDiagnosticChart {
  asset: string;
  title: string;
  subtitle: string;
  totalReturnPct: number;
  buyHoldPct: number;
  maxDrawdownPct: number;
  worstRelativeDate: string;
  worstRelativePct: number;
  maxDrawdownDate: string;
  points: AssetDiagnosticPoint[];
}

export interface PurePerpSignalSystemCardData {
  meta: {
    id: "pure-perp-signal";
    title: string;
    shortTitle: string;
    generatedFrom: string[];
    window: string;
    status: string;
  };
  profileMetrics: MetricRow[];
  rotatingPortfolioMetrics: MetricRow[];
  benchmarks: MetricRow[];
  profileConfigs: MetricRow[];
  charts: {
    btc: ChartPoint[];
    eth: ChartPoint[];
    sol: ChartPoint[];
  };
  assetDiagnostics: AssetDiagnosticChart[];
  trafficStates: TrafficState[];
  governors: string[];
  useCases: string[];
  nonUseCases: string[];
  failureModes: string[];
  productionControls: string[];
  artifactLinks: string[];
}

const btcProfile: ChartPoint[] = [
  { timestamp: "2022-01-01", normalized_value: 100.7019, drawdown_pct: 0, target_long_fraction: 0.2798, target_short_fraction: 0 },
  { timestamp: "2022-04-01", normalized_value: 101.4178, drawdown_pct: -2.1309, target_long_fraction: 0.5294, target_short_fraction: 0 },
  { timestamp: "2022-06-30", normalized_value: 86.6301, drawdown_pct: -16.4011, target_long_fraction: 0, target_short_fraction: 0.5238 },
  { timestamp: "2022-09-28", normalized_value: 74.5674, drawdown_pct: -28.0418, target_long_fraction: 0, target_short_fraction: 0.2462 },
  { timestamp: "2022-12-27", normalized_value: 77.8597, drawdown_pct: -24.8647, target_long_fraction: 0, target_short_fraction: 0.947 },
  { timestamp: "2023-03-27", normalized_value: 91.7675, drawdown_pct: -11.4436, target_long_fraction: 0.5019, target_short_fraction: 0 },
  { timestamp: "2023-06-25", normalized_value: 91.7208, drawdown_pct: -11.4885, target_long_fraction: 1.2403, target_short_fraction: 0 },
  { timestamp: "2023-09-23", normalized_value: 80.6109, drawdown_pct: -22.2097, target_long_fraction: 0.4202, target_short_fraction: 0 },
  { timestamp: "2023-12-22", normalized_value: 129.437, drawdown_pct: -2.865, target_long_fraction: 1.0817, target_short_fraction: 0 },
  { timestamp: "2024-03-21", normalized_value: 180.8897, drawdown_pct: -7.63, target_long_fraction: 0.4968, target_short_fraction: 0 },
  { timestamp: "2024-06-19", normalized_value: 162.4621, drawdown_pct: -17.04, target_long_fraction: 0.4894, target_short_fraction: 0 },
  { timestamp: "2024-09-17", normalized_value: 148.9215, drawdown_pct: -23.9544, target_long_fraction: 0.1967, target_short_fraction: 0 },
  { timestamp: "2024-12-16", normalized_value: 219.8813, drawdown_pct: 0, target_long_fraction: 0.8135, target_short_fraction: 0 },
  { timestamp: "2025-03-16", normalized_value: 195.2416, drawdown_pct: -12.9857, target_long_fraction: 0.1409, target_short_fraction: 0 },
  { timestamp: "2025-06-14", normalized_value: 220.2441, drawdown_pct: -8.2686, target_long_fraction: 0.5019, target_short_fraction: 0 },
  { timestamp: "2025-09-12", normalized_value: 228.4578, drawdown_pct: -10.6807, target_long_fraction: 0.4717, target_short_fraction: 0 },
  { timestamp: "2025-12-11", normalized_value: 211.2624, drawdown_pct: -17.4035, target_long_fraction: 0, target_short_fraction: 0.2967 },
  { timestamp: "2026-03-11", normalized_value: 280.0898, drawdown_pct: -3.7464, target_long_fraction: 0, target_short_fraction: 0.2546 },
  { timestamp: "2026-06-09", normalized_value: 311.734, drawdown_pct: -1.0056, target_long_fraction: 0, target_short_fraction: 0.5676 },
  { timestamp: "2026-07-01", normalized_value: 311.1636, drawdown_pct: -1.1867, target_long_fraction: 0, target_short_fraction: 0.4759 },
];

const ethProfile: ChartPoint[] = [
  { timestamp: "2022-01-01", normalized_value: 100.4953, drawdown_pct: 0, target_long_fraction: 0.2788, target_short_fraction: 0 },
  { timestamp: "2022-04-01", normalized_value: 96.2925, drawdown_pct: -6.9891, target_long_fraction: 0.7998, target_short_fraction: 0 },
  { timestamp: "2022-06-30", normalized_value: 94.68, drawdown_pct: -8.5467, target_long_fraction: 0, target_short_fraction: 0.4704 },
  { timestamp: "2022-09-28", normalized_value: 89.1986, drawdown_pct: -13.8413, target_long_fraction: 0.0573, target_short_fraction: 0 },
  { timestamp: "2022-12-27", normalized_value: 92.7372, drawdown_pct: -25.1384, target_long_fraction: 0.6401, target_short_fraction: 0 },
  { timestamp: "2023-03-27", normalized_value: 105.6678, drawdown_pct: -27.9161, target_long_fraction: 0.4714, target_short_fraction: 0 },
  { timestamp: "2023-06-25", normalized_value: 108.9298, drawdown_pct: -25.6909, target_long_fraction: 2.0166, target_short_fraction: 0 },
  { timestamp: "2023-09-23", normalized_value: 96.5842, drawdown_pct: -34.1127, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2023-12-22", normalized_value: 113.4366, drawdown_pct: -22.6164, target_long_fraction: 1.8344, target_short_fraction: 0 },
  { timestamp: "2024-03-21", normalized_value: 223.3355, drawdown_pct: -10.308, target_long_fraction: 0.3736, target_short_fraction: 0 },
  { timestamp: "2024-06-19", normalized_value: 216.9792, drawdown_pct: -12.8607, target_long_fraction: 0, target_short_fraction: 0.5114 },
  { timestamp: "2024-09-17", normalized_value: 238.4717, drawdown_pct: -4.2293, target_long_fraction: 0, target_short_fraction: 0.4585 },
  { timestamp: "2024-12-16", normalized_value: 271.9233, drawdown_pct: -1.4173, target_long_fraction: 1.2764, target_short_fraction: 0 },
  { timestamp: "2025-03-16", normalized_value: 237.1416, drawdown_pct: -14.027, target_long_fraction: 0, target_short_fraction: 0.5335 },
  { timestamp: "2025-06-14", normalized_value: 200.6331, drawdown_pct: -27.2628, target_long_fraction: 0.4355, target_short_fraction: 0 },
  { timestamp: "2025-09-12", normalized_value: 316.602, drawdown_pct: -12.1747, target_long_fraction: 1.2966, target_short_fraction: 0 },
  { timestamp: "2025-12-11", normalized_value: 238.7735, drawdown_pct: -33.7643, target_long_fraction: 0.0666, target_short_fraction: 0 },
  { timestamp: "2026-03-11", normalized_value: 258.5083, drawdown_pct: -28.2899, target_long_fraction: 0.44, target_short_fraction: 0 },
  { timestamp: "2026-06-09", normalized_value: 287.5546, drawdown_pct: -20.2325, target_long_fraction: 0, target_short_fraction: 0.5225 },
  { timestamp: "2026-07-01", normalized_value: 265.096, drawdown_pct: -26.4625, target_long_fraction: 0, target_short_fraction: 0.48 },
];

const solProfile: ChartPoint[] = [
  { timestamp: "2022-01-01", normalized_value: 101.4514, drawdown_pct: 0, target_long_fraction: 0.2807, target_short_fraction: 0 },
  { timestamp: "2022-04-01", normalized_value: 95.6772, drawdown_pct: -5.6916, target_long_fraction: 0.984, target_short_fraction: 0 },
  { timestamp: "2022-06-30", normalized_value: 62.7451, drawdown_pct: -38.1525, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2022-09-28", normalized_value: 60.5503, drawdown_pct: -40.3159, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2022-12-27", normalized_value: 57.6224, drawdown_pct: -43.202, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2023-03-27", normalized_value: 60.0916, drawdown_pct: -40.7681, target_long_fraction: 0.3053, target_short_fraction: 0 },
  { timestamp: "2023-06-25", normalized_value: 52.2812, drawdown_pct: -48.4668, target_long_fraction: 0.4916, target_short_fraction: 0 },
  { timestamp: "2023-09-23", normalized_value: 57.4422, drawdown_pct: -43.3796, target_long_fraction: 0.5711, target_short_fraction: 0 },
  { timestamp: "2023-12-22", normalized_value: 289.0714, drawdown_pct: 0, target_long_fraction: 0.9394, target_short_fraction: 0 },
  { timestamp: "2024-03-21", normalized_value: 540.4171, drawdown_pct: -11.4501, target_long_fraction: 1.0708, target_short_fraction: 0 },
  { timestamp: "2024-06-19", normalized_value: 401.7944, drawdown_pct: -34.6399, target_long_fraction: 0.4746, target_short_fraction: 0 },
  { timestamp: "2024-09-17", normalized_value: 388.0121, drawdown_pct: -36.8818, target_long_fraction: 0.151, target_short_fraction: 0 },
  { timestamp: "2024-12-16", normalized_value: 586.6897, drawdown_pct: -15.1049, target_long_fraction: 0.4541, target_short_fraction: 0 },
  { timestamp: "2025-03-16", normalized_value: 481.796, drawdown_pct: -30.2832, target_long_fraction: 0.1453, target_short_fraction: 0 },
  { timestamp: "2025-06-14", normalized_value: 520.0901, drawdown_pct: -24.742, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2025-09-12", normalized_value: 541.8358, drawdown_pct: -21.5953, target_long_fraction: 0.535, target_short_fraction: 0 },
  { timestamp: "2025-12-11", normalized_value: 516.741, drawdown_pct: -25.2266, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2026-03-11", normalized_value: 516.741, drawdown_pct: -25.2266, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2026-06-09", normalized_value: 497.0481, drawdown_pct: -28.0762, target_long_fraction: 0, target_short_fraction: 0 },
  { timestamp: "2026-07-01", normalized_value: 497.0481, drawdown_pct: -28.0762, target_long_fraction: 0, target_short_fraction: 0 },
];


export const purePerpSignalSystemCardData: PurePerpSignalSystemCardData = {
  meta: {
    id: "pure-perp-signal",
    title: "Pure Perp Signal Strategy System Card",
    shortTitle: "Pure Perp Signal",
    generatedFrom: [
      "reports/profile_diagnostics_20260706",
      "reports/multi_asset_signal_family_review_20260706",
      "reports/rotating_long_portfolio_20260706_002459",
      "arblab/perps/profiles.py",
    ],
    window: "2022-01-01 to 2026-07-01",
    status: "Research checkpoint before universe design",
  },
  profileMetrics: [
    { name: "BTC profile", asset: "BTC", total_return_pct: 211.16, buy_hold_pct: 25.55, max_drawdown_pct: 32.84, sharpe: 0.99, sortino: 1.2, trades: 244, liquidations: 0, archetype: "macro_trend" },
    { name: "ETH profile", asset: "ETH", total_return_pct: 165.1, buy_hold_pct: -57.69, max_drawdown_pct: 39.07, sharpe: 0.7, sortino: 0.74, trades: 506, liquidations: 0, archetype: "fast_transition" },
    { name: "SOL profile", asset: "SOL", total_return_pct: 397.05, buy_hold_pct: -57.27, max_drawdown_pct: 50.18, sharpe: 0.93, sortino: 0.93, trades: 141, liquidations: 0, archetype: "high_beta_long_only" },
  ],
  rotatingPortfolioMetrics: [
    { name: "Selector A", rule: "highest positive target_exposure", total_return_pct: 220.79, max_drawdown_pct: 52.13, sharpe: 0.73, sortino: 0.82, average_abs_exposure: 0.71, trades: 456, rotations: 340 },
    { name: "Selector B", rule: "supertrend_score * vol_target_confidence", total_return_pct: 200.47, max_drawdown_pct: 53.77, sharpe: 0.71, sortino: 0.75, average_abs_exposure: 0.69, trades: 457, rotations: 357 },
  ],
  benchmarks: [
    { name: "BTC buy-and-hold", total_return_pct: 25.55 },
    { name: "ETH buy-and-hold", total_return_pct: -57.69 },
    { name: "SOL buy-and-hold", total_return_pct: -57.27 },
    { name: "Equal-weight buy-and-hold", total_return_pct: -20.85 },
  ],
  profileConfigs: [
    { asset: "BTC", supertrend: "1w 7/4, 1d 21/4, 4h 7/4", rsi_filter: "on", target_vol_pct: 45, vol_window_hours: 1008, long_cap: 1.5, short_cap: 1, bull_floor: 0.5, short_rule: "weekly bull blocks shorts" },
    { asset: "ETH", supertrend: "1w 14/2, 1d 7/2, 4h 10/2", rsi_filter: "off", target_vol_pct: 90, vol_window_hours: 672, long_cap: 2, short_cap: 0.5, bull_floor: 0.25, short_rule: "short only when weekly and daily bearish" },
    { asset: "SOL", supertrend: "1w 7/4, 1d 21/4, 4h 7/4", rsi_filter: "on", target_vol_pct: 110, vol_window_hours: 1344, long_cap: 1.5, short_cap: 0, bull_floor: 0, short_rule: "near long-only" },
  ],
  charts: {
    btc: btcProfile,
    eth: ethProfile,
    sol: solProfile,
  },
  assetDiagnostics: assetDiagnostics as AssetDiagnosticChart[],
  trafficStates: [
    { state: "Weekly Bull", meaning: "Higher-timeframe trend supports long participation.", behavior: "Allow long floors and block or limit shorts depending on asset profile." },
    { state: "Daily/4h Agreement", meaning: "Intermediate trend confirms the weekly state.", behavior: "Increase conviction and allow higher target exposure through volatility targeting." },
    { state: "Mixed Pullback", meaning: "Weekly trend is constructive but lower timeframes weaken.", behavior: "Use small pullback floors for BTC/SOL-like assets instead of full de-risking." },
    { state: "Bear Confirmation", meaning: "Weekly and daily trend are bearish.", behavior: "Allow ETH/BTC shorts subject to profile caps; SOL remains near long-only in the current checkpoint." },
    { state: "Flat / No Trade", meaning: "The score is inside the no-trade zone or profile filters reject exposure.", behavior: "Hold cash rather than forcing a position." },
  ],
  governors: [
    "Pure signal generation is stateless: price-derived features map to target exposure without reading account history.",
    "Volatility targeting scales the raw signal by realized volatility, then clips by asset-specific long and short caps.",
    "Short governors prevent shorting weekly bull regimes and require stronger bearish confirmation for ETH.",
    "Bull and pullback floors preserve small long exposure when the weekly regime is constructive.",
    "Venue mechanics, fees, liquidation checks, funding, and deadbands are modeled outside the signal layer.",
  ],
  useCases: [
    "Use as a research-grade signal family for liquid crypto perpetual futures.",
    "Use when the execution venue supports linear perp exposure, fees, funding, and liquidation accounting.",
    "Use asset archetypes to initialize parameters before universe-wide calibration.",
    "Use the profile diagnostics as a baseline before testing broader tradable universes.",
  ],
  nonUseCases: [
    "Do not treat one universal parameter set as validated across assets.",
    "Do not trade illiquid or newly listed perps without a separate universe eligibility layer.",
    "Do not assume RSI overbought means immediate bearishness in crypto trend regimes.",
    "Do not use the rotating portfolio selector as final production allocation logic yet.",
  ],
  failureModes: [
    "Overfit profiles: BTC, ETH, and SOL required materially different settings over the current research window.",
    "Short-side asymmetry: SOL-like high-beta assets punished symmetric long/short assumptions.",
    "Turnover pressure: rotating portfolio selectors changed assets hundreds of times over the test window.",
    "Universe bias: selecting today’s surviving perps historically would overstate performance.",
    "Execution gaps: funding, spread, latency, and liquidation behavior remain deployment-sensitive.",
  ],
  productionControls: [
    "Exchange adapter for symbol status, min notional, lot size, leverage limits, funding, and delist warnings.",
    "Historical universe reconstruction to avoid survivor-only backtests.",
    "Liquidity, open-interest, spread/slippage, and data-gap filters before signal ranking.",
    "Per-asset archetype classification with out-of-sample validation.",
    "Portfolio-level concentration, turnover, and kill-switch controls before live use.",
  ],
  artifactLinks: [
    "reports/profile_diagnostics_20260706/report.md",
    "reports/multi_asset_signal_family_review_20260706/report.md",
    "reports/rotating_long_portfolio_20260706_002459/report.md",
    "arblab/perps/profiles.py",
    "arblab/perps/signal.py",
    "arblab/perps/rotating_portfolio.py",
  ],
};

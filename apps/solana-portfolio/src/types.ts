export type Mode = "demo" | "live" | "imported";
export type Asset = "SOL" | "ETH" | "BTC";
export type Interval = "1h" | "4h" | "1d" | "1w" | "1M";
export type View = "overview" | "analytics" | "positions" | "activity";
export type Trade = {
  id: string; time: number; type: string; asset: string | null; amount: number | null;
  price: number | null; value: number | null; funding: string; protocol: string;
  signature: string | null; feeSol?: number; externalSolOutflowSol?: number; quote?: string; mint?: string; transactionName?: string; source?: string;
  executionLegs?: { mint: string; asset: string; side: "buy" | "sell"; amount: number; price: number; quote: string; quoteMint: string; value: number; delta?: number }[];
  obligation?: string;
  liquidationRole?: "collateral-seized" | "debt-repaid" | "unknown" | null;
  venues?: string[];
  deltas?: Record<string, number>;
};
export type Position = {
  id: string; symbol: string; amount: number; price: number | null; value: number | null;
  kind: "supplied" | "borrowed" | "wallet"; obligation: string | null; apy: number | null; mint?: string;
};
export type Loan = {
  address: string; supplied: number; debt: number; ltv: number | null; liquidationLtv: number | null;
  health: number | null; borrowHealth: number | null; liquidationBuffer: number | null;
};
export type Observation = { time: number; equity: number; externalFlow: number | null; solPrice: number | null; valuationComplete?: boolean; unpricedCount?: number };
export type HealthPoint = { time: number; health: number | null; status: "available" | "no-debt" | "unavailable"; adjustedDebt?: number | null; liquidationLimit?: number | null; source?: string };
export type HealthSeries = { id: string; label: string; healthHistory?: HealthPoint[] };
export type Metrics = { pnl: number | null; returnPct: number | null; solPnl: number | null; drawdownPct: number | null };
export type Portfolio = {
  mode: Mode; wallet: string | null; asOf: number; source: string; historyComplete: boolean;
  summary: {
    netEquity: number | null; knownEquity: number; supplied: number | null; debt: number | null;
    walletValue: number | null; health: number | null; unpricedCount: number;
    loans: Loan[]; positions: Position[]; exposure: Record<string, number>;
  };
  metrics: Metrics; metricsByRange: Record<string, Metrics>; history: Observation[]; trades: Trade[]; warnings: string[];
  healthHistory?: HealthPoint[];
  kaminoHistory?: Observation[];
  kaminoSeries?: { id: string; label: string; history: Observation[]; healthHistory?: HealthPoint[] }[];
  kaminoHistoryWarnings?: string[];
  historyStatus?: { discovered: number; processed: number; missing: number; failed: number; oldest: number | null; discoveryComplete: boolean; decodingComplete: boolean; error?: string | null };
  prices: Record<string, number | null>;
};
export type Candle = { time: number; endTime?: number; open: number; high: number; low: number; close: number; volume: number };
export type Trend = { time: number; value: number | null; direction: "bullish" | "bearish" | null; atr: number | null };
export type CombinedTrend = { time: number; score: number | null; bullishCount: number; bearishCount: number; directions: Record<"1d" | "1w" | "1M", "bullish" | "bearish" | null> };
export type ChartData = { candles: Candle[]; indicator: Trend[]; combined?: CombinedTrend[]; source: string; asset: Asset; interval: Interval };

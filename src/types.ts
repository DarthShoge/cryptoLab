export type NumericValue = number | null;

export type MetricRow = Record<string, string | number | boolean | null>;

export interface ChartPoint {
  timestamp: string;
  portfolio_value?: NumericValue;
  normalized_value?: NumericValue;
  drawdown_pct?: NumericValue;
  target_long_fraction?: NumericValue;
  target_short_fraction?: NumericValue;
  health_factor?: NumericValue;
}

export interface TrafficState {
  state: string;
  meaning: string;
  behavior: string;
}

export interface StrategySystemCardData {
  meta: {
    title: string;
    generatedFrom: string[];
    topCandidate: string;
    window: string;
  };
  topCandidate: MetricRow;
  latestStrategies: MetricRow[];
  scenarioStrategies: MetricRow[];
  benchmarks: MetricRow[];
  regimes: MetricRow[];
  charts: {
    topEquity: ChartPoint[];
    controlEquity: ChartPoint[];
    solOnlyEquity: ChartPoint[];
    btcOnlyEquity: ChartPoint[];
    ethOnlyEquity: ChartPoint[];
  };
  trafficStates: TrafficState[];
  governors: string[];
  useCases: string[];
  nonUseCases: string[];
  failureModes: string[];
  productionControls: string[];
  artifactLinks: string[];
}

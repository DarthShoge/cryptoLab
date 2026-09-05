import type { components } from "./api.generated";
import type { Config } from "./labApi";

type Schemas = components["schemas"];
export type Explicit = Required<Schemas["ExplicitUniverse"]>;
export type Liquidity = Required<Schemas["LiquidityUniverse"]>;
export type MarketUniverse = Explicit | Liquidity;
export type AssetClass = MarketUniverse["classes"][number];
export type ConfigV2 = Required<Schemas["LabConfigV2"]> & {
  market_universe: MarketUniverse;
};
export type WireConfig = Config | ConfigV2;
export const classes: [AssetClass, string][] = [
  ["crypto", "Crypto"],
  ["commodity", "Commodities"],
  ["equity", "Equities"],
  ["index", "Indices"],
];

export function generalUniverse(previous: MarketUniverse): Liquidity {
  return {
    mode: "liquidity",
    general: true,
    classes: [],
    top_n: 3,
    min_volume_usd: 0,
    lookback_days: 30,
    reselection: previous.reselection,
    metric: "traded_notional_usd",
    publication_lag_days: 1,
  };
}
export function changeMode(
  previous: MarketUniverse,
  mode: MarketUniverse["mode"],
): MarketUniverse {
  if (mode === previous.mode) return previous;
  if (mode === "liquidity")
    return {
      ...generalUniverse(previous),
      general: previous.general,
      classes: previous.classes,
    };
  return {
    mode: "explicit",
    general: false,
    classes: previous.classes,
    instrument_ids: [],
    allocation: "equal",
    weights: null,
    reselection: previous.reselection,
  };
}
export function changeClass(
  previous: MarketUniverse,
  assetClass: AssetClass,
  checked: boolean,
): MarketUniverse {
  const selected = checked
    ? [...new Set([...previous.classes, assetClass])].sort()
    : previous.classes.filter((c) => c !== assetClass);
  const next = { ...previous, general: false, classes: selected };
  return next.mode === "explicit"
    ? { ...next, instrument_ids: [], allocation: "equal", weights: null }
    : next;
}
export function marketSummary(universe: MarketUniverse): string {
  return universe.mode === "explicit"
    ? universe.instrument_ids.join(" + ")
    : `${universe.general ? "General" : universe.classes.join(" + ")} · top ${universe.top_n} assets by ${universe.lookback_days}d USD volume (1d lag) · assets ${universe.reselection}`;
}
export function universeOf(config: WireConfig): MarketUniverse {
  return config.schema_version === "hyperliquid_copy_lab_v2"
    ? config.market_universe
    : {
        mode: "explicit",
        general: false,
        classes: ["crypto"],
        instrument_ids: config.coins,
        allocation: "custom",
        weights: config.asset_weights,
        reselection: "daily",
      };
}
// Flat form view is an editor adapter, never the persisted v2 configuration.
export function editorConfig(
  config: WireConfig,
  candidates: string[] = [],
): Config {
  if (config.schema_version === "hyperliquid_copy_lab_v1") return config;
  const u = config.market_universe;
  const coins = u.mode === "explicit" ? u.instrument_ids : candidates;
  return {
    ...config.trader,
    ...config.follower,
    schema_version: "hyperliquid_copy_lab_v1",
    start: config.start,
    end: config.end,
    benchmark: config.benchmark,
    split: config.split,
    coins,
    asset_weights:
      u.mode === "explicit" && u.weights
        ? u.weights
        : Object.fromEntries(coins.map((id) => [id, 1 / coins.length])),
  } as Config;
}
export function wireConfig(
  c: Config,
  market_universe: MarketUniverse,
): ConfigV2 {
  const {
    scope,
    lookback_days,
    min_active_days,
    min_episodes,
    min_notional,
    min_volume,
    min_minutes,
    metric_weights,
    metric_directions,
    selection,
    top_fraction,
    top_n,
    min_cohort,
    max_cohort,
    reselection,
    aggregation,
    initial_equity,
    gross_cap,
    asset_cap,
    update_minutes,
    latency_seconds,
    fee_bps,
    deadband,
    min_trade_usd,
    min_known,
    min_known_weight,
    scale_lookback_days,
    scale_quantile,
    trim,
    start,
    end,
    benchmark,
    split,
  } = c;
  return {
    schema_version: "hyperliquid_copy_lab_v2",
    market_universe,
    start,
    end,
    benchmark,
    split,
    trader: {
      scope,
      lookback_days,
      min_active_days,
      min_episodes,
      min_notional,
      min_volume,
      min_minutes,
      metric_weights,
      metric_directions,
      selection,
      top_fraction,
      top_n,
      min_cohort,
      max_cohort,
      reselection,
    },
    follower: {
      aggregation,
      initial_equity,
      gross_cap,
      asset_cap,
      update_minutes,
      latency_seconds,
      fee_bps,
      deadband,
      min_trade_usd,
      min_known,
      min_known_weight,
      scale_lookback_days,
      scale_quantile,
      trim,
    },
  };
}

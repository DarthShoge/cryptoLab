// Generated from Python OpenAPI models. Do not edit.
export interface paths {
    "/api/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Health */
        get: operations["health_api_health_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/bootstrap": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Bootstrap */
        get: operations["bootstrap_api_lab_bootstrap_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/compare": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Comparison */
        get: operations["comparison_api_lab_compare_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/datasets": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Datasets */
        get: operations["datasets_api_lab_datasets_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/datasets/{identifier}/instruments": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Instruments */
        get: operations["instruments_api_lab_datasets__identifier__instruments_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Experiments */
        get: operations["experiments_api_lab_experiments_get"];
        put?: never;
        /** Submit */
        post: operations["submit_api_lab_experiments_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Detail */
        get: operations["detail_api_lab_experiments__identifier__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/cancel": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Cancel */
        post: operations["cancel_api_lab_experiments__identifier__cancel_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/clone": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Clone */
        post: operations["clone_api_lab_experiments__identifier__clone_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/diagnostics": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Diagnostics */
        get: operations["diagnostics_api_lab_experiments__identifier__diagnostics_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/market-universe": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Market History */
        get: operations["market_history_api_lab_experiments__identifier__market_universe_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/metadata": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        /** Annotate */
        patch: operations["annotate_api_lab_experiments__identifier__metadata_patch"];
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/preview-info": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Preview Metadata */
        get: operations["preview_metadata_api_lab_experiments__identifier__preview_info_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/resume": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Resume */
        post: operations["resume_api_lab_experiments__identifier__resume_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/experiments/{identifier}/universe": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** History */
        get: operations["history_api_lab_experiments__identifier__universe_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/preflight": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Preflight */
        post: operations["preflight_api_lab_preflight_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/lab/previews": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /** Preview */
        post: operations["preview_api_lab_previews_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Runs */
        get: operations["runs_api_runs_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Detail */
        get: operations["detail_api_runs__run_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/analytics": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Analytics */
        get: operations["analytics_api_runs__run_id__analytics_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/artifacts/{name}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Artifact */
        get: operations["artifact_api_runs__run_id__artifacts__name__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/cohorts": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Cohorts */
        get: operations["cohorts_api_runs__run_id__cohorts_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/equity": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Equity */
        get: operations["equity_api_runs__run_id__equity_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/fills": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Fills */
        get: operations["fills_api_runs__run_id__fills_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/funding": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Funding */
        get: operations["funding_api_runs__run_id__funding_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{run_id}/traders": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /** Traders */
        get: operations["traders_api_runs__run_id__traders_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** AccountingCheck */
        AccountingCheck: {
            /** Difference Usd */
            difference_usd?: number | null;
            /** Name */
            name: string;
            /** Passed */
            passed?: boolean | null;
            /** Reason */
            reason?: string | null;
        };
        /** Analytics */
        Analytics: {
            /** Benchmark */
            benchmark?: {
                [key: string]: components["schemas"]["Metric"];
            } | null;
            /** Conventions */
            conventions: string;
            /** Metrics */
            metrics: {
                [key: string]: components["schemas"]["Metric"];
            };
            /** Warnings */
            warnings: string[];
        };
        /** Annotation */
        Annotation: {
            /** Name */
            name: string;
            /** Notes */
            notes: string;
        };
        /** Bootstrap */
        Bootstrap: {
            defaults: components["schemas"]["LabConfig"];
            /** Enabled */
            enabled: boolean;
            /** Limits */
            limits: {
                [key: string]: number;
            };
            /** Metrics */
            metrics: string[];
            /** Restrictions */
            restrictions: string[];
            /** Token */
            token: string;
        };
        /** Comparison */
        Comparison: {
            /** Differences */
            differences: string[];
            /** Series */
            series: components["schemas"]["ComparisonSeries"][];
            /**
             * Units
             * @enum {string}
             */
            units: "usd" | "growth";
            /** Warnings */
            warnings: string[];
        };
        /** ComparisonSeries */
        ComparisonSeries: {
            analytics: components["schemas"]["Analytics"];
            /** Config */
            config: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            curve: components["schemas"]["Page_EquityRow_"];
            /** Id */
            id: string;
            /** Market Membership Turnover */
            market_membership_turnover?: number | null;
            /** Mean Selected Assets */
            mean_selected_assets?: number | null;
            /** Membership Turnover */
            membership_turnover?: number | null;
            /** Name */
            name: string;
            /** Synthetic */
            synthetic: boolean;
        };
        /** Dataset */
        Dataset: {
            /** Available */
            available: boolean;
            /** Catalogue Hash */
            catalogue_hash?: string | null;
            /** Coins */
            coins?: string[];
            /** Coverage End */
            coverage_end?: string | null;
            /** Coverage Note */
            coverage_note: string;
            /** Coverage Start */
            coverage_start?: string | null;
            /** Dataset Hash */
            dataset_hash?: string | null;
            /** Default Config */
            default_config?: (components["schemas"]["LabConfig"] | components["schemas"]["LabConfigV2"] | components["schemas"]["LabConfigProxy"] | components["schemas"]["LabConfigProxyScheduled"]) | null;
            /** Fee Semantics */
            fee_semantics?: string | null;
            /** Id */
            id: string;
            /**
             * Liquidity Available
             * @default false
             */
            liquidity_available: boolean;
            /** Name */
            name: string;
            /**
             * Pricing Mode
             * @default order_book
             * @enum {string}
             */
            pricing_mode: "order_book" | "hourly_proxy";
            /** Proxy Mappings */
            proxy_mappings?: {
                [key: string]: components["schemas"]["JsonValue"];
            }[];
            /**
             * Rows
             * @default 0
             */
            rows: number;
            /** Supported Classes */
            supported_classes?: string[];
            /** Synthetic */
            synthetic: boolean;
        };
        /** DiagnosticCohort */
        DiagnosticCohort: {
            /** Candidate Count */
            candidate_count?: number | null;
            /** Coin */
            coin?: string | null;
            /**
             * Decision Time
             * Format: date-time
             */
            decision_time: string;
            /** Eligible Count */
            eligible_count?: number | null;
            /** Membership Turnover */
            membership_turnover?: number | null;
            /** Selected Count */
            selected_count?: number | null;
        };
        /** DiagnosticContext */
        DiagnosticContext: {
            /** Assets */
            assets?: string[];
            /**
             * End
             * Format: date-time
             */
            end: string;
            /** Exposure Reason */
            exposure_reason?: string | null;
            /** Fees Usd */
            fees_usd?: number | null;
            /** Fills */
            fills?: number | null;
            /** Funding Usd */
            funding_usd?: number | null;
            /** Max Gross Usd */
            max_gross_usd?: number | null;
            /** Mean Net Usd */
            mean_net_usd?: number | null;
            /** Return Value */
            return_value: number;
            /**
             * Start
             * Format: date-time
             */
            start: string;
        };
        /** DiagnosticMetric */
        DiagnosticMetric: {
            /** Reason */
            reason?: string | null;
            /**
             * Unit
             * @default ratio
             */
            unit: string;
            /** Value */
            value?: number | null;
        };
        /** DiagnosticPeriod */
        DiagnosticPeriod: {
            /**
             * End
             * Format: date-time
             */
            end: string;
            /** Partial */
            partial: boolean;
            /** Return Value */
            return_value: number;
            /**
             * Start
             * Format: date-time
             */
            start: string;
        };
        /** DiagnosticPoint */
        DiagnosticPoint: {
            /** Drawdown */
            drawdown: number;
            /** Equity */
            equity: number;
            /** Gross Exposure */
            gross_exposure?: number | null;
            /** Growth */
            growth: number;
            /** Net Exposure */
            net_exposure?: number | null;
            /**
             * Time
             * Format: date-time
             */
            time: string;
        };
        /** DiagnosticSeries */
        DiagnosticSeries: {
            /** Max Drawdown */
            max_drawdown?: number | null;
            /** Months */
            months?: components["schemas"]["DiagnosticPeriod"][];
            /** Points */
            points?: components["schemas"]["DiagnosticPoint"][];
            /** Reason */
            reason?: string | null;
            /**
             * Samples
             * @default 0
             */
            samples: number;
            /** Weeks */
            weeks?: components["schemas"]["DiagnosticPeriod"][];
        };
        /** Diagnostics */
        Diagnostics: {
            /** Accounting */
            accounting: {
                [key: string]: components["schemas"]["DiagnosticMetric"];
            };
            /** Benchmark Metrics */
            benchmark_metrics: {
                [key: string]: components["schemas"]["DiagnosticMetric"];
            };
            /** Checks */
            checks: components["schemas"]["AccountingCheck"][];
            /** Cohort Reason */
            cohort_reason?: string | null;
            /** Cohorts */
            cohorts: components["schemas"]["DiagnosticCohort"][];
            /** Config */
            config: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Config Hash */
            config_hash: string;
            /** Dataset Id */
            dataset_id: string;
            /** Experiment Id */
            experiment_id: string;
            /** Methods */
            methods: string[];
            /** Name */
            name: string;
            /** Positions */
            positions: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Provenance */
            provenance: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Reconciliation */
            reconciliation: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Research Eligible */
            research_eligible: boolean | null;
            /** Run Id */
            run_id: string;
            /**
             * Schema Version
             * @default saved_diagnostics_v1
             */
            schema_version: string;
            /** Series */
            series: {
                [key: string]: components["schemas"]["DiagnosticSeries"];
            };
            /** Statistics */
            statistics: {
                [key: string]: components["schemas"]["DiagnosticMetric"];
            };
            /** Stored Metrics */
            stored_metrics: {
                [key: string]: components["schemas"]["DiagnosticMetric"];
            };
            /**
             * Synthetic
             * @default false
             */
            synthetic: boolean;
            /** Warnings */
            warnings: string[];
            /** Worst Weeks */
            worst_weeks: components["schemas"]["DiagnosticContext"][];
        };
        /** EquityRow */
        EquityRow: {
            /** Cash */
            cash: number;
            /** Drawdown */
            drawdown: number;
            /** Equity */
            equity: number;
            /** Gross Exposure */
            gross_exposure: number;
            /** Net Exposure */
            net_exposure: number;
            /**
             * Time
             * Format: date-time
             */
            time: string;
            /** Unrealized Pnl */
            unrealized_pnl: number;
        };
        /** Experiment */
        Experiment: {
            /** Artifact Hashes */
            artifact_hashes: {
                [key: string]: string;
            };
            /** Config */
            config: components["schemas"]["LabConfig"] | components["schemas"]["LabConfigV2"] | components["schemas"]["LabConfigProxy"] | components["schemas"]["LabConfigProxyScheduled"];
            /** Config Hash */
            config_hash: string;
            /**
             * Created At
             * Format: date-time
             */
            created_at: string;
            /** Dataset Id */
            dataset_id: string;
            /** Error */
            error?: string | null;
            /** Id */
            id: string;
            /**
             * Kind
             * @enum {string}
             */
            kind: "backtest" | "cohort_preview";
            /** Name */
            name: string;
            /**
             * Needs Resume
             * @default false
             */
            needs_resume: boolean;
            /** Notes */
            notes: string;
            /** Parent Id */
            parent_id?: string | null;
            /** Preview Date */
            preview_date?: string | null;
            /** Preview Scope */
            preview_scope?: string | null;
            /** Provenance */
            provenance: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Run Id */
            run_id?: string | null;
            /**
             * Status
             * @enum {string}
             */
            status: "queued" | "running" | "completed" | "failed" | "cancelled";
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
        };
        /** ExplicitUniverse */
        ExplicitUniverse: {
            /**
             * Allocation
             * @default equal
             * @enum {string}
             */
            allocation: "equal" | "custom";
            /** Classes */
            classes?: ("crypto" | "commodity" | "equity" | "index")[];
            /**
             * General
             * @default false
             */
            general: boolean;
            /** Instrument Ids */
            instrument_ids?: string[];
            /**
             * Mode
             * @default explicit
             * @constant
             */
            mode: "explicit";
            /**
             * Reselection
             * @default daily
             * @enum {string}
             */
            reselection: "daily" | "weekly" | "monthly";
            /** Weights */
            weights?: {
                [key: string]: number;
            } | null;
        };
        /** FollowerSettings */
        FollowerSettings: {
            /**
             * Aggregation
             * @default direction_equal
             * @enum {string}
             */
            aggregation: "direction_equal" | "direction_score_weighted" | "conviction_trimmed";
            /**
             * Asset Cap
             * @default 0.5
             */
            asset_cap: number;
            /**
             * Deadband
             * @default 0.02
             */
            deadband: number;
            /**
             * Fee Bps
             * @default 4.5
             */
            fee_bps: number;
            /**
             * Gross Cap
             * @default 1
             */
            gross_cap: number;
            /**
             * Initial Equity
             * @default 10000
             */
            initial_equity: number;
            /**
             * Latency Seconds
             * @default 5
             */
            latency_seconds: number;
            /**
             * Min Known
             * @default 5
             */
            min_known: number;
            /**
             * Min Known Weight
             * @default 0.6
             */
            min_known_weight: number;
            /**
             * Min Trade Usd
             * @default 10
             */
            min_trade_usd: number;
            /**
             * Scale Lookback Days
             * @default 30
             */
            scale_lookback_days: number;
            /**
             * Scale Quantile
             * @default 0.95
             */
            scale_quantile: number;
            /**
             * Trim
             * @default 0.1
             */
            trim: number;
            /**
             * Update Minutes
             * @default 1
             */
            update_minutes: number;
        };
        /** HTTPValidationError */
        HTTPValidationError: {
            /** Detail */
            detail?: components["schemas"]["ValidationError"][];
        };
        /** Health */
        Health: {
            /**
             * Schema Version
             * @default 1
             */
            schema_version: string;
            /**
             * Status
             * @default ok
             */
            status: string;
        };
        /** InstrumentRow */
        InstrumentRow: {
            /** Asset Class */
            asset_class: string;
            /** Availability Basis */
            availability_basis?: string | null;
            /** Calendar */
            calendar?: string | null;
            /** Delisted At */
            delisted_at?: string | null;
            /** Display Name */
            display_name: string;
            /**
             * Effective From
             * Format: date-time
             */
            effective_from: string;
            /** Effective To */
            effective_to?: string | null;
            /** Instrument Id */
            instrument_id: string;
            /** Known At */
            known_at?: string | null;
            /** Listed At */
            listed_at?: string | null;
            /** Proxy Ticker */
            proxy_ticker?: string | null;
            /** Proxy Unit */
            proxy_unit?: string | null;
            /** Supported */
            supported: boolean;
            /** Venue */
            venue: string;
        };
        JsonValue: unknown;
        /** LabConfig */
        LabConfig: {
            /**
             * Aggregation
             * @default direction_equal
             * @enum {string}
             */
            aggregation: "direction_equal" | "direction_score_weighted" | "conviction_trimmed";
            /**
             * Asset Cap
             * @default 0.5
             */
            asset_cap: number;
            /** Asset Weights */
            asset_weights?: {
                [key: string]: number;
            };
            /**
             * Benchmark
             * @default btc_perp_buy_hold
             * @constant
             */
            benchmark: "btc_perp_buy_hold";
            /** Coins */
            coins?: string[];
            /**
             * Deadband
             * @default 0.02
             */
            deadband: number;
            /**
             * End
             * @default 2026-01-08
             */
            end: string;
            /**
             * Fee Bps
             * @default 4.5
             */
            fee_bps: number;
            /**
             * Gross Cap
             * @default 1
             */
            gross_cap: number;
            /**
             * Initial Equity
             * @default 10000
             */
            initial_equity: number;
            /**
             * Latency Seconds
             * @default 5
             */
            latency_seconds: number;
            /**
             * Lookback Days
             * @default 90
             */
            lookback_days: number;
            /**
             * Max Cohort
             * @default 25
             */
            max_cohort: number;
            /** Metric Directions */
            metric_directions?: {
                [key: string]: "asc" | "desc";
            };
            /** Metric Weights */
            metric_weights?: {
                [key: string]: number;
            };
            /**
             * Min Active Days
             * @default 30
             */
            min_active_days: number;
            /**
             * Min Cohort
             * @default 5
             */
            min_cohort: number;
            /**
             * Min Episodes
             * @default 20
             */
            min_episodes: number;
            /**
             * Min Known
             * @default 5
             */
            min_known: number;
            /**
             * Min Known Weight
             * @default 0.6
             */
            min_known_weight: number;
            /**
             * Min Minutes
             * @default 15
             */
            min_minutes: number;
            /**
             * Min Notional
             * @default 100000
             */
            min_notional: number;
            /**
             * Min Trade Usd
             * @default 10
             */
            min_trade_usd: number;
            /**
             * Min Volume
             * @default 0
             */
            min_volume: number;
            /**
             * Reselection
             * @default daily
             * @enum {string}
             */
            reselection: "daily" | "weekly" | "monthly";
            /**
             * Scale Lookback Days
             * @default 30
             */
            scale_lookback_days: number;
            /**
             * Scale Quantile
             * @default 0.95
             */
            scale_quantile: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            schema_version: "hyperliquid_copy_lab_v1";
            /**
             * Scope
             * @default per_asset
             * @enum {string}
             */
            scope: "per_asset" | "pooled";
            /**
             * Selection
             * @default fraction
             * @enum {string}
             */
            selection: "fraction" | "n";
            /**
             * Split
             * @default development
             * @constant
             */
            split: "development";
            /**
             * Start
             * @default 2026-01-05
             */
            start: string;
            /**
             * Top Fraction
             * @default 0.05
             */
            top_fraction: number | null;
            /** Top N */
            top_n?: number | null;
            /**
             * Trim
             * @default 0.1
             */
            trim: number;
            /**
             * Update Minutes
             * @default 1
             */
            update_minutes: number;
        };
        /** LabConfigProxy */
        LabConfigProxy: {
            /**
             * Benchmark
             * @default btc_perp_buy_hold
             * @constant
             */
            benchmark: "btc_perp_buy_hold";
            /**
             * End
             * @default 2026-08-08
             */
            end: string;
            follower?: components["schemas"]["FollowerSettings"];
            /** Market Universe */
            market_universe?: components["schemas"]["ExplicitUniverse"] | components["schemas"]["LiquidityUniverse"];
            proxy?: components["schemas"]["ProxySettings"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            schema_version: "hyperliquid_copy_lab_proxy_v1";
            /**
             * Split
             * @default development
             * @constant
             */
            split: "development";
            /**
             * Start
             * @default 2026-08-03
             */
            start: string;
            trader?: components["schemas"]["TraderSettings"];
        };
        /** LabConfigProxyScheduled */
        LabConfigProxyScheduled: {
            /**
             * Benchmark
             * @default btc_perp_buy_hold
             * @constant
             */
            benchmark: "btc_perp_buy_hold";
            /**
             * End
             * @default 2026-08-08
             */
            end: string;
            follower?: components["schemas"]["FollowerSettings"];
            /** Market Universe */
            market_universe?: components["schemas"]["ExplicitUniverse"] | components["schemas"]["LiquidityUniverse"];
            proxy?: components["schemas"]["ProxySettings"];
            /**
             * Rebalance
             * @default weekly
             * @enum {string}
             */
            rebalance: "daily" | "weekly";
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            schema_version: "hyperliquid_copy_lab_proxy_v2";
            /**
             * Split
             * @default development
             * @constant
             */
            split: "development";
            /**
             * Start
             * @default 2026-08-03
             */
            start: string;
            trader?: components["schemas"]["TraderSettings"];
        };
        /** LabConfigV2 */
        LabConfigV2: {
            /**
             * Benchmark
             * @default btc_perp_buy_hold
             * @constant
             */
            benchmark: "btc_perp_buy_hold";
            /**
             * End
             * @default 2026-01-08
             */
            end: string;
            follower?: components["schemas"]["FollowerSettings"];
            /** Market Universe */
            market_universe?: components["schemas"]["ExplicitUniverse"] | components["schemas"]["LiquidityUniverse"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            schema_version: "hyperliquid_copy_lab_v2";
            /**
             * Split
             * @default development
             * @constant
             */
            split: "development";
            /**
             * Start
             * @default 2026-01-05
             */
            start: string;
            trader?: components["schemas"]["TraderSettings"];
        };
        /** LiquidityUniverse */
        LiquidityUniverse: {
            /** Classes */
            classes?: ("crypto" | "commodity" | "equity" | "index")[];
            /**
             * General
             * @default true
             */
            general: boolean;
            /**
             * Lookback Days
             * @default 30
             */
            lookback_days: number;
            /**
             * Metric
             * @default traded_notional_usd
             * @constant
             */
            metric: "traded_notional_usd";
            /**
             * Min Volume Usd
             * @default 0
             */
            min_volume_usd: number;
            /**
             * Mode
             * @default liquidity
             * @constant
             */
            mode: "liquidity";
            /**
             * Publication Lag Days
             * @default 1
             * @constant
             */
            publication_lag_days: 1;
            /**
             * Reselection
             * @default daily
             * @enum {string}
             */
            reselection: "daily" | "weekly" | "monthly";
            /**
             * Top N
             * @default 3
             */
            top_n: number;
        };
        /** MarketRow */
        MarketRow: {
            /** Asset Class */
            asset_class?: string | null;
            /** Availability Basis */
            availability_basis?: string | null;
            /** Budget */
            budget?: number | null;
            /** Candidate Count */
            candidate_count?: number | null;
            /**
             * Decision Time
             * Format: date-time
             */
            decision_time: string;
            /** Display Name */
            display_name?: string | null;
            /**
             * Effective At
             * Format: date-time
             */
            effective_at: string;
            /** Eligible */
            eligible?: boolean | null;
            /** Eligible Count */
            eligible_count?: number | null;
            /** Entries */
            entries?: string[] | null;
            /** Exits */
            exits?: string[] | null;
            /** Instrument Id */
            instrument_id?: string | null;
            /** Members */
            members?: string[] | null;
            /** Membership Turnover */
            membership_turnover?: number | null;
            /** Proxy Ticker */
            proxy_ticker?: string | null;
            /** Rank */
            rank?: number | null;
            /** Reasons */
            reasons?: string[] | null;
            /** Requested Count */
            requested_count?: number | null;
            /** Retention */
            retention?: number | null;
            /** Selected */
            selected?: boolean | null;
            /** Selected Count */
            selected_count?: number | null;
            /** Venue */
            venue?: string | null;
            /** Volume Usd */
            volume_usd?: number | null;
            /** Window End */
            window_end?: string | null;
            /** Window Start */
            window_start?: string | null;
        };
        /** Metric */
        Metric: {
            /** Description */
            description: string;
            /** End */
            end: string | null;
            /** Group */
            group: string;
            /** Reason */
            reason?: string | null;
            /** Samples */
            samples: number;
            /**
             * Source
             * @default stored summary
             * @enum {string}
             */
            source: "stored summary" | "Python-derived";
            /** Start */
            start: string | null;
            /**
             * Unit
             * @enum {string}
             */
            unit: "usd" | "percent" | "ratio" | "minutes" | "count";
            /** Value */
            value: number | null;
        };
        /** Page[EquityRow] */
        Page_EquityRow_: {
            /**
             * Available
             * @default true
             */
            available: boolean;
            /**
             * Downsampled
             * @default false
             */
            downsampled: boolean;
            /** Reason */
            reason?: string | null;
            /** Rows */
            rows: components["schemas"]["EquityRow"][];
            /** Total */
            total: number;
        };
        /** Page[InstrumentRow] */
        Page_InstrumentRow_: {
            /**
             * Available
             * @default true
             */
            available: boolean;
            /**
             * Downsampled
             * @default false
             */
            downsampled: boolean;
            /** Reason */
            reason?: string | null;
            /** Rows */
            rows: components["schemas"]["InstrumentRow"][];
            /** Total */
            total: number;
        };
        /** Page[MarketRow] */
        Page_MarketRow_: {
            /**
             * Available
             * @default true
             */
            available: boolean;
            /**
             * Downsampled
             * @default false
             */
            downsampled: boolean;
            /** Reason */
            reason?: string | null;
            /** Rows */
            rows: components["schemas"]["MarketRow"][];
            /** Total */
            total: number;
        };
        /** Page[Record] */
        Page_Record_: {
            /**
             * Available
             * @default true
             */
            available: boolean;
            /**
             * Downsampled
             * @default false
             */
            downsampled: boolean;
            /** Reason */
            reason?: string | null;
            /** Rows */
            rows: components["schemas"]["Record"][];
            /** Total */
            total: number;
        };
        /** Page[UniverseRow] */
        Page_UniverseRow_: {
            /**
             * Available
             * @default true
             */
            available: boolean;
            /**
             * Downsampled
             * @default false
             */
            downsampled: boolean;
            /** Reason */
            reason?: string | null;
            /** Rows */
            rows: components["schemas"]["UniverseRow"][];
            /** Total */
            total: number;
        };
        /** Position */
        Position: {
            /** Coin */
            coin: string;
            /** Entry */
            entry: number;
            /** Qty */
            qty: number;
        };
        /** Preflight */
        Preflight: {
            /** Config Hash */
            config_hash: string;
            /** Estimate Notes */
            estimate_notes?: string[];
            /** Estimates */
            estimates: {
                [key: string]: number;
            };
            /** Issues */
            issues: components["schemas"]["PublicValidationIssue"][];
            /** Ready */
            ready: boolean;
            /** Required End */
            required_end: string;
            /** Required Start */
            required_start: string;
        };
        /** Preview */
        Preview: {
            /** Config */
            config: components["schemas"]["LabConfig"] | components["schemas"]["LabConfigV2"] | components["schemas"]["LabConfigProxy"] | components["schemas"]["LabConfigProxyScheduled"];
            /** Dataset Id */
            dataset_id: string;
            /** Decision Date */
            decision_date: string;
            /**
             * Name
             * @default Untitled hypothesis
             */
            name: string;
            /** Parent Id */
            parent_id?: string | null;
            /** Scope */
            scope?: string | null;
        };
        /** PreviewInfo */
        PreviewInfo: {
            /** Hypothetical */
            hypothetical?: boolean | null;
        };
        /** ProxySettings */
        ProxySettings: {
            /**
             * Max Mark Age Seconds
             * @default 345600
             */
            max_mark_age_seconds: number;
            /**
             * Max Wait Seconds
             * @default 345600
             */
            max_wait_seconds: number;
            /**
             * Slippage Bps
             * @default 5
             */
            slippage_bps: number;
        };
        /** PublicValidationIssue */
        PublicValidationIssue: {
            /** Available */
            available?: string | null;
            /** Code */
            code: string;
            /** Field */
            field: string;
            /** Message */
            message: string;
            /** Required */
            required?: string | null;
        };
        /**
         * Record
         * @description Typed union of available ledger columns, rather than arbitrary JSON rows.
         */
        Record: {
            /** Arrival Mid */
            arrival_mid?: number | null;
            /** Book Time */
            book_time?: string | null;
            /** Cash Delta */
            cash_delta?: number | null;
            /** Coin */
            coin?: string | null;
            /** Cutoff Address */
            cutoff_address?: string | null;
            /** Decision Time */
            decision_time?: string | null;
            /** Exclusions */
            exclusions?: string[];
            /** Fee */
            fee?: number | null;
            /** Filled Qty */
            filled_qty?: number | null;
            /** Mark */
            mark?: number | null;
            /** Members */
            members?: string[];
            /** Metrics */
            metrics?: {
                [key: string]: number;
            } | null;
            /** Qty */
            qty?: number | null;
            /** Rate */
            rate?: number | null;
            /** Reason */
            reason?: string | null;
            /** Requested Notional */
            requested_notional?: number | null;
            /** Requested Qty */
            requested_qty?: number | null;
            /** Score */
            score?: number | null;
            /** Selected */
            selected?: boolean | null;
            /** Signal Mid */
            signal_mid?: number | null;
            /** Signal Time */
            signal_time?: string | null;
            /** Spread Cost */
            spread_cost?: number | null;
            /** Time */
            time?: string | null;
            /** Unfilled Qty */
            unfilled_qty?: number | null;
            /** User */
            user?: string | null;
            /** Vwap */
            vwap?: number | null;
        };
        /** Run */
        Run: {
            /**
             * Available
             * @default true
             */
            available: boolean;
            /** End */
            end?: string | null;
            /** Id */
            id: string;
            /** Initial Equity */
            initial_equity?: number | null;
            /**
             * Mode
             * @default unavailable
             */
            mode: string;
            /**
             * Scenario Count
             * @default 0
             */
            scenario_count: number;
            /** Start */
            start?: string | null;
            /**
             * Synthetic
             * @default false
             */
            synthetic: boolean;
            /** Warnings */
            warnings?: string[];
        };
        /** RunDetail */
        RunDetail: {
            /** Artifacts */
            artifacts: string[];
            /**
             * Available
             * @default true
             */
            available: boolean;
            /** Config */
            config: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** End */
            end?: string | null;
            /** Id */
            id: string;
            /** Initial Equity */
            initial_equity?: number | null;
            /**
             * Mode
             * @default unavailable
             */
            mode: string;
            /** Provenance */
            provenance: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /** Reconciliation */
            reconciliation: {
                [key: string]: components["schemas"]["JsonValue"];
            };
            /**
             * Scenario Count
             * @default 0
             */
            scenario_count: number;
            /** Scenarios */
            scenarios: components["schemas"]["Scenario"][];
            /** Start */
            start?: string | null;
            /**
             * Synthetic
             * @default false
             */
            synthetic: boolean;
            /** Warnings */
            warnings?: string[];
        };
        /** Scenario */
        Scenario: {
            /** Latency Seconds */
            latency_seconds: number | null;
            /** Metrics */
            metrics: {
                [key: string]: number | null;
            };
            /** Name */
            name: string;
            /** Positions */
            positions?: components["schemas"]["Position"][];
            /**
             * Scenario Type
             * @enum {string}
             */
            scenario_type: "strategy" | "control";
            /** Warnings */
            warnings?: string[];
        };
        /** Submission */
        Submission: {
            /** Config */
            config: components["schemas"]["LabConfig"] | components["schemas"]["LabConfigV2"] | components["schemas"]["LabConfigProxy"] | components["schemas"]["LabConfigProxyScheduled"];
            /** Dataset Id */
            dataset_id: string;
            /**
             * Name
             * @default Untitled hypothesis
             */
            name: string;
            /** Parent Id */
            parent_id?: string | null;
        };
        /** TraderSettings */
        TraderSettings: {
            /**
             * Lookback Days
             * @default 90
             */
            lookback_days: number;
            /**
             * Max Cohort
             * @default 25
             */
            max_cohort: number;
            /** Metric Directions */
            metric_directions?: {
                [key: string]: "asc" | "desc";
            };
            /** Metric Weights */
            metric_weights?: {
                [key: string]: number;
            };
            /**
             * Min Active Days
             * @default 30
             */
            min_active_days: number;
            /**
             * Min Cohort
             * @default 5
             */
            min_cohort: number;
            /**
             * Min Episodes
             * @default 20
             */
            min_episodes: number;
            /**
             * Min Minutes
             * @default 15
             */
            min_minutes: number;
            /**
             * Min Notional
             * @default 100000
             */
            min_notional: number;
            /**
             * Min Volume
             * @default 0
             */
            min_volume: number;
            /**
             * Reselection
             * @default daily
             * @enum {string}
             */
            reselection: "daily" | "weekly" | "monthly";
            /**
             * Scope
             * @default per_asset
             * @enum {string}
             */
            scope: "per_asset" | "pooled";
            /**
             * Selection
             * @default fraction
             * @enum {string}
             */
            selection: "fraction" | "n";
            /**
             * Top Fraction
             * @default 0.05
             */
            top_fraction: number | null;
            /** Top N */
            top_n?: number | null;
        };
        /** UniverseRow */
        UniverseRow: {
            /** Aggregate Signal */
            aggregate_signal?: number | null;
            /** Candidate Count */
            candidate_count?: number | null;
            /** Coin */
            coin?: string | null;
            /** Decision Time */
            decision_time?: string | null;
            /** Decision Trigger */
            decision_trigger?: string | null;
            /** Effective Weight */
            effective_weight?: number | null;
            /** Eligible */
            eligible?: boolean | null;
            /** Eligible Count */
            eligible_count?: number | null;
            /** Entries */
            entries?: string[];
            /** Exits */
            exits?: string[];
            /** Known */
            known?: boolean | null;
            /** Market Decision Time */
            market_decision_time?: string | null;
            /** Members */
            members?: string[];
            /** Membership Turnover */
            membership_turnover?: number | null;
            /** Metrics */
            metrics?: {
                [key: string]: number | null;
            } | null;
            /** Nominal Weight */
            nominal_weight?: number | null;
            /** Percentiles */
            percentiles?: {
                [key: string]: number | null;
            } | null;
            /** Portfolio Target */
            portfolio_target?: number | null;
            /** Position Qty */
            position_qty?: number | null;
            /** Rank */
            rank?: number | null;
            /** Reason */
            reason?: string | null;
            /** Reasons */
            reasons?: string[];
            /** Requested Count */
            requested_count?: number | null;
            /** Retention */
            retention?: number | null;
            /** Score */
            score?: number | null;
            /** Selected */
            selected?: boolean | null;
            /** Selected Count */
            selected_count?: number | null;
            /** Signal Input */
            signal_input?: number | null;
            /** Target Contribution */
            target_contribution?: number | null;
            /** Time */
            time?: string | null;
            /** User */
            user?: string | null;
            /** Weight */
            weight?: number | null;
        };
        /** ValidationError */
        ValidationError: {
            /** Context */
            ctx?: Record<string, never>;
            /** Input */
            input?: unknown;
            /** Location */
            loc: (string | number)[];
            /** Message */
            msg: string;
            /** Error Type */
            type: string;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    health_api_health_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Health"];
                };
            };
        };
    };
    bootstrap_api_lab_bootstrap_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Bootstrap"];
                };
            };
        };
    };
    comparison_api_lab_compare_get: {
        parameters: {
            query: {
                ids: string;
                units?: "usd" | "growth";
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Comparison"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    datasets_api_lab_datasets_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Dataset"][];
                };
            };
        };
    };
    instruments_api_lab_datasets__identifier__instruments_get: {
        parameters: {
            query?: {
                page?: number;
                page_size?: number;
                search?: string;
                asset_class?: ("crypto" | "commodity" | "equity" | "index") | null;
            };
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_InstrumentRow_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    experiments_api_lab_experiments_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"][];
                };
            };
        };
    };
    submit_api_lab_experiments_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Submission"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    detail_api_lab_experiments__identifier__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cancel_api_lab_experiments__identifier__cancel_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    clone_api_lab_experiments__identifier__clone_post: {
        parameters: {
            query?: {
                upgrade?: boolean;
            };
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Submission"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    diagnostics_api_lab_experiments__identifier__diagnostics_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Diagnostics"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    market_history_api_lab_experiments__identifier__market_universe_get: {
        parameters: {
            query?: {
                table?: "rankings" | "cohorts";
                page?: number;
                page_size?: number;
                decision_date?: string | null;
                asset_class?: ("crypto" | "commodity" | "equity" | "index") | null;
                instrument_id?: string | null;
            };
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_MarketRow_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    annotate_api_lab_experiments__identifier__metadata_patch: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Annotation"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    preview_metadata_api_lab_experiments__identifier__preview_info_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["PreviewInfo"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    resume_api_lab_experiments__identifier__resume_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    history_api_lab_experiments__identifier__universe_get: {
        parameters: {
            query?: {
                table?: "rankings" | "cohorts" | "contributions";
                page?: number;
                page_size?: number;
                decision_date?: string | null;
                scope?: string | null;
                wallet?: string | null;
                at?: string | null;
                selection?: ("selected" | "eligible" | "excluded") | null;
            };
            header?: never;
            path: {
                identifier: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_UniverseRow_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    preflight_api_lab_preflight_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Submission"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Preflight"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    preview_api_lab_previews_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["Preview"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Experiment"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    runs_api_runs_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Run"][];
                };
            };
        };
    };
    detail_api_runs__run_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RunDetail"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    analytics_api_runs__run_id__analytics_get: {
        parameters: {
            query?: {
                scenario_type?: "strategy" | "control";
                name?: string;
                latency_seconds?: number | null;
                benchmark_type?: "strategy" | "control";
                benchmark_name?: string | null;
                benchmark_latency?: number | null;
            };
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Analytics"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    artifact_api_runs__run_id__artifacts__name__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                run_id: string;
                name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    cohorts_api_runs__run_id__cohorts_get: {
        parameters: {
            query?: {
                page?: number;
                page_size?: number;
                decision_date?: string | null;
                scope?: string | null;
            };
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_Record_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    equity_api_runs__run_id__equity_get: {
        parameters: {
            query?: {
                scenario_type?: "strategy" | "control";
                name?: string;
                latency_seconds?: number | null;
                max_points?: number;
            };
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_EquityRow_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    fills_api_runs__run_id__fills_get: {
        parameters: {
            query?: {
                scenario_type?: "strategy" | "control";
                name?: string;
                latency_seconds?: number | null;
                page?: number;
                page_size?: number;
                coin?: string | null;
                reason?: string | null;
            };
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_Record_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    funding_api_runs__run_id__funding_get: {
        parameters: {
            query?: {
                scenario_type?: "strategy" | "control";
                name?: string;
                latency_seconds?: number | null;
                page?: number;
                page_size?: number;
                coin?: string | null;
            };
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_Record_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    traders_api_runs__run_id__traders_get: {
        parameters: {
            query?: {
                page?: number;
                page_size?: number;
                wallet?: string | null;
                decision_date?: string | null;
                scope?: string | null;
            };
            header?: never;
            path: {
                run_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_Record_"];
                };
            };
            /** @description Validation Error */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}

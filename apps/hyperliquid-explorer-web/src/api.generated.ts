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
        JsonValue: unknown;
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
        /** Position */
        Position: {
            /** Coin */
            coin: string;
            /** Entry */
            entry: number;
            /** Qty */
            qty: number;
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

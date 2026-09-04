import type { StrategySystemCardData } from "../types";

export const strategySystemCardData = {
  "meta": {
    "title": "SOL/ETH Traffic-Light Governor System Card",
    "generatedFrom": [
      "reports/latest_strategy_presets_20260627_035218",
      "reports/btc_eth_directional_best_mechanics_20260628_172432"
    ],
    "topCandidate": "barbell_deep70_rec1.85_dd12_gy_cd12_thr5",
    "window": "2021-01-01 to 2026-06-01"
  },
  "topCandidate": {
    "name": "barbell_deep70_rec1.85_dd12_gy_cd12_thr5",
    "final_portfolio_value_usd": 95988.04193610512,
    "final_sol_equiv": 1158.8559934336006,
    "max_drawdown_pct": 56.87744418391819,
    "post_2024_drawdown_pct": 48.3027306209345,
    "sortino_ratio": 1.960741580362372,
    "min_health_factor": 1.2200396693252995,
    "bars_below_hf_1_5": 71,
    "total_actions": 4268,
    "total_interest_paid": 3694.670141587844,
    "total_liquidations": 0,
    "liquidated": false,
    "avg_target_long_fraction": 0.5279057951200706,
    "max_target_long_fraction": 1.85,
    "avg_target_short_fraction": 0.0,
    "boost_share": 0.0,
    "hedge_gate_active_share": 1.0,
    "final_collateral_SOL": 0.0,
    "final_collateral_ETH": 0.0,
    "final_collateral_USDC": 95988.04193610512,
    "final_debt_USDC_value": 0.0,
    "final_debt_ETH_value": 0.0,
    "config": "{'name': 'barbell_deep70_rec1.85_dd12_gy_cd12_thr5', 'enable_protective_short_hedge': True, 'protective_hedge_floors': {4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0}, 'enable_relative_strength_hedge_gate': False, 'relative_strength_lookback_bars': 24, 'relative_strength_min_underperformance': 0.005, 'enable_realized_vol_governor': True, 'realized_vol_lookback_bars': 336, 'realized_vol_target': 0.018, 'realized_vol_min_long_fraction': 0.7, 'enable_drawdown_governor': True, 'drawdown_exposure_tiers': [{'drawdown': 0.0, 'target_long_fraction': 1.075}, {'drawdown': 0.3, 'target_long_fraction': 1.125}, {'drawdown': 0.42, 'target_long_fraction': 0.85}, {'drawdown': 0.5, 'target_long_fraction': 0.7}], 'enable_recovery_boost': False, 'recovery_boost_target_long_fraction': 1.25, 'recovery_boost_min_drawdown': 0.22, 'recovery_boost_max_worsening': -0.012, 'recovery_boost_min_green': 4, 'rebalance_threshold': 0.01, 'enable_traffic_light_state_machine': True, 'target_short_fraction': 0.0, 'protective_short_symbols': ['SOL', 'ETH'], 'state_machine_recovery_min_drawdown': 0.12, 'state_machine_recovery_max_worsening': -0.02, 'state_machine_recovery_min_green': 3, 'state_machine_targets': {'green': {}, 'yellow': {'target_short_fraction': 0.0}, 'orange': {'target_short_fraction': 0.0}, 'red': {'target_short_fraction': 0.0}, 'recovery': {'target_long_fraction': 1.85, 'target_short_fraction': 0.0}}, 'rebalance_cooldown_bars': 12, 'cooldown_force_rotation': True, 'cooldown_force_state_change': True, 'rebalance_cooldown_states': ['green', 'yellow'], 'rebalance_threshold_by_state': {'green': 0.05, 'yellow': 0.05}}",
    "directional_overlap_count": 0,
    "buy_hold_final_usd": 8283.0,
    "buy_hold_final_sol": 100.0,
    "buy_hold_max_drawdown_pct": 96.79616158489398,
    "strategy_vs_buy_hold_usd": 87705.04193610513,
    "strategy_vs_buy_hold_sol": 1058.8559934336006,
    "recovery_boost_share": 0.0,
    "sharpe_ratio_check": 1.829329785206102,
    "buy_hold_sharpe_ratio": 1.212049498762636,
    "information_ratio_vs_sol": 0.1528035828810902,
    "action_turnover_per_year": 788.7274797090755,
    "estimated_turnover_multiple": 1144.6842606726912,
    "estimated_annualized_turnover_multiple": 211.5379409519724,
    "annualized_return_pct": 228.1758349867475
  },
  "latestStrategies": [
    {
      "name": "barbell_deep70_rec1.85_dd12_gy_cd12_thr5",
      "final_portfolio_value_usd": 95988.04193610512,
      "final_sol_equiv": 1158.8559934336006,
      "max_drawdown_pct": 56.87744418391819,
      "post_2024_drawdown_pct": 48.3027306209345,
      "sortino_ratio": 1.960741580362372,
      "min_health_factor": 1.2200396693252995,
      "bars_below_hf_1_5": 71,
      "total_actions": 4268,
      "total_interest_paid": 3694.670141587844,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.5279057951200706,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "boost_share": 0.0,
      "hedge_gate_active_share": 1.0,
      "final_collateral_SOL": 0.0,
      "final_collateral_ETH": 0.0,
      "final_collateral_USDC": 95988.04193610512,
      "final_debt_USDC_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "config": "{'name': 'barbell_deep70_rec1.85_dd12_gy_cd12_thr5', 'enable_protective_short_hedge': True, 'protective_hedge_floors': {4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0}, 'enable_relative_strength_hedge_gate': False, 'relative_strength_lookback_bars': 24, 'relative_strength_min_underperformance': 0.005, 'enable_realized_vol_governor': True, 'realized_vol_lookback_bars': 336, 'realized_vol_target': 0.018, 'realized_vol_min_long_fraction': 0.7, 'enable_drawdown_governor': True, 'drawdown_exposure_tiers': [{'drawdown': 0.0, 'target_long_fraction': 1.075}, {'drawdown': 0.3, 'target_long_fraction': 1.125}, {'drawdown': 0.42, 'target_long_fraction': 0.85}, {'drawdown': 0.5, 'target_long_fraction': 0.7}], 'enable_recovery_boost': False, 'recovery_boost_target_long_fraction': 1.25, 'recovery_boost_min_drawdown': 0.22, 'recovery_boost_max_worsening': -0.012, 'recovery_boost_min_green': 4, 'rebalance_threshold': 0.01, 'enable_traffic_light_state_machine': True, 'target_short_fraction': 0.0, 'protective_short_symbols': ['SOL', 'ETH'], 'state_machine_recovery_min_drawdown': 0.12, 'state_machine_recovery_max_worsening': -0.02, 'state_machine_recovery_min_green': 3, 'state_machine_targets': {'green': {}, 'yellow': {'target_short_fraction': 0.0}, 'orange': {'target_short_fraction': 0.0}, 'red': {'target_short_fraction': 0.0}, 'recovery': {'target_long_fraction': 1.85, 'target_short_fraction': 0.0}}, 'rebalance_cooldown_bars': 12, 'cooldown_force_rotation': True, 'cooldown_force_state_change': True, 'rebalance_cooldown_states': ['green', 'yellow'], 'rebalance_threshold_by_state': {'green': 0.05, 'yellow': 0.05}}",
      "directional_overlap_count": 0,
      "buy_hold_final_usd": 8283.0,
      "buy_hold_final_sol": 100.0,
      "buy_hold_max_drawdown_pct": 96.79616158489398,
      "strategy_vs_buy_hold_usd": 87705.04193610513,
      "strategy_vs_buy_hold_sol": 1058.8559934336006,
      "recovery_boost_share": 0.0,
      "sharpe_ratio_check": 1.829329785206102,
      "buy_hold_sharpe_ratio": 1.212049498762636,
      "information_ratio_vs_sol": 0.1528035828810902,
      "action_turnover_per_year": 788.7274797090755,
      "estimated_turnover_multiple": 1144.6842606726912,
      "estimated_annualized_turnover_multiple": 211.5379409519724,
      "annualized_return_pct": 228.1758349867475
    },
    {
      "name": "barbell_deep70_rec1.85_dd12",
      "final_portfolio_value_usd": 90737.5247238562,
      "final_sol_equiv": 1095.4669168641335,
      "max_drawdown_pct": 56.94255129025917,
      "post_2024_drawdown_pct": 48.61274794500897,
      "sortino_ratio": 1.9516427243561327,
      "min_health_factor": 1.2200396692841993,
      "bars_below_hf_1_5": 37,
      "total_actions": 5407,
      "total_interest_paid": 3539.9937051249303,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.527028925857075,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "boost_share": 0.0,
      "hedge_gate_active_share": 1.0,
      "final_collateral_SOL": 1.1368683772161605e-13,
      "final_collateral_ETH": -3.552713678800501e-15,
      "final_collateral_USDC": 90737.5247238562,
      "final_debt_USDC_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "config": "{'name': 'barbell_deep70_rec1.85_dd12', 'enable_protective_short_hedge': True, 'protective_hedge_floors': {4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0}, 'enable_relative_strength_hedge_gate': False, 'relative_strength_lookback_bars': 24, 'relative_strength_min_underperformance': 0.005, 'enable_realized_vol_governor': True, 'realized_vol_lookback_bars': 336, 'realized_vol_target': 0.018, 'realized_vol_min_long_fraction': 0.7, 'enable_drawdown_governor': True, 'drawdown_exposure_tiers': [{'drawdown': 0.0, 'target_long_fraction': 1.075}, {'drawdown': 0.3, 'target_long_fraction': 1.125}, {'drawdown': 0.42, 'target_long_fraction': 0.85}, {'drawdown': 0.5, 'target_long_fraction': 0.7}], 'enable_recovery_boost': False, 'recovery_boost_target_long_fraction': 1.25, 'recovery_boost_min_drawdown': 0.22, 'recovery_boost_max_worsening': -0.012, 'recovery_boost_min_green': 4, 'rebalance_threshold': 0.01, 'enable_traffic_light_state_machine': True, 'target_short_fraction': 0.0, 'protective_short_symbols': ['SOL', 'ETH'], 'state_machine_recovery_min_drawdown': 0.12, 'state_machine_recovery_max_worsening': -0.02, 'state_machine_recovery_min_green': 3, 'state_machine_targets': {'green': {}, 'yellow': {'target_short_fraction': 0.0}, 'orange': {'target_short_fraction': 0.0}, 'red': {'target_short_fraction': 0.0}, 'recovery': {'target_long_fraction': 1.85, 'target_short_fraction': 0.0}}}",
      "directional_overlap_count": 0,
      "buy_hold_final_usd": 8283.0,
      "buy_hold_final_sol": 100.0,
      "buy_hold_max_drawdown_pct": 96.79616158489398,
      "strategy_vs_buy_hold_usd": 82454.52472385619,
      "strategy_vs_buy_hold_sol": 995.4669168641336,
      "recovery_boost_share": 0.0,
      "sharpe_ratio_check": 1.8149219007816468,
      "buy_hold_sharpe_ratio": 1.212049498762636,
      "information_ratio_vs_sol": 0.1422357983002174,
      "action_turnover_per_year": 999.214967850743,
      "estimated_turnover_multiple": 1161.5263249647442,
      "estimated_annualized_turnover_multiple": 214.65035869381143,
      "annualized_return_pct": 224.78287943762703
    },
    {
      "name": "soft_mid_rec1.85_dd12",
      "final_portfolio_value_usd": 97805.49121552768,
      "final_sol_equiv": 1180.7979139868123,
      "max_drawdown_pct": 58.07382096306169,
      "post_2024_drawdown_pct": 48.925984503335066,
      "sortino_ratio": 1.969400270884745,
      "min_health_factor": 1.219671378405909,
      "bars_below_hf_1_5": 42,
      "total_actions": 5439,
      "total_interest_paid": 3842.077178393376,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.5306764385290742,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "boost_share": 0.0,
      "hedge_gate_active_share": 1.0,
      "final_collateral_SOL": -1.1368683772161605e-13,
      "final_collateral_ETH": 0.0,
      "final_collateral_USDC": 97805.49121552768,
      "final_debt_USDC_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "config": "{'name': 'soft_mid_rec1.85_dd12', 'enable_protective_short_hedge': True, 'protective_hedge_floors': {4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0}, 'enable_relative_strength_hedge_gate': False, 'relative_strength_lookback_bars': 24, 'relative_strength_min_underperformance': 0.005, 'enable_realized_vol_governor': True, 'realized_vol_lookback_bars': 336, 'realized_vol_target': 0.018, 'realized_vol_min_long_fraction': 0.7, 'enable_drawdown_governor': True, 'drawdown_exposure_tiers': [{'drawdown': 0.0, 'target_long_fraction': 1.075}, {'drawdown': 0.32, 'target_long_fraction': 1.075}, {'drawdown': 0.43, 'target_long_fraction': 0.9}, {'drawdown': 0.52, 'target_long_fraction': 0.75}], 'enable_recovery_boost': False, 'recovery_boost_target_long_fraction': 1.25, 'recovery_boost_min_drawdown': 0.22, 'recovery_boost_max_worsening': -0.012, 'recovery_boost_min_green': 4, 'rebalance_threshold': 0.01, 'enable_traffic_light_state_machine': True, 'target_short_fraction': 0.0, 'protective_short_symbols': ['SOL', 'ETH'], 'state_machine_recovery_min_drawdown': 0.12, 'state_machine_recovery_max_worsening': -0.02, 'state_machine_recovery_min_green': 3, 'state_machine_targets': {'green': {}, 'yellow': {'target_short_fraction': 0.0}, 'orange': {'target_short_fraction': 0.0}, 'red': {'target_short_fraction': 0.0}, 'recovery': {'target_long_fraction': 1.85, 'target_short_fraction': 0.0}}}",
      "directional_overlap_count": 0,
      "buy_hold_final_usd": 8283.0,
      "buy_hold_final_sol": 100.0,
      "buy_hold_max_drawdown_pct": 96.79616158489398,
      "strategy_vs_buy_hold_usd": 89522.49121552767,
      "strategy_vs_buy_hold_sol": 1080.7979139868123,
      "recovery_boost_share": 0.0,
      "sharpe_ratio_check": 1.8274562924087967,
      "buy_hold_sharpe_ratio": 1.212049498762636,
      "information_ratio_vs_sol": 0.161849785988636,
      "action_turnover_per_year": 1005.1285759460312,
      "estimated_turnover_multiple": 1171.4176137021375,
      "estimated_annualized_turnover_multiple": 216.47827135475785,
      "annualized_return_pct": 229.31505764451634
    }
  ],
  "scenarioStrategies": [
    {
      "name": "control_best_SOL_ETH",
      "directional_symbols": "SOL,ETH",
      "final_portfolio_value_usd": 6211260.625553627,
      "final_sol_equiv": 74988.0553610241,
      "final_eth_equiv": 3081.838519406991,
      "final_btc_equiv": 84.06659843748564,
      "total_return_pct": 62012.606255536266,
      "max_drawdown_pct": 56.87744418391837,
      "post_2024_drawdown_pct": 48.30273062093458,
      "sortino_ratio": 1.960912792346719,
      "sharpe_ratio_check": 1.8300151242063565,
      "information_ratio_vs_sol": 0.1528636858301042,
      "information_ratio_vs_eth": 1.2176234116094298,
      "information_ratio_vs_btc": 1.4626413685968995,
      "information_ratio_vs_50_50_btc_eth": 1.365860628486742,
      "min_health_factor": 1.2200396693252986,
      "bars_below_hf_1_5": 71,
      "total_actions": 4270,
      "action_turnover_per_year": 789.097080215031,
      "estimated_turnover_multiple": 1144.6842629183236,
      "estimated_annualized_turnover_multiple": 211.53794136696584,
      "total_interest_paid": 239077.27780919307,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.5279057951200706,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "max_target_short_fraction": 0.0,
      "directional_overlap_count": 0,
      "selected_SOL_share": 0.3749130388953304,
      "selected_ETH_share": 0.1436281226942131,
      "selected_BTC_share": 0.0,
      "selected_cash_share": 0.4814588384104564,
      "strategy_vs_buy_hold_sol_usd": 5674866.376104074,
      "strategy_vs_buy_hold_eth_usd": 6183804.933317191,
      "strategy_vs_buy_hold_btc_usd": 6185778.760150712,
      "strategy_vs_buy_hold_50_50_btc_eth_usd": 6184791.846733952,
      "final_collateral_SOL_value": 0.0,
      "final_collateral_ETH_value": 0.0,
      "final_collateral_BTC_value": 0.0,
      "final_collateral_USDC_value": 6211260.625553627,
      "final_debt_SOL_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "final_debt_BTC_value": 0.0,
      "final_debt_USDC_value": 0.0,
      "annualized_return_pct": 228.18995619494794
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "directional_symbols": "SOL",
      "final_portfolio_value_usd": 3272506.669119119,
      "final_sol_equiv": 39508.71265385874,
      "final_eth_equiv": 1623.718229825308,
      "final_btc_equiv": 44.29189509533896,
      "total_return_pct": 32625.066691191198,
      "max_drawdown_pct": 57.29655864380445,
      "post_2024_drawdown_pct": 41.69566447616847,
      "sortino_ratio": 1.587792075342726,
      "sharpe_ratio_check": 1.688829612696528,
      "information_ratio_vs_sol": 0.0138375679533816,
      "information_ratio_vs_eth": 1.0228524767627196,
      "information_ratio_vs_btc": 1.2730379156716431,
      "information_ratio_vs_50_50_btc_eth": 1.1610496250587818,
      "min_health_factor": 1.5481114004337622,
      "bars_below_hf_1_5": 0,
      "total_actions": 2401,
      "action_turnover_per_year": 443.7054073995994,
      "estimated_turnover_multiple": 587.6047711692338,
      "estimated_annualized_turnover_multiple": 108.5895103630126,
      "total_interest_paid": 12569.943400831926,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.4349181458005873,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "max_target_short_fraction": 0.0,
      "directional_overlap_count": 0,
      "selected_SOL_share": 0.4308422051227996,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "selected_cash_share": 0.5691577948772004,
      "strategy_vs_buy_hold_sol_usd": 2736112.419669566,
      "strategy_vs_buy_hold_eth_usd": 3245050.976882684,
      "strategy_vs_buy_hold_btc_usd": 3247024.803716205,
      "strategy_vs_buy_hold_50_50_btc_eth_usd": 3246037.890299445,
      "final_collateral_SOL_value": 0.0,
      "final_collateral_ETH_value": 0.0,
      "final_collateral_BTC_value": 0.0,
      "final_collateral_USDC_value": 3272506.669119119,
      "final_debt_SOL_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "final_debt_BTC_value": 0.0,
      "final_debt_USDC_value": 0.0,
      "annualized_return_pct": 191.54769487321337
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "directional_symbols": "ETH",
      "final_portfolio_value_usd": 45724.38215113376,
      "final_sol_equiv": 552.0268278538423,
      "final_eth_equiv": 22.68704707216973,
      "final_btc_equiv": 0.6188587961173954,
      "total_return_pct": 357.2438215113376,
      "max_drawdown_pct": 40.275612582301406,
      "post_2024_drawdown_pct": 40.275612582301406,
      "sortino_ratio": 0.6885841863885798,
      "sharpe_ratio_check": 0.8476929473445093,
      "information_ratio_vs_sol": -0.9378971883365887,
      "information_ratio_vs_eth": -0.1586822190950071,
      "information_ratio_vs_btc": 0.0635469293519558,
      "information_ratio_vs_50_50_btc_eth": -0.0530185152217618,
      "min_health_factor": 1.651880780074768,
      "bars_below_hf_1_5": 0,
      "total_actions": 1567,
      "action_turnover_per_year": 289.5819964161484,
      "estimated_turnover_multiple": 532.6330824541964,
      "estimated_annualized_turnover_multiple": 98.43072838185908,
      "total_interest_paid": 271.8059901735376,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.3887643484568422,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "max_target_short_fraction": 0.0,
      "directional_overlap_count": 0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.370970802150311,
      "selected_BTC_share": 0.0,
      "selected_cash_share": 0.629029197849689,
      "strategy_vs_buy_hold_sol_usd": -490669.86729841935,
      "strategy_vs_buy_hold_eth_usd": 18268.689914698545,
      "strategy_vs_buy_hold_btc_usd": 20242.516748219547,
      "strategy_vs_buy_hold_50_50_btc_eth_usd": 19255.60333145905,
      "final_collateral_SOL_value": 0.0,
      "final_collateral_ETH_value": 0.0,
      "final_collateral_BTC_value": 0.0,
      "final_collateral_USDC_value": 45724.38215113376,
      "final_debt_SOL_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "final_debt_BTC_value": 0.0,
      "final_debt_USDC_value": 0.0,
      "annualized_return_pct": 32.448928829676845
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "directional_symbols": "BTC",
      "final_portfolio_value_usd": 18460.892622963707,
      "final_sol_equiv": 222.8768878783497,
      "final_eth_equiv": 9.159733171398656,
      "final_btc_equiv": 0.2498598175944198,
      "total_return_pct": 84.60892622963708,
      "max_drawdown_pct": 46.73474800219964,
      "post_2024_drawdown_pct": 46.73474800219964,
      "sortino_ratio": 0.4068508555896571,
      "sharpe_ratio_check": 0.4968992771753863,
      "information_ratio_vs_sol": -1.1207963495056077,
      "information_ratio_vs_eth": -0.4393824227227772,
      "information_ratio_vs_btc": -0.3515170046760606,
      "information_ratio_vs_50_50_btc_eth": -0.3950763847940552,
      "min_health_factor": 1.5486425339366512,
      "bars_below_hf_1_5": 0,
      "total_actions": 1609,
      "action_turnover_per_year": 297.3436070412143,
      "estimated_turnover_multiple": 559.2530875436856,
      "estimated_annualized_turnover_multiple": 103.35011205666592,
      "total_interest_paid": 162.88414845155933,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.4067813850532307,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "max_target_short_fraction": 0.0,
      "directional_overlap_count": 0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.3927901338673975,
      "selected_cash_share": 0.6072098661326025,
      "strategy_vs_buy_hold_sol_usd": -517933.3568265894,
      "strategy_vs_buy_hold_eth_usd": -8994.799613471507,
      "strategy_vs_buy_hold_btc_usd": -7020.972779950505,
      "strategy_vs_buy_hold_50_50_btc_eth_usd": -8007.886196711006,
      "final_collateral_SOL_value": 0.0,
      "final_collateral_ETH_value": 0.0,
      "final_collateral_BTC_value": 0.0,
      "final_collateral_USDC_value": 18460.892622963707,
      "final_debt_SOL_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "final_debt_BTC_value": 0.0,
      "final_debt_USDC_value": 0.0,
      "annualized_return_pct": 12.015074194205578
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "directional_symbols": "BTC,ETH",
      "final_portfolio_value_usd": 11826.324791755773,
      "final_sol_equiv": 142.7782783020134,
      "final_eth_equiv": 5.867862497397973,
      "final_btc_equiv": 0.1600639479157578,
      "total_return_pct": 18.26324791755771,
      "max_drawdown_pct": 51.91481540450313,
      "post_2024_drawdown_pct": 47.05380413897544,
      "sortino_ratio": 0.2692954984712405,
      "sharpe_ratio_check": 0.2906003722587716,
      "information_ratio_vs_sol": -1.1859299886290402,
      "information_ratio_vs_eth": -0.5436862878674698,
      "information_ratio_vs_btc": -0.4436403842286885,
      "information_ratio_vs_50_50_btc_eth": -0.5024708515687364,
      "min_health_factor": 1.1951254734668637,
      "bars_below_hf_1_5": 64,
      "total_actions": 3471,
      "action_turnover_per_year": 641.4416780858016,
      "estimated_turnover_multiple": 1181.4724232131896,
      "estimated_annualized_turnover_multiple": 218.33640269604345,
      "total_interest_paid": 1036.2066413810244,
      "total_liquidations": 0,
      "liquidated": false,
      "avg_target_long_fraction": 0.4892513306429917,
      "max_target_long_fraction": 1.85,
      "avg_target_short_fraction": 0.0,
      "max_target_short_fraction": 0.0,
      "directional_overlap_count": 0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1311057236217982,
      "selected_BTC_share": 0.3447454411299673,
      "selected_cash_share": 0.5241488352482344,
      "strategy_vs_buy_hold_sol_usd": -524567.9246577973,
      "strategy_vs_buy_hold_eth_usd": -15629.367444679445,
      "strategy_vs_buy_hold_btc_usd": -13655.540611158442,
      "strategy_vs_buy_hold_50_50_btc_eth_usd": -14642.454027918942,
      "final_collateral_SOL_value": 0.0,
      "final_collateral_ETH_value": -8.950351571002102e-13,
      "final_collateral_BTC_value": -1.0253603521803713e-12,
      "final_collateral_USDC_value": 11826.324791755773,
      "final_debt_SOL_value": 0.0,
      "final_debt_ETH_value": 0.0,
      "final_debt_BTC_value": 0.0,
      "final_debt_USDC_value": 0.0,
      "annualized_return_pct": 3.1680826895659653
    }
  ],
  "benchmarks": [
    {
      "name": "buy_hold_SOL",
      "final_portfolio_value_usd": 536394.2494495531,
      "total_return_pct": 5263.942494495531,
      "max_drawdown_pct": 96.79616158489398,
      "post_2024_drawdown_pct": 73.62003912800446,
      "sharpe_ratio": 1.212049498762636,
      "annualized_return_pct": 108.6995601794551
    },
    {
      "name": "buy_hold_ETH",
      "final_portfolio_value_usd": 27455.69223643521,
      "total_return_pct": 174.55692236435212,
      "max_drawdown_pct": 81.3430141271089,
      "post_2024_drawdown_pct": 65.2821619611125,
      "sharpe_ratio": 0.6284775187726214,
      "annualized_return_pct": 20.5139077881352
    },
    {
      "name": "buy_hold_BTC",
      "final_portfolio_value_usd": 25481.865402914213,
      "total_return_pct": 154.81865402914212,
      "max_drawdown_pct": 77.19848663243955,
      "post_2024_drawdown_pct": 50.08379415223316,
      "sharpe_ratio": 0.587916091107796,
      "annualized_return_pct": 18.86419769022951
    },
    {
      "name": "buy_hold_50_50_BTC_ETH",
      "final_portfolio_value_usd": 26468.778819674717,
      "total_return_pct": 164.68778819674714,
      "max_drawdown_pct": 79.40824755262553,
      "post_2024_drawdown_pct": 56.91320327868495,
      "sharpe_ratio": 0.6056047324763361,
      "annualized_return_pct": 19.701592581158422
    }
  ],
  "regimes": [
    {
      "name": "control_best_SOL_ETH",
      "regime": "full_2021_2026",
      "return_pct": 62079.516145282185,
      "drawdown_pct": 56.87744418391837,
      "avg_target_long_fraction": 0.5279057951200706,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.3749130388953304,
      "selected_ETH_share": 0.1436281226942131,
      "selected_BTC_share": 0.0,
      "actions": 4270,
      "min_health_factor": 1.2200396693252986
    },
    {
      "name": "control_best_SOL_ETH",
      "regime": "bull_2021",
      "return_pct": 8219.881375544448,
      "drawdown_pct": 42.62513438304102,
      "avg_target_long_fraction": 0.6859369817530431,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.5587058420029725,
      "selected_ETH_share": 0.1247284783354292,
      "selected_BTC_share": 0.0,
      "actions": 1362,
      "min_health_factor": 1.2200396693252986
    },
    {
      "name": "control_best_SOL_ETH",
      "regime": "crash_2022",
      "return_pct": -37.446428280229625,
      "drawdown_pct": 56.87744418391837,
      "avg_target_long_fraction": 0.3151307776400324,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.23162100456621,
      "selected_ETH_share": 0.1131278538812785,
      "selected_BTC_share": 0.0,
      "actions": 571,
      "min_health_factor": 1.44641602316492
    },
    {
      "name": "control_best_SOL_ETH",
      "regime": "recovery_2023",
      "return_pct": 474.2022345681754,
      "drawdown_pct": 40.42189727346842,
      "avg_target_long_fraction": 0.6656924306427675,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.4533622559652928,
      "selected_ETH_share": 0.1808425619362941,
      "selected_BTC_share": 0.0,
      "actions": 829,
      "min_health_factor": 1.4258588335572695
    },
    {
      "name": "control_best_SOL_ETH",
      "regime": "post_2024",
      "return_pct": 107.58809352024072,
      "drawdown_pct": 48.30273062093458,
      "avg_target_long_fraction": 0.4936451887193537,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.3258066039964098,
      "selected_ETH_share": 0.1486607775520808,
      "selected_BTC_share": 0.0,
      "actions": 1508,
      "min_health_factor": 1.4212108231569136
    },
    {
      "name": "control_best_SOL_ETH",
      "regime": "ytd_2026",
      "return_pct": -30.909472079344447,
      "drawdown_pct": 38.69919227058388,
      "avg_target_long_fraction": 0.3926965517241379,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.2686896551724138,
      "selected_ETH_share": 0.1332413793103448,
      "selected_BTC_share": 0.0,
      "actions": 271,
      "min_health_factor": 1.471898765934366
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "regime": "full_2021_2026",
      "return_pct": 18.39064525855165,
      "drawdown_pct": 51.91481540450313,
      "avg_target_long_fraction": 0.4892513306429917,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1311057236217982,
      "selected_BTC_share": 0.3447454411299673,
      "actions": 3471,
      "min_health_factor": 1.1951254734668637
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "regime": "bull_2021",
      "return_pct": 107.2309757571507,
      "drawdown_pct": 40.60607037586595,
      "avg_target_long_fraction": 0.548978258035903,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1839487824396936,
      "selected_BTC_share": 0.3376014633588658,
      "actions": 687,
      "min_health_factor": 1.46277483386223
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "regime": "crash_2022",
      "return_pct": -35.440291993306985,
      "drawdown_pct": 43.08676638003004,
      "avg_target_long_fraction": 0.3308246627865606,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.08162100456621,
      "selected_BTC_share": 0.2401826484018265,
      "actions": 508,
      "min_health_factor": 1.4811427467391256
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "regime": "recovery_2023",
      "return_pct": 17.40524905437657,
      "drawdown_pct": 33.27221629115179,
      "avg_target_long_fraction": 0.6007877611599497,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1322068729306998,
      "selected_BTC_share": 0.4408037447197168,
      "actions": 716,
      "min_health_factor": 1.492553945794816
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "regime": "post_2024",
      "return_pct": -24.97664605380301,
      "drawdown_pct": 47.05380413897544,
      "avg_target_long_fraction": 0.4839812934007275,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1292928338608342,
      "selected_BTC_share": 0.3512211252302896,
      "actions": 1560,
      "min_health_factor": 1.1951254734668637
    },
    {
      "name": "best_mechanics_BTC_ETH_directional",
      "regime": "ytd_2026",
      "return_pct": -23.01281503677197,
      "drawdown_pct": 26.043291218762405,
      "avg_target_long_fraction": 0.3733172413793103,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0902068965517241,
      "selected_BTC_share": 0.3470344827586207,
      "actions": 244,
      "min_health_factor": 1.496214507184311
    },
    {
      "name": "best_mechanics_BTC_ETH_no_short",
      "regime": "full_2021_2026",
      "return_pct": 18.39064525855165,
      "drawdown_pct": 51.91481540450313,
      "avg_target_long_fraction": 0.4892513306429917,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1311057236217982,
      "selected_BTC_share": 0.3447454411299673,
      "actions": 3471,
      "min_health_factor": 1.1951254734668637
    },
    {
      "name": "best_mechanics_BTC_ETH_no_short",
      "regime": "bull_2021",
      "return_pct": 107.2309757571507,
      "drawdown_pct": 40.60607037586595,
      "avg_target_long_fraction": 0.548978258035903,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1839487824396936,
      "selected_BTC_share": 0.3376014633588658,
      "actions": 687,
      "min_health_factor": 1.46277483386223
    },
    {
      "name": "best_mechanics_BTC_ETH_no_short",
      "regime": "crash_2022",
      "return_pct": -35.440291993306985,
      "drawdown_pct": 43.08676638003004,
      "avg_target_long_fraction": 0.3308246627865606,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.08162100456621,
      "selected_BTC_share": 0.2401826484018265,
      "actions": 508,
      "min_health_factor": 1.4811427467391256
    },
    {
      "name": "best_mechanics_BTC_ETH_no_short",
      "regime": "recovery_2023",
      "return_pct": 17.40524905437657,
      "drawdown_pct": 33.27221629115179,
      "avg_target_long_fraction": 0.6007877611599497,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1322068729306998,
      "selected_BTC_share": 0.4408037447197168,
      "actions": 716,
      "min_health_factor": 1.492553945794816
    },
    {
      "name": "best_mechanics_BTC_ETH_no_short",
      "regime": "post_2024",
      "return_pct": -24.97664605380301,
      "drawdown_pct": 47.05380413897544,
      "avg_target_long_fraction": 0.4839812934007275,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.1292928338608342,
      "selected_BTC_share": 0.3512211252302896,
      "actions": 1560,
      "min_health_factor": 1.1951254734668637
    },
    {
      "name": "best_mechanics_BTC_ETH_no_short",
      "regime": "ytd_2026",
      "return_pct": -23.01281503677197,
      "drawdown_pct": 26.043291218762405,
      "avg_target_long_fraction": 0.3733172413793103,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0902068965517241,
      "selected_BTC_share": 0.3470344827586207,
      "actions": 244,
      "min_health_factor": 1.496214507184311
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "regime": "full_2021_2026",
      "return_pct": 84.80779347486778,
      "drawdown_pct": 46.73474800219964,
      "avg_target_long_fraction": 0.4067813850532307,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.3927901338673975,
      "actions": 1609,
      "min_health_factor": 1.5486425339366512
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "regime": "bull_2021",
      "return_pct": 129.0441687758618,
      "drawdown_pct": 29.68416041684264,
      "avg_target_long_fraction": 0.4091059791928661,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.3885903738424602,
      "actions": 235,
      "min_health_factor": 1.5486425339366516
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "regime": "crash_2022",
      "return_pct": -35.75446123902464,
      "drawdown_pct": 43.33122851492336,
      "avg_target_long_fraction": 0.2776369863013699,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.2699771689497717,
      "actions": 252,
      "min_health_factor": 1.5486425339366512
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "regime": "recovery_2023",
      "return_pct": 60.92296329392439,
      "drawdown_pct": 27.8858364598224,
      "avg_target_long_fraction": 0.5349811622331316,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.5106747345587396,
      "actions": 348,
      "min_health_factor": 1.5486425339366514
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "regime": "post_2024",
      "return_pct": -22.308043797624688,
      "drawdown_pct": 46.73474800219964,
      "avg_target_long_fraction": 0.4062178185081959,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.3965704568000378,
      "actions": 774,
      "min_health_factor": 1.5486425339366516
    },
    {
      "name": "best_mechanics_BTC_only_directional",
      "regime": "ytd_2026",
      "return_pct": -15.430844599013538,
      "drawdown_pct": 19.851159686741365,
      "avg_target_long_fraction": 0.3509862068965517,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.383448275862069,
      "actions": 113,
      "min_health_factor": 9.90904884280766
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "regime": "full_2021_2026",
      "return_pct": 357.73638067973667,
      "drawdown_pct": 40.275612582301406,
      "avg_target_long_fraction": 0.3887643484568422,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.370970802150311,
      "selected_BTC_share": 0.0,
      "actions": 1567,
      "min_health_factor": 1.651880780074768
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "regime": "bull_2021",
      "return_pct": 199.72021218188195,
      "drawdown_pct": 39.96629722750317,
      "avg_target_long_fraction": 0.4764333855081793,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.4546701726306162,
      "selected_BTC_share": 0.0,
      "actions": 376,
      "min_health_factor": 1.651880780074768
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "regime": "crash_2022",
      "return_pct": 0.3655346631191714,
      "drawdown_pct": 21.77234233701768,
      "avg_target_long_fraction": 0.2682019458915834,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.2567351598173516,
      "selected_BTC_share": 0.0,
      "actions": 218,
      "min_health_factor": 1.651885369532428
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "regime": "recovery_2023",
      "return_pct": 3.9890663940039817,
      "drawdown_pct": 31.75637354530269,
      "avg_target_long_fraction": 0.4943486699394908,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.4710583399931499,
      "selected_BTC_share": 0.0,
      "actions": 368,
      "min_health_factor": 1.6518853695324285
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "regime": "post_2024",
      "return_pct": 45.4360436844435,
      "drawdown_pct": 40.275612582301406,
      "avg_target_long_fraction": 0.3587427370211158,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.3422457366904435,
      "selected_BTC_share": 0.0,
      "actions": 605,
      "min_health_factor": 1.651885369532428
    },
    {
      "name": "best_mechanics_ETH_only_directional",
      "regime": "ytd_2026",
      "return_pct": -23.90462131833018,
      "drawdown_pct": 28.33475723338918,
      "avg_target_long_fraction": 0.3373931034482758,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.0,
      "selected_ETH_share": 0.3224827586206896,
      "selected_BTC_share": 0.0,
      "actions": 116,
      "min_health_factor": 1.651885369532428
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "regime": "full_2021_2026",
      "return_pct": 32660.31928702057,
      "drawdown_pct": 57.29655864380445,
      "avg_target_long_fraction": 0.4349181458005873,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.4308422051227996,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "actions": 2401,
      "min_health_factor": 1.5481114004337622
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "regime": "bull_2021",
      "return_pct": 5166.574115897348,
      "drawdown_pct": 57.29655864380445,
      "avg_target_long_fraction": 0.6145340578376716,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.6366754315765405,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "actions": 847,
      "min_health_factor": 1.5481114004337622
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "regime": "crash_2022",
      "return_pct": -28.767267409140405,
      "drawdown_pct": 56.11028368170328,
      "avg_target_long_fraction": 0.2561401646283958,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.267351598173516,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "actions": 280,
      "min_health_factor": 1.5481552829718803
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "regime": "recovery_2023",
      "return_pct": 450.9697960213596,
      "drawdown_pct": 52.590917420319,
      "avg_target_long_fraction": 0.5371446512158923,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.5259732846215321,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "actions": 537,
      "min_health_factor": 1.5481550065215954
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "regime": "post_2024",
      "return_pct": 58.12081848152661,
      "drawdown_pct": 41.69566447616847,
      "avg_target_long_fraction": 0.3923839104350701,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.374084746563371,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "actions": 737,
      "min_health_factor": 1.5486425339366512
    },
    {
      "name": "best_mechanics_SOL_only_directional",
      "regime": "ytd_2026",
      "return_pct": -23.49733753615861,
      "drawdown_pct": 33.359712385458145,
      "avg_target_long_fraction": 0.3375999999999999,
      "avg_target_short_fraction": 0.0,
      "selected_SOL_share": 0.3230344827586207,
      "selected_ETH_share": 0.0,
      "selected_BTC_share": 0.0,
      "actions": 116,
      "min_health_factor": 1.5487696835372091
    }
  ],
  "charts": {
    "topEquity": [
      {
        "timestamp": "2021-01-01T00:00:00+00:00",
        "portfolio_value": 154.4084069069069,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.198717948717944,
        "drawdown_pct": 0.0,
        "normalized_value": 100.0
      },
      {
        "timestamp": "2021-01-04T19:00:00+00:00",
        "portfolio_value": 251.47230065760343,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 12.582676401148548,
        "drawdown_pct": 0.0,
        "normalized_value": 162.86179340559906
      },
      {
        "timestamp": "2021-01-08T14:00:00+00:00",
        "portfolio_value": 395.337450135008,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 5.090152214626071,
        "drawdown_pct": -6.542725525168246,
        "normalized_value": 256.03363058680975
      },
      {
        "timestamp": "2021-01-12T09:00:00+00:00",
        "portfolio_value": 389.04388114411086,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.7933059321465,
        "drawdown_pct": -12.549343440794278,
        "normalized_value": 251.95770679679774
      },
      {
        "timestamp": "2021-01-16T04:00:00+00:00",
        "portfolio_value": 402.7545482120225,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.425020249864009,
        "normalized_value": 260.8371890365036
      },
      {
        "timestamp": "2021-01-19T23:00:00+00:00",
        "portfolio_value": 416.1453730598931,
        "target_long_fraction": 1.057608310045551,
        "target_short_fraction": 0.0,
        "health_factor": 1.809521085030944,
        "drawdown_pct": -9.424056854014212,
        "normalized_value": 269.50953085785534
      },
      {
        "timestamp": "2021-01-23T18:00:00+00:00",
        "portfolio_value": 375.1480932881149,
        "target_long_fraction": 0.8043114602276694,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -18.34730224406739,
        "normalized_value": 242.958334201513
      },
      {
        "timestamp": "2021-01-27T13:00:00+00:00",
        "portfolio_value": 392.20090582744433,
        "target_long_fraction": 0.9507749469447978,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.635679626028205,
        "normalized_value": 254.0023005767445
      },
      {
        "timestamp": "2021-01-31T08:00:00+00:00",
        "portfolio_value": 506.3036146734226,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 15.248051943317863,
        "drawdown_pct": 0.0,
        "normalized_value": 327.89899514906205
      },
      {
        "timestamp": "2021-02-04T03:00:00+00:00",
        "portfolio_value": 644.1825135390039,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 19.19114412286062,
        "drawdown_pct": -2.3404502505472298,
        "normalized_value": 417.1939381042787
      },
      {
        "timestamp": "2021-02-07T22:00:00+00:00",
        "portfolio_value": 671.8788305040838,
        "target_long_fraction": 1.020745601080005,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -12.270733201741251,
        "normalized_value": 435.1309905743414
      },
      {
        "timestamp": "2021-02-11T18:00:00+00:00",
        "portfolio_value": 912.9931876073558,
        "target_long_fraction": 1.030441645507973,
        "target_short_fraction": 0.0,
        "health_factor": 12.52959187547367,
        "drawdown_pct": -5.156666878503699,
        "normalized_value": 591.2846365663245
      },
      {
        "timestamp": "2021-02-15T13:00:00+00:00",
        "portfolio_value": 848.9615132900573,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5758391793609043,
        "drawdown_pct": -14.719689654138172,
        "normalized_value": 549.815602852439
      },
      {
        "timestamp": "2021-02-19T08:00:00+00:00",
        "portfolio_value": 749.4393180943732,
        "target_long_fraction": 1.0309258052742405,
        "target_short_fraction": 0.0,
        "health_factor": 4.24042153904228,
        "drawdown_pct": -24.71694342798457,
        "normalized_value": 485.361731985365
      },
      {
        "timestamp": "2021-02-23T03:00:00+00:00",
        "portfolio_value": 1115.7736163155214,
        "target_long_fraction": 0.9307102526066408,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -4.489320815007837,
        "normalized_value": 722.6119604926845
      },
      {
        "timestamp": "2021-02-26T22:00:00+00:00",
        "portfolio_value": 1198.480563465121,
        "target_long_fraction": 0.7326304973533242,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -18.8624522030885,
        "normalized_value": 776.1757196210741
      },
      {
        "timestamp": "2021-03-02T17:00:00+00:00",
        "portfolio_value": 1157.3819905046644,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.644839777504046,
        "normalized_value": 749.5589221398108
      },
      {
        "timestamp": "2021-03-06T13:00:00+00:00",
        "portfolio_value": 1039.9064318112078,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.597975647226153,
        "normalized_value": 673.4778582607677
      },
      {
        "timestamp": "2021-03-10T08:00:00+00:00",
        "portfolio_value": 1158.2493570102065,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.5693647245923,
        "drawdown_pct": -21.586118765711092,
        "normalized_value": 750.1206574254194
      },
      {
        "timestamp": "2021-03-14T03:00:00+00:00",
        "portfolio_value": 1184.0671012431087,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5450795372323078,
        "drawdown_pct": -19.83824857026215,
        "normalized_value": 766.8410839553476
      },
      {
        "timestamp": "2021-03-17T22:00:00+00:00",
        "portfolio_value": 1135.574587694783,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.121208474462676,
        "normalized_value": 735.4357255815889
      },
      {
        "timestamp": "2021-03-21T17:00:00+00:00",
        "portfolio_value": 1105.3706280978658,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.166026963938474,
        "normalized_value": 715.8746406627301
      },
      {
        "timestamp": "2021-03-25T12:00:00+00:00",
        "portfolio_value": 1160.959562294763,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.402636932531124,
        "normalized_value": 751.8758761591961
      },
      {
        "timestamp": "2021-03-29T07:00:00+00:00",
        "portfolio_value": 1471.7479898605493,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.31388576107992,
        "drawdown_pct": -6.089955088984652,
        "normalized_value": 953.1527585462809
      },
      {
        "timestamp": "2021-04-02T02:00:00+00:00",
        "portfolio_value": 1638.5952823274522,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5826218163229582,
        "drawdown_pct": -3.1898487001137994,
        "normalized_value": 1061.2085929462146
      },
      {
        "timestamp": "2021-04-05T21:00:00+00:00",
        "portfolio_value": 2133.549825844946,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5864176946060642,
        "drawdown_pct": -3.40956745595297,
        "normalized_value": 1381.757553609932
      },
      {
        "timestamp": "2021-04-09T16:00:00+00:00",
        "portfolio_value": 2532.114854191452,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 7.099166690091142,
        "drawdown_pct": -1.9799162787782727,
        "normalized_value": 1639.8814707790286
      },
      {
        "timestamp": "2021-04-13T11:00:00+00:00",
        "portfolio_value": 2315.742411201696,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.6115590883164563,
        "drawdown_pct": -10.719862935897972,
        "normalized_value": 1499.7515080884564
      },
      {
        "timestamp": "2021-04-17T06:00:00+00:00",
        "portfolio_value": 2255.2033814359197,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 4.813885584878547,
        "drawdown_pct": -13.053858655400951,
        "normalized_value": 1460.5444267005396
      },
      {
        "timestamp": "2021-04-21T03:00:00+00:00",
        "portfolio_value": 2450.112536096636,
        "target_long_fraction": 0.8731652096454842,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -6.822692920614654,
        "normalized_value": 1586.7740527715007
      },
      {
        "timestamp": "2021-04-24T22:00:00+00:00",
        "portfolio_value": 2916.572792519579,
        "target_long_fraction": 0.7728857186560426,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -3.6291867668714626,
        "normalized_value": 1888.8691690718535
      },
      {
        "timestamp": "2021-04-28T20:00:00+00:00",
        "portfolio_value": 3208.4503272924458,
        "target_long_fraction": 0.7449134069463353,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -3.380031576665568,
        "normalized_value": 2077.89873075164
      },
      {
        "timestamp": "2021-05-02T15:00:00+00:00",
        "portfolio_value": 3333.9739213173266,
        "target_long_fraction": 0.7715141038188117,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -3.9910650865662047,
        "normalized_value": 2159.191968949842
      },
      {
        "timestamp": "2021-05-06T10:00:00+00:00",
        "portfolio_value": 3233.055423375595,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.6327304724809948,
        "drawdown_pct": -6.897229840434712,
        "normalized_value": 2093.8338061636828
      },
      {
        "timestamp": "2021-05-10T05:00:00+00:00",
        "portfolio_value": 3874.677174207269,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.7878046163707992,
        "drawdown_pct": 0.0,
        "normalized_value": 2509.369309498361
      },
      {
        "timestamp": "2021-05-14T00:00:00+00:00",
        "portfolio_value": 3369.1594036650263,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -17.274374715768797,
        "normalized_value": 2181.979253044362
      },
      {
        "timestamp": "2021-05-17T19:00:00+00:00",
        "portfolio_value": 3559.505974892811,
        "target_long_fraction": 1.85,
        "target_short_fraction": 0.0,
        "health_factor": 1.5501543487007798,
        "drawdown_pct": -12.6006453848269,
        "normalized_value": 2305.253998921732
      },
      {
        "timestamp": "2021-05-21T14:00:00+00:00",
        "portfolio_value": 3352.9498237715097,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -18.985585760080152,
        "normalized_value": 2171.4813920676024
      },
      {
        "timestamp": "2021-05-25T09:00:00+00:00",
        "portfolio_value": 3352.9498237715097,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -18.985585760080152,
        "normalized_value": 2171.4813920676024
      },
      {
        "timestamp": "2021-05-29T04:00:00+00:00",
        "portfolio_value": 3122.6750395633226,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -24.54951535562233,
        "normalized_value": 2022.3478126071136
      },
      {
        "timestamp": "2021-06-01T23:00:00+00:00",
        "portfolio_value": 2968.769232142717,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -28.268207698675994,
        "normalized_value": 1922.6733126860079
      },
      {
        "timestamp": "2021-06-05T18:00:00+00:00",
        "portfolio_value": 2932.346932068373,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.148248099183544,
        "normalized_value": 1899.0850244548476
      },
      {
        "timestamp": "2021-06-09T13:00:00+00:00",
        "portfolio_value": 2920.8114602780324,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.42696968441274,
        "normalized_value": 1891.61427074304
      },
      {
        "timestamp": "2021-06-13T08:00:00+00:00",
        "portfolio_value": 2920.8114602780324,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.42696968441274,
        "normalized_value": 1891.61427074304
      },
      {
        "timestamp": "2021-06-17T03:00:00+00:00",
        "portfolio_value": 2830.693651138421,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -31.60440803087099,
        "normalized_value": 1833.2509918615062
      },
      {
        "timestamp": "2021-06-20T22:00:00+00:00",
        "portfolio_value": 2733.533012465742,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -33.95201614997775,
        "normalized_value": 1770.3265432391865
      },
      {
        "timestamp": "2021-06-24T17:00:00+00:00",
        "portfolio_value": 2733.533012465742,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -33.95201614997775,
        "normalized_value": 1770.3265432391865
      },
      {
        "timestamp": "2021-06-28T12:00:00+00:00",
        "portfolio_value": 2733.533012465742,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -33.95201614997775,
        "normalized_value": 1770.3265432391865
      },
      {
        "timestamp": "2021-07-02T07:00:00+00:00",
        "portfolio_value": 2495.0862106890822,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 14.694182813062652,
        "drawdown_pct": -39.71339910786206,
        "normalized_value": 1615.900494455185
      },
      {
        "timestamp": "2021-07-06T02:00:00+00:00",
        "portfolio_value": 2464.0921062235007,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -40.46228273277285,
        "normalized_value": 1595.8276855411805
      },
      {
        "timestamp": "2021-07-09T21:00:00+00:00",
        "portfolio_value": 2556.526642590105,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -38.228867318623976,
        "normalized_value": 1655.691353730137
      },
      {
        "timestamp": "2021-07-13T16:00:00+00:00",
        "portfolio_value": 2556.526642590105,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -38.228867318623976,
        "normalized_value": 1655.691353730137
      },
      {
        "timestamp": "2021-07-17T11:00:00+00:00",
        "portfolio_value": 2556.526642590105,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -38.228867318623976,
        "normalized_value": 1655.691353730137
      },
      {
        "timestamp": "2021-07-21T06:00:00+00:00",
        "portfolio_value": 2556.526642590105,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -38.228867318623976,
        "normalized_value": 1655.691353730137
      },
      {
        "timestamp": "2021-07-25T01:00:00+00:00",
        "portfolio_value": 2504.508110843672,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.5401564260086,
        "drawdown_pct": -39.48574591823201,
        "normalized_value": 1622.0024291511827
      },
      {
        "timestamp": "2021-07-28T20:00:00+00:00",
        "portfolio_value": 2582.8245164089917,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -37.59345463571184,
        "normalized_value": 1672.7227280870668
      },
      {
        "timestamp": "2021-08-01T15:00:00+00:00",
        "portfolio_value": 3158.216607770887,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.911426434148638,
        "drawdown_pct": -23.690755314213423,
        "normalized_value": 2045.3657097019218
      },
      {
        "timestamp": "2021-08-05T10:00:00+00:00",
        "portfolio_value": 3184.0768093674087,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.058814641293292,
        "drawdown_pct": -23.06591772505086,
        "normalized_value": 2062.1136330271797
      },
      {
        "timestamp": "2021-08-09T05:00:00+00:00",
        "portfolio_value": 3318.44415933088,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.715348711584528,
        "drawdown_pct": -19.819316158548332,
        "normalized_value": 2149.134380572669
      },
      {
        "timestamp": "2021-08-13T00:00:00+00:00",
        "portfolio_value": 3683.6065579710425,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.48939359111245,
        "drawdown_pct": -10.996214298049733,
        "normalized_value": 2385.625648085273
      },
      {
        "timestamp": "2021-08-16T23:00:00+00:00",
        "portfolio_value": 5629.243525286846,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 14.56711976403012,
        "drawdown_pct": -10.315348117692052,
        "normalized_value": 3645.68460879253
      },
      {
        "timestamp": "2021-08-20T18:00:00+00:00",
        "portfolio_value": 7011.832505180886,
        "target_long_fraction": 1.063348081486983,
        "target_short_fraction": 0.0,
        "health_factor": 11.93948571274264,
        "drawdown_pct": -2.9699925552868196,
        "normalized_value": 4541.095038567643
      },
      {
        "timestamp": "2021-08-24T13:00:00+00:00",
        "portfolio_value": 6689.128623446346,
        "target_long_fraction": 1.0600730681552,
        "target_short_fraction": 0.0,
        "health_factor": 8.868229263957744,
        "drawdown_pct": -9.496528737570177,
        "normalized_value": 4332.101313291338
      },
      {
        "timestamp": "2021-08-28T08:00:00+00:00",
        "portfolio_value": 6593.038666939783,
        "target_long_fraction": 0.9840872138750928,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.79661954264417,
        "normalized_value": 4269.870273912442
      },
      {
        "timestamp": "2021-09-01T03:00:00+00:00",
        "portfolio_value": 8330.636357238038,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5587267696396965,
        "drawdown_pct": -12.215571509927361,
        "normalized_value": 5395.196106297887
      },
      {
        "timestamp": "2021-09-04T22:00:00+00:00",
        "portfolio_value": 10561.768103464066,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 5.338520075926297,
        "drawdown_pct": -6.184549583641719,
        "normalized_value": 6840.150944521936
      },
      {
        "timestamp": "2021-09-08T17:00:00+00:00",
        "portfolio_value": 11802.375001199563,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.780007066683032,
        "normalized_value": 7643.609073899219
      },
      {
        "timestamp": "2021-09-12T12:00:00+00:00",
        "portfolio_value": 10103.693606421017,
        "target_long_fraction": 0.7934454447248385,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -27.04547197898005,
        "normalized_value": 6543.486723823627
      },
      {
        "timestamp": "2021-09-16T07:00:00+00:00",
        "portfolio_value": 9011.549526036768,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -34.931385687284525,
        "normalized_value": 5836.178033667459
      },
      {
        "timestamp": "2021-09-20T02:00:00+00:00",
        "portfolio_value": 8891.029121832933,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -35.801612907952176,
        "normalized_value": 5758.12502695747
      },
      {
        "timestamp": "2021-09-23T21:00:00+00:00",
        "portfolio_value": 8891.029121832933,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -35.801612907952176,
        "normalized_value": 5758.12502695747
      },
      {
        "timestamp": "2021-09-27T16:00:00+00:00",
        "portfolio_value": 8637.17242906745,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -37.63460546761503,
        "normalized_value": 5593.719022225789
      },
      {
        "timestamp": "2021-10-01T13:00:00+00:00",
        "portfolio_value": 8847.47133196246,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 27.815039951612214,
        "drawdown_pct": -36.11612541450915,
        "normalized_value": 5729.915559129249
      },
      {
        "timestamp": "2021-10-05T08:00:00+00:00",
        "portfolio_value": 9947.83265730116,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.147596081371752,
        "drawdown_pct": -28.170878431598638,
        "normalized_value": 6442.5460093625115
      },
      {
        "timestamp": "2021-10-09T03:00:00+00:00",
        "portfolio_value": 8941.255432943828,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 4.9659453423459405,
        "drawdown_pct": -35.4389498102704,
        "normalized_value": 5790.653250074996
      },
      {
        "timestamp": "2021-10-12T22:00:00+00:00",
        "portfolio_value": 8603.960531370125,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -37.87441463202308,
        "normalized_value": 5572.2098969374565
      },
      {
        "timestamp": "2021-10-16T17:00:00+00:00",
        "portfolio_value": 8614.00329770802,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.502364268497712,
        "drawdown_pct": -37.801900034218875,
        "normalized_value": 5578.713925143609
      },
      {
        "timestamp": "2021-10-20T12:00:00+00:00",
        "portfolio_value": 8217.00744216813,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5398417448898418,
        "drawdown_pct": -40.66844037039927,
        "normalized_value": 5321.606256272159
      },
      {
        "timestamp": "2021-10-24T07:00:00+00:00",
        "portfolio_value": 10652.30337193457,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5537225394195102,
        "drawdown_pct": -23.08419127611628,
        "normalized_value": 6898.7845838969515
      },
      {
        "timestamp": "2021-10-28T02:00:00+00:00",
        "portfolio_value": 10367.04737307986,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.14390503746633,
        "normalized_value": 6714.043348255106
      },
      {
        "timestamp": "2021-10-31T21:00:00+00:00",
        "portfolio_value": 10701.845227281468,
        "target_long_fraction": 1.85,
        "target_short_fraction": 0.0,
        "health_factor": 1.4566190993674817,
        "drawdown_pct": -22.726470346037626,
        "normalized_value": 6930.8695307850885
      },
      {
        "timestamp": "2021-11-04T16:00:00+00:00",
        "portfolio_value": 12309.256219852276,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.390354813179082,
        "drawdown_pct": -11.120030674878429,
        "normalized_value": 7971.8821445218
      },
      {
        "timestamp": "2021-11-08T11:00:00+00:00",
        "portfolio_value": 12493.178846628609,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5572860238640474,
        "drawdown_pct": -9.792002633694924,
        "normalized_value": 8090.996531141448
      },
      {
        "timestamp": "2021-11-12T06:00:00+00:00",
        "portfolio_value": 11983.909050603286,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 16.101434803062613,
        "drawdown_pct": -13.469225939514192,
        "normalized_value": 7761.176538676685
      },
      {
        "timestamp": "2021-11-16T01:00:00+00:00",
        "portfolio_value": 11930.101100748569,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.857750546291156,
        "normalized_value": 7726.3287276458
      },
      {
        "timestamp": "2021-11-19T20:00:00+00:00",
        "portfolio_value": 11930.101100748569,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.857750546291156,
        "normalized_value": 7726.3287276458
      },
      {
        "timestamp": "2021-11-23T15:00:00+00:00",
        "portfolio_value": 11930.101100748569,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.857750546291156,
        "normalized_value": 7726.3287276458
      },
      {
        "timestamp": "2021-11-27T10:00:00+00:00",
        "portfolio_value": 11930.101100748569,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.857750546291156,
        "normalized_value": 7726.3287276458
      },
      {
        "timestamp": "2021-12-01T05:00:00+00:00",
        "portfolio_value": 12295.865226192886,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.21672141592905,
        "normalized_value": 7963.209693372515
      },
      {
        "timestamp": "2021-12-05T00:00:00+00:00",
        "portfolio_value": 12370.996887984227,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.674227241131542,
        "normalized_value": 8011.867446726993
      },
      {
        "timestamp": "2021-12-08T19:00:00+00:00",
        "portfolio_value": 12370.996887984227,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.674227241131542,
        "normalized_value": 8011.867446726993
      },
      {
        "timestamp": "2021-12-12T14:00:00+00:00",
        "portfolio_value": 12370.996887984227,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.674227241131542,
        "normalized_value": 8011.867446726993
      },
      {
        "timestamp": "2021-12-16T09:00:00+00:00",
        "portfolio_value": 12538.395236109678,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -9.465514075955047,
        "normalized_value": 8120.280163028363
      },
      {
        "timestamp": "2021-12-20T04:00:00+00:00",
        "portfolio_value": 12608.198295970698,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -8.961495497704869,
        "normalized_value": 8165.486937231471
      },
      {
        "timestamp": "2021-12-23T23:00:00+00:00",
        "portfolio_value": 12518.514572495444,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -9.609063998114149,
        "normalized_value": 8107.404786607816
      },
      {
        "timestamp": "2021-12-27T18:00:00+00:00",
        "portfolio_value": 13335.383386361378,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -3.7107973748263254,
        "normalized_value": 8636.436094053677
      },
      {
        "timestamp": "2021-12-31T13:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-04T08:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-08T03:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-11T22:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-15T17:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-19T12:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-23T07:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-27T02:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-01-30T21:00:00+00:00",
        "portfolio_value": 12843.604644867875,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.261725122044344,
        "normalized_value": 8317.943887997826
      },
      {
        "timestamp": "2022-02-03T16:00:00+00:00",
        "portfolio_value": 12642.538326186426,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -8.713540562192586,
        "normalized_value": 8187.726678514749
      },
      {
        "timestamp": "2022-02-07T11:00:00+00:00",
        "portfolio_value": 14497.427990499207,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -1.401006711409396,
        "normalized_value": 9389.01467925884
      },
      {
        "timestamp": "2022-02-11T06:00:00+00:00",
        "portfolio_value": 14120.372180332335,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6158587957492807,
        "drawdown_pct": -5.947206010304247,
        "normalized_value": 9144.820844402295
      },
      {
        "timestamp": "2022-02-15T01:00:00+00:00",
        "portfolio_value": 13363.584103255342,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 36.83106287593012,
        "drawdown_pct": -10.988010331759831,
        "normalized_value": 8654.699812628902
      },
      {
        "timestamp": "2022-02-18T20:00:00+00:00",
        "portfolio_value": 12485.437084367872,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.837160736954377,
        "normalized_value": 8085.982709409965
      },
      {
        "timestamp": "2022-02-22T15:00:00+00:00",
        "portfolio_value": 12485.437084367872,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.837160736954377,
        "normalized_value": 8085.982709409965
      },
      {
        "timestamp": "2022-02-26T10:00:00+00:00",
        "portfolio_value": 12485.437084367872,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.837160736954377,
        "normalized_value": 8085.982709409965
      },
      {
        "timestamp": "2022-03-02T05:00:00+00:00",
        "portfolio_value": 13064.996683746864,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 28.1314882011397,
        "drawdown_pct": -12.976837587606624,
        "normalized_value": 8461.324707289918
      },
      {
        "timestamp": "2022-03-06T00:00:00+00:00",
        "portfolio_value": 11717.6908805688,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.950954808392627,
        "normalized_value": 7588.764831718921
      },
      {
        "timestamp": "2022-03-09T19:00:00+00:00",
        "portfolio_value": 11717.6908805688,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.950954808392627,
        "normalized_value": 7588.764831718921
      },
      {
        "timestamp": "2022-03-13T14:00:00+00:00",
        "portfolio_value": 11717.6908805688,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.950954808392627,
        "normalized_value": 7588.764831718921
      },
      {
        "timestamp": "2022-03-17T09:00:00+00:00",
        "portfolio_value": 11705.97318968823,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.029003853584246,
        "normalized_value": 7581.176066887201
      },
      {
        "timestamp": "2022-03-21T04:00:00+00:00",
        "portfolio_value": 12066.003896105403,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.630915940024583,
        "normalized_value": 7814.343880498695
      },
      {
        "timestamp": "2022-03-24T23:00:00+00:00",
        "portfolio_value": 13112.306959053014,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.6789467800428688,
        "drawdown_pct": -12.661713912384883,
        "normalized_value": 8491.964409009444
      },
      {
        "timestamp": "2022-03-28T18:00:00+00:00",
        "portfolio_value": 14243.125468616368,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 5.089178149816647,
        "drawdown_pct": -5.129572481450322,
        "normalized_value": 9224.319940820045
      },
      {
        "timestamp": "2022-04-01T13:00:00+00:00",
        "portfolio_value": 16370.849581842733,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 5.7266169864840295,
        "drawdown_pct": -0.06910451358347551,
        "normalized_value": 10602.304569927172
      },
      {
        "timestamp": "2022-04-05T08:00:00+00:00",
        "portfolio_value": 17190.741591742146,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5803779644266596,
        "drawdown_pct": -5.318278262587885,
        "normalized_value": 11133.293799285473
      },
      {
        "timestamp": "2022-04-09T03:00:00+00:00",
        "portfolio_value": 16001.37233829436,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.868986275793759,
        "normalized_value": 10363.018865897384
      },
      {
        "timestamp": "2022-04-12T22:00:00+00:00",
        "portfolio_value": 16001.37233829436,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.868986275793759,
        "normalized_value": 10363.018865897384
      },
      {
        "timestamp": "2022-04-16T17:00:00+00:00",
        "portfolio_value": 16001.37233829436,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.868986275793759,
        "normalized_value": 10363.018865897384
      },
      {
        "timestamp": "2022-04-20T12:00:00+00:00",
        "portfolio_value": 16141.579707178174,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.557676596652397,
        "drawdown_pct": -11.096763913230363,
        "normalized_value": 10453.82180318068
      },
      {
        "timestamp": "2022-04-24T07:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-04-28T02:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-01T21:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-05T16:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-09T11:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-13T06:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-17T01:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-20T20:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-24T15:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-05-28T10:00:00+00:00",
        "portfolio_value": 15087.530360437984,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.90216831727864,
        "normalized_value": 9771.184524644621
      },
      {
        "timestamp": "2022-06-01T05:00:00+00:00",
        "portfolio_value": 14614.230993003988,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.508966794629107,
        "normalized_value": 9464.660173467715
      },
      {
        "timestamp": "2022-06-05T00:00:00+00:00",
        "portfolio_value": 14614.230993003988,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.508966794629107,
        "normalized_value": 9464.660173467715
      },
      {
        "timestamp": "2022-06-08T19:00:00+00:00",
        "portfolio_value": 14614.230993003988,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.508966794629107,
        "normalized_value": 9464.660173467715
      },
      {
        "timestamp": "2022-06-12T14:00:00+00:00",
        "portfolio_value": 14614.230993003988,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.508966794629107,
        "normalized_value": 9464.660173467715
      },
      {
        "timestamp": "2022-06-16T09:00:00+00:00",
        "portfolio_value": 14614.230993003988,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.508966794629107,
        "normalized_value": 9464.660173467715
      },
      {
        "timestamp": "2022-06-20T04:00:00+00:00",
        "portfolio_value": 14614.230993003988,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.508966794629107,
        "normalized_value": 9464.660173467715
      },
      {
        "timestamp": "2022-06-23T23:00:00+00:00",
        "portfolio_value": 13804.618723516587,
        "target_long_fraction": 0.8915503096446132,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.968081208379793,
        "normalized_value": 8940.328444577124
      },
      {
        "timestamp": "2022-06-27T18:00:00+00:00",
        "portfolio_value": 13869.098597649783,
        "target_long_fraction": 0.9388191696481456,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.612944376861407,
        "normalized_value": 8982.087747341042
      },
      {
        "timestamp": "2022-07-01T13:00:00+00:00",
        "portfolio_value": 13239.91421666436,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -27.078313230442703,
        "normalized_value": 8574.607096779859
      },
      {
        "timestamp": "2022-07-05T08:00:00+00:00",
        "portfolio_value": 12577.423859458711,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -30.727121940637918,
        "normalized_value": 8145.556392562008
      },
      {
        "timestamp": "2022-07-09T03:00:00+00:00",
        "portfolio_value": 12943.327213091912,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 28.206861229170265,
        "drawdown_pct": -28.71183020196583,
        "normalized_value": 8382.527527076598
      },
      {
        "timestamp": "2022-07-12T22:00:00+00:00",
        "portfolio_value": 11836.265180667631,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -34.809213420756166,
        "normalized_value": 7665.55747693436
      },
      {
        "timestamp": "2022-07-16T17:00:00+00:00",
        "portfolio_value": 12501.94595373285,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.100802971365171,
        "drawdown_pct": -31.142832806229176,
        "normalized_value": 8096.674400164167
      },
      {
        "timestamp": "2022-07-20T12:00:00+00:00",
        "portfolio_value": 14469.698345158698,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.707178574225717,
        "drawdown_pct": -20.30501156513701,
        "normalized_value": 9371.05604222865
      },
      {
        "timestamp": "2022-07-24T07:00:00+00:00",
        "portfolio_value": 13661.49814864306,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.643011585324177,
        "drawdown_pct": -24.75634868205196,
        "normalized_value": 8847.638818577801
      },
      {
        "timestamp": "2022-07-28T02:00:00+00:00",
        "portfolio_value": 12282.54391978061,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5998463607385878,
        "drawdown_pct": -32.35123688911452,
        "normalized_value": 7954.582373993261
      },
      {
        "timestamp": "2022-07-31T21:00:00+00:00",
        "portfolio_value": 12485.042309115694,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.452278120542514,
        "drawdown_pct": -31.23593327937912,
        "normalized_value": 8085.727039877401
      },
      {
        "timestamp": "2022-08-04T16:00:00+00:00",
        "portfolio_value": 11093.559184308064,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -38.899827086533314,
        "normalized_value": 7184.556467185359
      },
      {
        "timestamp": "2022-08-08T11:00:00+00:00",
        "portfolio_value": 11450.02876189603,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5155678397331265,
        "drawdown_pct": -36.93649390669833,
        "normalized_value": 7415.417975783711
      },
      {
        "timestamp": "2022-08-12T06:00:00+00:00",
        "portfolio_value": 10316.81408123469,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": 2.045558194860032,
        "drawdown_pct": -43.177918483440735,
        "normalized_value": 6681.510604182786
      },
      {
        "timestamp": "2022-08-16T01:00:00+00:00",
        "portfolio_value": 10031.617208703225,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -44.74870184850369,
        "normalized_value": 6496.80766070678
      },
      {
        "timestamp": "2022-08-19T20:00:00+00:00",
        "portfolio_value": 9926.972000272495,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -45.32505793255717,
        "normalized_value": 6429.035956738731
      },
      {
        "timestamp": "2022-08-23T15:00:00+00:00",
        "portfolio_value": 9926.972000272495,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -45.32505793255717,
        "normalized_value": 6429.035956738731
      },
      {
        "timestamp": "2022-08-27T10:00:00+00:00",
        "portfolio_value": 9926.972000272495,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -45.32505793255717,
        "normalized_value": 6429.035956738731
      },
      {
        "timestamp": "2022-08-31T05:00:00+00:00",
        "portfolio_value": 9926.972000272495,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -45.32505793255717,
        "normalized_value": 6429.035956738731
      },
      {
        "timestamp": "2022-09-04T00:00:00+00:00",
        "portfolio_value": 9693.206037662714,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.612574454481376,
        "normalized_value": 6277.641374479542
      },
      {
        "timestamp": "2022-09-07T19:00:00+00:00",
        "portfolio_value": 9804.302278353596,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.00068791712457,
        "normalized_value": 6349.590980667671
      },
      {
        "timestamp": "2022-09-11T14:00:00+00:00",
        "portfolio_value": 9590.190972341505,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -47.17995217332216,
        "normalized_value": 6210.925405197301
      },
      {
        "timestamp": "2022-09-15T09:00:00+00:00",
        "portfolio_value": 8759.260635471785,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -51.75648044691713,
        "normalized_value": 5672.7873895834955
      },
      {
        "timestamp": "2022-09-19T04:00:00+00:00",
        "portfolio_value": 8759.260635471785,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -51.75648044691713,
        "normalized_value": 5672.7873895834955
      },
      {
        "timestamp": "2022-09-22T23:00:00+00:00",
        "portfolio_value": 8759.260635471785,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -51.75648044691713,
        "normalized_value": 5672.7873895834955
      },
      {
        "timestamp": "2022-09-26T18:00:00+00:00",
        "portfolio_value": 8596.975560734856,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.65030054225019,
        "normalized_value": 5567.686198535802
      },
      {
        "timestamp": "2022-09-30T13:00:00+00:00",
        "portfolio_value": 8664.079454348432,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.28071135677437,
        "normalized_value": 5611.144903251298
      },
      {
        "timestamp": "2022-10-04T08:00:00+00:00",
        "portfolio_value": 8425.169151365813,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -53.596561444246284,
        "normalized_value": 5456.418675730113
      },
      {
        "timestamp": "2022-10-08T03:00:00+00:00",
        "portfolio_value": 8302.929524163721,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -54.26982258927991,
        "normalized_value": 5377.252243247074
      },
      {
        "timestamp": "2022-10-11T22:00:00+00:00",
        "portfolio_value": 8200.980733074868,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -54.83132757251909,
        "normalized_value": 5311.226828484316
      },
      {
        "timestamp": "2022-10-15T17:00:00+00:00",
        "portfolio_value": 7902.385179599357,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -56.47590706638906,
        "normalized_value": 5117.846455318796
      },
      {
        "timestamp": "2022-10-19T12:00:00+00:00",
        "portfolio_value": 7834.970678145942,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -56.84720699162003,
        "normalized_value": 5074.1865906755065
      },
      {
        "timestamp": "2022-10-23T07:00:00+00:00",
        "portfolio_value": 7834.970678145942,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -56.84720699162003,
        "normalized_value": 5074.1865906755065
      },
      {
        "timestamp": "2022-10-27T02:00:00+00:00",
        "portfolio_value": 8731.925789736399,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -51.9070330128923,
        "normalized_value": 5655.084437857644
      },
      {
        "timestamp": "2022-10-30T21:00:00+00:00",
        "portfolio_value": 8945.078606792584,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": 2.5668856074440707,
        "drawdown_pct": -50.73304784160953,
        "normalized_value": 5793.129264124581
      },
      {
        "timestamp": "2022-11-03T16:00:00+00:00",
        "portfolio_value": 8525.888736823279,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -53.04182657647466,
        "normalized_value": 5521.648016201316
      },
      {
        "timestamp": "2022-11-07T11:00:00+00:00",
        "portfolio_value": 8598.950401890776,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.639423678107796,
        "normalized_value": 5568.965171096609
      },
      {
        "timestamp": "2022-11-11T06:00:00+00:00",
        "portfolio_value": 8598.950401890776,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.639423678107796,
        "normalized_value": 5568.965171096609
      },
      {
        "timestamp": "2022-11-15T01:00:00+00:00",
        "portfolio_value": 8598.950401890776,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.639423678107796,
        "normalized_value": 5568.965171096609
      },
      {
        "timestamp": "2022-11-18T20:00:00+00:00",
        "portfolio_value": 8598.950401890776,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.639423678107796,
        "normalized_value": 5568.965171096609
      },
      {
        "timestamp": "2022-11-22T15:00:00+00:00",
        "portfolio_value": 8598.950401890776,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.639423678107796,
        "normalized_value": 5568.965171096609
      },
      {
        "timestamp": "2022-11-26T10:00:00+00:00",
        "portfolio_value": 8895.165447226922,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -51.00795534689722,
        "normalized_value": 5760.8038483227365
      },
      {
        "timestamp": "2022-11-30T05:00:00+00:00",
        "portfolio_value": 8553.21302327739,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.891332168021435,
        "normalized_value": 5539.344129386775
      },
      {
        "timestamp": "2022-12-04T00:00:00+00:00",
        "portfolio_value": 8557.957703031876,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -52.865199819637866,
        "normalized_value": 5542.416941191217
      },
      {
        "timestamp": "2022-12-07T19:00:00+00:00",
        "portfolio_value": 8474.171196367635,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -53.326672099178396,
        "normalized_value": 5488.1540235543835
      },
      {
        "timestamp": "2022-12-11T14:00:00+00:00",
        "portfolio_value": 8362.610289318702,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -53.941117886849035,
        "normalized_value": 5415.903484037974
      },
      {
        "timestamp": "2022-12-15T09:00:00+00:00",
        "portfolio_value": 8120.005592968186,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -55.27731564361614,
        "normalized_value": 5258.784644973219
      },
      {
        "timestamp": "2022-12-19T04:00:00+00:00",
        "portfolio_value": 8034.133442931223,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -55.75027505443636,
        "normalized_value": 5203.170995589001
      },
      {
        "timestamp": "2022-12-22T23:00:00+00:00",
        "portfolio_value": 8034.133442931223,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -55.75027505443636,
        "normalized_value": 5203.170995589001
      },
      {
        "timestamp": "2022-12-26T18:00:00+00:00",
        "portfolio_value": 8034.133442931223,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -55.75027505443636,
        "normalized_value": 5203.170995589001
      },
      {
        "timestamp": "2022-12-30T13:00:00+00:00",
        "portfolio_value": 8034.133442931223,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -55.75027505443636,
        "normalized_value": 5203.170995589001
      },
      {
        "timestamp": "2023-01-03T08:00:00+00:00",
        "portfolio_value": 8320.001523021481,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -54.17579486879295,
        "normalized_value": 5388.3086353177805
      },
      {
        "timestamp": "2023-01-07T03:00:00+00:00",
        "portfolio_value": 9010.09703127512,
        "target_long_fraction": 0.7,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -50.3749448277397,
        "normalized_value": 5835.23734993738
      },
      {
        "timestamp": "2023-01-10T22:00:00+00:00",
        "portfolio_value": 11452.261172161838,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.79653118893057,
        "drawdown_pct": -36.92419842505998,
        "normalized_value": 7416.863758633574
      },
      {
        "timestamp": "2023-01-14T17:00:00+00:00",
        "portfolio_value": 16862.81058785643,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.388003658838567,
        "drawdown_pct": -7.124429084719513,
        "normalized_value": 10920.914816524886
      },
      {
        "timestamp": "2023-01-18T12:00:00+00:00",
        "portfolio_value": 16471.98648296794,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.553132127711602,
        "drawdown_pct": -9.80945524359261,
        "normalized_value": 10667.80417785084
      },
      {
        "timestamp": "2023-01-22T07:00:00+00:00",
        "portfolio_value": 18510.50481940512,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 8.629132019625786,
        "drawdown_pct": -2.0830283255012194,
        "normalized_value": 11988.01619044301
      },
      {
        "timestamp": "2023-01-26T02:00:00+00:00",
        "portfolio_value": 18131.083327368684,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.594444017793585,
        "drawdown_pct": -4.090094251087451,
        "normalized_value": 11742.290261630604
      },
      {
        "timestamp": "2023-01-29T21:00:00+00:00",
        "portfolio_value": 19409.630209274685,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.282517126681626,
        "drawdown_pct": 0.0,
        "normalized_value": 12570.319581741936
      },
      {
        "timestamp": "2023-02-02T16:00:00+00:00",
        "portfolio_value": 18187.31723927549,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5610843652057589,
        "drawdown_pct": -6.297456246307701,
        "normalized_value": 11778.709206060688
      },
      {
        "timestamp": "2023-02-06T11:00:00+00:00",
        "portfolio_value": 17305.487182812027,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5347996606066054,
        "drawdown_pct": -10.840716715237654,
        "normalized_value": 11207.606845685244
      },
      {
        "timestamp": "2023-02-10T06:00:00+00:00",
        "portfolio_value": 16436.751335488032,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -15.31651474929231,
        "normalized_value": 10644.984728971254
      },
      {
        "timestamp": "2023-02-14T01:00:00+00:00",
        "portfolio_value": 15388.8384814334,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.715447355199956,
        "normalized_value": 9966.321646405793
      },
      {
        "timestamp": "2023-02-17T20:00:00+00:00",
        "portfolio_value": 16100.312310345353,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 4.602190612775138,
        "drawdown_pct": -17.04987608340941,
        "normalized_value": 10427.095669766388
      },
      {
        "timestamp": "2023-02-21T15:00:00+00:00",
        "portfolio_value": 17182.04243295504,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 10.418872652518292,
        "drawdown_pct": -11.476714148089307,
        "normalized_value": 11127.659935844118
      },
      {
        "timestamp": "2023-02-25T10:00:00+00:00",
        "portfolio_value": 15556.916203116129,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.849497206379425,
        "normalized_value": 10075.174347531105
      },
      {
        "timestamp": "2023-03-01T05:00:00+00:00",
        "portfolio_value": 15556.916203116129,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.849497206379425,
        "normalized_value": 10075.174347531105
      },
      {
        "timestamp": "2023-03-05T00:00:00+00:00",
        "portfolio_value": 15265.975533087349,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.348447299152227,
        "normalized_value": 9886.751530498746
      },
      {
        "timestamp": "2023-03-08T19:00:00+00:00",
        "portfolio_value": 15265.975533087349,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.348447299152227,
        "normalized_value": 9886.751530498746
      },
      {
        "timestamp": "2023-03-12T14:00:00+00:00",
        "portfolio_value": 15265.975533087349,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.348447299152227,
        "normalized_value": 9886.751530498746
      },
      {
        "timestamp": "2023-03-16T09:00:00+00:00",
        "portfolio_value": 14951.61200676431,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 4.741288305796723,
        "drawdown_pct": -22.968073860470348,
        "normalized_value": 9683.159295710282
      },
      {
        "timestamp": "2023-03-20T04:00:00+00:00",
        "portfolio_value": 15961.451359792189,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5878658295357189,
        "drawdown_pct": -17.765299041270865,
        "normalized_value": 10337.164717602052
      },
      {
        "timestamp": "2023-03-23T23:00:00+00:00",
        "portfolio_value": 16003.74352242269,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.676277015767202,
        "drawdown_pct": -17.547406365447024,
        "normalized_value": 10364.554523298319
      },
      {
        "timestamp": "2023-03-27T19:00:00+00:00",
        "portfolio_value": 14325.477527244891,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5719826851840286,
        "drawdown_pct": -26.193969834625623,
        "normalized_value": 9277.653862384415
      },
      {
        "timestamp": "2023-03-31T14:00:00+00:00",
        "portfolio_value": 15455.196778280086,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.6280286953174958,
        "drawdown_pct": -20.373563990441276,
        "normalized_value": 10009.297477952772
      },
      {
        "timestamp": "2023-04-04T09:00:00+00:00",
        "portfolio_value": 15378.565155181734,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.62346309097685,
        "drawdown_pct": -20.76837637105909,
        "normalized_value": 9959.66829996083
      },
      {
        "timestamp": "2023-04-08T04:00:00+00:00",
        "portfolio_value": 15685.97083017093,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6381541287518566,
        "drawdown_pct": -19.18459722805251,
        "normalized_value": 10158.754399705731
      },
      {
        "timestamp": "2023-04-11T23:00:00+00:00",
        "portfolio_value": 16590.354484260264,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.957599237355891,
        "drawdown_pct": -14.525138782228112,
        "normalized_value": 10744.463217124323
      },
      {
        "timestamp": "2023-04-15T18:00:00+00:00",
        "portfolio_value": 17421.286993082413,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 10.412590300924542,
        "drawdown_pct": -10.244106635489446,
        "normalized_value": 11282.60263936648
      },
      {
        "timestamp": "2023-04-19T13:00:00+00:00",
        "portfolio_value": 16594.25509522506,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.505042515979142,
        "normalized_value": 10746.989382015816
      },
      {
        "timestamp": "2023-04-23T08:00:00+00:00",
        "portfolio_value": 16594.25509522506,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.505042515979142,
        "normalized_value": 10746.989382015816
      },
      {
        "timestamp": "2023-04-27T03:00:00+00:00",
        "portfolio_value": 15518.833053331022,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.045704704278634,
        "normalized_value": 10050.510438001835
      },
      {
        "timestamp": "2023-04-30T22:00:00+00:00",
        "portfolio_value": 15456.6843697356,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.21757724995468,
        "drawdown_pct": -20.365899797772613,
        "normalized_value": 10010.260891464584
      },
      {
        "timestamp": "2023-05-04T17:00:00+00:00",
        "portfolio_value": 14223.01989321896,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -26.721839932722464,
        "normalized_value": 9211.298904077188
      },
      {
        "timestamp": "2023-05-08T12:00:00+00:00",
        "portfolio_value": 13672.22693564666,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.55957023275213,
        "normalized_value": 8854.587136495535
      },
      {
        "timestamp": "2023-05-12T07:00:00+00:00",
        "portfolio_value": 13672.22693564666,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.55957023275213,
        "normalized_value": 8854.587136495535
      },
      {
        "timestamp": "2023-05-16T02:00:00+00:00",
        "portfolio_value": 13672.22693564666,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.55957023275213,
        "normalized_value": 8854.587136495535
      },
      {
        "timestamp": "2023-05-19T21:00:00+00:00",
        "portfolio_value": 13672.22693564666,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.55957023275213,
        "normalized_value": 8854.587136495535
      },
      {
        "timestamp": "2023-05-23T16:00:00+00:00",
        "portfolio_value": 13612.635437680094,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.866590497043877,
        "normalized_value": 8815.993708093352
      },
      {
        "timestamp": "2023-05-27T11:00:00+00:00",
        "portfolio_value": 13433.540566967564,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -30.78930189742363,
        "normalized_value": 8700.005936248452
      },
      {
        "timestamp": "2023-05-31T06:00:00+00:00",
        "portfolio_value": 12983.306200074883,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -33.108946125769315,
        "normalized_value": 8408.419243585968
      },
      {
        "timestamp": "2023-06-04T01:00:00+00:00",
        "portfolio_value": 13021.790136912814,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5488750597688914,
        "drawdown_pct": -32.910673740242146,
        "normalized_value": 8433.342716089077
      },
      {
        "timestamp": "2023-06-07T20:00:00+00:00",
        "portfolio_value": 12060.29534265985,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -37.86437344438964,
        "normalized_value": 7810.646832157931
      },
      {
        "timestamp": "2023-06-11T15:00:00+00:00",
        "portfolio_value": 12060.29534265985,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -37.86437344438964,
        "normalized_value": 7810.646832157931
      },
      {
        "timestamp": "2023-06-15T10:00:00+00:00",
        "portfolio_value": 12060.29534265985,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -37.86437344438964,
        "normalized_value": 7810.646832157931
      },
      {
        "timestamp": "2023-06-19T05:00:00+00:00",
        "portfolio_value": 11875.06489398374,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -38.81869584352325,
        "normalized_value": 7690.685456746689
      },
      {
        "timestamp": "2023-06-23T00:00:00+00:00",
        "portfolio_value": 12480.27857087559,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 12.037555987363742,
        "drawdown_pct": -35.70058555308271,
        "normalized_value": 8082.6418851662465
      },
      {
        "timestamp": "2023-06-26T19:00:00+00:00",
        "portfolio_value": 12142.345652239792,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 26.08767283354349,
        "drawdown_pct": -37.44164354848089,
        "normalized_value": 7863.7853310412265
      },
      {
        "timestamp": "2023-06-30T14:00:00+00:00",
        "portfolio_value": 12122.035505317992,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5552819715556314,
        "drawdown_pct": -37.54628308412797,
        "normalized_value": 7850.631807001539
      },
      {
        "timestamp": "2023-07-04T09:00:00+00:00",
        "portfolio_value": 12654.528039238668,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 6.062232701960545,
        "drawdown_pct": -34.80283806132567,
        "normalized_value": 8195.491613917178
      },
      {
        "timestamp": "2023-07-08T04:00:00+00:00",
        "portfolio_value": 14370.619882259009,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.470947776763817,
        "drawdown_pct": -25.96139273486951,
        "normalized_value": 9306.889547097706
      },
      {
        "timestamp": "2023-07-11T23:00:00+00:00",
        "portfolio_value": 14669.069778168652,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.533645736682391,
        "drawdown_pct": -24.4237544970888,
        "normalized_value": 9500.175587597805
      },
      {
        "timestamp": "2023-07-15T18:00:00+00:00",
        "portfolio_value": 19535.17979852396,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.152859506003155,
        "drawdown_pct": -5.433478546187481,
        "normalized_value": 12651.629655308701
      },
      {
        "timestamp": "2023-07-19T13:00:00+00:00",
        "portfolio_value": 17586.315477802927,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.86760311001466,
        "normalized_value": 11389.480553611145
      },
      {
        "timestamp": "2023-07-23T08:00:00+00:00",
        "portfolio_value": 17101.66441235542,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -17.213717445840615,
        "normalized_value": 11075.604466708894
      },
      {
        "timestamp": "2023-07-27T03:00:00+00:00",
        "portfolio_value": 16861.036847511208,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -18.378555036684716,
        "normalized_value": 10919.766083511733
      },
      {
        "timestamp": "2023-07-30T22:00:00+00:00",
        "portfolio_value": 16837.419221890865,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -18.492883992084607,
        "normalized_value": 10904.470526687175
      },
      {
        "timestamp": "2023-08-03T17:00:00+00:00",
        "portfolio_value": 16549.910058667705,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.88466751967543,
        "normalized_value": 10718.27006715099
      },
      {
        "timestamp": "2023-08-07T12:00:00+00:00",
        "portfolio_value": 16434.30699872716,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.444282499457696,
        "normalized_value": 10643.401695501874
      },
      {
        "timestamp": "2023-08-11T07:00:00+00:00",
        "portfolio_value": 16358.703237328573,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.111762669153476,
        "drawdown_pct": -20.810267599058804,
        "normalized_value": 10594.438194800665
      },
      {
        "timestamp": "2023-08-15T02:00:00+00:00",
        "portfolio_value": 16641.416661438965,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.091756910192792,
        "drawdown_pct": -19.44169943832776,
        "normalized_value": 10777.532774800342
      },
      {
        "timestamp": "2023-08-18T21:00:00+00:00",
        "portfolio_value": 16013.186307722795,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.48285697234366,
        "normalized_value": 10370.669983906493
      },
      {
        "timestamp": "2023-08-22T16:00:00+00:00",
        "portfolio_value": 16013.186307722795,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.48285697234366,
        "normalized_value": 10370.669983906493
      },
      {
        "timestamp": "2023-08-26T11:00:00+00:00",
        "portfolio_value": 16013.186307722795,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.48285697234366,
        "normalized_value": 10370.669983906493
      },
      {
        "timestamp": "2023-08-30T06:00:00+00:00",
        "portfolio_value": 15769.786357889623,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5403581551736831,
        "drawdown_pct": -23.66111521286903,
        "normalized_value": 10213.03611233892
      },
      {
        "timestamp": "2023-09-03T01:00:00+00:00",
        "portfolio_value": 14652.06639107809,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.071809678492734,
        "normalized_value": 9489.163630780704
      },
      {
        "timestamp": "2023-09-06T20:00:00+00:00",
        "portfolio_value": 14652.06639107809,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.071809678492734,
        "normalized_value": 9489.163630780704
      },
      {
        "timestamp": "2023-09-10T15:00:00+00:00",
        "portfolio_value": 14652.06639107809,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.071809678492734,
        "normalized_value": 9489.163630780704
      },
      {
        "timestamp": "2023-09-14T10:00:00+00:00",
        "portfolio_value": 14659.82540004044,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.034249620430003,
        "normalized_value": 9494.188622047552
      },
      {
        "timestamp": "2023-09-18T05:00:00+00:00",
        "portfolio_value": 14404.867155090926,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -30.268459624612625,
        "normalized_value": 9329.069215626094
      },
      {
        "timestamp": "2023-09-22T00:00:00+00:00",
        "portfolio_value": 14236.732856832688,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 36.79827345518623,
        "drawdown_pct": -31.082369498355416,
        "normalized_value": 9220.179873636052
      },
      {
        "timestamp": "2023-09-25T19:00:00+00:00",
        "portfolio_value": 14236.451401815577,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 36.76770508489611,
        "drawdown_pct": -31.08373197477941,
        "normalized_value": 9219.997594042117
      },
      {
        "timestamp": "2023-09-29T14:00:00+00:00",
        "portfolio_value": 14980.213053994768,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5672873078371004,
        "drawdown_pct": -27.48330684623051,
        "normalized_value": 9701.682281474717
      },
      {
        "timestamp": "2023-10-03T09:00:00+00:00",
        "portfolio_value": 17338.790030796503,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.457140755367238,
        "drawdown_pct": -16.06583218884127,
        "normalized_value": 11229.174873392802
      },
      {
        "timestamp": "2023-10-07T04:00:00+00:00",
        "portfolio_value": 17003.596432650873,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.35732961182572,
        "drawdown_pct": -17.688448049925608,
        "normalized_value": 11012.092393973324
      },
      {
        "timestamp": "2023-10-10T23:00:00+00:00",
        "portfolio_value": 15712.679304595447,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.937560857915805,
        "normalized_value": 10176.051692618428
      },
      {
        "timestamp": "2023-10-14T18:00:00+00:00",
        "portfolio_value": 15712.679304595447,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.937560857915805,
        "normalized_value": 10176.051692618428
      },
      {
        "timestamp": "2023-10-18T13:00:00+00:00",
        "portfolio_value": 16467.698070584374,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.519726209140137,
        "drawdown_pct": -20.282641933785005,
        "normalized_value": 10665.02685991235
      },
      {
        "timestamp": "2023-10-22T08:00:00+00:00",
        "portfolio_value": 19957.83448074112,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 12.588432416167723,
        "drawdown_pct": -5.653902462010035,
        "normalized_value": 12925.354830435972
      },
      {
        "timestamp": "2023-10-26T03:00:00+00:00",
        "portfolio_value": 23128.568744613724,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 14.46378818387021,
        "drawdown_pct": -2.076194261143835,
        "normalized_value": 14978.827388950382
      },
      {
        "timestamp": "2023-10-29T22:00:00+00:00",
        "portfolio_value": 23283.559488763924,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 14.544440955552664,
        "drawdown_pct": -1.419980567634612,
        "normalized_value": 15079.204529842487
      },
      {
        "timestamp": "2023-11-02T17:00:00+00:00",
        "portfolio_value": 28647.544851777475,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.59157785562154,
        "drawdown_pct": -10.235430410482218,
        "normalized_value": 18553.09916450931
      },
      {
        "timestamp": "2023-11-06T12:00:00+00:00",
        "portfolio_value": 28337.77689114292,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5679417424938782,
        "drawdown_pct": -11.206061150493376,
        "normalized_value": 18352.48317031586
      },
      {
        "timestamp": "2023-11-10T07:00:00+00:00",
        "portfolio_value": 31252.478606980607,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.591477282567174,
        "drawdown_pct": -5.279313471014194,
        "normalized_value": 20240.140568138097
      },
      {
        "timestamp": "2023-11-14T02:00:00+00:00",
        "portfolio_value": 33297.136510522665,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.836053472361268,
        "normalized_value": 21564.328767796673
      },
      {
        "timestamp": "2023-11-17T21:00:00+00:00",
        "portfolio_value": 34060.27222062038,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.18561260853382,
        "normalized_value": 22058.560737016986
      },
      {
        "timestamp": "2023-11-21T16:00:00+00:00",
        "portfolio_value": 32424.13554039899,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 23.49204093618681,
        "drawdown_pct": -23.067653910034338,
        "normalized_value": 20998.944416250317
      },
      {
        "timestamp": "2023-11-25T11:00:00+00:00",
        "portfolio_value": 31267.873473102445,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5505968053218424,
        "drawdown_pct": -25.81110264195579,
        "normalized_value": 20250.110793484127
      },
      {
        "timestamp": "2023-11-29T06:00:00+00:00",
        "portfolio_value": 32430.97144688639,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 25.24497989653097,
        "drawdown_pct": -23.051434439107442,
        "normalized_value": 21003.371575771183
      },
      {
        "timestamp": "2023-12-03T01:00:00+00:00",
        "portfolio_value": 35544.72785145308,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.104125448100952,
        "drawdown_pct": -15.6634630602713,
        "normalized_value": 23019.94338487221
      },
      {
        "timestamp": "2023-12-06T20:00:00+00:00",
        "portfolio_value": 33207.438238055285,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 3.998626777980237,
        "drawdown_pct": -21.20912126992077,
        "normalized_value": 21506.237194763697
      },
      {
        "timestamp": "2023-12-10T15:00:00+00:00",
        "portfolio_value": 38868.6873168877,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 3.768393640861519,
        "drawdown_pct": -7.776745474070175,
        "normalized_value": 25172.64966040463
      },
      {
        "timestamp": "2023-12-14T10:00:00+00:00",
        "portfolio_value": 34885.48341930044,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -17.227644185441406,
        "normalized_value": 22592.99484925906
      },
      {
        "timestamp": "2023-12-18T05:00:00+00:00",
        "portfolio_value": 33633.806350351406,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 27.660268360923283,
        "drawdown_pct": -20.197482913794936,
        "normalized_value": 21782.367310239326
      },
      {
        "timestamp": "2023-12-22T00:00:00+00:00",
        "portfolio_value": 43865.47732222166,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 12.01254145417388,
        "drawdown_pct": -0.730440093342121,
        "normalized_value": 28408.736415931184
      },
      {
        "timestamp": "2023-12-25T19:00:00+00:00",
        "portfolio_value": 57437.201271531645,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 15.496494287448469,
        "drawdown_pct": -2.065664159804175,
        "normalized_value": 37198.234488722264
      },
      {
        "timestamp": "2023-12-29T14:00:00+00:00",
        "portfolio_value": 50315.05705402631,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 28.3567664074694,
        "drawdown_pct": -14.209404597331183,
        "normalized_value": 32585.69793052871
      },
      {
        "timestamp": "2024-01-02T09:00:00+00:00",
        "portfolio_value": 52284.77516601754,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 7.719221017618776,
        "drawdown_pct": -10.850901208937898,
        "normalized_value": 33861.3526383574
      },
      {
        "timestamp": "2024-01-06T04:00:00+00:00",
        "portfolio_value": 42820.87157716141,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -26.987500693454454,
        "normalized_value": 27732.215126717932
      },
      {
        "timestamp": "2024-01-09T23:00:00+00:00",
        "portfolio_value": 41103.79533500595,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 4.101457378283381,
        "drawdown_pct": -29.915232505586147,
        "normalized_value": 26620.179664043484
      },
      {
        "timestamp": "2024-01-13T18:00:00+00:00",
        "portfolio_value": 43418.13065991256,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.7679670233152414,
        "drawdown_pct": -25.969133323546583,
        "normalized_value": 28119.019896429232
      },
      {
        "timestamp": "2024-01-17T13:00:00+00:00",
        "portfolio_value": 44335.311360423446,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 28.014066838007043,
        "drawdown_pct": -24.405277829868492,
        "normalized_value": 28713.016505087893
      },
      {
        "timestamp": "2024-01-21T08:00:00+00:00",
        "portfolio_value": 41776.70857362448,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -28.76786965288315,
        "normalized_value": 27055.980571583597
      },
      {
        "timestamp": "2024-01-25T03:00:00+00:00",
        "portfolio_value": 41776.70857362448,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -28.76786965288315,
        "normalized_value": 27055.980571583597
      },
      {
        "timestamp": "2024-01-28T22:00:00+00:00",
        "portfolio_value": 42469.32336997234,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -27.586914303879396,
        "normalized_value": 27504.540860639263
      },
      {
        "timestamp": "2024-02-01T17:00:00+00:00",
        "portfolio_value": 42910.0285377628,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -26.83548200994043,
        "normalized_value": 27789.956128252357
      },
      {
        "timestamp": "2024-02-05T12:00:00+00:00",
        "portfolio_value": 42910.0285377628,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -26.83548200994043,
        "normalized_value": 27789.956128252357
      },
      {
        "timestamp": "2024-02-09T07:00:00+00:00",
        "portfolio_value": 44911.28196893902,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.20908455277699,
        "drawdown_pct": -23.423208756868508,
        "normalized_value": 29086.034153578254
      },
      {
        "timestamp": "2024-02-13T02:00:00+00:00",
        "portfolio_value": 48030.40530086358,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.562013965372612,
        "drawdown_pct": -18.10489127006063,
        "normalized_value": 31106.08176264729
      },
      {
        "timestamp": "2024-02-16T21:00:00+00:00",
        "portfolio_value": 46901.18950087073,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6052344092532005,
        "drawdown_pct": -20.03028103391335,
        "normalized_value": 30374.76419865373
      },
      {
        "timestamp": "2024-02-20T16:00:00+00:00",
        "portfolio_value": 48026.95932831427,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 9.89926526942505,
        "drawdown_pct": -18.11076688769989,
        "normalized_value": 31103.850036656233
      },
      {
        "timestamp": "2024-02-24T11:00:00+00:00",
        "portfolio_value": 49128.50883336364,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 26.182392099077703,
        "drawdown_pct": -16.23255003896955,
        "normalized_value": 31817.249991435572
      },
      {
        "timestamp": "2024-02-28T06:00:00+00:00",
        "portfolio_value": 54208.25993241141,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 28.787721950351067,
        "drawdown_pct": -7.571228820219251,
        "normalized_value": 35107.0650998256
      },
      {
        "timestamp": "2024-03-03T01:00:00+00:00",
        "portfolio_value": 64244.90542680436,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5896390281214423,
        "drawdown_pct": -5.775571180095309,
        "normalized_value": 41607.1292449366
      },
      {
        "timestamp": "2024-03-06T20:00:00+00:00",
        "portfolio_value": 59547.336111046,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.718552911359087,
        "normalized_value": 38564.827721425296
      },
      {
        "timestamp": "2024-03-10T15:00:00+00:00",
        "portfolio_value": 58984.78570497749,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5443246452682815,
        "drawdown_pct": -14.533663481635129,
        "normalized_value": 38200.501440662825
      },
      {
        "timestamp": "2024-03-14T10:00:00+00:00",
        "portfolio_value": 72115.13700826673,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.795277283394148,
        "drawdown_pct": 0.0,
        "normalized_value": 46704.151964824734
      },
      {
        "timestamp": "2024-03-18T05:00:00+00:00",
        "portfolio_value": 85485.52779669943,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.9710131545297345,
        "drawdown_pct": -0.23451302784898734,
        "normalized_value": 55363.260012286
      },
      {
        "timestamp": "2024-03-22T00:00:00+00:00",
        "portfolio_value": 70687.41592875104,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.65050161270682,
        "normalized_value": 45779.512492068265
      },
      {
        "timestamp": "2024-03-25T19:00:00+00:00",
        "portfolio_value": 74757.72928909332,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.4771067111036,
        "drawdown_pct": -15.02382750833377,
        "normalized_value": 48415.582277307534
      },
      {
        "timestamp": "2024-03-29T14:00:00+00:00",
        "portfolio_value": 73157.55026027528,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.395726588474822,
        "drawdown_pct": -16.84273092425422,
        "normalized_value": 47379.253322898476
      },
      {
        "timestamp": "2024-04-02T09:00:00+00:00",
        "portfolio_value": 73518.79540047527,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.43210810789363,
        "normalized_value": 47613.20764406298
      },
      {
        "timestamp": "2024-04-06T04:00:00+00:00",
        "portfolio_value": 73518.79540047527,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.43210810789363,
        "normalized_value": 47613.20764406298
      },
      {
        "timestamp": "2024-04-09T23:00:00+00:00",
        "portfolio_value": 70648.5847571157,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.694640518079098,
        "normalized_value": 45754.364138806166
      },
      {
        "timestamp": "2024-04-13T18:00:00+00:00",
        "portfolio_value": 70648.5847571157,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.694640518079098,
        "normalized_value": 45754.364138806166
      },
      {
        "timestamp": "2024-04-17T13:00:00+00:00",
        "portfolio_value": 70648.5847571157,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.694640518079098,
        "normalized_value": 45754.364138806166
      },
      {
        "timestamp": "2024-04-21T08:00:00+00:00",
        "portfolio_value": 70648.5847571157,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.694640518079098,
        "normalized_value": 45754.364138806166
      },
      {
        "timestamp": "2024-04-25T03:00:00+00:00",
        "portfolio_value": 70648.5847571157,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.694640518079098,
        "normalized_value": 45754.364138806166
      },
      {
        "timestamp": "2024-04-28T22:00:00+00:00",
        "portfolio_value": 70135.03771773331,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.278382991481237,
        "normalized_value": 45421.774061834505
      },
      {
        "timestamp": "2024-05-02T17:00:00+00:00",
        "portfolio_value": 69892.64736483652,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.553905063030818,
        "normalized_value": 45264.79403869177
      },
      {
        "timestamp": "2024-05-06T12:00:00+00:00",
        "portfolio_value": 73948.33970795202,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 27.316979113981358,
        "drawdown_pct": -15.94385048541013,
        "normalized_value": 47891.39476876775
      },
      {
        "timestamp": "2024-05-10T07:00:00+00:00",
        "portfolio_value": 72620.11470872954,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -17.453627169573537,
        "normalized_value": 47031.1922540023
      },
      {
        "timestamp": "2024-05-14T02:00:00+00:00",
        "portfolio_value": 69472.69050029207,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.031264759478887,
        "normalized_value": 44992.816059670426
      },
      {
        "timestamp": "2024-05-17T21:00:00+00:00",
        "portfolio_value": 75999.44760358935,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.612381886829722,
        "normalized_value": 49219.76019700116
      },
      {
        "timestamp": "2024-05-21T16:00:00+00:00",
        "portfolio_value": 79784.84924374013,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5523952858370602,
        "drawdown_pct": -9.309563358463603,
        "normalized_value": 51671.31171286714
      },
      {
        "timestamp": "2024-05-25T11:00:00+00:00",
        "portfolio_value": 68131.50519151885,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 4.128020315715329,
        "drawdown_pct": -22.55577326482522,
        "normalized_value": 44124.220019053406
      },
      {
        "timestamp": "2024-05-29T06:00:00+00:00",
        "portfolio_value": 69769.94851939664,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.7016889277707168,
        "drawdown_pct": -20.693375300473527,
        "normalized_value": 45185.330201263634
      },
      {
        "timestamp": "2024-06-02T01:00:00+00:00",
        "portfolio_value": 64184.34238562849,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -27.042463679791627,
        "normalized_value": 41567.90661296398
      },
      {
        "timestamp": "2024-06-05T20:00:00+00:00",
        "portfolio_value": 65845.12303924536,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.220002290524407,
        "drawdown_pct": -25.154675157660066,
        "normalized_value": 42643.4831873782
      },
      {
        "timestamp": "2024-06-09T15:00:00+00:00",
        "portfolio_value": 61583.54758452991,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.99875448390284,
        "normalized_value": 39883.54573314052
      },
      {
        "timestamp": "2024-06-13T10:00:00+00:00",
        "portfolio_value": 61583.54758452991,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.99875448390284,
        "normalized_value": 39883.54573314052
      },
      {
        "timestamp": "2024-06-17T05:00:00+00:00",
        "portfolio_value": 61583.54758452991,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.99875448390284,
        "normalized_value": 39883.54573314052
      },
      {
        "timestamp": "2024-06-21T00:00:00+00:00",
        "portfolio_value": 61583.54758452991,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.99875448390284,
        "normalized_value": 39883.54573314052
      },
      {
        "timestamp": "2024-06-24T19:00:00+00:00",
        "portfolio_value": 61583.54758452991,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -29.99875448390284,
        "normalized_value": 39883.54573314052
      },
      {
        "timestamp": "2024-06-28T14:00:00+00:00",
        "portfolio_value": 64777.905919030105,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -26.36776746203937,
        "normalized_value": 41952.31802247971
      },
      {
        "timestamp": "2024-07-02T09:00:00+00:00",
        "portfolio_value": 68083.0773541273,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.610820579701112,
        "normalized_value": 44092.856547101546
      },
      {
        "timestamp": "2024-07-06T04:00:00+00:00",
        "portfolio_value": 65173.86950704335,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.91768062185037,
        "normalized_value": 42208.75716070097
      },
      {
        "timestamp": "2024-07-09T23:00:00+00:00",
        "portfolio_value": 63226.07468911682,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -28.131714541827215,
        "normalized_value": 40947.3007044467
      },
      {
        "timestamp": "2024-07-13T18:00:00+00:00",
        "portfolio_value": 61491.32660739543,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -30.103580911014145,
        "normalized_value": 39823.82037298569
      },
      {
        "timestamp": "2024-07-17T13:00:00+00:00",
        "portfolio_value": 71821.20111494494,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.153850067166765,
        "drawdown_pct": -18.361742223320647,
        "normalized_value": 46513.78934195343
      },
      {
        "timestamp": "2024-07-21T08:00:00+00:00",
        "portfolio_value": 76298.41307600735,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.872563282978495,
        "drawdown_pct": -13.272551587075675,
        "normalized_value": 49413.38014192958
      },
      {
        "timestamp": "2024-07-25T03:00:00+00:00",
        "portfolio_value": 75548.6889946934,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.139902838874544,
        "drawdown_pct": -14.124753539445367,
        "normalized_value": 48927.83398784875
      },
      {
        "timestamp": "2024-07-28T22:00:00+00:00",
        "portfolio_value": 78389.2091581221,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 29.55171152925062,
        "drawdown_pct": -10.895969925109597,
        "normalized_value": 50767.44895463049
      },
      {
        "timestamp": "2024-08-01T17:00:00+00:00",
        "portfolio_value": 76608.19063687118,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -12.920431322103246,
        "normalized_value": 49614.00235354957
      },
      {
        "timestamp": "2024-08-05T12:00:00+00:00",
        "portfolio_value": 76608.19063687118,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -12.920431322103246,
        "normalized_value": 49614.00235354957
      },
      {
        "timestamp": "2024-08-09T07:00:00+00:00",
        "portfolio_value": 76227.71150309745,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 27.648850419853083,
        "drawdown_pct": -13.352917177525727,
        "normalized_value": 49367.59146090749
      },
      {
        "timestamp": "2024-08-13T02:00:00+00:00",
        "portfolio_value": 73496.28370690819,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.45769686739485,
        "normalized_value": 47598.628325476624
      },
      {
        "timestamp": "2024-08-16T21:00:00+00:00",
        "portfolio_value": 73496.28370690819,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.45769686739485,
        "normalized_value": 47598.628325476624
      },
      {
        "timestamp": "2024-08-20T16:00:00+00:00",
        "portfolio_value": 73496.28370690819,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -16.45769686739485,
        "normalized_value": 47598.628325476624
      },
      {
        "timestamp": "2024-08-24T11:00:00+00:00",
        "portfolio_value": 76566.41949341523,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -12.967912056540221,
        "normalized_value": 49586.94997713257
      },
      {
        "timestamp": "2024-08-28T06:00:00+00:00",
        "portfolio_value": 76040.42590108425,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 175.73115782673602,
        "drawdown_pct": -13.565802370444521,
        "normalized_value": 49246.299100106095
      },
      {
        "timestamp": "2024-09-01T01:00:00+00:00",
        "portfolio_value": 76040.11691914461,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 175.58517774127097,
        "drawdown_pct": -13.566153586336835,
        "normalized_value": 49246.09899316514
      },
      {
        "timestamp": "2024-09-04T20:00:00+00:00",
        "portfolio_value": 76039.80768031992,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 175.43931892163965,
        "drawdown_pct": -13.566505094227157,
        "normalized_value": 49245.898719856916
      },
      {
        "timestamp": "2024-09-08T15:00:00+00:00",
        "portfolio_value": 76039.49818439658,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 175.29358126710605,
        "drawdown_pct": -13.566856894358295,
        "normalized_value": 49245.698280043085
      },
      {
        "timestamp": "2024-09-12T10:00:00+00:00",
        "portfolio_value": 76039.18843116086,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 175.14796467701868,
        "drawdown_pct": -13.567208986973183,
        "normalized_value": 49245.49767358524
      },
      {
        "timestamp": "2024-09-16T05:00:00+00:00",
        "portfolio_value": 76038.87842039882,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 175.0024690508088,
        "drawdown_pct": -13.567561372315007,
        "normalized_value": 49245.29690034481
      },
      {
        "timestamp": "2024-09-20T00:00:00+00:00",
        "portfolio_value": 76296.13751497913,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.536455159960075,
        "drawdown_pct": -13.27513818873263,
        "normalized_value": 49411.906413216355
      },
      {
        "timestamp": "2024-09-23T19:00:00+00:00",
        "portfolio_value": 77814.11856610354,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.506861404319812,
        "drawdown_pct": -11.549668182272063,
        "normalized_value": 50395.00123397932
      },
      {
        "timestamp": "2024-09-27T14:00:00+00:00",
        "portfolio_value": 86104.90191940284,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5747352773022618,
        "drawdown_pct": -2.125638814063567,
        "normalized_value": 55764.38721456121
      },
      {
        "timestamp": "2024-10-01T09:00:00+00:00",
        "portfolio_value": 82576.22439057034,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 8.674537389385161,
        "drawdown_pct": -6.136642267605914,
        "normalized_value": 53479.098738681816
      },
      {
        "timestamp": "2024-10-05T04:00:00+00:00",
        "portfolio_value": 78459.52156080691,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.816046699569364,
        "normalized_value": 50812.98559612127
      },
      {
        "timestamp": "2024-10-08T23:00:00+00:00",
        "portfolio_value": 77024.95382155658,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -12.446701841462508,
        "normalized_value": 49883.91199968475
      },
      {
        "timestamp": "2024-10-12T18:00:00+00:00",
        "portfolio_value": 77121.70943525727,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -12.336720949887106,
        "normalized_value": 49946.57414071637
      },
      {
        "timestamp": "2024-10-16T13:00:00+00:00",
        "portfolio_value": 80534.10316851406,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.055898532043985,
        "drawdown_pct": -8.457895827131342,
        "normalized_value": 52156.553377995944
      },
      {
        "timestamp": "2024-10-20T08:00:00+00:00",
        "portfolio_value": 82925.31644040335,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.53398913389112,
        "drawdown_pct": -5.739833716514635,
        "normalized_value": 53705.18231588204
      },
      {
        "timestamp": "2024-10-24T03:00:00+00:00",
        "portfolio_value": 91524.05246143015,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.543425108473864,
        "drawdown_pct": -0.2503569974648569,
        "normalized_value": 59274.00864682852
      },
      {
        "timestamp": "2024-10-27T22:00:00+00:00",
        "portfolio_value": 89399.37759580124,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -4.994992184128951,
        "normalized_value": 57897.99881148978
      },
      {
        "timestamp": "2024-10-31T17:00:00+00:00",
        "portfolio_value": 83100.82451759299,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 13.111166735369844,
        "drawdown_pct": -11.6884849188251,
        "normalized_value": 53818.84716140788
      },
      {
        "timestamp": "2024-11-04T12:00:00+00:00",
        "portfolio_value": 81096.42712880838,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -13.818564508960154,
        "normalized_value": 52520.733004972695
      },
      {
        "timestamp": "2024-11-08T07:00:00+00:00",
        "portfolio_value": 88259.81079958632,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.152774213522202,
        "drawdown_pct": -6.2060135054473005,
        "normalized_value": 57159.977599405145
      },
      {
        "timestamp": "2024-11-12T02:00:00+00:00",
        "portfolio_value": 96262.8007734206,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 6.841295626200674,
        "drawdown_pct": -2.1954718200716354,
        "normalized_value": 62342.97905259629
      },
      {
        "timestamp": "2024-11-15T21:00:00+00:00",
        "portfolio_value": 95491.61897708908,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 30.96150219409695,
        "drawdown_pct": -4.239360511131954,
        "normalized_value": 61843.536171357
      },
      {
        "timestamp": "2024-11-19T16:00:00+00:00",
        "portfolio_value": 106410.78201930544,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.305664054232958,
        "drawdown_pct": -2.281037338508879,
        "normalized_value": 68915.14791902534
      },
      {
        "timestamp": "2024-11-23T11:00:00+00:00",
        "portfolio_value": 113015.59847602015,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.953841020646651,
        "drawdown_pct": -2.566292562129448,
        "normalized_value": 73192.64587980462
      },
      {
        "timestamp": "2024-11-27T06:00:00+00:00",
        "portfolio_value": 110183.68733451348,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5805635499838495,
        "drawdown_pct": -5.1401814517541045,
        "normalized_value": 71358.60640084412
      },
      {
        "timestamp": "2024-12-01T01:00:00+00:00",
        "portfolio_value": 119856.97092722775,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.6391826404705223,
        "drawdown_pct": -0.3085094666345118,
        "normalized_value": 77623.34533992682
      },
      {
        "timestamp": "2024-12-04T20:00:00+00:00",
        "portfolio_value": 125616.9359966008,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.6764517793840068,
        "drawdown_pct": 0.0,
        "normalized_value": 81353.68955158995
      },
      {
        "timestamp": "2024-12-08T15:00:00+00:00",
        "portfolio_value": 122122.49277656295,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6851313017443308,
        "drawdown_pct": -4.140505046440915,
        "normalized_value": 79090.57234829891
      },
      {
        "timestamp": "2024-12-12T10:00:00+00:00",
        "portfolio_value": 118003.05763403037,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -7.374036914969078,
        "normalized_value": 76422.68966946511
      },
      {
        "timestamp": "2024-12-16T05:00:00+00:00",
        "portfolio_value": 115182.90508507677,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -9.587702824403955,
        "normalized_value": 74596.26544461452
      },
      {
        "timestamp": "2024-12-20T00:00:00+00:00",
        "portfolio_value": 114404.32624309033,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.198844743287799,
        "normalized_value": 74092.03199153846
      },
      {
        "timestamp": "2024-12-23T19:00:00+00:00",
        "portfolio_value": 114404.32624309033,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.198844743287799,
        "normalized_value": 74092.03199153846
      },
      {
        "timestamp": "2024-12-27T14:00:00+00:00",
        "portfolio_value": 114404.32624309033,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.198844743287799,
        "normalized_value": 74092.03199153846
      },
      {
        "timestamp": "2024-12-31T09:00:00+00:00",
        "portfolio_value": 114404.32624309033,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.198844743287799,
        "normalized_value": 74092.03199153846
      },
      {
        "timestamp": "2025-01-04T04:00:00+00:00",
        "portfolio_value": 119860.7266214982,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -5.915868097080923,
        "normalized_value": 77625.77765196584
      },
      {
        "timestamp": "2025-01-07T23:00:00+00:00",
        "portfolio_value": 110161.54186782015,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 25.87654562015068,
        "drawdown_pct": -13.529199030719314,
        "normalized_value": 71344.26426291461
      },
      {
        "timestamp": "2025-01-11T18:00:00+00:00",
        "portfolio_value": 110158.41340800616,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 25.85504994238992,
        "drawdown_pct": -13.531654701013402,
        "normalized_value": 71342.23816869044
      },
      {
        "timestamp": "2025-01-15T13:00:00+00:00",
        "portfolio_value": 110155.28234721618,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 25.833572121114667,
        "drawdown_pct": -13.534112412931758,
        "normalized_value": 71340.21038998803
      },
      {
        "timestamp": "2025-01-19T08:00:00+00:00",
        "portfolio_value": 153845.29190755743,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.231883451065542,
        "drawdown_pct": -0.07056676569064047,
        "normalized_value": 99635.30806992331
      },
      {
        "timestamp": "2025-01-23T03:00:00+00:00",
        "portfolio_value": 135682.12729196216,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 26.867591995926475,
        "drawdown_pct": -15.681932582877572,
        "normalized_value": 87872.24090315572
      },
      {
        "timestamp": "2025-01-26T22:00:00+00:00",
        "portfolio_value": 134732.51307007842,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 26.66296037054468,
        "drawdown_pct": -16.272059208831454,
        "normalized_value": 87257.23927150476
      },
      {
        "timestamp": "2025-01-30T17:00:00+00:00",
        "portfolio_value": 120459.42620030658,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.141901722032983,
        "normalized_value": 78013.5153346487
      },
      {
        "timestamp": "2025-02-03T12:00:00+00:00",
        "portfolio_value": 113254.08806393738,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 54.08572343811572,
        "drawdown_pct": -29.61957463939689,
        "normalized_value": 73347.0996383108
      },
      {
        "timestamp": "2025-02-07T07:00:00+00:00",
        "portfolio_value": 113252.57621298265,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 54.04079436220465,
        "drawdown_pct": -29.620514161449428,
        "normalized_value": 73346.12051354356
      },
      {
        "timestamp": "2025-02-11T02:00:00+00:00",
        "portfolio_value": 113251.06310508732,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 53.99590260893128,
        "drawdown_pct": -29.621454464612963,
        "normalized_value": 73345.14057473994
      },
      {
        "timestamp": "2025-02-14T21:00:00+00:00",
        "portfolio_value": 113249.54873920644,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 53.95104814729171,
        "drawdown_pct": -29.62239554953687,
        "normalized_value": 73344.1598212232
      },
      {
        "timestamp": "2025-02-18T16:00:00+00:00",
        "portfolio_value": 113248.03311429411,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 53.90623094630765,
        "drawdown_pct": -29.62333741687111,
        "normalized_value": 73343.17825231599
      },
      {
        "timestamp": "2025-02-22T11:00:00+00:00",
        "portfolio_value": 113246.51622930358,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 53.8614509750267,
        "drawdown_pct": -29.624280067266177,
        "normalized_value": 73342.19586734037
      },
      {
        "timestamp": "2025-02-26T06:00:00+00:00",
        "portfolio_value": 113244.99808318724,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 53.81670820252209,
        "drawdown_pct": -29.6252235013731,
        "normalized_value": 73341.21266561792
      },
      {
        "timestamp": "2025-03-02T01:00:00+00:00",
        "portfolio_value": 113243.4786748966,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 53.77200259789279,
        "drawdown_pct": -29.626167719843444,
        "normalized_value": 73340.22864646955
      },
      {
        "timestamp": "2025-03-05T20:00:00+00:00",
        "portfolio_value": 101890.71907391996,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 30.053324972716307,
        "drawdown_pct": -36.68120708656657,
        "normalized_value": 65987.80540191056
      },
      {
        "timestamp": "2025-03-09T15:00:00+00:00",
        "portfolio_value": 101888.23941389854,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 30.028359639292884,
        "drawdown_pct": -36.68274804221807,
        "normalized_value": 65986.19949192737
      },
      {
        "timestamp": "2025-03-13T10:00:00+00:00",
        "portfolio_value": 101885.75769230796,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 30.00341504460216,
        "drawdown_pct": -36.684290279007556,
        "normalized_value": 65984.59224680367
      },
      {
        "timestamp": "2025-03-17T05:00:00+00:00",
        "portfolio_value": 101883.2739074343,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 29.9784911714164,
        "drawdown_pct": -36.685833798000125,
        "normalized_value": 65982.9836654295
      },
      {
        "timestamp": "2025-03-21T00:00:00+00:00",
        "portfolio_value": 97726.89191003138,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -39.26876866542164,
        "normalized_value": 63291.1729792997
      },
      {
        "timestamp": "2025-03-24T19:00:00+00:00",
        "portfolio_value": 101102.0315996714,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5562513225215795,
        "drawdown_pct": -37.171327671731284,
        "normalized_value": 65477.02526367363
      },
      {
        "timestamp": "2025-03-28T14:00:00+00:00",
        "portfolio_value": 98465.50173569568,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 91.22724962185828,
        "drawdown_pct": -38.80976824791417,
        "normalized_value": 63769.521173196685
      },
      {
        "timestamp": "2025-04-01T09:00:00+00:00",
        "portfolio_value": 98464.727510202,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 91.15146703520008,
        "drawdown_pct": -38.81024938127104,
        "normalized_value": 63769.0197591162
      },
      {
        "timestamp": "2025-04-05T04:00:00+00:00",
        "portfolio_value": 98463.95264102354,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 91.07574740122833,
        "drawdown_pct": -38.810730914638285,
        "normalized_value": 63768.51792816413
      },
      {
        "timestamp": "2025-04-08T23:00:00+00:00",
        "portfolio_value": 98463.17712762515,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 91.0000906676481,
        "drawdown_pct": -38.81121284834845,
        "normalized_value": 63768.01567999389
      },
      {
        "timestamp": "2025-04-12T18:00:00+00:00",
        "portfolio_value": 107435.18581386744,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 82.61039293381091,
        "drawdown_pct": -33.23566322827405,
        "normalized_value": 69578.58575578744
      },
      {
        "timestamp": "2025-04-16T13:00:00+00:00",
        "portfolio_value": 105043.88585472728,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -34.721708554903344,
        "normalized_value": 68029.90067636565
      },
      {
        "timestamp": "2025-04-20T08:00:00+00:00",
        "portfolio_value": 110787.95479214372,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.110021711646292,
        "drawdown_pct": -31.15212425091102,
        "normalized_value": 71749.95002632076
      },
      {
        "timestamp": "2025-04-24T03:00:00+00:00",
        "portfolio_value": 117657.21055682776,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.653269889655602,
        "drawdown_pct": -26.88330578356968,
        "normalized_value": 76198.70764404914
      },
      {
        "timestamp": "2025-04-27T22:00:00+00:00",
        "portfolio_value": 115485.34686861336,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 24.05621425928429,
        "drawdown_pct": -28.232985012062368,
        "normalized_value": 74792.1367637966
      },
      {
        "timestamp": "2025-05-01T17:00:00+00:00",
        "portfolio_value": 116734.09783828918,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5959454387705525,
        "drawdown_pct": -27.456963360944282,
        "normalized_value": 75600.86926398275
      },
      {
        "timestamp": "2025-05-05T12:00:00+00:00",
        "portfolio_value": 113252.43948253576,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.616391093541386,
        "drawdown_pct": -29.620599130982768,
        "normalized_value": 73346.03196237615
      },
      {
        "timestamp": "2025-05-09T07:00:00+00:00",
        "portfolio_value": 125600.30634159267,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.156553832603864,
        "drawdown_pct": -21.94716202427188,
        "normalized_value": 81342.9196360515
      },
      {
        "timestamp": "2025-05-13T02:00:00+00:00",
        "portfolio_value": 125006.99101391336,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 8.894869448866753,
        "drawdown_pct": -22.315870879279935,
        "normalized_value": 80958.66897278481
      },
      {
        "timestamp": "2025-05-16T21:00:00+00:00",
        "portfolio_value": 125397.28479359394,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.073327385273707,
        "normalized_value": 81211.43615528407
      },
      {
        "timestamp": "2025-05-20T16:00:00+00:00",
        "portfolio_value": 118144.62018744524,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5223450079538097,
        "drawdown_pct": -26.580410782478452,
        "normalized_value": 76514.37026914918
      },
      {
        "timestamp": "2025-05-24T11:00:00+00:00",
        "portfolio_value": 126338.4395423352,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6067021050429664,
        "drawdown_pct": -21.488457799734626,
        "normalized_value": 81820.95915185814
      },
      {
        "timestamp": "2025-05-28T06:00:00+00:00",
        "portfolio_value": 127876.92627670662,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5320515663176564,
        "drawdown_pct": -20.532383254190826,
        "normalized_value": 82817.33413246331
      },
      {
        "timestamp": "2025-06-01T01:00:00+00:00",
        "portfolio_value": 121946.72589193365,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -24.21763676414191,
        "normalized_value": 78976.73989050061
      },
      {
        "timestamp": "2025-06-04T20:00:00+00:00",
        "portfolio_value": 121070.2016200327,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -24.762342497505408,
        "normalized_value": 78409.07373199321
      },
      {
        "timestamp": "2025-06-08T15:00:00+00:00",
        "portfolio_value": 120195.30844994965,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.306034601765433,
        "normalized_value": 77842.46392906288
      },
      {
        "timestamp": "2025-06-12T10:00:00+00:00",
        "portfolio_value": 127234.61320696311,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 30.843974142666017,
        "drawdown_pct": -20.931541181608882,
        "normalized_value": 82401.35090809732
      },
      {
        "timestamp": "2025-06-16T05:00:00+00:00",
        "portfolio_value": 115626.54219788918,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -28.145240829903106,
        "normalized_value": 74883.57953696176
      },
      {
        "timestamp": "2025-06-20T00:00:00+00:00",
        "portfolio_value": 111617.17030344807,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -30.636817992224234,
        "normalized_value": 72286.97746408476
      },
      {
        "timestamp": "2025-06-23T19:00:00+00:00",
        "portfolio_value": 111617.17030344807,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -30.636817992224234,
        "normalized_value": 72286.97746408476
      },
      {
        "timestamp": "2025-06-27T14:00:00+00:00",
        "portfolio_value": 107602.04574616117,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -33.131970079432975,
        "normalized_value": 69686.64977615801
      },
      {
        "timestamp": "2025-07-01T09:00:00+00:00",
        "portfolio_value": 109757.35244468236,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5922508502568502,
        "drawdown_pct": -31.792579998089142,
        "normalized_value": 71082.49780133751
      },
      {
        "timestamp": "2025-07-05T04:00:00+00:00",
        "portfolio_value": 106986.83420996332,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -33.51428607653774,
        "normalized_value": 69288.21840281396
      },
      {
        "timestamp": "2025-07-08T23:00:00+00:00",
        "portfolio_value": 104130.4525227282,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 22.845034139703312,
        "drawdown_pct": -35.28935099098349,
        "normalized_value": 67438.33098770888
      },
      {
        "timestamp": "2025-07-12T18:00:00+00:00",
        "portfolio_value": 111520.8182506348,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6428702003625406,
        "drawdown_pct": -30.696694845919243,
        "normalized_value": 72224.57668245415
      },
      {
        "timestamp": "2025-07-16T13:00:00+00:00",
        "portfolio_value": 116745.22873073284,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.593214378342095,
        "drawdown_pct": -27.450046198321548,
        "normalized_value": 75608.07799870557
      },
      {
        "timestamp": "2025-07-20T08:00:00+00:00",
        "portfolio_value": 128823.28182882303,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 4.021164925169464,
        "drawdown_pct": -19.944281690362775,
        "normalized_value": 83430.22534161033
      },
      {
        "timestamp": "2025-07-24T03:00:00+00:00",
        "portfolio_value": 137171.39670954068,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 12.213013833017463,
        "drawdown_pct": -14.756443561884872,
        "normalized_value": 88836.74111879256
      },
      {
        "timestamp": "2025-07-27T22:00:00+00:00",
        "portfolio_value": 140192.5961175332,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5777662902052536,
        "drawdown_pct": -12.878954607016654,
        "normalized_value": 90793.36995041699
      },
      {
        "timestamp": "2025-07-31T17:00:00+00:00",
        "portfolio_value": 136334.8775623724,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.6835063875421676,
        "drawdown_pct": -15.276288579459477,
        "normalized_value": 88294.98360446717
      },
      {
        "timestamp": "2025-08-04T12:00:00+00:00",
        "portfolio_value": 132721.15008756256,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -17.521997155302046,
        "normalized_value": 85954.61396579293
      },
      {
        "timestamp": "2025-08-08T07:00:00+00:00",
        "portfolio_value": 138674.6565724103,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.164174461878742,
        "drawdown_pct": -13.822260342674655,
        "normalized_value": 89810.30201031575
      },
      {
        "timestamp": "2025-08-12T02:00:00+00:00",
        "portfolio_value": 151695.39315789408,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 18.640599723532134,
        "drawdown_pct": -5.730676232462581,
        "normalized_value": 98242.96241159429
      },
      {
        "timestamp": "2025-08-15T21:00:00+00:00",
        "portfolio_value": 155752.3038526362,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 18.25088762023894,
        "drawdown_pct": -11.547140970127792,
        "normalized_value": 100870.35218654871
      },
      {
        "timestamp": "2025-08-19T16:00:00+00:00",
        "portfolio_value": 149053.44092771443,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -15.351473640026295,
        "normalized_value": 96531.94661711587
      },
      {
        "timestamp": "2025-08-23T11:00:00+00:00",
        "portfolio_value": 153455.6370700344,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5171725215216836,
        "drawdown_pct": -12.851434634716357,
        "normalized_value": 99382.954687534
      },
      {
        "timestamp": "2025-08-27T06:00:00+00:00",
        "portfolio_value": 144462.18886595257,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 11.37549290149694,
        "drawdown_pct": -17.95887886835517,
        "normalized_value": 93558.49966967737
      },
      {
        "timestamp": "2025-08-31T01:00:00+00:00",
        "portfolio_value": 147457.92283100603,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 11.49713277445949,
        "drawdown_pct": -16.257579898469093,
        "normalized_value": 95498.63623676181
      },
      {
        "timestamp": "2025-09-03T20:00:00+00:00",
        "portfolio_value": 149463.6850227537,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.101942691651209,
        "drawdown_pct": -15.118493053487567,
        "normalized_value": 96797.63428481302
      },
      {
        "timestamp": "2025-09-07T15:00:00+00:00",
        "portfolio_value": 146038.12410896958,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.20209953644956,
        "drawdown_pct": -17.06389385405529,
        "normalized_value": 94579.12754518363
      },
      {
        "timestamp": "2025-09-11T10:00:00+00:00",
        "portfolio_value": 161751.850819819,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 11.21447740002058,
        "drawdown_pct": -8.13995488681044,
        "normalized_value": 104755.85757279361
      },
      {
        "timestamp": "2025-09-15T05:00:00+00:00",
        "portfolio_value": 175682.88305638926,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 12.10951608439554,
        "drawdown_pct": -2.011462654026418,
        "normalized_value": 113778.05559661578
      },
      {
        "timestamp": "2025-09-19T00:00:00+00:00",
        "portfolio_value": 164442.73853096325,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.27593925454048,
        "drawdown_pct": -8.280743431086107,
        "normalized_value": 106498.56560602045
      },
      {
        "timestamp": "2025-09-22T19:00:00+00:00",
        "portfolio_value": 152976.6718739942,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.676034089352505,
        "normalized_value": 99072.7609580378
      },
      {
        "timestamp": "2025-09-26T14:00:00+00:00",
        "portfolio_value": 152976.6718739942,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.676034089352505,
        "normalized_value": 99072.7609580378
      },
      {
        "timestamp": "2025-09-30T09:00:00+00:00",
        "portfolio_value": 152976.6718739942,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -14.676034089352505,
        "normalized_value": 99072.7609580378
      },
      {
        "timestamp": "2025-10-04T04:00:00+00:00",
        "portfolio_value": 161194.82639195395,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.5512169792759287,
        "drawdown_pct": -10.092292481243728,
        "normalized_value": 104395.11009859625
      },
      {
        "timestamp": "2025-10-07T23:00:00+00:00",
        "portfolio_value": 161193.52620160347,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.093017672841782,
        "normalized_value": 104394.26805224882
      },
      {
        "timestamp": "2025-10-11T18:00:00+00:00",
        "portfolio_value": 161193.52620160347,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.093017672841782,
        "normalized_value": 104394.26805224882
      },
      {
        "timestamp": "2025-10-15T13:00:00+00:00",
        "portfolio_value": 161193.52620160347,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.093017672841782,
        "normalized_value": 104394.26805224882
      },
      {
        "timestamp": "2025-10-19T08:00:00+00:00",
        "portfolio_value": 161193.52620160347,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.093017672841782,
        "normalized_value": 104394.26805224882
      },
      {
        "timestamp": "2025-10-23T03:00:00+00:00",
        "portfolio_value": 161193.52620160347,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.093017672841782,
        "normalized_value": 104394.26805224882
      },
      {
        "timestamp": "2025-10-26T22:00:00+00:00",
        "portfolio_value": 161193.52620160347,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.093017672841782,
        "normalized_value": 104394.26805224882
      },
      {
        "timestamp": "2025-10-30T17:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-03T12:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-07T07:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-11T02:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-14T21:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-18T16:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-22T11:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-26T06:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-11-30T01:00:00+00:00",
        "portfolio_value": 158071.8633635718,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -11.834150162612124,
        "normalized_value": 102372.57577488876
      },
      {
        "timestamp": "2025-12-03T20:00:00+00:00",
        "portfolio_value": 160553.80824839996,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -10.449825493033643,
        "normalized_value": 103979.96551133264
      },
      {
        "timestamp": "2025-12-07T15:00:00+00:00",
        "portfolio_value": 155364.34116236298,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 55.85901357199885,
        "drawdown_pct": -13.344292389974413,
        "normalized_value": 100619.09469478072
      },
      {
        "timestamp": "2025-12-11T10:00:00+00:00",
        "portfolio_value": 152796.41333657267,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 15.36020648517146,
        "drawdown_pct": -14.776574734626355,
        "normalized_value": 98956.01955707885
      },
      {
        "timestamp": "2025-12-15T05:00:00+00:00",
        "portfolio_value": 144994.87775418838,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.12794378931699,
        "normalized_value": 93903.48664215289
      },
      {
        "timestamp": "2025-12-19T00:00:00+00:00",
        "portfolio_value": 144994.87775418838,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -19.12794378931699,
        "normalized_value": 93903.48664215289
      },
      {
        "timestamp": "2025-12-22T19:00:00+00:00",
        "portfolio_value": 143010.2176774192,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.234903868001144,
        "normalized_value": 92618.15502289348
      },
      {
        "timestamp": "2025-12-26T14:00:00+00:00",
        "portfolio_value": 142669.23616462806,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -20.425089042100865,
        "normalized_value": 92397.32409819085
      },
      {
        "timestamp": "2025-12-30T09:00:00+00:00",
        "portfolio_value": 138930.82716973743,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -22.510216648373046,
        "normalized_value": 89976.20657630324
      },
      {
        "timestamp": "2026-01-03T04:00:00+00:00",
        "portfolio_value": 138035.85204762852,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.680819697253314,
        "drawdown_pct": -23.009396202185002,
        "normalized_value": 89396.59103590816
      },
      {
        "timestamp": "2026-01-06T23:00:00+00:00",
        "portfolio_value": 145980.3682579734,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5624465828822154,
        "drawdown_pct": -18.578278555264298,
        "normalized_value": 94541.72294257606
      },
      {
        "timestamp": "2026-01-10T18:00:00+00:00",
        "portfolio_value": 140010.56273208783,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5543149432661596,
        "drawdown_pct": -21.907985476875318,
        "normalized_value": 90675.47909907551
      },
      {
        "timestamp": "2026-01-14T13:00:00+00:00",
        "portfolio_value": 146924.37566576936,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 5.086908944244838,
        "drawdown_pct": -18.0517508507753,
        "normalized_value": 95153.09341566508
      },
      {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "portfolio_value": 145251.2384247478,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 1.58162038905572,
        "drawdown_pct": -18.984956568797116,
        "normalized_value": 94069.51430586292
      },
      {
        "timestamp": "2026-01-22T03:00:00+00:00",
        "portfolio_value": 140233.7697000228,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.783489999951954,
        "normalized_value": 90820.03532655445
      },
      {
        "timestamp": "2026-01-25T22:00:00+00:00",
        "portfolio_value": 140233.7697000228,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -21.783489999951954,
        "normalized_value": 90820.03532655445
      },
      {
        "timestamp": "2026-01-29T17:00:00+00:00",
        "portfolio_value": 136633.5360512296,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.791549198501016,
        "normalized_value": 88488.40473667099
      },
      {
        "timestamp": "2026-02-02T12:00:00+00:00",
        "portfolio_value": 136633.5360512296,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.791549198501016,
        "normalized_value": 88488.40473667099
      },
      {
        "timestamp": "2026-02-06T07:00:00+00:00",
        "portfolio_value": 136633.53605122957,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.791549198501034,
        "normalized_value": 88488.40473667096
      },
      {
        "timestamp": "2026-02-10T02:00:00+00:00",
        "portfolio_value": 136633.5360512296,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.791549198501016,
        "normalized_value": 88488.40473667099
      },
      {
        "timestamp": "2026-02-13T21:00:00+00:00",
        "portfolio_value": 136633.5360512296,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -23.791549198501016,
        "normalized_value": 88488.40473667099
      },
      {
        "timestamp": "2026-02-17T16:00:00+00:00",
        "portfolio_value": 133404.7953684486,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.592405215154624,
        "normalized_value": 86397.36529946752
      },
      {
        "timestamp": "2026-02-21T11:00:00+00:00",
        "portfolio_value": 133404.7953684486,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.592405215154624,
        "normalized_value": 86397.36529946752
      },
      {
        "timestamp": "2026-02-25T06:00:00+00:00",
        "portfolio_value": 133404.7953684486,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -25.592405215154624,
        "normalized_value": 86397.36529946752
      },
      {
        "timestamp": "2026-03-01T01:00:00+00:00",
        "portfolio_value": 129108.23893312224,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -27.988844033766945,
        "normalized_value": 83614.77300323539
      },
      {
        "timestamp": "2026-03-04T20:00:00+00:00",
        "portfolio_value": 120924.3350397006,
        "target_long_fraction": 1.025,
        "target_short_fraction": 0.0,
        "health_factor": 1.5532143369892126,
        "drawdown_pct": -32.55348208128241,
        "normalized_value": 78314.60570188132
      },
      {
        "timestamp": "2026-03-08T15:00:00+00:00",
        "portfolio_value": 107551.15472567624,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": 8.688624541001722,
        "drawdown_pct": -40.012480680563186,
        "normalized_value": 69653.69106522744
      },
      {
        "timestamp": "2026-03-12T10:00:00+00:00",
        "portfolio_value": 110538.70107453511,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 4.725820449645115,
        "drawdown_pct": -38.346152738506234,
        "normalized_value": 71588.52506080132
      },
      {
        "timestamp": "2026-03-16T05:00:00+00:00",
        "portfolio_value": 120029.90184860854,
        "target_long_fraction": 1.075,
        "target_short_fraction": 0.0,
        "health_factor": 10.293446021335273,
        "drawdown_pct": -33.05235936872237,
        "normalized_value": 77735.34113396739
      },
      {
        "timestamp": "2026-03-20T00:00:00+00:00",
        "portfolio_value": 113048.64212645794,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -36.946212982146534,
        "normalized_value": 73214.04604259352
      },
      {
        "timestamp": "2026-03-23T19:00:00+00:00",
        "portfolio_value": 113048.64212645794,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -36.946212982146534,
        "normalized_value": 73214.04604259352
      },
      {
        "timestamp": "2026-03-27T14:00:00+00:00",
        "portfolio_value": 104979.05173175802,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -41.4470917587315,
        "normalized_value": 67987.91195031893
      },
      {
        "timestamp": "2026-03-31T09:00:00+00:00",
        "portfolio_value": 104979.05173175802,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -41.4470917587315,
        "normalized_value": 67987.91195031893
      },
      {
        "timestamp": "2026-04-04T04:00:00+00:00",
        "portfolio_value": 101137.50761320542,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -43.589743807572106,
        "normalized_value": 65500.00070538997
      },
      {
        "timestamp": "2026-04-07T23:00:00+00:00",
        "portfolio_value": 101822.39847389676,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -43.2077404358536,
        "normalized_value": 65943.55871781365
      },
      {
        "timestamp": "2026-04-11T18:00:00+00:00",
        "portfolio_value": 101697.07875087204,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -43.2776385166806,
        "normalized_value": 65862.39751322956
      },
      {
        "timestamp": "2026-04-15T13:00:00+00:00",
        "portfolio_value": 100722.38075343824,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -43.82128414373882,
        "normalized_value": 65231.15079748471
      },
      {
        "timestamp": "2026-04-19T08:00:00+00:00",
        "portfolio_value": 101294.73595338668,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -43.50204843955636,
        "normalized_value": 65601.82698760532
      },
      {
        "timestamp": "2026-04-23T03:00:00+00:00",
        "portfolio_value": 96037.33953048692,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.43440346905886,
        "normalized_value": 62196.96288194205
      },
      {
        "timestamp": "2026-04-26T22:00:00+00:00",
        "portfolio_value": 96067.36967143118,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": 1.89377811490502,
        "drawdown_pct": -46.41765391704657,
        "normalized_value": 62216.41139614268
      },
      {
        "timestamp": "2026-04-30T17:00:00+00:00",
        "portfolio_value": 94604.77047393112,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -47.233430352366256,
        "normalized_value": 61269.18369863663
      },
      {
        "timestamp": "2026-05-04T12:00:00+00:00",
        "portfolio_value": 92687.63037649542,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -48.302730620934504,
        "normalized_value": 60027.58025499023
      },
      {
        "timestamp": "2026-05-08T07:00:00+00:00",
        "portfolio_value": 93753.16668328732,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": 1.8894055791273048,
        "drawdown_pct": -47.70841919813041,
        "normalized_value": 60717.65686942892
      },
      {
        "timestamp": "2026-05-12T02:00:00+00:00",
        "portfolio_value": 101227.47513234043,
        "target_long_fraction": 0.85,
        "target_short_fraction": 0.0,
        "health_factor": 1.9684573388044049,
        "drawdown_pct": -43.53956370205861,
        "normalized_value": 65558.26665148528
      },
      {
        "timestamp": "2026-05-15T21:00:00+00:00",
        "portfolio_value": 95988.04193610512,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.461899597788694,
        "normalized_value": 62165.036126547486
      },
      {
        "timestamp": "2026-05-19T16:00:00+00:00",
        "portfolio_value": 95988.04193610512,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.461899597788694,
        "normalized_value": 62165.036126547486
      },
      {
        "timestamp": "2026-05-23T11:00:00+00:00",
        "portfolio_value": 95988.04193610512,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.461899597788694,
        "normalized_value": 62165.036126547486
      },
      {
        "timestamp": "2026-05-27T06:00:00+00:00",
        "portfolio_value": 95988.04193610512,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.461899597788694,
        "normalized_value": 62165.036126547486
      },
      {
        "timestamp": "2026-05-31T01:00:00+00:00",
        "portfolio_value": 95988.04193610512,
        "target_long_fraction": 0.0,
        "target_short_fraction": 0.0,
        "health_factor": null,
        "drawdown_pct": -46.461899597788694,
        "normalized_value": 62165.036126547486
      }
    ],
    "controlEquity": [
      {
        "timestamp": "2021-01-01T00:00:00+00:00",
        "portfolio_value": 9989.23923923924,
        "drawdown_pct": 0.0,
        "normalized_value": 100.0
      },
      {
        "timestamp": "2021-01-04T19:00:00+00:00",
        "portfolio_value": 16272.441894652273,
        "drawdown_pct": 0.0,
        "normalized_value": 162.89971142879094
      },
      {
        "timestamp": "2021-01-08T14:00:00+00:00",
        "portfolio_value": 25581.76923188909,
        "drawdown_pct": -6.542725525168266,
        "normalized_value": 256.0932681579999
      },
      {
        "timestamp": "2021-01-12T09:00:00+00:00",
        "portfolio_value": 25174.520615510555,
        "drawdown_pct": -12.549343440794233,
        "normalized_value": 252.0163949685101
      },
      {
        "timestamp": "2021-01-16T04:00:00+00:00",
        "portfolio_value": 26061.719945669625,
        "drawdown_pct": -10.425020249864044,
        "normalized_value": 260.8979454941399
      },
      {
        "timestamp": "2021-01-19T23:00:00+00:00",
        "portfolio_value": 26928.22270417603,
        "drawdown_pct": -9.424056854014214,
        "normalized_value": 269.5723073524749
      },
      {
        "timestamp": "2021-01-23T18:00:00+00:00",
        "portfolio_value": 24275.342361322957,
        "drawdown_pct": -18.34730224406736,
        "normalized_value": 243.01492616140123
      },
      {
        "timestamp": "2021-01-27T13:00:00+00:00",
        "portfolio_value": 25378.80755286202,
        "drawdown_pct": -14.635679626028155,
        "normalized_value": 254.06146499295198
      },
      {
        "timestamp": "2021-01-31T08:00:00+00:00",
        "portfolio_value": 32762.244577192567,
        "drawdown_pct": 0.0,
        "normalized_value": 327.9753722235175
      },
      {
        "timestamp": "2021-02-04T03:00:00+00:00",
        "portfolio_value": 41684.207754528135,
        "drawdown_pct": -2.3404502505472373,
        "normalized_value": 417.2911145303866
      },
      {
        "timestamp": "2021-02-07T22:00:00+00:00",
        "portfolio_value": 43476.400194004775,
        "drawdown_pct": -12.270733201741239,
        "normalized_value": 435.23234505409494
      },
      {
        "timestamp": "2021-02-11T18:00:00+00:00",
        "portfolio_value": 59078.59482495813,
        "drawdown_pct": -5.156666878503699,
        "normalized_value": 591.4223637060217
      },
      {
        "timestamp": "2021-02-15T13:00:00+00:00",
        "portfolio_value": 54935.18894383756,
        "drawdown_pct": -14.719689654138133,
        "normalized_value": 549.9436706655682
      },
      {
        "timestamp": "2021-02-19T08:00:00+00:00",
        "portfolio_value": 48495.2378841098,
        "drawdown_pct": -24.716943427984532,
        "normalized_value": 485.4747866445444
      },
      {
        "timestamp": "2021-02-23T03:00:00+00:00",
        "portfolio_value": 72200.25109654163,
        "drawdown_pct": -4.489320815007864,
        "normalized_value": 722.7802775303263
      },
      {
        "timestamp": "2021-02-26T22:00:00+00:00",
        "portfolio_value": 77552.10945231492,
        "drawdown_pct": -18.862452203088452,
        "normalized_value": 776.3565131935026
      },
      {
        "timestamp": "2021-03-02T17:00:00+00:00",
        "portfolio_value": 74892.67455973063,
        "drawdown_pct": -21.644839777503957,
        "normalized_value": 749.7335158971956
      },
      {
        "timestamp": "2021-03-06T13:00:00+00:00",
        "portfolio_value": 67290.98483401156,
        "drawdown_pct": -29.59797564722609,
        "normalized_value": 673.6347305576826
      },
      {
        "timestamp": "2021-03-10T08:00:00+00:00",
        "portfolio_value": 74948.80071164633,
        "drawdown_pct": -21.586118765711017,
        "normalized_value": 750.2953820270529
      },
      {
        "timestamp": "2021-03-14T03:00:00+00:00",
        "portfolio_value": 76619.43316710551,
        "drawdown_pct": -19.83824857026213,
        "normalized_value": 767.019703223573
      },
      {
        "timestamp": "2021-03-17T22:00:00+00:00",
        "portfolio_value": 73481.54605156941,
        "drawdown_pct": -23.121208474462687,
        "normalized_value": 735.6070296416849
      },
      {
        "timestamp": "2021-03-21T17:00:00+00:00",
        "portfolio_value": 71527.08733779522,
        "drawdown_pct": -25.166026963938474,
        "normalized_value": 716.0413883854742
      },
      {
        "timestamp": "2021-03-25T12:00:00+00:00",
        "portfolio_value": 75124.17455021605,
        "drawdown_pct": -21.402636932531124,
        "normalized_value": 752.0510096015815
      },
      {
        "timestamp": "2021-03-29T07:00:00+00:00",
        "portfolio_value": 95234.88713566562,
        "drawdown_pct": -6.0899550889846426,
        "normalized_value": 953.3747751437227
      },
      {
        "timestamp": "2021-04-02T02:00:00+00:00",
        "portfolio_value": 106031.35716752376,
        "drawdown_pct": -3.1898487001138003,
        "normalized_value": 1061.455778844665
      },
      {
        "timestamp": "2021-04-05T21:00:00+00:00",
        "portfolio_value": 138059.2181966662,
        "drawdown_pct": -3.4095674559529527,
        "normalized_value": 1382.0794045491348
      },
      {
        "timestamp": "2021-04-09T16:00:00+00:00",
        "portfolio_value": 163849.83979241882,
        "drawdown_pct": -1.9799162787782754,
        "normalized_value": 1640.2634461770813
      },
      {
        "timestamp": "2021-04-13T11:00:00+00:00",
        "portfolio_value": 149848.66205726177,
        "drawdown_pct": -10.71986293589801,
        "normalized_value": 1500.1008432016886
      },
      {
        "timestamp": "2021-04-17T06:00:00+00:00",
        "portfolio_value": 145931.26063611722,
        "drawdown_pct": -13.05385865540096,
        "normalized_value": 1460.8846293607346
      },
      {
        "timestamp": "2021-04-21T03:00:00+00:00",
        "portfolio_value": 158543.57706101018,
        "drawdown_pct": -6.822692920614669,
        "normalized_value": 1587.1436579296958
      },
      {
        "timestamp": "2021-04-24T22:00:00+00:00",
        "portfolio_value": 188727.610047474,
        "drawdown_pct": -3.6291867668714595,
        "normalized_value": 1889.3091408415114
      },
      {
        "timestamp": "2021-04-28T20:00:00+00:00",
        "portfolio_value": 207614.62349884896,
        "drawdown_pct": -3.3800315766655653,
        "normalized_value": 2078.382732924319
      },
      {
        "timestamp": "2021-05-02T15:00:00+00:00",
        "portfolio_value": 215737.09106271176,
        "drawdown_pct": -3.9910650865662043,
        "normalized_value": 2159.6949066477846
      },
      {
        "timestamp": "2021-05-06T10:00:00+00:00",
        "portfolio_value": 209206.78707888068,
        "drawdown_pct": -6.897229840434721,
        "normalized_value": 2094.321520072168
      },
      {
        "timestamp": "2021-05-10T05:00:00+00:00",
        "portfolio_value": 250725.29122851585,
        "drawdown_pct": 0.0,
        "normalized_value": 2509.9538135359603
      },
      {
        "timestamp": "2021-05-14T00:00:00+00:00",
        "portfolio_value": 218013.8975970385,
        "drawdown_pct": -17.274374715768808,
        "normalized_value": 2182.487498553914
      },
      {
        "timestamp": "2021-05-17T19:00:00+00:00",
        "portfolio_value": 230330.9752165951,
        "drawdown_pct": -12.600645384826967,
        "normalized_value": 2305.79095865299
      },
      {
        "timestamp": "2021-05-21T14:00:00+00:00",
        "portfolio_value": 216964.9968869229,
        "drawdown_pct": -18.985585760080166,
        "normalized_value": 2171.9871923244327
      },
      {
        "timestamp": "2021-05-25T09:00:00+00:00",
        "portfolio_value": 216964.99688692292,
        "drawdown_pct": -18.98558576008016,
        "normalized_value": 2171.987192324433
      },
      {
        "timestamp": "2021-05-29T04:00:00+00:00",
        "portfolio_value": 202064.21683806804,
        "drawdown_pct": -24.549515355622315,
        "normalized_value": 2022.8188753787106
      },
      {
        "timestamp": "2021-06-01T23:00:00+00:00",
        "portfolio_value": 192105.1733739668,
        "drawdown_pct": -28.26820769867599,
        "normalized_value": 1923.1211584096282
      },
      {
        "timestamp": "2021-06-05T18:00:00+00:00",
        "portfolio_value": 189748.3339825095,
        "drawdown_pct": -29.148248099183515,
        "normalized_value": 1899.5273757900343
      },
      {
        "timestamp": "2021-06-09T13:00:00+00:00",
        "portfolio_value": 189001.88869324917,
        "drawdown_pct": -29.426969684412725,
        "normalized_value": 1892.0548819255548
      },
      {
        "timestamp": "2021-06-13T08:00:00+00:00",
        "portfolio_value": 189001.88869324917,
        "drawdown_pct": -29.426969684412725,
        "normalized_value": 1892.0548819255548
      },
      {
        "timestamp": "2021-06-17T03:00:00+00:00",
        "portfolio_value": 183170.48315272757,
        "drawdown_pct": -31.604408030870974,
        "normalized_value": 1833.6780085635176
      },
      {
        "timestamp": "2021-06-20T22:00:00+00:00",
        "portfolio_value": 176883.34532629943,
        "drawdown_pct": -33.95201614997775,
        "normalized_value": 1770.7389030335257
      },
      {
        "timestamp": "2021-06-24T17:00:00+00:00",
        "portfolio_value": 176883.34532629943,
        "drawdown_pct": -33.95201614997775,
        "normalized_value": 1770.7389030335257
      },
      {
        "timestamp": "2021-06-28T12:00:00+00:00",
        "portfolio_value": 176883.34532629943,
        "drawdown_pct": -33.95201614997775,
        "normalized_value": 1770.7389030335257
      },
      {
        "timestamp": "2021-07-02T07:00:00+00:00",
        "portfolio_value": 161453.76471093044,
        "drawdown_pct": -39.71339910786207,
        "normalized_value": 1616.2768839964876
      },
      {
        "timestamp": "2021-07-06T02:00:00+00:00",
        "portfolio_value": 159448.1767563442,
        "drawdown_pct": -40.46228273277288,
        "normalized_value": 1596.1993995499447
      },
      {
        "timestamp": "2021-07-09T21:00:00+00:00",
        "portfolio_value": 165429.4946850645,
        "drawdown_pct": -38.22886731862406,
        "normalized_value": 1656.0770117030781
      },
      {
        "timestamp": "2021-07-13T16:00:00+00:00",
        "portfolio_value": 165429.4946850645,
        "drawdown_pct": -38.22886731862406,
        "normalized_value": 1656.0770117030781
      },
      {
        "timestamp": "2021-07-17T11:00:00+00:00",
        "portfolio_value": 165429.4946850645,
        "drawdown_pct": -38.22886731862406,
        "normalized_value": 1656.0770117030781
      },
      {
        "timestamp": "2021-07-21T06:00:00+00:00",
        "portfolio_value": 165429.4946850645,
        "drawdown_pct": -38.22886731862406,
        "normalized_value": 1656.0770117030781
      },
      {
        "timestamp": "2021-07-25T01:00:00+00:00",
        "portfolio_value": 162063.44354454026,
        "drawdown_pct": -39.48574591823204,
        "normalized_value": 1622.3802400079737
      },
      {
        "timestamp": "2021-07-28T20:00:00+00:00",
        "portfolio_value": 167131.19569794455,
        "drawdown_pct": -37.593454635711865,
        "normalized_value": 1673.1123531552632
      },
      {
        "timestamp": "2021-08-01T15:00:00+00:00",
        "portfolio_value": 204364.06522256797,
        "drawdown_pct": -23.69075531421345,
        "normalized_value": 2045.8421340015072
      },
      {
        "timestamp": "2021-08-05T10:00:00+00:00",
        "portfolio_value": 206037.4450384857,
        "drawdown_pct": -23.06591772505093,
        "normalized_value": 2062.593958398148
      },
      {
        "timestamp": "2021-08-09T05:00:00+00:00",
        "portfolio_value": 214732.1804800488,
        "drawdown_pct": -19.81931615854838,
        "normalized_value": 2149.634975569995
      },
      {
        "timestamp": "2021-08-13T00:00:00+00:00",
        "portfolio_value": 238361.3616036323,
        "drawdown_pct": -10.996214298049745,
        "normalized_value": 2386.181328677292
      },
      {
        "timestamp": "2021-08-16T23:00:00+00:00",
        "portfolio_value": 364260.9845457747,
        "drawdown_pct": -10.315348117692043,
        "normalized_value": 3646.533793233248
      },
      {
        "timestamp": "2021-08-20T18:00:00+00:00",
        "portfolio_value": 453726.5087811431,
        "drawdown_pct": -2.9699925552868365,
        "normalized_value": 4542.152789762376
      },
      {
        "timestamp": "2021-08-24T13:00:00+00:00",
        "portfolio_value": 432844.7627438052,
        "drawdown_pct": -9.496528737570243,
        "normalized_value": 4333.110383857117
      },
      {
        "timestamp": "2021-08-28T08:00:00+00:00",
        "portfolio_value": 426626.9073597181,
        "drawdown_pct": -10.796619542644265,
        "normalized_value": 4270.864849085436
      },
      {
        "timestamp": "2021-09-01T03:00:00+00:00",
        "portfolio_value": 539064.580835614,
        "drawdown_pct": -12.215571509927415,
        "normalized_value": 5396.452802112167
      },
      {
        "timestamp": "2021-09-04T22:00:00+00:00",
        "portfolio_value": 683438.1974469528,
        "drawdown_pct": -6.184549583641705,
        "normalized_value": 6841.744211734406
      },
      {
        "timestamp": "2021-09-08T17:00:00+00:00",
        "portfolio_value": 763716.2468817366,
        "drawdown_pct": -14.780007066683002,
        "normalized_value": 7645.389489539341
      },
      {
        "timestamp": "2021-09-12T12:00:00+00:00",
        "portfolio_value": 653796.7959800113,
        "drawdown_pct": -27.045471978980025,
        "normalized_value": 6545.010889435894
      },
      {
        "timestamp": "2021-09-16T07:00:00+00:00",
        "portfolio_value": 583125.5812422666,
        "drawdown_pct": -34.931385687284504,
        "normalized_value": 5837.537446812379
      },
      {
        "timestamp": "2021-09-20T02:00:00+00:00",
        "portfolio_value": 575326.8635467291,
        "drawdown_pct": -35.801612907952176,
        "normalized_value": 5759.466259319913
      },
      {
        "timestamp": "2021-09-23T21:00:00+00:00",
        "portfolio_value": 575326.8635467291,
        "drawdown_pct": -35.801612907952176,
        "normalized_value": 5759.466259319913
      },
      {
        "timestamp": "2021-09-27T16:00:00+00:00",
        "portfolio_value": 558900.1290441429,
        "drawdown_pct": -37.63460546761504,
        "normalized_value": 5595.021959717401
      },
      {
        "timestamp": "2021-10-01T13:00:00+00:00",
        "portfolio_value": 572508.2959449575,
        "drawdown_pct": -36.116125414509135,
        "normalized_value": 5731.2502206980735
      },
      {
        "timestamp": "2021-10-05T08:00:00+00:00",
        "portfolio_value": 643711.2378541984,
        "drawdown_pct": -28.170878431598577,
        "normalized_value": 6444.046662989144
      },
      {
        "timestamp": "2021-10-09T03:00:00+00:00",
        "portfolio_value": 578576.9424344474,
        "drawdown_pct": -35.438949810270344,
        "normalized_value": 5792.002059192954
      },
      {
        "timestamp": "2021-10-12T22:00:00+00:00",
        "portfolio_value": 556751.0305907693,
        "drawdown_pct": -37.87441463202302,
        "normalized_value": 5573.507824337285
      },
      {
        "timestamp": "2021-10-16T17:00:00+00:00",
        "portfolio_value": 557400.8848629061,
        "drawdown_pct": -37.80190003421878,
        "normalized_value": 5580.013367518031
      },
      {
        "timestamp": "2021-10-20T12:00:00+00:00",
        "portfolio_value": 531711.8023867338,
        "drawdown_pct": -40.668440370399175,
        "normalized_value": 5322.845810901091
      },
      {
        "timestamp": "2021-10-24T07:00:00+00:00",
        "portfolio_value": 689296.616234672,
        "drawdown_pct": -23.084191276116197,
        "normalized_value": 6900.391508564646
      },
      {
        "timestamp": "2021-10-28T02:00:00+00:00",
        "portfolio_value": 670838.0737105039,
        "drawdown_pct": -25.14390503746627,
        "normalized_value": 6715.607241393826
      },
      {
        "timestamp": "2021-10-31T21:00:00+00:00",
        "portfolio_value": 692502.4048852818,
        "drawdown_pct": -22.726470346037598,
        "normalized_value": 6932.483928956551
      },
      {
        "timestamp": "2021-11-04T16:00:00+00:00",
        "portfolio_value": 796515.8674568277,
        "drawdown_pct": -11.120030674878437,
        "normalized_value": 7973.739024369274
      },
      {
        "timestamp": "2021-11-08T11:00:00+00:00",
        "portfolio_value": 808417.2600345052,
        "drawdown_pct": -9.792002633695015,
        "normalized_value": 8092.881156143704
      },
      {
        "timestamp": "2021-11-12T06:00:00+00:00",
        "portfolio_value": 775463.0777422853,
        "drawdown_pct": -13.469225939514113,
        "normalized_value": 7762.984339149165
      },
      {
        "timestamp": "2021-11-16T01:00:00+00:00",
        "portfolio_value": 771981.2356968268,
        "drawdown_pct": -13.85775054629107,
        "normalized_value": 7728.128411064259
      },
      {
        "timestamp": "2021-11-19T20:00:00+00:00",
        "portfolio_value": 771981.2356968268,
        "drawdown_pct": -13.85775054629107,
        "normalized_value": 7728.128411064259
      },
      {
        "timestamp": "2021-11-23T15:00:00+00:00",
        "portfolio_value": 771981.2356968268,
        "drawdown_pct": -13.85775054629107,
        "normalized_value": 7728.128411064259
      },
      {
        "timestamp": "2021-11-27T10:00:00+00:00",
        "portfolio_value": 771981.2356968268,
        "drawdown_pct": -13.85775054629107,
        "normalized_value": 7728.128411064259
      },
      {
        "timestamp": "2021-12-01T05:00:00+00:00",
        "portfolio_value": 795649.3537747496,
        "drawdown_pct": -11.21672141592896,
        "normalized_value": 7965.064553157549
      },
      {
        "timestamp": "2021-12-05T00:00:00+00:00",
        "portfolio_value": 800511.0253247079,
        "drawdown_pct": -10.674227241131456,
        "normalized_value": 8013.733640297449
      },
      {
        "timestamp": "2021-12-08T19:00:00+00:00",
        "portfolio_value": 800511.0253247079,
        "drawdown_pct": -10.674227241131456,
        "normalized_value": 8013.733640297449
      },
      {
        "timestamp": "2021-12-12T14:00:00+00:00",
        "portfolio_value": 800511.0253247079,
        "drawdown_pct": -10.674227241131456,
        "normalized_value": 8013.733640297449
      },
      {
        "timestamp": "2021-12-16T09:00:00+00:00",
        "portfolio_value": 811343.1534473593,
        "drawdown_pct": -9.465514075954951,
        "normalized_value": 8122.171609027852
      },
      {
        "timestamp": "2021-12-20T04:00:00+00:00",
        "portfolio_value": 815860.0181370937,
        "drawdown_pct": -8.961495497704746,
        "normalized_value": 8167.3889131844235
      },
      {
        "timestamp": "2021-12-23T23:00:00+00:00",
        "portfolio_value": 810056.7017120572,
        "drawdown_pct": -9.609063998114035,
        "normalized_value": 8109.293233563095
      },
      {
        "timestamp": "2021-12-27T18:00:00+00:00",
        "portfolio_value": 862915.2140586841,
        "drawdown_pct": -3.710797374826212,
        "normalized_value": 8638.447767563948
      },
      {
        "timestamp": "2021-12-31T13:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-04T08:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-08T03:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-11T22:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-15T17:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-19T12:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-23T07:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-27T02:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-01-30T21:00:00+00:00",
        "portfolio_value": 831092.8550240434,
        "drawdown_pct": -7.261725122044221,
        "normalized_value": 8319.881375544448
      },
      {
        "timestamp": "2022-02-03T16:00:00+00:00",
        "portfolio_value": 818082.1165699511,
        "drawdown_pct": -8.713540562192476,
        "normalized_value": 8189.633834740898
      },
      {
        "timestamp": "2022-02-07T11:00:00+00:00",
        "portfolio_value": 938109.600247152,
        "drawdown_pct": -1.4010067114093745,
        "normalized_value": 9391.201649892575
      },
      {
        "timestamp": "2022-02-11T06:00:00+00:00",
        "portfolio_value": 913710.8120222111,
        "drawdown_pct": -5.947206010304276,
        "normalized_value": 9146.950935292622
      },
      {
        "timestamp": "2022-02-15T01:00:00+00:00",
        "portfolio_value": 864740.0455577202,
        "drawdown_pct": -10.988010331759872,
        "normalized_value": 8656.715740282712
      },
      {
        "timestamp": "2022-02-18T20:00:00+00:00",
        "portfolio_value": 807916.3007261115,
        "drawdown_pct": -16.83716073695439,
        "normalized_value": 8087.866166549444
      },
      {
        "timestamp": "2022-02-22T15:00:00+00:00",
        "portfolio_value": 807916.3007261115,
        "drawdown_pct": -16.83716073695439,
        "normalized_value": 8087.866166549444
      },
      {
        "timestamp": "2022-02-26T10:00:00+00:00",
        "portfolio_value": 807916.3007261115,
        "drawdown_pct": -16.83716073695439,
        "normalized_value": 8087.866166549444
      },
      {
        "timestamp": "2022-03-02T05:00:00+00:00",
        "portfolio_value": 845418.8442427358,
        "drawdown_pct": -12.976837587606656,
        "normalized_value": 8463.295592339035
      },
      {
        "timestamp": "2022-03-06T00:00:00+00:00",
        "portfolio_value": 758236.4482164653,
        "drawdown_pct": -21.950954808392655,
        "normalized_value": 7590.532472563056
      },
      {
        "timestamp": "2022-03-09T19:00:00+00:00",
        "portfolio_value": 758236.4482164653,
        "drawdown_pct": -21.950954808392655,
        "normalized_value": 7590.532472563056
      },
      {
        "timestamp": "2022-03-13T14:00:00+00:00",
        "portfolio_value": 758236.4482164653,
        "drawdown_pct": -21.950954808392655,
        "normalized_value": 7590.532472563056
      },
      {
        "timestamp": "2022-03-17T09:00:00+00:00",
        "portfolio_value": 757478.2117682489,
        "drawdown_pct": -22.029003853584253,
        "normalized_value": 7582.9419400904935
      },
      {
        "timestamp": "2022-03-21T04:00:00+00:00",
        "portfolio_value": 780775.3278011793,
        "drawdown_pct": -19.63091594002461,
        "normalized_value": 7816.164065169007
      },
      {
        "timestamp": "2022-03-24T23:00:00+00:00",
        "portfolio_value": 848480.2302681828,
        "drawdown_pct": -12.66171391238489,
        "normalized_value": 8493.94243092331
      },
      {
        "timestamp": "2022-03-28T18:00:00+00:00",
        "portfolio_value": 921654.0167255986,
        "drawdown_pct": -5.129572481450376,
        "normalized_value": 9226.468549328587
      },
      {
        "timestamp": "2022-04-01T13:00:00+00:00",
        "portfolio_value": 1059336.2606797058,
        "drawdown_pct": -0.06910451358345782,
        "normalized_value": 10604.774150552657
      },
      {
        "timestamp": "2022-04-05T08:00:00+00:00",
        "portfolio_value": 1112390.4000868204,
        "drawdown_pct": -5.3182782625879925,
        "normalized_value": 11135.88706252207
      },
      {
        "timestamp": "2022-04-09T03:00:00+00:00",
        "portfolio_value": 1035427.8715866364,
        "drawdown_pct": -11.8689862757939,
        "normalized_value": 10365.432710023797
      },
      {
        "timestamp": "2022-04-12T22:00:00+00:00",
        "portfolio_value": 1035427.8715866366,
        "drawdown_pct": -11.868986275793878,
        "normalized_value": 10365.432710023799
      },
      {
        "timestamp": "2022-04-16T17:00:00+00:00",
        "portfolio_value": 1035427.8715866366,
        "drawdown_pct": -11.868986275793878,
        "normalized_value": 10365.432710023799
      },
      {
        "timestamp": "2022-04-20T12:00:00+00:00",
        "portfolio_value": 1044500.5070128303,
        "drawdown_pct": -11.096763913230507,
        "normalized_value": 10456.256797913846
      },
      {
        "timestamp": "2022-04-24T07:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-04-28T02:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-01T21:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-05T16:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-09T11:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-13T06:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-17T01:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-20T20:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-24T15:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-05-28T10:00:00+00:00",
        "portfolio_value": 976294.3526550216,
        "drawdown_pct": -16.902168317278758,
        "normalized_value": 9773.460513589363
      },
      {
        "timestamp": "2022-06-01T05:00:00+00:00",
        "portfolio_value": 945667.7697417138,
        "drawdown_pct": -19.508966794629213,
        "normalized_value": 9466.86476410524
      },
      {
        "timestamp": "2022-06-05T00:00:00+00:00",
        "portfolio_value": 945667.7697417138,
        "drawdown_pct": -19.508966794629213,
        "normalized_value": 9466.86476410524
      },
      {
        "timestamp": "2022-06-08T19:00:00+00:00",
        "portfolio_value": 945667.7697417138,
        "drawdown_pct": -19.508966794629213,
        "normalized_value": 9466.86476410524
      },
      {
        "timestamp": "2022-06-12T14:00:00+00:00",
        "portfolio_value": 945667.7697417138,
        "drawdown_pct": -19.508966794629213,
        "normalized_value": 9466.86476410524
      },
      {
        "timestamp": "2022-06-16T09:00:00+00:00",
        "portfolio_value": 945667.7697417138,
        "drawdown_pct": -19.508966794629213,
        "normalized_value": 9466.86476410524
      },
      {
        "timestamp": "2022-06-20T04:00:00+00:00",
        "portfolio_value": 945667.7697417138,
        "drawdown_pct": -19.508966794629213,
        "normalized_value": 9466.86476410524
      },
      {
        "timestamp": "2022-06-23T23:00:00+00:00",
        "portfolio_value": 893278.8188890693,
        "drawdown_pct": -23.968081208379907,
        "normalized_value": 8942.410903326205
      },
      {
        "timestamp": "2022-06-27T18:00:00+00:00",
        "portfolio_value": 897451.2271939578,
        "drawdown_pct": -23.612944376861503,
        "normalized_value": 8984.179933028672
      },
      {
        "timestamp": "2022-07-01T13:00:00+00:00",
        "portfolio_value": 856737.5289769499,
        "drawdown_pct": -27.07831323044281,
        "normalized_value": 8576.604368544458
      },
      {
        "timestamp": "2022-07-05T08:00:00+00:00",
        "portfolio_value": 813868.6446084213,
        "drawdown_pct": -30.727121940638007,
        "normalized_value": 8147.453726119827
      },
      {
        "timestamp": "2022-07-09T03:00:00+00:00",
        "portfolio_value": 837545.7719603128,
        "drawdown_pct": -28.711830201965956,
        "normalized_value": 8384.480058003883
      },
      {
        "timestamp": "2022-07-12T22:00:00+00:00",
        "portfolio_value": 765909.2360611905,
        "drawdown_pct": -34.80921342075628,
        "normalized_value": 7667.34300498664
      },
      {
        "timestamp": "2022-07-16T17:00:00+00:00",
        "portfolio_value": 808984.5680664034,
        "drawdown_pct": -31.142832806229322,
        "normalized_value": 8098.560347704857
      },
      {
        "timestamp": "2022-07-20T12:00:00+00:00",
        "portfolio_value": 936315.2511720974,
        "drawdown_pct": -20.305011565137196,
        "normalized_value": 9373.238829781048
      },
      {
        "timestamp": "2022-07-24T07:00:00+00:00",
        "portfolio_value": 884017.673714232,
        "drawdown_pct": -24.756348682052163,
        "normalized_value": 8849.699687256234
      },
      {
        "timestamp": "2022-07-28T02:00:00+00:00",
        "portfolio_value": 794787.3494632661,
        "drawdown_pct": -32.35123688911474,
        "normalized_value": 7956.435224228302
      },
      {
        "timestamp": "2022-07-31T21:00:00+00:00",
        "portfolio_value": 807890.7553359708,
        "drawdown_pct": -31.235933279379374,
        "normalized_value": 8087.610437464086
      },
      {
        "timestamp": "2022-08-04T16:00:00+00:00",
        "portfolio_value": 717849.7026182475,
        "drawdown_pct": -38.899827086533534,
        "normalized_value": 7186.229956315647
      },
      {
        "timestamp": "2022-08-08T11:00:00+00:00",
        "portfolio_value": 740916.3826631814,
        "drawdown_pct": -36.936493906698566,
        "normalized_value": 7417.145239176473
      },
      {
        "timestamp": "2022-08-12T06:00:00+00:00",
        "portfolio_value": 667587.5431086002,
        "drawdown_pct": -43.17791848344096,
        "normalized_value": 6683.066919512905
      },
      {
        "timestamp": "2022-08-16T01:00:00+00:00",
        "portfolio_value": 649132.8265714624,
        "drawdown_pct": -44.748701848503906,
        "normalized_value": 6498.320953427272
      },
      {
        "timestamp": "2022-08-19T20:00:00+00:00",
        "portfolio_value": 642361.372027038,
        "drawdown_pct": -45.32505793255739,
        "normalized_value": 6430.533463486844
      },
      {
        "timestamp": "2022-08-23T15:00:00+00:00",
        "portfolio_value": 642361.372027038,
        "drawdown_pct": -45.32505793255739,
        "normalized_value": 6430.533463486844
      },
      {
        "timestamp": "2022-08-27T10:00:00+00:00",
        "portfolio_value": 642361.372027038,
        "drawdown_pct": -45.32505793255739,
        "normalized_value": 6430.533463486844
      },
      {
        "timestamp": "2022-08-31T05:00:00+00:00",
        "portfolio_value": 642361.372027038,
        "drawdown_pct": -45.32505793255739,
        "normalized_value": 6430.533463486844
      },
      {
        "timestamp": "2022-09-04T00:00:00+00:00",
        "portfolio_value": 627234.682390851,
        "drawdown_pct": -46.612574454481596,
        "normalized_value": 6279.103617090063
      },
      {
        "timestamp": "2022-09-07T19:00:00+00:00",
        "portfolio_value": 634423.5747938197,
        "drawdown_pct": -46.000687917124786,
        "normalized_value": 6351.0699824037465
      },
      {
        "timestamp": "2022-09-11T14:00:00+00:00",
        "portfolio_value": 620568.7122745486,
        "drawdown_pct": -47.179952173322384,
        "normalized_value": 6212.372107746314
      },
      {
        "timestamp": "2022-09-15T09:00:00+00:00",
        "portfolio_value": 566800.2971691297,
        "drawdown_pct": -51.75648044691733,
        "normalized_value": 5674.108744364161
      },
      {
        "timestamp": "2022-09-19T04:00:00+00:00",
        "portfolio_value": 566800.2971691297,
        "drawdown_pct": -51.75648044691733,
        "normalized_value": 5674.108744364161
      },
      {
        "timestamp": "2022-09-22T23:00:00+00:00",
        "portfolio_value": 566800.2971691297,
        "drawdown_pct": -51.75648044691733,
        "normalized_value": 5674.108744364161
      },
      {
        "timestamp": "2022-09-26T18:00:00+00:00",
        "portfolio_value": 556299.0422784479,
        "drawdown_pct": -52.650300542250385,
        "normalized_value": 5568.983072236586
      },
      {
        "timestamp": "2022-09-30T13:00:00+00:00",
        "portfolio_value": 560641.2474512629,
        "drawdown_pct": -52.280711356774546,
        "normalized_value": 5612.4518997300565
      },
      {
        "timestamp": "2022-10-04T08:00:00+00:00",
        "portfolio_value": 545181.6742791921,
        "drawdown_pct": -53.59656144424647,
        "normalized_value": 5457.689632035603
      },
      {
        "timestamp": "2022-10-08T03:00:00+00:00",
        "portfolio_value": 537271.7079124637,
        "drawdown_pct": -54.26982258928009,
        "normalized_value": 5378.504759421311
      },
      {
        "timestamp": "2022-10-11T22:00:00+00:00",
        "portfolio_value": 530674.7350068754,
        "drawdown_pct": -54.831327572519264,
        "normalized_value": 5312.4639654469875
      },
      {
        "timestamp": "2022-10-15T17:00:00+00:00",
        "portfolio_value": 511353.0073535248,
        "drawdown_pct": -56.47590706638923,
        "normalized_value": 5119.0385484497465
      },
      {
        "timestamp": "2022-10-19T12:00:00+00:00",
        "portfolio_value": 506990.70315371983,
        "drawdown_pct": -56.84720699162019,
        "normalized_value": 5075.368514172569
      },
      {
        "timestamp": "2022-10-23T07:00:00+00:00",
        "portfolio_value": 506990.70315371983,
        "drawdown_pct": -56.84720699162019,
        "normalized_value": 5075.368514172569
      },
      {
        "timestamp": "2022-10-27T02:00:00+00:00",
        "portfolio_value": 565031.495060063,
        "drawdown_pct": -51.90703301289249,
        "normalized_value": 5656.401669113439
      },
      {
        "timestamp": "2022-10-30T21:00:00+00:00",
        "portfolio_value": 578824.3350128585,
        "drawdown_pct": -50.73304784160971,
        "normalized_value": 5794.478649977159
      },
      {
        "timestamp": "2022-11-03T16:00:00+00:00",
        "portfolio_value": 551699.1068963767,
        "drawdown_pct": -53.041826576474826,
        "normalized_value": 5522.934166289855
      },
      {
        "timestamp": "2022-11-07T11:00:00+00:00",
        "portfolio_value": 556426.8316662307,
        "drawdown_pct": -52.63942367810799,
        "normalized_value": 5570.262342706761
      },
      {
        "timestamp": "2022-11-11T06:00:00+00:00",
        "portfolio_value": 556426.8316662307,
        "drawdown_pct": -52.63942367810799,
        "normalized_value": 5570.262342706761
      },
      {
        "timestamp": "2022-11-15T01:00:00+00:00",
        "portfolio_value": 556426.8316662307,
        "drawdown_pct": -52.63942367810799,
        "normalized_value": 5570.262342706761
      },
      {
        "timestamp": "2022-11-18T20:00:00+00:00",
        "portfolio_value": 556426.8316662307,
        "drawdown_pct": -52.63942367810799,
        "normalized_value": 5570.262342706761
      },
      {
        "timestamp": "2022-11-22T15:00:00+00:00",
        "portfolio_value": 556426.8316662307,
        "drawdown_pct": -52.63942367810799,
        "normalized_value": 5570.262342706761
      },
      {
        "timestamp": "2022-11-26T10:00:00+00:00",
        "portfolio_value": 575594.5197519788,
        "drawdown_pct": -51.00795534689742,
        "normalized_value": 5762.145704659437
      },
      {
        "timestamp": "2022-11-30T05:00:00+00:00",
        "portfolio_value": 553467.2257281654,
        "drawdown_pct": -52.89133216802162,
        "normalized_value": 5540.634401407292
      },
      {
        "timestamp": "2022-12-04T00:00:00+00:00",
        "portfolio_value": 553774.2477482574,
        "drawdown_pct": -52.86519981963805,
        "normalized_value": 5543.707928957678
      },
      {
        "timestamp": "2022-12-07T19:00:00+00:00",
        "portfolio_value": 548352.5325085332,
        "drawdown_pct": -53.326672099178566,
        "normalized_value": 5489.4323719320055
      },
      {
        "timestamp": "2022-12-11T14:00:00+00:00",
        "portfolio_value": 541133.5721533949,
        "drawdown_pct": -53.94111788684921,
        "normalized_value": 5417.165003194043
      },
      {
        "timestamp": "2022-12-15T09:00:00+00:00",
        "portfolio_value": 525434.9396193611,
        "drawdown_pct": -55.27731564361632,
        "normalized_value": 5260.009566648212
      },
      {
        "timestamp": "2022-12-19T04:00:00+00:00",
        "portfolio_value": 519878.2651253522,
        "drawdown_pct": -55.750275054436536,
        "normalized_value": 5204.382963251014
      },
      {
        "timestamp": "2022-12-22T23:00:00+00:00",
        "portfolio_value": 519878.2651253522,
        "drawdown_pct": -55.750275054436536,
        "normalized_value": 5204.382963251014
      },
      {
        "timestamp": "2022-12-26T18:00:00+00:00",
        "portfolio_value": 519878.2651253522,
        "drawdown_pct": -55.750275054436536,
        "normalized_value": 5204.382963251014
      },
      {
        "timestamp": "2022-12-30T13:00:00+00:00",
        "portfolio_value": 519878.2651253522,
        "drawdown_pct": -55.750275054436536,
        "normalized_value": 5204.382963251014
      },
      {
        "timestamp": "2023-01-03T08:00:00+00:00",
        "portfolio_value": 538376.4146255696,
        "drawdown_pct": -54.17579486879313,
        "normalized_value": 5389.563726842638
      },
      {
        "timestamp": "2023-01-07T03:00:00+00:00",
        "portfolio_value": 583031.5922063398,
        "drawdown_pct": -50.37494482773991,
        "normalized_value": 5836.596543970072
      },
      {
        "timestamp": "2023-01-10T22:00:00+00:00",
        "portfolio_value": 741060.8390111226,
        "drawdown_pct": -36.9241984250602,
        "normalized_value": 7418.591358790606
      },
      {
        "timestamp": "2023-01-14T17:00:00+00:00",
        "portfolio_value": 1091170.4138130122,
        "drawdown_pct": -7.124429084719861,
        "normalized_value": 10923.458610609005
      },
      {
        "timestamp": "2023-01-18T12:00:00+00:00",
        "portfolio_value": 1065880.697247828,
        "drawdown_pct": -9.809455243592618,
        "normalized_value": 10670.289015212365
      },
      {
        "timestamp": "2023-01-22T07:00:00+00:00",
        "portfolio_value": 1197790.552081721,
        "drawdown_pct": -2.083028325501267,
        "normalized_value": 11990.808543023166
      },
      {
        "timestamp": "2023-01-26T02:00:00+00:00",
        "portfolio_value": 1173238.6836777015,
        "drawdown_pct": -4.090094251087536,
        "normalized_value": 11745.025377598755
      },
      {
        "timestamp": "2023-01-29T21:00:00+00:00",
        "portfolio_value": 1255971.7798564243,
        "drawdown_pct": 0.0,
        "normalized_value": 12573.247569472333
      },
      {
        "timestamp": "2023-02-02T16:00:00+00:00",
        "portfolio_value": 1176877.5065539929,
        "drawdown_pct": -6.297456246307785,
        "normalized_value": 11781.452805044857
      },
      {
        "timestamp": "2023-02-06T11:00:00+00:00",
        "portfolio_value": 1119815.437178861,
        "drawdown_pct": -10.840716715237669,
        "normalized_value": 11210.217418560333
      },
      {
        "timestamp": "2023-02-10T06:00:00+00:00",
        "portfolio_value": 1063600.6769477655,
        "drawdown_pct": -15.316514749292342,
        "normalized_value": 10647.464251029061
      },
      {
        "timestamp": "2023-02-14T01:00:00+00:00",
        "portfolio_value": 995791.6070040984,
        "drawdown_pct": -20.715447355199995,
        "normalized_value": 9968.64308837933
      },
      {
        "timestamp": "2023-02-17T20:00:00+00:00",
        "portfolio_value": 1041830.147748312,
        "drawdown_pct": -17.049876083409437,
        "normalized_value": 10429.524439217012
      },
      {
        "timestamp": "2023-02-21T15:00:00+00:00",
        "portfolio_value": 1111827.488901632,
        "drawdown_pct": -11.476714148089393,
        "normalized_value": 11130.251886792397
      },
      {
        "timestamp": "2023-02-25T10:00:00+00:00",
        "portfolio_value": 1006667.6965009084,
        "drawdown_pct": -19.84949720637951,
        "normalized_value": 10077.521144418743
      },
      {
        "timestamp": "2023-03-01T05:00:00+00:00",
        "portfolio_value": 1006667.6965009084,
        "drawdown_pct": -19.84949720637951,
        "normalized_value": 10077.521144418743
      },
      {
        "timestamp": "2023-03-05T00:00:00+00:00",
        "portfolio_value": 987841.3063415504,
        "drawdown_pct": -21.348447299152298,
        "normalized_value": 9889.054438311585
      },
      {
        "timestamp": "2023-03-08T19:00:00+00:00",
        "portfolio_value": 987841.3063415504,
        "drawdown_pct": -21.348447299152298,
        "normalized_value": 9889.054438311585
      },
      {
        "timestamp": "2023-03-12T14:00:00+00:00",
        "portfolio_value": 987841.3063415504,
        "drawdown_pct": -21.348447299152298,
        "normalized_value": 9889.054438311585
      },
      {
        "timestamp": "2023-03-16T09:00:00+00:00",
        "portfolio_value": 967499.2537923356,
        "drawdown_pct": -22.968073860470437,
        "normalized_value": 9685.414781056124
      },
      {
        "timestamp": "2023-03-20T04:00:00+00:00",
        "portfolio_value": 1032844.6372909568,
        "drawdown_pct": -17.76529904127099,
        "normalized_value": 10339.572539556239
      },
      {
        "timestamp": "2023-03-23T23:00:00+00:00",
        "portfolio_value": 1035581.3078096788,
        "drawdown_pct": -17.547406365447102,
        "normalized_value": 10366.968725123323
      },
      {
        "timestamp": "2023-03-27T19:00:00+00:00",
        "portfolio_value": 926982.910709421,
        "drawdown_pct": -26.193969834625698,
        "normalized_value": 9279.81489389194
      },
      {
        "timestamp": "2023-03-31T14:00:00+00:00",
        "portfolio_value": 1000085.5655854902,
        "drawdown_pct": -20.373563990441376,
        "normalized_value": 10011.628930229273
      },
      {
        "timestamp": "2023-04-04T09:00:00+00:00",
        "portfolio_value": 995126.8335015508,
        "drawdown_pct": -20.76837637105921,
        "normalized_value": 9961.988192179264
      },
      {
        "timestamp": "2023-04-08T04:00:00+00:00",
        "portfolio_value": 1015018.6525929648,
        "drawdown_pct": -19.18459722805268,
        "normalized_value": 10161.120664783144
      },
      {
        "timestamp": "2023-04-11T23:00:00+00:00",
        "portfolio_value": 1073540.1357666564,
        "drawdown_pct": -14.525138782228254,
        "normalized_value": 10746.965910573337
      },
      {
        "timestamp": "2023-04-15T18:00:00+00:00",
        "portfolio_value": 1127308.6914162757,
        "drawdown_pct": -10.244106635489587,
        "normalized_value": 11285.230680911485
      },
      {
        "timestamp": "2023-04-19T13:00:00+00:00",
        "portfolio_value": 1073792.5391995483,
        "drawdown_pct": -14.505042515979277,
        "normalized_value": 10749.492663881041
      },
      {
        "timestamp": "2023-04-23T08:00:00+00:00",
        "portfolio_value": 1073792.5391995483,
        "drawdown_pct": -14.505042515979277,
        "normalized_value": 10749.492663881041
      },
      {
        "timestamp": "2023-04-27T03:00:00+00:00",
        "portfolio_value": 1004203.3856973316,
        "drawdown_pct": -20.04570470427874,
        "normalized_value": 10052.851489958006
      },
      {
        "timestamp": "2023-04-30T22:00:00+00:00",
        "portfolio_value": 1000181.8256825624,
        "drawdown_pct": -20.365899797772716,
        "normalized_value": 10012.592568147706
      },
      {
        "timestamp": "2023-05-04T17:00:00+00:00",
        "portfolio_value": 920353.0112430244,
        "drawdown_pct": -26.721839932722535,
        "normalized_value": 9213.44447961301
      },
      {
        "timestamp": "2023-05-08T12:00:00+00:00",
        "portfolio_value": 884711.9194862165,
        "drawdown_pct": -29.55957023275222,
        "normalized_value": 8856.649623636347
      },
      {
        "timestamp": "2023-05-12T07:00:00+00:00",
        "portfolio_value": 884711.9194862165,
        "drawdown_pct": -29.55957023275222,
        "normalized_value": 8856.649623636347
      },
      {
        "timestamp": "2023-05-16T02:00:00+00:00",
        "portfolio_value": 884711.9194862165,
        "drawdown_pct": -29.55957023275222,
        "normalized_value": 8856.649623636347
      },
      {
        "timestamp": "2023-05-19T21:00:00+00:00",
        "portfolio_value": 884711.9194862165,
        "drawdown_pct": -29.55957023275222,
        "normalized_value": 8856.649623636347
      },
      {
        "timestamp": "2023-05-23T16:00:00+00:00",
        "portfolio_value": 880855.8316082716,
        "drawdown_pct": -29.86659049704396,
        "normalized_value": 8818.047205718498
      },
      {
        "timestamp": "2023-05-27T11:00:00+00:00",
        "portfolio_value": 869266.836809984,
        "drawdown_pct": -30.78930189742371,
        "normalized_value": 8702.032416997008
      },
      {
        "timestamp": "2023-05-31T06:00:00+00:00",
        "portfolio_value": 840132.7599088938,
        "drawdown_pct": -33.10894612576939,
        "normalized_value": 8410.37780543613
      },
      {
        "timestamp": "2023-06-04T01:00:00+00:00",
        "portfolio_value": 842623.0051183633,
        "drawdown_pct": -32.91067374024222,
        "normalized_value": 8435.307083330361
      },
      {
        "timestamp": "2023-06-07T20:00:00+00:00",
        "portfolio_value": 780405.9347754394,
        "drawdown_pct": -37.86437344438972,
        "normalized_value": 7812.4661556796755
      },
      {
        "timestamp": "2023-06-11T15:00:00+00:00",
        "portfolio_value": 780405.9347754394,
        "drawdown_pct": -37.86437344438972,
        "normalized_value": 7812.4661556796755
      },
      {
        "timestamp": "2023-06-15T10:00:00+00:00",
        "portfolio_value": 780405.9347754394,
        "drawdown_pct": -37.86437344438972,
        "normalized_value": 7812.4661556796755
      },
      {
        "timestamp": "2023-06-19T05:00:00+00:00",
        "portfolio_value": 768419.9147534727,
        "drawdown_pct": -38.81869584352332,
        "normalized_value": 7692.47683782568
      },
      {
        "timestamp": "2023-06-23T00:00:00+00:00",
        "portfolio_value": 807582.5000662052,
        "drawdown_pct": -35.70058555308277,
        "normalized_value": 8084.524564131963
      },
      {
        "timestamp": "2023-06-26T19:00:00+00:00",
        "portfolio_value": 785715.3029730703,
        "drawdown_pct": -37.44164354848093,
        "normalized_value": 7865.617032042461
      },
      {
        "timestamp": "2023-06-30T14:00:00+00:00",
        "portfolio_value": 784401.0599347701,
        "drawdown_pct": -37.54628308412802,
        "normalized_value": 7852.460444170007
      },
      {
        "timestamp": "2023-07-04T09:00:00+00:00",
        "portfolio_value": 818857.955217043,
        "drawdown_pct": -34.80283806132568,
        "normalized_value": 8197.40057881931
      },
      {
        "timestamp": "2023-07-08T04:00:00+00:00",
        "portfolio_value": 929904.0134487668,
        "drawdown_pct": -25.96139273486955,
        "normalized_value": 9309.057388434181
      },
      {
        "timestamp": "2023-07-11T23:00:00+00:00",
        "portfolio_value": 949216.3157915744,
        "drawdown_pct": -24.42375449708882,
        "normalized_value": 9502.388450793223
      },
      {
        "timestamp": "2023-07-15T18:00:00+00:00",
        "portfolio_value": 1264095.9295372514,
        "drawdown_pct": -5.433478546187482,
        "normalized_value": 12654.576582485799
      },
      {
        "timestamp": "2023-07-19T13:00:00+00:00",
        "portfolio_value": 1137987.468778166,
        "drawdown_pct": -14.867603110014652,
        "normalized_value": 11392.133490085806
      },
      {
        "timestamp": "2023-07-23T08:00:00+00:00",
        "portfolio_value": 1106626.33233629,
        "drawdown_pct": -17.213717445840622,
        "normalized_value": 11078.184292446364
      },
      {
        "timestamp": "2023-07-27T03:00:00+00:00",
        "portfolio_value": 1091055.6373956162,
        "drawdown_pct": -18.378555036684723,
        "normalized_value": 10922.309610023003
      },
      {
        "timestamp": "2023-07-30T22:00:00+00:00",
        "portfolio_value": 1089527.371737459,
        "drawdown_pct": -18.492883992084614,
        "normalized_value": 10907.010490424847
      },
      {
        "timestamp": "2023-08-03T17:00:00+00:00",
        "portfolio_value": 1070923.0298945175,
        "drawdown_pct": -19.88466751967546,
        "normalized_value": 10720.766659464618
      },
      {
        "timestamp": "2023-08-07T12:00:00+00:00",
        "portfolio_value": 1063442.5071135638,
        "drawdown_pct": -20.444282499457692,
        "normalized_value": 10645.880848825816
      },
      {
        "timestamp": "2023-08-11T07:00:00+00:00",
        "portfolio_value": 1058550.286615605,
        "drawdown_pct": -20.810267599058765,
        "normalized_value": 10596.90594312187
      },
      {
        "timestamp": "2023-08-15T02:00:00+00:00",
        "portfolio_value": 1076844.3024541736,
        "drawdown_pct": -19.441699438327735,
        "normalized_value": 10780.04317109722
      },
      {
        "timestamp": "2023-08-18T21:00:00+00:00",
        "portfolio_value": 1036192.3380937348,
        "drawdown_pct": -22.482856972343644,
        "normalized_value": 10373.08561019757
      },
      {
        "timestamp": "2023-08-22T16:00:00+00:00",
        "portfolio_value": 1036192.3380937348,
        "drawdown_pct": -22.482856972343644,
        "normalized_value": 10373.08561019757
      },
      {
        "timestamp": "2023-08-26T11:00:00+00:00",
        "portfolio_value": 1036192.3380937348,
        "drawdown_pct": -22.482856972343644,
        "normalized_value": 10373.08561019757
      },
      {
        "timestamp": "2023-08-30T06:00:00+00:00",
        "portfolio_value": 1020442.2457471602,
        "drawdown_pct": -23.661115212869042,
        "normalized_value": 10215.415021183086
      },
      {
        "timestamp": "2023-09-03T01:00:00+00:00",
        "portfolio_value": 948116.0488561676,
        "drawdown_pct": -29.071809678492794,
        "normalized_value": 9491.37392897574
      },
      {
        "timestamp": "2023-09-06T20:00:00+00:00",
        "portfolio_value": 948116.0488561676,
        "drawdown_pct": -29.071809678492794,
        "normalized_value": 9491.37392897574
      },
      {
        "timestamp": "2023-09-10T15:00:00+00:00",
        "portfolio_value": 948116.0488561676,
        "drawdown_pct": -29.071809678492794,
        "normalized_value": 9491.37392897574
      },
      {
        "timestamp": "2023-09-14T10:00:00+00:00",
        "portfolio_value": 948618.1241760628,
        "drawdown_pct": -29.034249620430046,
        "normalized_value": 9496.400090707084
      },
      {
        "timestamp": "2023-09-18T05:00:00+00:00",
        "portfolio_value": 932120.1096726592,
        "drawdown_pct": -30.26845962461268,
        "normalized_value": 9331.242223243094
      },
      {
        "timestamp": "2023-09-22T00:00:00+00:00",
        "portfolio_value": 921240.3591796586,
        "drawdown_pct": -31.0823694983555,
        "normalized_value": 9222.327517804233
      },
      {
        "timestamp": "2023-09-25T19:00:00+00:00",
        "portfolio_value": 921222.1465936908,
        "drawdown_pct": -31.083731974779482,
        "normalized_value": 9222.145195752157
      },
      {
        "timestamp": "2023-09-29T14:00:00+00:00",
        "portfolio_value": 969349.9901437492,
        "drawdown_pct": -27.483306846230565,
        "normalized_value": 9703.942081354866
      },
      {
        "timestamp": "2023-10-03T09:00:00+00:00",
        "portfolio_value": 1121970.420906338,
        "drawdown_pct": -16.065832188841327,
        "normalized_value": 11231.790470079732
      },
      {
        "timestamp": "2023-10-07T04:00:00+00:00",
        "portfolio_value": 1100280.48165864,
        "drawdown_pct": -17.688448049925675,
        "normalized_value": 11014.657425928615
      },
      {
        "timestamp": "2023-10-10T23:00:00+00:00",
        "portfolio_value": 1016746.923033903,
        "drawdown_pct": -23.93756085791587,
        "normalized_value": 10178.421986731159
      },
      {
        "timestamp": "2023-10-14T18:00:00+00:00",
        "portfolio_value": 1016746.923033903,
        "drawdown_pct": -23.93756085791587,
        "normalized_value": 10178.421986731159
      },
      {
        "timestamp": "2023-10-18T13:00:00+00:00",
        "portfolio_value": 1065603.1996924344,
        "drawdown_pct": -20.282641933785072,
        "normalized_value": 10667.51105035691
      },
      {
        "timestamp": "2023-10-22T08:00:00+00:00",
        "portfolio_value": 1291445.3611217467,
        "drawdown_pct": -5.653902462010028,
        "normalized_value": 12928.365516052058
      },
      {
        "timestamp": "2023-10-26T03:00:00+00:00",
        "portfolio_value": 1496619.4274955047,
        "drawdown_pct": -2.076194261143846,
        "normalized_value": 14982.316387183497
      },
      {
        "timestamp": "2023-10-29T22:00:00+00:00",
        "portfolio_value": 1506648.6757961025,
        "drawdown_pct": -1.4199805676346198,
        "normalized_value": 15082.7169087888
      },
      {
        "timestamp": "2023-11-02T17:00:00+00:00",
        "portfolio_value": 1853745.1516624345,
        "drawdown_pct": -10.23543041048222,
        "normalized_value": 18557.420713087376
      },
      {
        "timestamp": "2023-11-06T12:00:00+00:00",
        "portfolio_value": 1833700.4721571652,
        "drawdown_pct": -11.206061150493442,
        "normalized_value": 18356.757989678663
      },
      {
        "timestamp": "2023-11-10T07:00:00+00:00",
        "portfolio_value": 2022307.0072802284,
        "drawdown_pct": -5.279313471014201,
        "normalized_value": 20244.855077014287
      },
      {
        "timestamp": "2023-11-14T02:00:00+00:00",
        "portfolio_value": 2154614.1454699147,
        "drawdown_pct": -13.836053472361284,
        "normalized_value": 21569.35171805942
      },
      {
        "timestamp": "2023-11-17T21:00:00+00:00",
        "portfolio_value": 2203995.6589634297,
        "drawdown_pct": -19.18561260853382,
        "normalized_value": 22063.69880807141
      },
      {
        "timestamp": "2023-11-21T16:00:00+00:00",
        "portfolio_value": 2098123.394721935,
        "drawdown_pct": -23.06765391003431,
        "normalized_value": 21003.835672292138
      },
      {
        "timestamp": "2023-11-25T11:00:00+00:00",
        "portfolio_value": 2023303.1889279587,
        "drawdown_pct": -25.81110264195577,
        "normalized_value": 20254.827624711583
      },
      {
        "timestamp": "2023-11-29T06:00:00+00:00",
        "portfolio_value": 2098565.737288245,
        "drawdown_pct": -23.05143443910744,
        "normalized_value": 21008.263863025346
      },
      {
        "timestamp": "2023-12-03T01:00:00+00:00",
        "portfolio_value": 2300052.840922721,
        "drawdown_pct": -15.663463060271237,
        "normalized_value": 23025.305389501198
      },
      {
        "timestamp": "2023-12-06T20:00:00+00:00",
        "portfolio_value": 2148809.8875986338,
        "drawdown_pct": -21.20912126992066,
        "normalized_value": 21511.246613834057
      },
      {
        "timestamp": "2023-12-10T15:00:00+00:00",
        "portfolio_value": 2515141.9096458154,
        "drawdown_pct": -7.7767454740701085,
        "normalized_value": 25178.51309202765
      },
      {
        "timestamp": "2023-12-14T10:00:00+00:00",
        "portfolio_value": 2257393.9961181707,
        "drawdown_pct": -17.227644185441314,
        "normalized_value": 22598.2574053366
      },
      {
        "timestamp": "2023-12-18T05:00:00+00:00",
        "portfolio_value": 2176399.6103857644,
        "drawdown_pct": -20.197482913794783,
        "normalized_value": 21787.441047928238
      },
      {
        "timestamp": "2023-12-22T00:00:00+00:00",
        "portfolio_value": 2838477.6542685707,
        "drawdown_pct": -0.7304400933421094,
        "normalized_value": 28415.35362491472
      },
      {
        "timestamp": "2023-12-25T19:00:00+00:00",
        "portfolio_value": 3716686.156983363,
        "drawdown_pct": -2.0656641598041605,
        "normalized_value": 37206.89902373806
      },
      {
        "timestamp": "2023-12-29T14:00:00+00:00",
        "portfolio_value": 3255821.5216035545,
        "drawdown_pct": -14.209404597331126,
        "normalized_value": 32593.28807357217
      },
      {
        "timestamp": "2024-01-02T09:00:00+00:00",
        "portfolio_value": 3383279.40391555,
        "drawdown_pct": -10.850901208937817,
        "normalized_value": 33869.239917946084
      },
      {
        "timestamp": "2024-01-06T04:00:00+00:00",
        "portfolio_value": 2770882.5830216883,
        "drawdown_pct": -26.987500693454418,
        "normalized_value": 27738.674754502255
      },
      {
        "timestamp": "2024-01-09T23:00:00+00:00",
        "portfolio_value": 2659772.8256096877,
        "drawdown_pct": -29.915232505586076,
        "normalized_value": 26626.380266894586
      },
      {
        "timestamp": "2024-01-13T18:00:00+00:00",
        "portfolio_value": 2809530.436953495,
        "drawdown_pct": -25.969133323546483,
        "normalized_value": 28125.569622131337
      },
      {
        "timestamp": "2024-01-17T13:00:00+00:00",
        "portfolio_value": 2868880.000260486,
        "drawdown_pct": -24.405277829868375,
        "normalized_value": 28719.704589625722
      },
      {
        "timestamp": "2024-01-21T08:00:00+00:00",
        "portfolio_value": 2703316.160999604,
        "drawdown_pct": -28.767869652883032,
        "normalized_value": 27062.28268495733
      },
      {
        "timestamp": "2024-01-25T03:00:00+00:00",
        "portfolio_value": 2703316.160999604,
        "drawdown_pct": -28.767869652883032,
        "normalized_value": 27062.28268495733
      },
      {
        "timestamp": "2024-01-28T22:00:00+00:00",
        "portfolio_value": 2748134.358417309,
        "drawdown_pct": -27.58691430387928,
        "normalized_value": 27510.94745656128
      },
      {
        "timestamp": "2024-02-01T17:00:00+00:00",
        "portfolio_value": 2776651.813310252,
        "drawdown_pct": -26.835482009940336,
        "normalized_value": 27796.429205570978
      },
      {
        "timestamp": "2024-02-05T12:00:00+00:00",
        "portfolio_value": 2776651.813310252,
        "drawdown_pct": -26.835482009940336,
        "normalized_value": 27796.429205570978
      },
      {
        "timestamp": "2024-02-09T07:00:00+00:00",
        "portfolio_value": 2906150.304873328,
        "drawdown_pct": -23.423208756868394,
        "normalized_value": 29092.80912461813
      },
      {
        "timestamp": "2024-02-13T02:00:00+00:00",
        "portfolio_value": 3107984.695356309,
        "drawdown_pct": -18.1048912700605,
        "normalized_value": 31113.327260675425
      },
      {
        "timestamp": "2024-02-16T21:00:00+00:00",
        "portfolio_value": 3034914.6181385946,
        "drawdown_pct": -20.030281033913166,
        "normalized_value": 30381.839351859668
      },
      {
        "timestamp": "2024-02-20T16:00:00+00:00",
        "portfolio_value": 3107761.710980542,
        "drawdown_pct": -18.110766887699732,
        "normalized_value": 31111.095014851428
      },
      {
        "timestamp": "2024-02-24T11:00:00+00:00",
        "portfolio_value": 3179041.538444526,
        "drawdown_pct": -16.232550038969375,
        "normalized_value": 31824.66114092824
      },
      {
        "timestamp": "2024-02-28T06:00:00+00:00",
        "portfolio_value": 3507745.586914757,
        "drawdown_pct": -7.571228820219054,
        "normalized_value": 35115.24254155214
      },
      {
        "timestamp": "2024-03-03T01:00:00+00:00",
        "portfolio_value": 4157203.7872753823,
        "drawdown_pct": -5.775571180095305,
        "normalized_value": 41616.82073791223
      },
      {
        "timestamp": "2024-03-06T20:00:00+00:00",
        "portfolio_value": 3853230.2220452335,
        "drawdown_pct": -13.718552911359128,
        "normalized_value": 38573.810575175376
      },
      {
        "timestamp": "2024-03-10T15:00:00+00:00",
        "portfolio_value": 3816828.321176909,
        "drawdown_pct": -14.533663481635203,
        "normalized_value": 38209.399432379505
      },
      {
        "timestamp": "2024-03-14T10:00:00+00:00",
        "portfolio_value": 4666476.17735565,
        "drawdown_pct": 0.0,
        "normalized_value": 46715.03070048645
      },
      {
        "timestamp": "2024-03-18T05:00:00+00:00",
        "portfolio_value": 5531656.674606936,
        "drawdown_pct": -0.23451302784897682,
        "normalized_value": 55376.15570240579
      },
      {
        "timestamp": "2024-03-22T00:00:00+00:00",
        "portfolio_value": 4574090.21399397,
        "drawdown_pct": -19.650501612706787,
        "normalized_value": 45790.175852694105
      },
      {
        "timestamp": "2024-03-25T19:00:00+00:00",
        "portfolio_value": 4837474.866903002,
        "drawdown_pct": -15.023827508333742,
        "normalized_value": 48426.859654143336
      },
      {
        "timestamp": "2024-03-29T14:00:00+00:00",
        "portfolio_value": 4733929.375245302,
        "drawdown_pct": -16.84273092425418,
        "normalized_value": 47390.28930901678
      },
      {
        "timestamp": "2024-04-02T09:00:00+00:00",
        "portfolio_value": 4757305.07569964,
        "drawdown_pct": -16.43210810789357,
        "normalized_value": 47624.298124848465
      },
      {
        "timestamp": "2024-04-06T04:00:00+00:00",
        "portfolio_value": 4757305.07569964,
        "drawdown_pct": -16.43210810789357,
        "normalized_value": 47624.298124848465
      },
      {
        "timestamp": "2024-04-09T23:00:00+00:00",
        "portfolio_value": 4571577.499675003,
        "drawdown_pct": -19.694640518079034,
        "normalized_value": 45765.021641659725
      },
      {
        "timestamp": "2024-04-13T18:00:00+00:00",
        "portfolio_value": 4571577.499675003,
        "drawdown_pct": -19.694640518079034,
        "normalized_value": 45765.021641659725
      },
      {
        "timestamp": "2024-04-17T13:00:00+00:00",
        "portfolio_value": 4571577.499675003,
        "drawdown_pct": -19.694640518079034,
        "normalized_value": 45765.021641659725
      },
      {
        "timestamp": "2024-04-21T08:00:00+00:00",
        "portfolio_value": 4571577.499675003,
        "drawdown_pct": -19.694640518079034,
        "normalized_value": 45765.021641659725
      },
      {
        "timestamp": "2024-04-25T03:00:00+00:00",
        "portfolio_value": 4571577.499675003,
        "drawdown_pct": -19.694640518079034,
        "normalized_value": 45765.021641659725
      },
      {
        "timestamp": "2024-04-28T22:00:00+00:00",
        "portfolio_value": 4538346.542560483,
        "drawdown_pct": -20.27838299148116,
        "normalized_value": 45432.3540949262
      },
      {
        "timestamp": "2024-05-02T17:00:00+00:00",
        "portfolio_value": 4522661.779910944,
        "drawdown_pct": -20.553905063030722,
        "normalized_value": 45275.337506636606
      },
      {
        "timestamp": "2024-05-06T12:00:00+00:00",
        "portfolio_value": 4785100.325921923,
        "drawdown_pct": -15.94385048541008,
        "normalized_value": 47902.55004730818
      },
      {
        "timestamp": "2024-05-10T07:00:00+00:00",
        "portfolio_value": 4699152.623758791,
        "drawdown_pct": -17.45362716957349,
        "normalized_value": 47042.14716672127
      },
      {
        "timestamp": "2024-05-14T02:00:00+00:00",
        "portfolio_value": 4495486.920578857,
        "drawdown_pct": -21.031264759478844,
        "normalized_value": 45003.29617614829
      },
      {
        "timestamp": "2024-05-17T21:00:00+00:00",
        "portfolio_value": 4917824.834662446,
        "drawdown_pct": -13.612381886829661,
        "normalized_value": 49231.22488992442
      },
      {
        "timestamp": "2024-05-21T16:00:00+00:00",
        "portfolio_value": 5162773.222869236,
        "drawdown_pct": -9.309563358463507,
        "normalized_value": 51683.34744240666
      },
      {
        "timestamp": "2024-05-25T11:00:00+00:00",
        "portfolio_value": 4408700.573738921,
        "drawdown_pct": -22.555773264825007,
        "normalized_value": 44134.497814617156
      },
      {
        "timestamp": "2024-05-29T06:00:00+00:00",
        "portfolio_value": 4514722.09813279,
        "drawdown_pct": -20.693375300473285,
        "normalized_value": 45195.85515980316
      },
      {
        "timestamp": "2024-06-02T01:00:00+00:00",
        "portfolio_value": 4153284.832107311,
        "drawdown_pct": -27.04246367979141,
        "normalized_value": 41577.588969864504
      },
      {
        "timestamp": "2024-06-05T20:00:00+00:00",
        "portfolio_value": 4260751.775628863,
        "drawdown_pct": -25.154675157659824,
        "normalized_value": 42653.416076891896
      },
      {
        "timestamp": "2024-06-09T15:00:00+00:00",
        "portfolio_value": 3984990.802795189,
        "drawdown_pct": -29.99875448390262,
        "normalized_value": 39892.83575411372
      },
      {
        "timestamp": "2024-06-13T10:00:00+00:00",
        "portfolio_value": 3984990.802795189,
        "drawdown_pct": -29.99875448390262,
        "normalized_value": 39892.83575411372
      },
      {
        "timestamp": "2024-06-17T05:00:00+00:00",
        "portfolio_value": 3984990.802795189,
        "drawdown_pct": -29.99875448390262,
        "normalized_value": 39892.83575411372
      },
      {
        "timestamp": "2024-06-21T00:00:00+00:00",
        "portfolio_value": 3984990.802795189,
        "drawdown_pct": -29.99875448390262,
        "normalized_value": 39892.83575411372
      },
      {
        "timestamp": "2024-06-24T19:00:00+00:00",
        "portfolio_value": 3984990.802795189,
        "drawdown_pct": -29.99875448390262,
        "normalized_value": 39892.83575411372
      },
      {
        "timestamp": "2024-06-28T14:00:00+00:00",
        "portfolio_value": 4191693.551875093,
        "drawdown_pct": -26.367767462039144,
        "normalized_value": 41962.0899198158
      },
      {
        "timestamp": "2024-07-02T09:00:00+00:00",
        "portfolio_value": 4405566.871732884,
        "drawdown_pct": -22.610820579700878,
        "normalized_value": 44103.12703721373
      },
      {
        "timestamp": "2024-07-06T04:00:00+00:00",
        "portfolio_value": 4217315.837669992,
        "drawdown_pct": -25.917680621850142,
        "normalized_value": 42218.5887900626
      },
      {
        "timestamp": "2024-07-09T23:00:00+00:00",
        "portfolio_value": 4091276.583037652,
        "drawdown_pct": -28.13171454182699,
        "normalized_value": 40956.838504442865
      },
      {
        "timestamp": "2024-07-13T18:00:00+00:00",
        "portfolio_value": 3979023.304005008,
        "drawdown_pct": -30.103580911013932,
        "normalized_value": 39833.09648221062
      },
      {
        "timestamp": "2024-07-17T13:00:00+00:00",
        "portfolio_value": 4647455.97021526,
        "drawdown_pct": -18.36174222332041,
        "normalized_value": 46524.62373670411
      },
      {
        "timestamp": "2024-07-21T08:00:00+00:00",
        "portfolio_value": 4937170.499286662,
        "drawdown_pct": -13.272551587075409,
        "normalized_value": 49424.88993448781
      },
      {
        "timestamp": "2024-07-25T03:00:00+00:00",
        "portfolio_value": 4888656.834747131,
        "drawdown_pct": -14.12475353944509,
        "normalized_value": 48939.23068279063
      },
      {
        "timestamp": "2024-07-28T22:00:00+00:00",
        "portfolio_value": 5072463.17865547,
        "drawdown_pct": -10.89596992510932,
        "normalized_value": 50779.274148626544
      },
      {
        "timestamp": "2024-08-01T17:00:00+00:00",
        "portfolio_value": 4957215.800010207,
        "drawdown_pct": -12.92043132210299,
        "normalized_value": 49625.55887677127
      },
      {
        "timestamp": "2024-08-05T12:00:00+00:00",
        "portfolio_value": 4957215.800010208,
        "drawdown_pct": -12.920431322102974,
        "normalized_value": 49625.558876771276
      },
      {
        "timestamp": "2024-08-09T07:00:00+00:00",
        "portfolio_value": 4932595.492992937,
        "drawdown_pct": -13.352917177525484,
        "normalized_value": 49379.09058796947
      },
      {
        "timestamp": "2024-08-13T02:00:00+00:00",
        "portfolio_value": 4755848.373457919,
        "drawdown_pct": -16.457696867394613,
        "normalized_value": 47609.71541032103
      },
      {
        "timestamp": "2024-08-16T21:00:00+00:00",
        "portfolio_value": 4755848.373457919,
        "drawdown_pct": -16.457696867394613,
        "normalized_value": 47609.71541032103
      },
      {
        "timestamp": "2024-08-20T16:00:00+00:00",
        "portfolio_value": 4755848.373457919,
        "drawdown_pct": -16.457696867394613,
        "normalized_value": 47609.71541032103
      },
      {
        "timestamp": "2024-08-24T11:00:00+00:00",
        "portfolio_value": 4954512.843960691,
        "drawdown_pct": -12.967912056539982,
        "normalized_value": 49598.5001990804
      },
      {
        "timestamp": "2024-08-28T06:00:00+00:00",
        "portfolio_value": 4920476.4866870055,
        "drawdown_pct": -13.565802370444263,
        "normalized_value": 49257.769974700685
      },
      {
        "timestamp": "2024-09-01T01:00:00+00:00",
        "portfolio_value": 4920456.492869891,
        "drawdown_pct": -13.566153586336565,
        "normalized_value": 49257.569821149096
      },
      {
        "timestamp": "2024-09-04T20:00:00+00:00",
        "portfolio_value": 4920436.48243008,
        "drawdown_pct": -13.566505094226905,
        "normalized_value": 49257.36950119147
      },
      {
        "timestamp": "2024-09-08T15:00:00+00:00",
        "portfolio_value": 4920416.455353752,
        "drawdown_pct": -13.566856894358045,
        "normalized_value": 49257.16901468946
      },
      {
        "timestamp": "2024-09-12T10:00:00+00:00",
        "portfolio_value": 4920396.411627077,
        "drawdown_pct": -13.56720898697293,
        "normalized_value": 49256.968361504616
      },
      {
        "timestamp": "2024-09-16T05:00:00+00:00",
        "portfolio_value": 4920376.35123621,
        "drawdown_pct": -13.56756137231475,
        "normalized_value": 49256.767541498346
      },
      {
        "timestamp": "2024-09-20T00:00:00+00:00",
        "portfolio_value": 4937023.250709336,
        "drawdown_pct": -13.275138188732313,
        "normalized_value": 49423.41586250096
      },
      {
        "timestamp": "2024-09-23T19:00:00+00:00",
        "portfolio_value": 5035249.818758946,
        "drawdown_pct": -11.549668182271732,
        "normalized_value": 50406.739674225886
      },
      {
        "timestamp": "2024-09-27T14:00:00+00:00",
        "portfolio_value": 5571735.563843963,
        "drawdown_pct": -2.125638814063216,
        "normalized_value": 55777.376338703994
      },
      {
        "timestamp": "2024-10-01T09:00:00+00:00",
        "portfolio_value": 5343399.457043258,
        "drawdown_pct": -6.136642267605564,
        "normalized_value": 53491.55555363594
      },
      {
        "timestamp": "2024-10-05T04:00:00+00:00",
        "portfolio_value": 5077013.0022530295,
        "drawdown_pct": -10.816046699569048,
        "normalized_value": 50824.82139690634
      },
      {
        "timestamp": "2024-10-08T23:00:00+00:00",
        "portfolio_value": 4984184.000496475,
        "drawdown_pct": -12.446701841462177,
        "normalized_value": 49895.53139259943
      },
      {
        "timestamp": "2024-10-12T18:00:00+00:00",
        "portfolio_value": 4990444.929686805,
        "drawdown_pct": -12.336720949886772,
        "normalized_value": 49958.20812943977
      },
      {
        "timestamp": "2024-10-16T13:00:00+00:00",
        "portfolio_value": 5211256.464194121,
        "drawdown_pct": -8.457895827130965,
        "normalized_value": 52168.702134227795
      },
      {
        "timestamp": "2024-10-20T08:00:00+00:00",
        "portfolio_value": 5365988.746918188,
        "drawdown_pct": -5.739833716514261,
        "normalized_value": 53717.69179218148
      },
      {
        "timestamp": "2024-10-24T03:00:00+00:00",
        "portfolio_value": 5922401.70627916,
        "drawdown_pct": -0.2503569974648487,
        "normalized_value": 59287.81526239828
      },
      {
        "timestamp": "2024-10-27T22:00:00+00:00",
        "portfolio_value": 5784916.775148166,
        "drawdown_pct": -4.994992184128948,
        "normalized_value": 57911.48491492865
      },
      {
        "timestamp": "2024-10-31T17:00:00+00:00",
        "portfolio_value": 5377345.644999725,
        "drawdown_pct": -11.68848491882509,
        "normalized_value": 53831.38311350778
      },
      {
        "timestamp": "2024-11-04T12:00:00+00:00",
        "portfolio_value": 5247643.712052623,
        "drawdown_pct": -13.818564508960145,
        "normalized_value": 52532.96658907804
      },
      {
        "timestamp": "2024-11-08T07:00:00+00:00",
        "portfolio_value": 5711176.898505724,
        "drawdown_pct": -6.206013505447295,
        "normalized_value": 57173.29179655003
      },
      {
        "timestamp": "2024-11-12T02:00:00+00:00",
        "portfolio_value": 6229039.910486593,
        "drawdown_pct": -2.195471820071677,
        "normalized_value": 62357.50051933869
      },
      {
        "timestamp": "2024-11-15T21:00:00+00:00",
        "portfolio_value": 6179137.745278491,
        "drawdown_pct": -4.239360511131971,
        "normalized_value": 61857.94130353696
      },
      {
        "timestamp": "2024-11-19T16:00:00+00:00",
        "portfolio_value": 6885702.5016808,
        "drawdown_pct": -2.2810373385088765,
        "normalized_value": 68931.20023227316
      },
      {
        "timestamp": "2024-11-23T11:00:00+00:00",
        "portfolio_value": 7313091.53440957,
        "drawdown_pct": -2.5662925621294588,
        "normalized_value": 73209.69454493234
      },
      {
        "timestamp": "2024-11-27T06:00:00+00:00",
        "portfolio_value": 7129842.269047792,
        "drawdown_pct": -5.140181451754081,
        "normalized_value": 71375.22786560858
      },
      {
        "timestamp": "2024-12-01T01:00:00+00:00",
        "portfolio_value": 7755787.796087866,
        "drawdown_pct": -0.30850946663453555,
        "normalized_value": 77641.42604195484
      },
      {
        "timestamp": "2024-12-04T20:00:00+00:00",
        "portfolio_value": 8128507.600746205,
        "drawdown_pct": 0.0,
        "normalized_value": 81372.63915770683
      },
      {
        "timestamp": "2024-12-08T15:00:00+00:00",
        "portfolio_value": 7902386.751283498,
        "drawdown_pct": -4.140505046440909,
        "normalized_value": 79108.99480955196
      },
      {
        "timestamp": "2024-12-12T10:00:00+00:00",
        "portfolio_value": 7635823.492108289,
        "drawdown_pct": -7.3740369149691,
        "normalized_value": 76440.49070437338
      },
      {
        "timestamp": "2024-12-16T05:00:00+00:00",
        "portfolio_value": 7453335.109888447,
        "drawdown_pct": -9.587702824403973,
        "normalized_value": 74613.6410529705
      },
      {
        "timestamp": "2024-12-20T00:00:00+00:00",
        "portfolio_value": 7402954.29153257,
        "drawdown_pct": -10.198844743287792,
        "normalized_value": 74109.29014947051
      },
      {
        "timestamp": "2024-12-23T19:00:00+00:00",
        "portfolio_value": 7402954.29153257,
        "drawdown_pct": -10.198844743287792,
        "normalized_value": 74109.29014947051
      },
      {
        "timestamp": "2024-12-27T14:00:00+00:00",
        "portfolio_value": 7402954.29153257,
        "drawdown_pct": -10.198844743287792,
        "normalized_value": 74109.29014947051
      },
      {
        "timestamp": "2024-12-31T09:00:00+00:00",
        "portfolio_value": 7402954.29153257,
        "drawdown_pct": -10.198844743287792,
        "normalized_value": 74109.29014947051
      },
      {
        "timestamp": "2025-01-04T04:00:00+00:00",
        "portfolio_value": 7756030.822151043,
        "drawdown_pct": -5.915868097080927,
        "normalized_value": 77643.85892054907
      },
      {
        "timestamp": "2025-01-07T23:00:00+00:00",
        "portfolio_value": 7128409.26486798,
        "drawdown_pct": -13.529199030719303,
        "normalized_value": 71360.88238698411
      },
      {
        "timestamp": "2025-01-11T18:00:00+00:00",
        "portfolio_value": 7128206.826325955,
        "drawdown_pct": -13.53165470101341,
        "normalized_value": 71358.8558208245
      },
      {
        "timestamp": "2025-01-15T13:00:00+00:00",
        "portfolio_value": 7128004.219478177,
        "drawdown_pct": -13.534112412931771,
        "normalized_value": 71356.82756979432
      },
      {
        "timestamp": "2025-01-19T08:00:00+00:00",
        "portfolio_value": 9955127.58441615,
        "drawdown_pct": -0.07056676569067848,
        "normalized_value": 99658.51598899449
      },
      {
        "timestamp": "2025-01-23T03:00:00+00:00",
        "portfolio_value": 8779812.962544896,
        "drawdown_pct": -15.681932582877547,
        "normalized_value": 87892.70886672195
      },
      {
        "timestamp": "2025-01-26T22:00:00+00:00",
        "portfolio_value": 8718364.668498246,
        "drawdown_pct": -16.272059208831404,
        "normalized_value": 87277.56398356336
      },
      {
        "timestamp": "2025-01-30T17:00:00+00:00",
        "portfolio_value": 7794771.888698309,
        "drawdown_pct": -25.141901722032944,
        "normalized_value": 78031.68691845189
      },
      {
        "timestamp": "2025-02-03T12:00:00+00:00",
        "portfolio_value": 7328523.88366013,
        "drawdown_pct": -29.619574639396866,
        "normalized_value": 73364.184280146
      },
      {
        "timestamp": "2025-02-07T07:00:00+00:00",
        "portfolio_value": 7328426.053762598,
        "drawdown_pct": -29.620514161449407,
        "normalized_value": 73363.20492731253
      },
      {
        "timestamp": "2025-02-11T02:00:00+00:00",
        "portfolio_value": 7328328.142530086,
        "drawdown_pct": -29.621454464612928,
        "normalized_value": 73362.22476025308
      },
      {
        "timestamp": "2025-02-14T21:00:00+00:00",
        "portfolio_value": 7328230.149894973,
        "drawdown_pct": -29.622395549536844,
        "normalized_value": 73361.24377829074
      },
      {
        "timestamp": "2025-02-18T16:00:00+00:00",
        "portfolio_value": 7328132.075789583,
        "drawdown_pct": -29.623337416871088,
        "normalized_value": 73360.261980748
      },
      {
        "timestamp": "2025-02-22T11:00:00+00:00",
        "portfolio_value": 7328033.920146181,
        "drawdown_pct": -29.624280067266152,
        "normalized_value": 73359.27936694676
      },
      {
        "timestamp": "2025-02-26T06:00:00+00:00",
        "portfolio_value": 7327935.682896977,
        "drawdown_pct": -29.62522350137307,
        "normalized_value": 73358.29593620844
      },
      {
        "timestamp": "2025-03-02T01:00:00+00:00",
        "portfolio_value": 7327837.363974125,
        "drawdown_pct": -29.626167719843423,
        "normalized_value": 73357.3116878538
      },
      {
        "timestamp": "2025-03-05T20:00:00+00:00",
        "portfolio_value": 6593215.141469986,
        "drawdown_pct": -36.68120708656655,
        "normalized_value": 66003.17585318051
      },
      {
        "timestamp": "2025-03-09T15:00:00+00:00",
        "portfolio_value": 6593054.685913802,
        "drawdown_pct": -36.68274804221806,
        "normalized_value": 66001.56956913484
      },
      {
        "timestamp": "2025-03-13T10:00:00+00:00",
        "portfolio_value": 6592894.096956176,
        "drawdown_pct": -36.684290279007534,
        "normalized_value": 65999.96194963768
      },
      {
        "timestamp": "2025-03-17T05:00:00+00:00",
        "portfolio_value": 6592733.3744862,
        "drawdown_pct": -36.685833798000104,
        "normalized_value": 65998.35299357881
      },
      {
        "timestamp": "2025-03-21T00:00:00+00:00",
        "portfolio_value": 6323779.332664892,
        "drawdown_pct": -39.268768665421625,
        "normalized_value": 63305.91530758551
      },
      {
        "timestamp": "2025-03-24T19:00:00+00:00",
        "portfolio_value": 6542180.206744175,
        "drawdown_pct": -37.17132767173128,
        "normalized_value": 65492.276739609
      },
      {
        "timestamp": "2025-03-28T14:00:00+00:00",
        "portfolio_value": 6371573.808260613,
        "drawdown_pct": -38.8097682479141,
        "normalized_value": 63784.37492248769
      },
      {
        "timestamp": "2025-04-01T09:00:00+00:00",
        "portfolio_value": 6371523.70914173,
        "drawdown_pct": -38.81024938127099,
        "normalized_value": 63783.87339161349
      },
      {
        "timestamp": "2025-04-05T04:00:00+00:00",
        "portfolio_value": 6371473.568370849,
        "drawdown_pct": -38.81073091463822,
        "normalized_value": 63783.3714437706
      },
      {
        "timestamp": "2025-04-08T23:00:00+00:00",
        "portfolio_value": 6371423.385913339,
        "drawdown_pct": -38.81121284834837,
        "normalized_value": 63782.869078612384
      },
      {
        "timestamp": "2025-04-12T18:00:00+00:00",
        "portfolio_value": 6951990.331138423,
        "drawdown_pct": -33.23566322827398,
        "normalized_value": 69594.7926027235
      },
      {
        "timestamp": "2025-04-16T13:00:00+00:00",
        "portfolio_value": 6797252.438995752,
        "drawdown_pct": -34.72170855490327,
        "normalized_value": 68045.74679015714
      },
      {
        "timestamp": "2025-04-20T08:00:00+00:00",
        "portfolio_value": 7168943.625749925,
        "drawdown_pct": -31.152124250910944,
        "normalized_value": 71766.66264623268
      },
      {
        "timestamp": "2025-04-24T03:00:00+00:00",
        "portfolio_value": 7613444.180167325,
        "drawdown_pct": -26.883305783569593,
        "normalized_value": 76216.45650712386
      },
      {
        "timestamp": "2025-04-27T22:00:00+00:00",
        "portfolio_value": 7472905.722057563,
        "drawdown_pct": -28.23298501206224,
        "normalized_value": 74809.55799619717
      },
      {
        "timestamp": "2025-05-01T17:00:00+00:00",
        "portfolio_value": 7553710.763733816,
        "drawdown_pct": -27.456963360944197,
        "normalized_value": 75618.47887336306
      },
      {
        "timestamp": "2025-05-05T12:00:00+00:00",
        "portfolio_value": 7328417.20611425,
        "drawdown_pct": -29.620599130982704,
        "normalized_value": 73363.11635551906
      },
      {
        "timestamp": "2025-05-09T07:00:00+00:00",
        "portfolio_value": 8127431.51752495,
        "drawdown_pct": -21.94716202427184,
        "normalized_value": 81361.86673354635
      },
      {
        "timestamp": "2025-05-13T02:00:00+00:00",
        "portfolio_value": 8089038.858824758,
        "drawdown_pct": -22.315870879279963,
        "normalized_value": 80977.52656728645
      },
      {
        "timestamp": "2025-05-16T21:00:00+00:00",
        "portfolio_value": 8114294.258739493,
        "drawdown_pct": -22.07332738527371,
        "normalized_value": 81230.35262650753
      },
      {
        "timestamp": "2025-05-20T16:00:00+00:00",
        "portfolio_value": 7644983.819752687,
        "drawdown_pct": -26.580410782478424,
        "normalized_value": 76532.19265909697
      },
      {
        "timestamp": "2025-05-24T11:00:00+00:00",
        "portfolio_value": 8175195.151345477,
        "drawdown_pct": -21.488457799734622,
        "normalized_value": 81840.01759845811
      },
      {
        "timestamp": "2025-05-28T06:00:00+00:00",
        "portfolio_value": 8274748.615333203,
        "drawdown_pct": -20.53238325419077,
        "normalized_value": 82836.62466335515
      },
      {
        "timestamp": "2025-06-01T01:00:00+00:00",
        "portfolio_value": 7891013.106111107,
        "drawdown_pct": -24.217636764141844,
        "normalized_value": 78995.13583691155
      },
      {
        "timestamp": "2025-06-04T20:00:00+00:00",
        "portfolio_value": 7834294.367114176,
        "drawdown_pct": -24.762342497505347,
        "normalized_value": 78427.33745268494
      },
      {
        "timestamp": "2025-06-08T15:00:00+00:00",
        "portfolio_value": 7777681.174582131,
        "drawdown_pct": -25.306034601765376,
        "normalized_value": 77860.59567009093
      },
      {
        "timestamp": "2025-06-12T10:00:00+00:00",
        "portfolio_value": 8233185.376841145,
        "drawdown_pct": -20.931541181608797,
        "normalized_value": 82420.54454457303
      },
      {
        "timestamp": "2025-06-16T05:00:00+00:00",
        "portfolio_value": 7482042.287108304,
        "drawdown_pct": -28.145240829903035,
        "normalized_value": 74901.02206900514
      },
      {
        "timestamp": "2025-06-20T00:00:00+00:00",
        "portfolio_value": 7222601.076736282,
        "drawdown_pct": -30.63681799222418,
        "normalized_value": 72303.8151730796
      },
      {
        "timestamp": "2025-06-23T19:00:00+00:00",
        "portfolio_value": 7222601.076736282,
        "drawdown_pct": -30.63681799222418,
        "normalized_value": 72303.8151730796
      },
      {
        "timestamp": "2025-06-27T14:00:00+00:00",
        "portfolio_value": 6962787.619076939,
        "drawdown_pct": -33.131970079432904,
        "normalized_value": 69702.8817943018
      },
      {
        "timestamp": "2025-07-01T09:00:00+00:00",
        "portfolio_value": 7102254.69604292,
        "drawdown_pct": -31.792579998089078,
        "normalized_value": 71099.05495249521
      },
      {
        "timestamp": "2025-07-05T04:00:00+00:00",
        "portfolio_value": 6922978.085367357,
        "drawdown_pct": -33.51428607653767,
        "normalized_value": 69304.35761487075
      },
      {
        "timestamp": "2025-07-08T23:00:00+00:00",
        "portfolio_value": 6738145.363002984,
        "drawdown_pct": -35.289350990983436,
        "normalized_value": 67454.03930796383
      },
      {
        "timestamp": "2025-07-12T18:00:00+00:00",
        "portfolio_value": 7216366.2614430515,
        "drawdown_pct": -30.696694845919197,
        "normalized_value": 72241.39985651837
      },
      {
        "timestamp": "2025-07-16T13:00:00+00:00",
        "portfolio_value": 7554431.029222807,
        "drawdown_pct": -27.45004619832148,
        "normalized_value": 75625.6892872068
      },
      {
        "timestamp": "2025-07-20T08:00:00+00:00",
        "portfolio_value": 8335986.045122091,
        "drawdown_pct": -19.94428169036269,
        "normalized_value": 83449.65863243198
      },
      {
        "timestamp": "2025-07-24T03:00:00+00:00",
        "portfolio_value": 8876181.638347296,
        "drawdown_pct": -14.756443561884744,
        "normalized_value": 88857.43374210436
      },
      {
        "timestamp": "2025-07-27T22:00:00+00:00",
        "portfolio_value": 9071679.499813229,
        "drawdown_pct": -12.878954607016418,
        "normalized_value": 90814.51832866613
      },
      {
        "timestamp": "2025-07-31T17:00:00+00:00",
        "portfolio_value": 8822051.578638542,
        "drawdown_pct": -15.276288579459308,
        "normalized_value": 88315.55003692566
      },
      {
        "timestamp": "2025-08-04T12:00:00+00:00",
        "portfolio_value": 8588212.001092946,
        "drawdown_pct": -17.5219971553019,
        "normalized_value": 85974.63525908113
      },
      {
        "timestamp": "2025-08-08T07:00:00+00:00",
        "portfolio_value": 8973455.617562678,
        "drawdown_pct": -13.822260342674465,
        "normalized_value": 89831.22140386417
      },
      {
        "timestamp": "2025-08-12T02:00:00+00:00",
        "portfolio_value": 9816010.448746296,
        "drawdown_pct": -5.730676232462361,
        "normalized_value": 98265.84601345341
      },
      {
        "timestamp": "2025-08-15T21:00:00+00:00",
        "portfolio_value": 10078527.832697228,
        "drawdown_pct": -11.54714097012785,
        "normalized_value": 100893.8477827946
      },
      {
        "timestamp": "2025-08-19T16:00:00+00:00",
        "portfolio_value": 9645053.176039014,
        "drawdown_pct": -15.351473640026331,
        "normalized_value": 96554.43167435403
      },
      {
        "timestamp": "2025-08-23T11:00:00+00:00",
        "portfolio_value": 9929913.52961261,
        "drawdown_pct": -12.851434634716401,
        "normalized_value": 99406.1038262695
      },
      {
        "timestamp": "2025-08-27T06:00:00+00:00",
        "portfolio_value": 9347959.26123454,
        "drawdown_pct": -17.958878868355217,
        "normalized_value": 93580.2921258943
      },
      {
        "timestamp": "2025-08-31T01:00:00+00:00",
        "portfolio_value": 9541809.28720086,
        "drawdown_pct": -16.25757989846914,
        "normalized_value": 95520.88060639487
      },
      {
        "timestamp": "2025-09-03T20:00:00+00:00",
        "portfolio_value": 9671599.534762315,
        "drawdown_pct": -15.118493053487613,
        "normalized_value": 96820.1812283243
      },
      {
        "timestamp": "2025-09-07T15:00:00+00:00",
        "portfolio_value": 9449935.969227914,
        "drawdown_pct": -17.063893854055337,
        "normalized_value": 94601.15773488674
      },
      {
        "timestamp": "2025-09-11T10:00:00+00:00",
        "portfolio_value": 10466750.66855034,
        "drawdown_pct": -8.1399548868105,
        "normalized_value": 104780.25821461322
      },
      {
        "timestamp": "2025-09-15T05:00:00+00:00",
        "portfolio_value": 11368209.540499471,
        "drawdown_pct": -2.0114626540264338,
        "normalized_value": 113804.55776695615
      },
      {
        "timestamp": "2025-09-19T00:00:00+00:00",
        "portfolio_value": 10640874.49221519,
        "drawdown_pct": -8.280743431086112,
        "normalized_value": 106523.37217449183
      },
      {
        "timestamp": "2025-09-22T19:00:00+00:00",
        "portfolio_value": 9898920.31834203,
        "drawdown_pct": -14.6760340893525,
        "normalized_value": 99095.83784376268
      },
      {
        "timestamp": "2025-09-26T14:00:00+00:00",
        "portfolio_value": 9898920.31834203,
        "drawdown_pct": -14.6760340893525,
        "normalized_value": 99095.83784376268
      },
      {
        "timestamp": "2025-09-30T09:00:00+00:00",
        "portfolio_value": 9898920.31834203,
        "drawdown_pct": -14.6760340893525,
        "normalized_value": 99095.83784376268
      },
      {
        "timestamp": "2025-10-04T04:00:00+00:00",
        "portfolio_value": 10430706.34650268,
        "drawdown_pct": -10.092292481243701,
        "normalized_value": 104419.42671198914
      },
      {
        "timestamp": "2025-10-07T23:00:00+00:00",
        "portfolio_value": 10430622.212885952,
        "drawdown_pct": -10.09301767284186,
        "normalized_value": 104418.58446950487
      },
      {
        "timestamp": "2025-10-11T18:00:00+00:00",
        "portfolio_value": 10430622.212885952,
        "drawdown_pct": -10.09301767284186,
        "normalized_value": 104418.58446950487
      },
      {
        "timestamp": "2025-10-15T13:00:00+00:00",
        "portfolio_value": 10430622.212885952,
        "drawdown_pct": -10.09301767284186,
        "normalized_value": 104418.58446950487
      },
      {
        "timestamp": "2025-10-19T08:00:00+00:00",
        "portfolio_value": 10430622.212885952,
        "drawdown_pct": -10.09301767284186,
        "normalized_value": 104418.58446950487
      },
      {
        "timestamp": "2025-10-23T03:00:00+00:00",
        "portfolio_value": 10430622.212885952,
        "drawdown_pct": -10.09301767284186,
        "normalized_value": 104418.58446950487
      },
      {
        "timestamp": "2025-10-26T22:00:00+00:00",
        "portfolio_value": 10430622.212885952,
        "drawdown_pct": -10.09301767284186,
        "normalized_value": 104418.58446950487
      },
      {
        "timestamp": "2025-10-30T17:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-03T12:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-07T07:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-11T02:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-14T21:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-18T16:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-22T11:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-26T06:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-11-30T01:00:00+00:00",
        "portfolio_value": 10228623.49428487,
        "drawdown_pct": -11.83415016261221,
        "normalized_value": 102396.42128206613
      },
      {
        "timestamp": "2025-12-03T20:00:00+00:00",
        "portfolio_value": 10389226.900990356,
        "drawdown_pct": -10.44982549303371,
        "normalized_value": 104004.18542565187
      },
      {
        "timestamp": "2025-12-07T15:00:00+00:00",
        "portfolio_value": 10053423.274528593,
        "drawdown_pct": -13.344292389974468,
        "normalized_value": 100642.53176595499
      },
      {
        "timestamp": "2025-12-11T10:00:00+00:00",
        "portfolio_value": 9887256.024193255,
        "drawdown_pct": -14.776574734626479,
        "normalized_value": 98979.06925038518
      },
      {
        "timestamp": "2025-12-15T05:00:00+00:00",
        "portfolio_value": 9382428.862347672,
        "drawdown_pct": -19.127943789317076,
        "normalized_value": 93925.35945572387
      },
      {
        "timestamp": "2025-12-19T00:00:00+00:00",
        "portfolio_value": 9382428.862347672,
        "drawdown_pct": -19.127943789317076,
        "normalized_value": 93925.35945572387
      },
      {
        "timestamp": "2025-12-22T19:00:00+00:00",
        "portfolio_value": 9254004.10504144,
        "drawdown_pct": -20.23490386800123,
        "normalized_value": 92639.72844588921
      },
      {
        "timestamp": "2025-12-26T14:00:00+00:00",
        "portfolio_value": 9231939.637408571,
        "drawdown_pct": -20.42508904210099,
        "normalized_value": 92418.84608333455
      },
      {
        "timestamp": "2025-12-30T09:00:00+00:00",
        "portfolio_value": 8990032.081802469,
        "drawdown_pct": -22.510216648373127,
        "normalized_value": 89997.16461378025
      },
      {
        "timestamp": "2026-01-03T04:00:00+00:00",
        "portfolio_value": 8932119.412425322,
        "drawdown_pct": -23.009396202185098,
        "normalized_value": 89417.41406431241
      },
      {
        "timestamp": "2026-01-06T23:00:00+00:00",
        "portfolio_value": 9446198.663664082,
        "drawdown_pct": -18.57827855526441,
        "normalized_value": 94563.744419675
      },
      {
        "timestamp": "2026-01-10T18:00:00+00:00",
        "portfolio_value": 9059900.357570617,
        "drawdown_pct": -21.907985476875464,
        "normalized_value": 90696.60001716607
      },
      {
        "timestamp": "2026-01-14T13:00:00+00:00",
        "portfolio_value": 9507284.14810575,
        "drawdown_pct": -18.0517508507754,
        "normalized_value": 95175.25729847077
      },
      {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "portfolio_value": 9399017.62597769,
        "drawdown_pct": -18.984956568797205,
        "normalized_value": 94091.42579203559
      },
      {
        "timestamp": "2026-01-22T03:00:00+00:00",
        "portfolio_value": 9074343.7884743,
        "drawdown_pct": -21.78348999995206,
        "normalized_value": 90841.18991593384
      },
      {
        "timestamp": "2026-01-25T22:00:00+00:00",
        "portfolio_value": 9074343.7884743,
        "drawdown_pct": -21.78348999995206,
        "normalized_value": 90841.18991593384
      },
      {
        "timestamp": "2026-01-29T17:00:00+00:00",
        "portfolio_value": 8841377.378758097,
        "drawdown_pct": -23.791549198501162,
        "normalized_value": 88509.01622245497
      },
      {
        "timestamp": "2026-02-02T12:00:00+00:00",
        "portfolio_value": 8841377.378758097,
        "drawdown_pct": -23.791549198501162,
        "normalized_value": 88509.01622245497
      },
      {
        "timestamp": "2026-02-06T07:00:00+00:00",
        "portfolio_value": 8841377.378758099,
        "drawdown_pct": -23.791549198501144,
        "normalized_value": 88509.016222455
      },
      {
        "timestamp": "2026-02-10T02:00:00+00:00",
        "portfolio_value": 8841377.378758097,
        "drawdown_pct": -23.791549198501162,
        "normalized_value": 88509.01622245497
      },
      {
        "timestamp": "2026-02-13T21:00:00+00:00",
        "portfolio_value": 8841377.378758097,
        "drawdown_pct": -23.791549198501162,
        "normalized_value": 88509.01622245497
      },
      {
        "timestamp": "2026-02-17T16:00:00+00:00",
        "portfolio_value": 8632449.792898705,
        "drawdown_pct": -25.59240521515474,
        "normalized_value": 86417.4897222317
      },
      {
        "timestamp": "2026-02-21T11:00:00+00:00",
        "portfolio_value": 8632449.792898705,
        "drawdown_pct": -25.59240521515474,
        "normalized_value": 86417.4897222317
      },
      {
        "timestamp": "2026-02-25T06:00:00+00:00",
        "portfolio_value": 8632449.792898705,
        "drawdown_pct": -25.59240521515474,
        "normalized_value": 86417.4897222317
      },
      {
        "timestamp": "2026-03-01T01:00:00+00:00",
        "portfolio_value": 8354425.246570569,
        "drawdown_pct": -27.988844033767062,
        "normalized_value": 83634.24928049702
      },
      {
        "timestamp": "2026-03-04T20:00:00+00:00",
        "portfolio_value": 7824855.531518339,
        "drawdown_pct": -32.55348208128252,
        "normalized_value": 78332.8474182611
      },
      {
        "timestamp": "2026-03-08T15:00:00+00:00",
        "portfolio_value": 6959494.527715172,
        "drawdown_pct": -40.012480680563286,
        "normalized_value": 69669.91540634273
      },
      {
        "timestamp": "2026-03-12T10:00:00+00:00",
        "portfolio_value": 7152814.7437482895,
        "drawdown_pct": -38.34615273850632,
        "normalized_value": 71605.20008021186
      },
      {
        "timestamp": "2026-03-16T05:00:00+00:00",
        "portfolio_value": 7766977.929788361,
        "drawdown_pct": -33.05235936872247,
        "normalized_value": 77753.447923026
      },
      {
        "timestamp": "2026-03-20T00:00:00+00:00",
        "portfolio_value": 7315229.745802882,
        "drawdown_pct": -36.946212982146626,
        "normalized_value": 73231.09969243259
      },
      {
        "timestamp": "2026-03-23T19:00:00+00:00",
        "portfolio_value": 7315229.745802882,
        "drawdown_pct": -36.946212982146626,
        "normalized_value": 73231.09969243259
      },
      {
        "timestamp": "2026-03-27T14:00:00+00:00",
        "portfolio_value": 6793057.107712094,
        "drawdown_pct": -41.44709175873159,
        "normalized_value": 68003.74828373255
      },
      {
        "timestamp": "2026-03-31T09:00:00+00:00",
        "portfolio_value": 6793057.107712094,
        "drawdown_pct": -41.44709175873159,
        "normalized_value": 68003.74828373255
      },
      {
        "timestamp": "2026-04-04T04:00:00+00:00",
        "portfolio_value": 6544475.813171509,
        "drawdown_pct": -43.5897438075722,
        "normalized_value": 65515.257532964264
      },
      {
        "timestamp": "2026-04-07T23:00:00+00:00",
        "portfolio_value": 6588794.204817057,
        "drawdown_pct": -43.207740435853694,
        "normalized_value": 65958.91886276263
      },
      {
        "timestamp": "2026-04-11T18:00:00+00:00",
        "portfolio_value": 6580684.929478915,
        "drawdown_pct": -43.2776385166807,
        "normalized_value": 65877.73875340768
      },
      {
        "timestamp": "2026-04-15T13:00:00+00:00",
        "portfolio_value": 6517613.497130119,
        "drawdown_pct": -43.82128414373893,
        "normalized_value": 65246.34500220948
      },
      {
        "timestamp": "2026-04-19T08:00:00+00:00",
        "portfolio_value": 6554649.853384131,
        "drawdown_pct": -43.50204843955648,
        "normalized_value": 65617.10753343935
      },
      {
        "timestamp": "2026-04-23T03:00:00+00:00",
        "portfolio_value": 6214450.608396712,
        "drawdown_pct": -46.43440346905896,
        "normalized_value": 62211.45033733311
      },
      {
        "timestamp": "2026-04-26T22:00:00+00:00",
        "portfolio_value": 6216393.819532855,
        "drawdown_pct": -46.41765391704668,
        "normalized_value": 62230.90338165015
      },
      {
        "timestamp": "2026-04-30T17:00:00+00:00",
        "portfolio_value": 6121750.938782715,
        "drawdown_pct": -47.233430352366334,
        "normalized_value": 61283.45504766322
      },
      {
        "timestamp": "2026-05-04T12:00:00+00:00",
        "portfolio_value": 5997695.311011926,
        "drawdown_pct": -48.30273062093458,
        "normalized_value": 60041.56239898704
      },
      {
        "timestamp": "2026-05-08T07:00:00+00:00",
        "portfolio_value": 6066644.771527847,
        "drawdown_pct": -47.70841919813051,
        "normalized_value": 60731.7997520487
      },
      {
        "timestamp": "2026-05-12T02:00:00+00:00",
        "portfolio_value": 6550297.493642435,
        "drawdown_pct": -43.53956370205871,
        "normalized_value": 65573.53705086849
      },
      {
        "timestamp": "2026-05-15T21:00:00+00:00",
        "portfolio_value": 6211260.625553627,
        "drawdown_pct": -46.46189959778874,
        "normalized_value": 62179.516145282185
      },
      {
        "timestamp": "2026-05-19T16:00:00+00:00",
        "portfolio_value": 6211260.625553627,
        "drawdown_pct": -46.46189959778874,
        "normalized_value": 62179.516145282185
      },
      {
        "timestamp": "2026-05-23T11:00:00+00:00",
        "portfolio_value": 6211260.625553627,
        "drawdown_pct": -46.46189959778874,
        "normalized_value": 62179.516145282185
      },
      {
        "timestamp": "2026-05-27T06:00:00+00:00",
        "portfolio_value": 6211260.625553627,
        "drawdown_pct": -46.46189959778874,
        "normalized_value": 62179.516145282185
      },
      {
        "timestamp": "2026-05-31T01:00:00+00:00",
        "portfolio_value": 6211260.625553627,
        "drawdown_pct": -46.46189959778874,
        "normalized_value": 62179.516145282185
      }
    ],
    "solOnlyEquity": [
      {
        "timestamp": "2021-01-01T00:00:00+00:00",
        "portfolio_value": 9989.23923923924,
        "drawdown_pct": 0.0,
        "normalized_value": 100.0
      },
      {
        "timestamp": "2021-01-04T19:00:00+00:00",
        "portfolio_value": 16272.441894652273,
        "drawdown_pct": 0.0,
        "normalized_value": 162.89971142879094
      },
      {
        "timestamp": "2021-01-08T14:00:00+00:00",
        "portfolio_value": 24320.82550026164,
        "drawdown_pct": -6.542445226279373,
        "normalized_value": 243.47024751119952
      },
      {
        "timestamp": "2021-01-12T09:00:00+00:00",
        "portfolio_value": 25141.89668915762,
        "drawdown_pct": -8.135837724386894,
        "normalized_value": 251.68980426854185
      },
      {
        "timestamp": "2021-01-16T04:00:00+00:00",
        "portfolio_value": 25254.43336569337,
        "drawdown_pct": -10.413029787196319,
        "normalized_value": 252.81638331866301
      },
      {
        "timestamp": "2021-01-19T23:00:00+00:00",
        "portfolio_value": 26272.73886883124,
        "drawdown_pct": -8.803962439280637,
        "normalized_value": 263.01040789600825
      },
      {
        "timestamp": "2021-01-23T18:00:00+00:00",
        "portfolio_value": 23992.88903987735,
        "drawdown_pct": -16.71761284596652,
        "normalized_value": 240.18735026016458
      },
      {
        "timestamp": "2021-01-27T13:00:00+00:00",
        "portfolio_value": 24819.10949122652,
        "drawdown_pct": -13.849695964865749,
        "normalized_value": 248.45845511170975
      },
      {
        "timestamp": "2021-01-31T08:00:00+00:00",
        "portfolio_value": 32039.709875176883,
        "drawdown_pct": 0.0,
        "normalized_value": 320.74224180476193
      },
      {
        "timestamp": "2021-02-04T03:00:00+00:00",
        "portfolio_value": 40764.90912229569,
        "drawdown_pct": -2.340450250547231,
        "normalized_value": 408.08822519902196
      },
      {
        "timestamp": "2021-02-07T22:00:00+00:00",
        "portfolio_value": 39156.31399696589,
        "drawdown_pct": -19.206244116741182,
        "normalized_value": 391.98494559179215
      },
      {
        "timestamp": "2021-02-11T18:00:00+00:00",
        "portfolio_value": 57282.23667755511,
        "drawdown_pct": -5.3310717604595705,
        "normalized_value": 573.4394312285748
      },
      {
        "timestamp": "2021-02-15T13:00:00+00:00",
        "portfolio_value": 54133.644282702575,
        "drawdown_pct": -13.584748885414863,
        "normalized_value": 541.9195895324787
      },
      {
        "timestamp": "2021-02-19T08:00:00+00:00",
        "portfolio_value": 55257.12297775358,
        "drawdown_pct": -11.791304257011364,
        "normalized_value": 553.1664789916659
      },
      {
        "timestamp": "2021-02-23T03:00:00+00:00",
        "portfolio_value": 81713.33772135671,
        "drawdown_pct": -4.517814648670118,
        "normalized_value": 818.0136221022157
      },
      {
        "timestamp": "2021-02-26T22:00:00+00:00",
        "portfolio_value": 87770.2736912752,
        "drawdown_pct": -18.862452203088523,
        "normalized_value": 878.6482292515362
      },
      {
        "timestamp": "2021-03-02T17:00:00+00:00",
        "portfolio_value": 84760.43514484885,
        "drawdown_pct": -21.644839777504057,
        "normalized_value": 848.517420745086
      },
      {
        "timestamp": "2021-03-06T13:00:00+00:00",
        "portfolio_value": 76157.15674978771,
        "drawdown_pct": -29.597975647226182,
        "normalized_value": 762.3919592458143
      },
      {
        "timestamp": "2021-03-10T08:00:00+00:00",
        "portfolio_value": 84823.95640493667,
        "drawdown_pct": -21.586118765711106,
        "normalized_value": 849.1533176193776
      },
      {
        "timestamp": "2021-03-14T03:00:00+00:00",
        "portfolio_value": 86650.91235347834,
        "drawdown_pct": -19.897224344360033,
        "normalized_value": 867.4425577185145
      },
      {
        "timestamp": "2021-03-17T22:00:00+00:00",
        "portfolio_value": 85355.76634836086,
        "drawdown_pct": -21.09449725322467,
        "normalized_value": 854.477145897863
      },
      {
        "timestamp": "2021-03-21T17:00:00+00:00",
        "portfolio_value": 83085.4776803278,
        "drawdown_pct": -23.193222112659974,
        "normalized_value": 831.7498028674246
      },
      {
        "timestamp": "2021-03-25T12:00:00+00:00",
        "portfolio_value": 85370.49802800902,
        "drawdown_pct": -21.08087883423938,
        "normalized_value": 854.6246213891926
      },
      {
        "timestamp": "2021-03-29T07:00:00+00:00",
        "portfolio_value": 108224.14746106011,
        "drawdown_pct": -6.089955088984677,
        "normalized_value": 1083.407303290318
      },
      {
        "timestamp": "2021-04-02T02:00:00+00:00",
        "portfolio_value": 111846.92204875704,
        "drawdown_pct": -10.136668775625541,
        "normalized_value": 1119.674074972651
      },
      {
        "timestamp": "2021-04-05T21:00:00+00:00",
        "portfolio_value": 140044.46308763407,
        "drawdown_pct": -8.872580001478504,
        "normalized_value": 1401.9532392168392
      },
      {
        "timestamp": "2021-04-09T16:00:00+00:00",
        "portfolio_value": 172680.4963447916,
        "drawdown_pct": -1.9761864510653464,
        "normalized_value": 1728.665138647161
      },
      {
        "timestamp": "2021-04-13T11:00:00+00:00",
        "portfolio_value": 163424.26687721792,
        "drawdown_pct": -10.801569676231521,
        "normalized_value": 1636.0031326035592
      },
      {
        "timestamp": "2021-04-17T06:00:00+00:00",
        "portfolio_value": 158426.66073250066,
        "drawdown_pct": -13.529307924677308,
        "normalized_value": 1585.9732351806815
      },
      {
        "timestamp": "2021-04-21T03:00:00+00:00",
        "portfolio_value": 179039.121992489,
        "drawdown_pct": -6.822650093673781,
        "normalized_value": 1792.3198924818648
      },
      {
        "timestamp": "2021-04-24T22:00:00+00:00",
        "portfolio_value": 213125.16233715223,
        "drawdown_pct": -3.6291867668714826,
        "normalized_value": 2133.547482774909
      },
      {
        "timestamp": "2021-04-28T20:00:00+00:00",
        "portfolio_value": 230456.79169203385,
        "drawdown_pct": -5.0272145119076415,
        "normalized_value": 2307.0504787468176
      },
      {
        "timestamp": "2021-05-02T15:00:00+00:00",
        "portfolio_value": 239537.2868599048,
        "drawdown_pct": -4.017380747456076,
        "normalized_value": 2397.953248721546
      },
      {
        "timestamp": "2021-05-06T10:00:00+00:00",
        "portfolio_value": 225851.9983443345,
        "drawdown_pct": -9.501077478643625,
        "normalized_value": 2260.9529408120866
      },
      {
        "timestamp": "2021-05-10T05:00:00+00:00",
        "portfolio_value": 205174.4422683669,
        "drawdown_pct": -17.786576650526143,
        "normalized_value": 2053.9546341267983
      },
      {
        "timestamp": "2021-05-14T00:00:00+00:00",
        "portfolio_value": 169831.99919614248,
        "drawdown_pct": -31.948297780982188,
        "normalized_value": 1700.149482144914
      },
      {
        "timestamp": "2021-05-17T19:00:00+00:00",
        "portfolio_value": 162788.9761052353,
        "drawdown_pct": -34.770437968771766,
        "normalized_value": 1629.6433813075141
      },
      {
        "timestamp": "2021-05-21T14:00:00+00:00",
        "portfolio_value": 144697.75446327546,
        "drawdown_pct": -42.0195926262226,
        "normalized_value": 1448.5362798688498
      },
      {
        "timestamp": "2021-05-25T09:00:00+00:00",
        "portfolio_value": 144697.75446327546,
        "drawdown_pct": -42.0195926262226,
        "normalized_value": 1448.5362798688498
      },
      {
        "timestamp": "2021-05-29T04:00:00+00:00",
        "portfolio_value": 144697.75446327546,
        "drawdown_pct": -42.0195926262226,
        "normalized_value": 1448.5362798688498
      },
      {
        "timestamp": "2021-06-01T23:00:00+00:00",
        "portfolio_value": 144697.75446327546,
        "drawdown_pct": -42.0195926262226,
        "normalized_value": 1448.5362798688498
      },
      {
        "timestamp": "2021-06-05T18:00:00+00:00",
        "portfolio_value": 139790.9558201932,
        "drawdown_pct": -43.98574742442449,
        "normalized_value": 1399.415435672751
      },
      {
        "timestamp": "2021-06-09T13:00:00+00:00",
        "portfolio_value": 138951.86210230607,
        "drawdown_pct": -44.32197237669367,
        "normalized_value": 1391.0154594804594
      },
      {
        "timestamp": "2021-06-13T08:00:00+00:00",
        "portfolio_value": 138951.86210230607,
        "drawdown_pct": -44.32197237669367,
        "normalized_value": 1391.0154594804594
      },
      {
        "timestamp": "2021-06-17T03:00:00+00:00",
        "portfolio_value": 135319.52274448783,
        "drawdown_pct": -45.77745118814654,
        "normalized_value": 1354.6529370618366
      },
      {
        "timestamp": "2021-06-20T22:00:00+00:00",
        "portfolio_value": 131357.03512654718,
        "drawdown_pct": -47.36522044658418,
        "normalized_value": 1314.985375568511
      },
      {
        "timestamp": "2021-06-24T17:00:00+00:00",
        "portfolio_value": 131357.03512654718,
        "drawdown_pct": -47.36522044658418,
        "normalized_value": 1314.985375568511
      },
      {
        "timestamp": "2021-06-28T12:00:00+00:00",
        "portfolio_value": 131357.03512654718,
        "drawdown_pct": -47.36522044658418,
        "normalized_value": 1314.985375568511
      },
      {
        "timestamp": "2021-07-02T07:00:00+00:00",
        "portfolio_value": 119807.29932787074,
        "drawdown_pct": -51.993200950847786,
        "normalized_value": 1199.3635997549202
      },
      {
        "timestamp": "2021-07-06T02:00:00+00:00",
        "portfolio_value": 119306.98051716702,
        "drawdown_pct": -52.19367875763178,
        "normalized_value": 1194.3550220372258
      },
      {
        "timestamp": "2021-07-09T21:00:00+00:00",
        "portfolio_value": 121237.1348189502,
        "drawdown_pct": -51.420265700001764,
        "normalized_value": 1213.67735735783
      },
      {
        "timestamp": "2021-07-13T16:00:00+00:00",
        "portfolio_value": 121237.1348189502,
        "drawdown_pct": -51.420265700001764,
        "normalized_value": 1213.67735735783
      },
      {
        "timestamp": "2021-07-17T11:00:00+00:00",
        "portfolio_value": 121237.1348189502,
        "drawdown_pct": -51.420265700001764,
        "normalized_value": 1213.67735735783
      },
      {
        "timestamp": "2021-07-21T06:00:00+00:00",
        "portfolio_value": 121237.1348189502,
        "drawdown_pct": -51.420265700001764,
        "normalized_value": 1213.67735735783
      },
      {
        "timestamp": "2021-07-25T01:00:00+00:00",
        "portfolio_value": 114826.84879777402,
        "drawdown_pct": -53.988868068911074,
        "normalized_value": 1149.505443284578
      },
      {
        "timestamp": "2021-07-28T20:00:00+00:00",
        "portfolio_value": 106646.7981558861,
        "drawdown_pct": -57.266615331223726,
        "normalized_value": 1067.6168184756393
      },
      {
        "timestamp": "2021-08-01T15:00:00+00:00",
        "portfolio_value": 121408.25619744376,
        "drawdown_pct": -51.35169734334536,
        "normalized_value": 1215.3904145225974
      },
      {
        "timestamp": "2021-08-05T10:00:00+00:00",
        "portfolio_value": 124526.81837903308,
        "drawdown_pct": -50.10208910733852,
        "normalized_value": 1246.6096305900148
      },
      {
        "timestamp": "2021-08-09T05:00:00+00:00",
        "portfolio_value": 125987.66431377945,
        "drawdown_pct": -49.51672796803751,
        "normalized_value": 1261.233826685028
      },
      {
        "timestamp": "2021-08-13T00:00:00+00:00",
        "portfolio_value": 136696.5619012253,
        "drawdown_pct": -45.2256714347327,
        "normalized_value": 1368.4381625805954
      },
      {
        "timestamp": "2021-08-16T23:00:00+00:00",
        "portfolio_value": 225929.0155631046,
        "drawdown_pct": -10.527602833388741,
        "normalized_value": 2261.723942656427
      },
      {
        "timestamp": "2021-08-20T18:00:00+00:00",
        "portfolio_value": 281614.8747254749,
        "drawdown_pct": -2.962400672056898,
        "normalized_value": 2819.182401991627
      },
      {
        "timestamp": "2021-08-24T13:00:00+00:00",
        "portfolio_value": 277238.34368705185,
        "drawdown_pct": -6.592314317681285,
        "normalized_value": 2775.3699460717467
      },
      {
        "timestamp": "2021-08-28T08:00:00+00:00",
        "portfolio_value": 257843.52219685947,
        "drawdown_pct": -13.126855555835348,
        "normalized_value": 2581.2128033135014
      },
      {
        "timestamp": "2021-09-01T03:00:00+00:00",
        "portfolio_value": 322024.58845162805,
        "drawdown_pct": -13.232368979462695,
        "normalized_value": 3223.7148469391627
      },
      {
        "timestamp": "2021-09-04T22:00:00+00:00",
        "portfolio_value": 412713.813773908,
        "drawdown_pct": -6.17224354454716,
        "normalized_value": 4131.584036477031
      },
      {
        "timestamp": "2021-09-08T17:00:00+00:00",
        "portfolio_value": 477495.08232829353,
        "drawdown_pct": -14.59467298271218,
        "normalized_value": 4780.094568689683
      },
      {
        "timestamp": "2021-09-12T12:00:00+00:00",
        "portfolio_value": 408770.60845190135,
        "drawdown_pct": -26.886812488913204,
        "normalized_value": 4092.109505658736
      },
      {
        "timestamp": "2021-09-16T07:00:00+00:00",
        "portfolio_value": 364585.1434480837,
        "drawdown_pct": -34.78987626427465,
        "normalized_value": 3649.778874210343
      },
      {
        "timestamp": "2021-09-20T02:00:00+00:00",
        "portfolio_value": 359709.1841329712,
        "drawdown_pct": -35.6619960310367,
        "normalized_value": 3600.9667555060573
      },
      {
        "timestamp": "2021-09-23T21:00:00+00:00",
        "portfolio_value": 359709.1841329712,
        "drawdown_pct": -35.6619960310367,
        "normalized_value": 3600.9667555060573
      },
      {
        "timestamp": "2021-09-27T16:00:00+00:00",
        "portfolio_value": 349438.7663230539,
        "drawdown_pct": -37.49897493222902,
        "normalized_value": 3498.1519408445615
      },
      {
        "timestamp": "2021-10-01T13:00:00+00:00",
        "portfolio_value": 357946.94301979465,
        "drawdown_pct": -35.97719253069547,
        "normalized_value": 3583.325360891599
      },
      {
        "timestamp": "2021-10-05T08:00:00+00:00",
        "portfolio_value": 402464.85755649337,
        "drawdown_pct": -28.014666444363172,
        "normalized_value": 4028.9840689324037
      },
      {
        "timestamp": "2021-10-09T03:00:00+00:00",
        "portfolio_value": 361873.1128959321,
        "drawdown_pct": -35.274953210110375,
        "normalized_value": 3622.629353739371
      },
      {
        "timestamp": "2021-10-12T22:00:00+00:00",
        "portfolio_value": 346800.89150690555,
        "drawdown_pct": -37.97078829668358,
        "normalized_value": 3471.7447765653596
      },
      {
        "timestamp": "2021-10-16T17:00:00+00:00",
        "portfolio_value": 345296.9714239359,
        "drawdown_pct": -38.239781195767684,
        "normalized_value": 3456.6893749782
      },
      {
        "timestamp": "2021-10-20T12:00:00+00:00",
        "portfolio_value": 348371.3755325734,
        "drawdown_pct": -37.689889693218234,
        "normalized_value": 3487.4665346297647
      },
      {
        "timestamp": "2021-10-24T07:00:00+00:00",
        "portfolio_value": 459235.04495993233,
        "drawdown_pct": -17.860684551226246,
        "normalized_value": 4597.297491444471
      },
      {
        "timestamp": "2021-10-28T02:00:00+00:00",
        "portfolio_value": 432831.7611707014,
        "drawdown_pct": -22.58320666675067,
        "normalized_value": 4332.980228068549
      },
      {
        "timestamp": "2021-10-31T21:00:00+00:00",
        "portfolio_value": 436108.5963568977,
        "drawdown_pct": -21.997108105705312,
        "normalized_value": 4365.783879154654
      },
      {
        "timestamp": "2021-11-04T16:00:00+00:00",
        "portfolio_value": 521563.28880113154,
        "drawdown_pct": -6.712582204884124,
        "normalized_value": 5221.251351678036
      },
      {
        "timestamp": "2021-11-08T11:00:00+00:00",
        "portfolio_value": 526970.6961682941,
        "drawdown_pct": -6.054373900750039,
        "normalized_value": 5275.383675848643
      },
      {
        "timestamp": "2021-11-12T06:00:00+00:00",
        "portfolio_value": 489870.1527166166,
        "drawdown_pct": -12.66846802122682,
        "normalized_value": 4903.978581194979
      },
      {
        "timestamp": "2021-11-16T01:00:00+00:00",
        "portfolio_value": 503421.1703193613,
        "drawdown_pct": -10.252662280550023,
        "normalized_value": 5039.634733562562
      },
      {
        "timestamp": "2021-11-19T20:00:00+00:00",
        "portfolio_value": 503421.1703193613,
        "drawdown_pct": -10.252662280550023,
        "normalized_value": 5039.634733562562
      },
      {
        "timestamp": "2021-11-23T15:00:00+00:00",
        "portfolio_value": 503421.1703193613,
        "drawdown_pct": -10.252662280550023,
        "normalized_value": 5039.634733562562
      },
      {
        "timestamp": "2021-11-27T10:00:00+00:00",
        "portfolio_value": 503421.1703193613,
        "drawdown_pct": -10.252662280550023,
        "normalized_value": 5039.634733562562
      },
      {
        "timestamp": "2021-12-01T05:00:00+00:00",
        "portfolio_value": 503421.1703193613,
        "drawdown_pct": -10.252662280550023,
        "normalized_value": 5039.634733562562
      },
      {
        "timestamp": "2021-12-05T00:00:00+00:00",
        "portfolio_value": 506732.0620529478,
        "drawdown_pct": -9.662413526453689,
        "normalized_value": 5072.779316991707
      },
      {
        "timestamp": "2021-12-08T19:00:00+00:00",
        "portfolio_value": 506732.0620529478,
        "drawdown_pct": -9.662413526453689,
        "normalized_value": 5072.779316991707
      },
      {
        "timestamp": "2021-12-12T14:00:00+00:00",
        "portfolio_value": 506732.0620529478,
        "drawdown_pct": -9.662413526453689,
        "normalized_value": 5072.779316991707
      },
      {
        "timestamp": "2021-12-16T09:00:00+00:00",
        "portfolio_value": 513588.9152958953,
        "drawdown_pct": -8.44000898733364,
        "normalized_value": 5141.421713862258
      },
      {
        "timestamp": "2021-12-20T04:00:00+00:00",
        "portfolio_value": 516448.13907399983,
        "drawdown_pct": -7.93028127392373,
        "normalized_value": 5170.044752210095
      },
      {
        "timestamp": "2021-12-23T23:00:00+00:00",
        "portfolio_value": 512774.5775542048,
        "drawdown_pct": -8.585184932705243,
        "normalized_value": 5133.269564112038
      },
      {
        "timestamp": "2021-12-27T18:00:00+00:00",
        "portfolio_value": 546234.582614838,
        "drawdown_pct": -2.6201072774168472,
        "normalized_value": 5468.230057691942
      },
      {
        "timestamp": "2021-12-31T13:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-04T08:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-08T03:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-11T22:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-15T17:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-19T12:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-23T07:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-27T02:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-01-30T21:00:00+00:00",
        "portfolio_value": 526090.688148835,
        "drawdown_pct": -6.2112572055743405,
        "normalized_value": 5266.574115897348
      },
      {
        "timestamp": "2022-02-03T16:00:00+00:00",
        "portfolio_value": 517854.75120718015,
        "drawdown_pct": -7.679517695433923,
        "normalized_value": 5184.126026063812
      },
      {
        "timestamp": "2022-02-07T11:00:00+00:00",
        "portfolio_value": 593833.4353010108,
        "drawdown_pct": -1.4010067114094018,
        "normalized_value": 5944.7313361796705
      },
      {
        "timestamp": "2022-02-11T06:00:00+00:00",
        "portfolio_value": 575674.0967649469,
        "drawdown_pct": -6.070939818631494,
        "normalized_value": 5762.9423320207625
      },
      {
        "timestamp": "2022-02-15T01:00:00+00:00",
        "portfolio_value": 575674.0967649469,
        "drawdown_pct": -6.070939818631494,
        "normalized_value": 5762.9423320207625
      },
      {
        "timestamp": "2022-02-18T20:00:00+00:00",
        "portfolio_value": 575674.0967649469,
        "drawdown_pct": -6.070939818631494,
        "normalized_value": 5762.9423320207625
      },
      {
        "timestamp": "2022-02-22T15:00:00+00:00",
        "portfolio_value": 575674.0967649469,
        "drawdown_pct": -6.070939818631494,
        "normalized_value": 5762.9423320207625
      },
      {
        "timestamp": "2022-02-26T10:00:00+00:00",
        "portfolio_value": 575674.0967649469,
        "drawdown_pct": -6.070939818631494,
        "normalized_value": 5762.9423320207625
      },
      {
        "timestamp": "2022-03-02T05:00:00+00:00",
        "portfolio_value": 604328.8487366368,
        "drawdown_pct": -1.395527223977239,
        "normalized_value": 6049.798530830475
      },
      {
        "timestamp": "2022-03-06T00:00:00+00:00",
        "portfolio_value": 556396.4343357688,
        "drawdown_pct": -9.624203942380612,
        "normalized_value": 5569.9580419513795
      },
      {
        "timestamp": "2022-03-09T19:00:00+00:00",
        "portfolio_value": 556396.4343357688,
        "drawdown_pct": -9.624203942380612,
        "normalized_value": 5569.9580419513795
      },
      {
        "timestamp": "2022-03-13T14:00:00+00:00",
        "portfolio_value": 556396.4343357688,
        "drawdown_pct": -9.624203942380612,
        "normalized_value": 5569.9580419513795
      },
      {
        "timestamp": "2022-03-17T09:00:00+00:00",
        "portfolio_value": 555840.0379014331,
        "drawdown_pct": -9.714579738438228,
        "normalized_value": 5564.388083909429
      },
      {
        "timestamp": "2022-03-21T04:00:00+00:00",
        "portfolio_value": 545531.1211185704,
        "drawdown_pct": -11.389063080257637,
        "normalized_value": 5461.187864793967
      },
      {
        "timestamp": "2022-03-24T23:00:00+00:00",
        "portfolio_value": 606184.6458151893,
        "drawdown_pct": -1.5370758282160109,
        "normalized_value": 6068.376492916544
      },
      {
        "timestamp": "2022-03-28T18:00:00+00:00",
        "portfolio_value": 669847.2066297238,
        "drawdown_pct": -0.6221164565200049,
        "normalized_value": 6705.6878966164195
      },
      {
        "timestamp": "2022-04-01T13:00:00+00:00",
        "portfolio_value": 769773.5198941943,
        "drawdown_pct": -0.06720849759211522,
        "normalized_value": 7706.027470744796
      },
      {
        "timestamp": "2022-04-05T08:00:00+00:00",
        "portfolio_value": 797552.7048735162,
        "drawdown_pct": -6.565593223567966,
        "normalized_value": 7984.118567714434
      },
      {
        "timestamp": "2022-04-09T03:00:00+00:00",
        "portfolio_value": 750131.3650479914,
        "drawdown_pct": -12.121069028573455,
        "normalized_value": 7509.394330064318
      },
      {
        "timestamp": "2022-04-12T22:00:00+00:00",
        "portfolio_value": 750131.3650479914,
        "drawdown_pct": -12.121069028573455,
        "normalized_value": 7509.394330064318
      },
      {
        "timestamp": "2022-04-16T17:00:00+00:00",
        "portfolio_value": 750131.3650479914,
        "drawdown_pct": -12.121069028573455,
        "normalized_value": 7509.394330064318
      },
      {
        "timestamp": "2022-04-20T12:00:00+00:00",
        "portfolio_value": 764253.4940936037,
        "drawdown_pct": -10.466642002329136,
        "normalized_value": 7650.76774907443
      },
      {
        "timestamp": "2022-04-24T07:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-04-28T02:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-01T21:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-05T16:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-09T11:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-13T06:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-17T01:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-20T20:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-24T15:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-05-28T10:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-01T05:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-05T00:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-08T19:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-12T14:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-16T09:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-20T04:00:00+00:00",
        "portfolio_value": 739328.6280934967,
        "drawdown_pct": -13.386624662373393,
        "normalized_value": 7401.250589627509
      },
      {
        "timestamp": "2022-06-23T23:00:00+00:00",
        "portfolio_value": 702088.3084746266,
        "drawdown_pct": -17.749379813840985,
        "normalized_value": 7028.446227583756
      },
      {
        "timestamp": "2022-06-27T18:00:00+00:00",
        "portfolio_value": 717301.2009435009,
        "drawdown_pct": -15.967168338038473,
        "normalized_value": 7180.739030914721
      },
      {
        "timestamp": "2022-07-01T13:00:00+00:00",
        "portfolio_value": 684710.9685537766,
        "drawdown_pct": -19.78515931397386,
        "normalized_value": 6854.485633541827
      },
      {
        "timestamp": "2022-07-05T08:00:00+00:00",
        "portfolio_value": 650449.8391599868,
        "drawdown_pct": -23.798898193973024,
        "normalized_value": 6511.505266636539
      },
      {
        "timestamp": "2022-07-09T03:00:00+00:00",
        "portfolio_value": 669372.7744270387,
        "drawdown_pct": -21.582050052976225,
        "normalized_value": 6700.938463838583
      },
      {
        "timestamp": "2022-07-12T22:00:00+00:00",
        "portfolio_value": 612120.3251992133,
        "drawdown_pct": -28.289253974938987,
        "normalized_value": 6127.797227988217
      },
      {
        "timestamp": "2022-07-16T17:00:00+00:00",
        "portfolio_value": 640640.7166069457,
        "drawdown_pct": -24.94805052101085,
        "normalized_value": 6413.308373778979
      },
      {
        "timestamp": "2022-07-20T12:00:00+00:00",
        "portfolio_value": 732555.5425026922,
        "drawdown_pct": -14.180101043753323,
        "normalized_value": 7333.446771653075
      },
      {
        "timestamp": "2022-07-24T07:00:00+00:00",
        "portfolio_value": 628595.1108736586,
        "drawdown_pct": -26.359209957967387,
        "normalized_value": 6292.722556933486
      },
      {
        "timestamp": "2022-07-28T02:00:00+00:00",
        "portfolio_value": 604478.3331534239,
        "drawdown_pct": -29.184523953995473,
        "normalized_value": 6051.294985297196
      },
      {
        "timestamp": "2022-07-31T21:00:00+00:00",
        "portfolio_value": 632686.3343161073,
        "drawdown_pct": -25.879917451028106,
        "normalized_value": 6333.6788634595905
      },
      {
        "timestamp": "2022-08-04T16:00:00+00:00",
        "portfolio_value": 566928.3050596549,
        "drawdown_pct": -33.58356188332714,
        "normalized_value": 5675.390202215549
      },
      {
        "timestamp": "2022-08-08T11:00:00+00:00",
        "portfolio_value": 582343.3677803773,
        "drawdown_pct": -31.777665881807582,
        "normalized_value": 5829.706885914241
      },
      {
        "timestamp": "2022-08-12T06:00:00+00:00",
        "portfolio_value": 548091.7374324669,
        "drawdown_pct": -35.79029193539255,
        "normalized_value": 5486.821611794819
      },
      {
        "timestamp": "2022-08-16T01:00:00+00:00",
        "portfolio_value": 554254.1031411221,
        "drawdown_pct": -35.06836223618987,
        "normalized_value": 5548.51165205783
      },
      {
        "timestamp": "2022-08-19T20:00:00+00:00",
        "portfolio_value": 548482.0923932625,
        "drawdown_pct": -35.74456130972819,
        "normalized_value": 5490.729366444064
      },
      {
        "timestamp": "2022-08-23T15:00:00+00:00",
        "portfolio_value": 548482.0923932625,
        "drawdown_pct": -35.74456130972819,
        "normalized_value": 5490.729366444064
      },
      {
        "timestamp": "2022-08-27T10:00:00+00:00",
        "portfolio_value": 548482.0923932625,
        "drawdown_pct": -35.74456130972819,
        "normalized_value": 5490.729366444064
      },
      {
        "timestamp": "2022-08-31T05:00:00+00:00",
        "portfolio_value": 548482.0923932625,
        "drawdown_pct": -35.74456130972819,
        "normalized_value": 5490.729366444064
      },
      {
        "timestamp": "2022-09-04T00:00:00+00:00",
        "portfolio_value": 548482.0923932625,
        "drawdown_pct": -35.74456130972819,
        "normalized_value": 5490.729366444064
      },
      {
        "timestamp": "2022-09-07T19:00:00+00:00",
        "portfolio_value": 548482.0923932625,
        "drawdown_pct": -35.74456130972819,
        "normalized_value": 5490.729366444064
      },
      {
        "timestamp": "2022-09-11T14:00:00+00:00",
        "portfolio_value": 534404.3853551688,
        "drawdown_pct": -37.39378423611183,
        "normalized_value": 5349.800646038667
      },
      {
        "timestamp": "2022-09-15T09:00:00+00:00",
        "portfolio_value": 474104.66937450046,
        "drawdown_pct": -44.45797968929457,
        "normalized_value": 4746.153916427847
      },
      {
        "timestamp": "2022-09-19T04:00:00+00:00",
        "portfolio_value": 474104.66937450046,
        "drawdown_pct": -44.45797968929457,
        "normalized_value": 4746.153916427847
      },
      {
        "timestamp": "2022-09-22T23:00:00+00:00",
        "portfolio_value": 474104.66937450046,
        "drawdown_pct": -44.45797968929457,
        "normalized_value": 4746.153916427847
      },
      {
        "timestamp": "2022-09-26T18:00:00+00:00",
        "portfolio_value": 463438.5546892463,
        "drawdown_pct": -45.7075298345532,
        "normalized_value": 4639.377870426706
      },
      {
        "timestamp": "2022-09-30T13:00:00+00:00",
        "portfolio_value": 467831.088113689,
        "drawdown_pct": -45.19293844485495,
        "normalized_value": 4683.350522590128
      },
      {
        "timestamp": "2022-10-04T08:00:00+00:00",
        "portfolio_value": 452135.68441241264,
        "drawdown_pct": -47.03167678149936,
        "normalized_value": 4526.227409153997
      },
      {
        "timestamp": "2022-10-08T03:00:00+00:00",
        "portfolio_value": 444193.5171924094,
        "drawdown_pct": -47.96211269900767,
        "normalized_value": 4446.72018112801
      },
      {
        "timestamp": "2022-10-11T22:00:00+00:00",
        "portfolio_value": 437574.5640532288,
        "drawdown_pct": -48.737532249667225,
        "normalized_value": 4380.459348038937
      },
      {
        "timestamp": "2022-10-15T17:00:00+00:00",
        "portfolio_value": 419259.9044053993,
        "drawdown_pct": -50.88312005728252,
        "normalized_value": 4197.115459588585
      },
      {
        "timestamp": "2022-10-19T12:00:00+00:00",
        "portfolio_value": 419259.9044053993,
        "drawdown_pct": -50.88312005728252,
        "normalized_value": 4197.115459588585
      },
      {
        "timestamp": "2022-10-23T07:00:00+00:00",
        "portfolio_value": 419259.9044053993,
        "drawdown_pct": -50.88312005728252,
        "normalized_value": 4197.115459588585
      },
      {
        "timestamp": "2022-10-27T02:00:00+00:00",
        "portfolio_value": 417837.7137023724,
        "drawdown_pct": -51.04973167285135,
        "normalized_value": 4182.878232218554
      },
      {
        "timestamp": "2022-10-30T21:00:00+00:00",
        "portfolio_value": 418470.0406704167,
        "drawdown_pct": -50.975653690560094,
        "normalized_value": 4189.208313548075
      },
      {
        "timestamp": "2022-11-03T16:00:00+00:00",
        "portfolio_value": 394857.0834751107,
        "drawdown_pct": -53.7419444125389,
        "normalized_value": 3952.8243744934302
      },
      {
        "timestamp": "2022-11-07T11:00:00+00:00",
        "portfolio_value": 401197.7151382559,
        "drawdown_pct": -52.999130609245746,
        "normalized_value": 4016.2989946450652
      },
      {
        "timestamp": "2022-11-11T06:00:00+00:00",
        "portfolio_value": 401197.7151382559,
        "drawdown_pct": -52.999130609245746,
        "normalized_value": 4016.2989946450652
      },
      {
        "timestamp": "2022-11-15T01:00:00+00:00",
        "portfolio_value": 401197.7151382559,
        "drawdown_pct": -52.999130609245746,
        "normalized_value": 4016.2989946450652
      },
      {
        "timestamp": "2022-11-18T20:00:00+00:00",
        "portfolio_value": 401197.7151382559,
        "drawdown_pct": -52.999130609245746,
        "normalized_value": 4016.2989946450652
      },
      {
        "timestamp": "2022-11-22T15:00:00+00:00",
        "portfolio_value": 401197.7151382559,
        "drawdown_pct": -52.999130609245746,
        "normalized_value": 4016.2989946450652
      },
      {
        "timestamp": "2022-11-26T10:00:00+00:00",
        "portfolio_value": 415018.09946705785,
        "drawdown_pct": -51.38005339554776,
        "normalized_value": 4154.65171598658
      },
      {
        "timestamp": "2022-11-30T05:00:00+00:00",
        "portfolio_value": 402619.89839290414,
        "drawdown_pct": -52.83251986626482,
        "normalized_value": 4030.5361474510732
      },
      {
        "timestamp": "2022-12-04T00:00:00+00:00",
        "portfolio_value": 402619.89839290414,
        "drawdown_pct": -52.83251986626482,
        "normalized_value": 4030.5361474510732
      },
      {
        "timestamp": "2022-12-07T19:00:00+00:00",
        "portfolio_value": 395592.7726384226,
        "drawdown_pct": -53.65575740555383,
        "normalized_value": 3960.1891912296433
      },
      {
        "timestamp": "2022-12-11T14:00:00+00:00",
        "portfolio_value": 391008.97837004304,
        "drawdown_pct": -54.192755268684614,
        "normalized_value": 3914.3018702975974
      },
      {
        "timestamp": "2022-12-15T09:00:00+00:00",
        "portfolio_value": 374748.7730744727,
        "drawdown_pct": -56.09766089632655,
        "normalized_value": 3751.524656676586
      },
      {
        "timestamp": "2022-12-19T04:00:00+00:00",
        "portfolio_value": 374748.7730744727,
        "drawdown_pct": -56.09766089632655,
        "normalized_value": 3751.524656676586
      },
      {
        "timestamp": "2022-12-22T23:00:00+00:00",
        "portfolio_value": 374748.7730744727,
        "drawdown_pct": -56.09766089632655,
        "normalized_value": 3751.524656676586
      },
      {
        "timestamp": "2022-12-26T18:00:00+00:00",
        "portfolio_value": 374748.7730744727,
        "drawdown_pct": -56.09766089632655,
        "normalized_value": 3751.524656676586
      },
      {
        "timestamp": "2022-12-30T13:00:00+00:00",
        "portfolio_value": 374748.7730744727,
        "drawdown_pct": -56.09766089632655,
        "normalized_value": 3751.524656676586
      },
      {
        "timestamp": "2023-01-03T08:00:00+00:00",
        "portfolio_value": 388082.9693553716,
        "drawdown_pct": -54.535541287511634,
        "normalized_value": 3885.0102601499734
      },
      {
        "timestamp": "2023-01-07T03:00:00+00:00",
        "portfolio_value": 420420.5974300939,
        "drawdown_pct": -50.74714325781936,
        "normalized_value": 4208.73489323009
      },
      {
        "timestamp": "2023-01-10T22:00:00+00:00",
        "portfolio_value": 542137.282939983,
        "drawdown_pct": -36.487864546936436,
        "normalized_value": 5427.212923386457
      },
      {
        "timestamp": "2023-01-14T17:00:00+00:00",
        "portfolio_value": 813667.562177637,
        "drawdown_pct": -4.677715314939766,
        "normalized_value": 8145.440735681132
      },
      {
        "timestamp": "2023-01-18T12:00:00+00:00",
        "portfolio_value": 806913.0123415464,
        "drawdown_pct": -8.436009634975147,
        "normalized_value": 8077.822474927522
      },
      {
        "timestamp": "2023-01-22T07:00:00+00:00",
        "portfolio_value": 899627.7841114396,
        "drawdown_pct": -2.0789564163039618,
        "normalized_value": 9005.968948842128
      },
      {
        "timestamp": "2023-01-26T02:00:00+00:00",
        "portfolio_value": 883988.7818979284,
        "drawdown_pct": -3.7812019943096393,
        "normalized_value": 8849.410457860366
      },
      {
        "timestamp": "2023-01-29T21:00:00+00:00",
        "portfolio_value": 946324.885774716,
        "drawdown_pct": 0.0,
        "normalized_value": 9473.443003120889
      },
      {
        "timestamp": "2023-02-02T16:00:00+00:00",
        "portfolio_value": 877052.0633724349,
        "drawdown_pct": -7.320194517083885,
        "normalized_value": 8779.968547827366
      },
      {
        "timestamp": "2023-02-06T11:00:00+00:00",
        "portfolio_value": 809234.5513218718,
        "drawdown_pct": -14.486603545315644,
        "normalized_value": 8101.0628731673205
      },
      {
        "timestamp": "2023-02-10T06:00:00+00:00",
        "portfolio_value": 809574.590421517,
        "drawdown_pct": -14.450670949147378,
        "normalized_value": 8104.466927184863
      },
      {
        "timestamp": "2023-02-14T01:00:00+00:00",
        "portfolio_value": 757960.7646537055,
        "drawdown_pct": -19.90480478242996,
        "normalized_value": 7587.772667174906
      },
      {
        "timestamp": "2023-02-17T20:00:00+00:00",
        "portfolio_value": 744260.0093260246,
        "drawdown_pct": -21.35259037209716,
        "normalized_value": 7450.617524530387
      },
      {
        "timestamp": "2023-02-21T15:00:00+00:00",
        "portfolio_value": 793580.7359711889,
        "drawdown_pct": -16.140772804308323,
        "normalized_value": 7944.356091241502
      },
      {
        "timestamp": "2023-02-25T10:00:00+00:00",
        "portfolio_value": 717710.7385234361,
        "drawdown_pct": -24.158103700730983,
        "normalized_value": 7184.838818397301
      },
      {
        "timestamp": "2023-03-01T05:00:00+00:00",
        "portfolio_value": 717710.7385234361,
        "drawdown_pct": -24.158103700730983,
        "normalized_value": 7184.838818397301
      },
      {
        "timestamp": "2023-03-05T00:00:00+00:00",
        "portfolio_value": 717710.7385234361,
        "drawdown_pct": -24.158103700730983,
        "normalized_value": 7184.838818397301
      },
      {
        "timestamp": "2023-03-08T19:00:00+00:00",
        "portfolio_value": 717710.7385234361,
        "drawdown_pct": -24.158103700730983,
        "normalized_value": 7184.838818397301
      },
      {
        "timestamp": "2023-03-12T14:00:00+00:00",
        "portfolio_value": 717710.7385234361,
        "drawdown_pct": -24.158103700730983,
        "normalized_value": 7184.838818397301
      },
      {
        "timestamp": "2023-03-16T09:00:00+00:00",
        "portfolio_value": 721153.0107234432,
        "drawdown_pct": -23.794352070423905,
        "normalized_value": 7219.298621767365
      },
      {
        "timestamp": "2023-03-20T04:00:00+00:00",
        "portfolio_value": 776202.5857574829,
        "drawdown_pct": -17.97715589799387,
        "normalized_value": 7770.387385542254
      },
      {
        "timestamp": "2023-03-23T23:00:00+00:00",
        "portfolio_value": 766240.4379906917,
        "drawdown_pct": -19.029875520667176,
        "normalized_value": 7670.65859210563
      },
      {
        "timestamp": "2023-03-27T19:00:00+00:00",
        "portfolio_value": 692814.5577493224,
        "drawdown_pct": -26.788931775566223,
        "normalized_value": 6935.6088202176825
      },
      {
        "timestamp": "2023-03-31T14:00:00+00:00",
        "portfolio_value": 663372.5425929667,
        "drawdown_pct": -29.90012705310061,
        "normalized_value": 6640.871508884673
      },
      {
        "timestamp": "2023-04-04T09:00:00+00:00",
        "portfolio_value": 652846.965257166,
        "drawdown_pct": -31.01238537939242,
        "normalized_value": 6535.502350295953
      },
      {
        "timestamp": "2023-04-08T04:00:00+00:00",
        "portfolio_value": 652846.965257166,
        "drawdown_pct": -31.01238537939242,
        "normalized_value": 6535.502350295953
      },
      {
        "timestamp": "2023-04-11T23:00:00+00:00",
        "portfolio_value": 705063.0093678181,
        "drawdown_pct": -25.49461395695909,
        "normalized_value": 7058.225281042667
      },
      {
        "timestamp": "2023-04-15T18:00:00+00:00",
        "portfolio_value": 720819.1209071833,
        "drawdown_pct": -23.829634859799842,
        "normalized_value": 7215.956126825924
      },
      {
        "timestamp": "2023-04-19T13:00:00+00:00",
        "portfolio_value": 685417.9920527791,
        "drawdown_pct": -27.570541327183168,
        "normalized_value": 6861.563484838302
      },
      {
        "timestamp": "2023-04-23T08:00:00+00:00",
        "portfolio_value": 685417.9920527791,
        "drawdown_pct": -27.570541327183168,
        "normalized_value": 6861.563484838302
      },
      {
        "timestamp": "2023-04-27T03:00:00+00:00",
        "portfolio_value": 640998.1845751653,
        "drawdown_pct": -32.26446918909807,
        "normalized_value": 6416.886904232183
      },
      {
        "timestamp": "2023-04-30T22:00:00+00:00",
        "portfolio_value": 638431.1620921282,
        "drawdown_pct": -32.53573147138874,
        "normalized_value": 6391.189026530411
      },
      {
        "timestamp": "2023-05-04T17:00:00+00:00",
        "portfolio_value": 587475.2244192049,
        "drawdown_pct": -37.92034498403116,
        "normalized_value": 5881.080734471886
      },
      {
        "timestamp": "2023-05-08T12:00:00+00:00",
        "portfolio_value": 543559.75734675,
        "drawdown_pct": -42.56097820974446,
        "normalized_value": 5441.452990850046
      },
      {
        "timestamp": "2023-05-12T07:00:00+00:00",
        "portfolio_value": 543559.75734675,
        "drawdown_pct": -42.56097820974446,
        "normalized_value": 5441.452990850046
      },
      {
        "timestamp": "2023-05-16T02:00:00+00:00",
        "portfolio_value": 543559.75734675,
        "drawdown_pct": -42.56097820974446,
        "normalized_value": 5441.452990850046
      },
      {
        "timestamp": "2023-05-19T21:00:00+00:00",
        "portfolio_value": 543559.75734675,
        "drawdown_pct": -42.56097820974446,
        "normalized_value": 5441.452990850046
      },
      {
        "timestamp": "2023-05-23T16:00:00+00:00",
        "portfolio_value": 543559.75734675,
        "drawdown_pct": -42.56097820974446,
        "normalized_value": 5441.452990850046
      },
      {
        "timestamp": "2023-05-27T11:00:00+00:00",
        "portfolio_value": 543559.75734675,
        "drawdown_pct": -42.56097820974446,
        "normalized_value": 5441.452990850046
      },
      {
        "timestamp": "2023-05-31T06:00:00+00:00",
        "portfolio_value": 524836.572210402,
        "drawdown_pct": -44.53949377219003,
        "normalized_value": 5254.019446733889
      },
      {
        "timestamp": "2023-06-04T01:00:00+00:00",
        "portfolio_value": 527999.6711434568,
        "drawdown_pct": -44.20524292656577,
        "normalized_value": 5285.684510081553
      },
      {
        "timestamp": "2023-06-07T20:00:00+00:00",
        "portfolio_value": 502162.9492684535,
        "drawdown_pct": -46.93545981754944,
        "normalized_value": 5027.038969052635
      },
      {
        "timestamp": "2023-06-11T15:00:00+00:00",
        "portfolio_value": 502162.9492684535,
        "drawdown_pct": -46.93545981754944,
        "normalized_value": 5027.038969052635
      },
      {
        "timestamp": "2023-06-15T10:00:00+00:00",
        "portfolio_value": 502162.9492684535,
        "drawdown_pct": -46.93545981754944,
        "normalized_value": 5027.038969052635
      },
      {
        "timestamp": "2023-06-19T05:00:00+00:00",
        "portfolio_value": 502162.9492684535,
        "drawdown_pct": -46.93545981754944,
        "normalized_value": 5027.038969052635
      },
      {
        "timestamp": "2023-06-23T00:00:00+00:00",
        "portfolio_value": 506892.8073128344,
        "drawdown_pct": -46.43564647485068,
        "normalized_value": 5074.388501195196
      },
      {
        "timestamp": "2023-06-26T19:00:00+00:00",
        "portfolio_value": 470192.62298095255,
        "drawdown_pct": -50.313826673169885,
        "normalized_value": 4706.991310549105
      },
      {
        "timestamp": "2023-06-30T14:00:00+00:00",
        "portfolio_value": 460675.8891977164,
        "drawdown_pct": -51.319478529767224,
        "normalized_value": 4611.721455104529
      },
      {
        "timestamp": "2023-07-04T09:00:00+00:00",
        "portfolio_value": 476732.9307154944,
        "drawdown_pct": -49.622699573707884,
        "normalized_value": 4772.464842395761
      },
      {
        "timestamp": "2023-07-08T04:00:00+00:00",
        "portfolio_value": 524935.959603571,
        "drawdown_pct": -44.52899131212975,
        "normalized_value": 5255.0143913016245
      },
      {
        "timestamp": "2023-07-11T23:00:00+00:00",
        "portfolio_value": 534163.4347768839,
        "drawdown_pct": -43.55390597811586,
        "normalized_value": 5347.388544651221
      },
      {
        "timestamp": "2023-07-15T18:00:00+00:00",
        "portfolio_value": 694676.8566014469,
        "drawdown_pct": -26.592139016534013,
        "normalized_value": 6954.251870078868
      },
      {
        "timestamp": "2023-07-19T13:00:00+00:00",
        "portfolio_value": 627366.042470917,
        "drawdown_pct": -33.70500428536031,
        "normalized_value": 6280.418632947828
      },
      {
        "timestamp": "2023-07-23T08:00:00+00:00",
        "portfolio_value": 610345.587469552,
        "drawdown_pct": -35.50358902694523,
        "normalized_value": 6110.03073259095
      },
      {
        "timestamp": "2023-07-27T03:00:00+00:00",
        "portfolio_value": 601757.7699983995,
        "drawdown_pct": -36.41108048154456,
        "normalized_value": 6024.06004688129
      },
      {
        "timestamp": "2023-07-30T22:00:00+00:00",
        "portfolio_value": 600914.8746382575,
        "drawdown_pct": -36.50015088144765,
        "normalized_value": 6015.6220133138195
      },
      {
        "timestamp": "2023-08-03T17:00:00+00:00",
        "portfolio_value": 600914.8746382575,
        "drawdown_pct": -36.50015088144765,
        "normalized_value": 6015.6220133138195
      },
      {
        "timestamp": "2023-08-07T12:00:00+00:00",
        "portfolio_value": 600914.8746382575,
        "drawdown_pct": -36.50015088144765,
        "normalized_value": 6015.6220133138195
      },
      {
        "timestamp": "2023-08-11T07:00:00+00:00",
        "portfolio_value": 599881.3272586064,
        "drawdown_pct": -36.60936785282688,
        "normalized_value": 6005.275405780472
      },
      {
        "timestamp": "2023-08-15T02:00:00+00:00",
        "portfolio_value": 610248.5612397311,
        "drawdown_pct": -35.513841978260295,
        "normalized_value": 6109.05942509198
      },
      {
        "timestamp": "2023-08-18T21:00:00+00:00",
        "portfolio_value": 587211.0592480424,
        "drawdown_pct": -37.94825983390285,
        "normalized_value": 5878.436237079884
      },
      {
        "timestamp": "2023-08-22T16:00:00+00:00",
        "portfolio_value": 587211.0592480424,
        "drawdown_pct": -37.94825983390285,
        "normalized_value": 5878.436237079884
      },
      {
        "timestamp": "2023-08-26T11:00:00+00:00",
        "portfolio_value": 587211.0592480424,
        "drawdown_pct": -37.94825983390285,
        "normalized_value": 5878.436237079884
      },
      {
        "timestamp": "2023-08-30T06:00:00+00:00",
        "portfolio_value": 583916.7699784169,
        "drawdown_pct": -38.29637381876638,
        "normalized_value": 5845.457857137946
      },
      {
        "timestamp": "2023-09-03T01:00:00+00:00",
        "portfolio_value": 563590.9439436187,
        "drawdown_pct": -40.4442435768525,
        "normalized_value": 5641.980639824386
      },
      {
        "timestamp": "2023-09-06T20:00:00+00:00",
        "portfolio_value": 563590.9439436187,
        "drawdown_pct": -40.4442435768525,
        "normalized_value": 5641.980639824386
      },
      {
        "timestamp": "2023-09-10T15:00:00+00:00",
        "portfolio_value": 563590.9439436187,
        "drawdown_pct": -40.4442435768525,
        "normalized_value": 5641.980639824386
      },
      {
        "timestamp": "2023-09-14T10:00:00+00:00",
        "portfolio_value": 563590.9439436187,
        "drawdown_pct": -40.4442435768525,
        "normalized_value": 5641.980639824386
      },
      {
        "timestamp": "2023-09-18T05:00:00+00:00",
        "portfolio_value": 563590.9439436187,
        "drawdown_pct": -40.4442435768525,
        "normalized_value": 5641.980639824386
      },
      {
        "timestamp": "2023-09-22T00:00:00+00:00",
        "portfolio_value": 551186.8454332083,
        "drawdown_pct": -41.7550089066954,
        "normalized_value": 5517.806033397049
      },
      {
        "timestamp": "2023-09-25T19:00:00+00:00",
        "portfolio_value": 551186.8454332083,
        "drawdown_pct": -41.7550089066954,
        "normalized_value": 5517.806033397049
      },
      {
        "timestamp": "2023-09-29T14:00:00+00:00",
        "portfolio_value": 569083.3507028985,
        "drawdown_pct": -39.863850221267924,
        "normalized_value": 5696.963873559592
      },
      {
        "timestamp": "2023-10-03T09:00:00+00:00",
        "portfolio_value": 668284.5461921667,
        "drawdown_pct": -29.381066033672937,
        "normalized_value": 6690.044458711573
      },
      {
        "timestamp": "2023-10-07T04:00:00+00:00",
        "portfolio_value": 655312.6119315265,
        "drawdown_pct": -30.751835676914506,
        "normalized_value": 6560.185377854998
      },
      {
        "timestamp": "2023-10-10T23:00:00+00:00",
        "portfolio_value": 605561.120969539,
        "drawdown_pct": -36.00917295186717,
        "normalized_value": 6062.134527630528
      },
      {
        "timestamp": "2023-10-14T18:00:00+00:00",
        "portfolio_value": 605561.120969539,
        "drawdown_pct": -36.00917295186717,
        "normalized_value": 6062.134527630528
      },
      {
        "timestamp": "2023-10-18T13:00:00+00:00",
        "portfolio_value": 634659.278032515,
        "drawdown_pct": -32.9343138310321,
        "normalized_value": 6353.429553869102
      },
      {
        "timestamp": "2023-10-22T08:00:00+00:00",
        "portfolio_value": 757337.7257223494,
        "drawdown_pct": -19.970642523857006,
        "normalized_value": 7581.535566266272
      },
      {
        "timestamp": "2023-10-26T03:00:00+00:00",
        "portfolio_value": 878873.774959401,
        "drawdown_pct": -7.127690693677118,
        "normalized_value": 8798.205287816636
      },
      {
        "timestamp": "2023-10-29T22:00:00+00:00",
        "portfolio_value": 882755.0571039829,
        "drawdown_pct": -6.717548024607978,
        "normalized_value": 8837.059919802377
      },
      {
        "timestamp": "2023-11-02T17:00:00+00:00",
        "portfolio_value": 1090435.5419063922,
        "drawdown_pct": -10.400502544072157,
        "normalized_value": 10916.101975243488
      },
      {
        "timestamp": "2023-11-06T12:00:00+00:00",
        "portfolio_value": 1096644.2538134372,
        "drawdown_pct": -9.890341745619734,
        "normalized_value": 10978.255976747989
      },
      {
        "timestamp": "2023-11-10T07:00:00+00:00",
        "portfolio_value": 1301179.7914702022,
        "drawdown_pct": -2.5969192560951733,
        "normalized_value": 13025.814682252994
      },
      {
        "timestamp": "2023-11-14T02:00:00+00:00",
        "portfolio_value": 1502275.785665352,
        "drawdown_pct": -12.912754423157521,
        "normalized_value": 15038.940901166785
      },
      {
        "timestamp": "2023-11-17T21:00:00+00:00",
        "portfolio_value": 1535061.9706868457,
        "drawdown_pct": -19.190458451773804,
        "normalized_value": 15367.155935728224
      },
      {
        "timestamp": "2023-11-21T16:00:00+00:00",
        "portfolio_value": 1516497.0082386665,
        "drawdown_pct": -20.167764992451207,
        "normalized_value": 15181.306322924345
      },
      {
        "timestamp": "2023-11-25T11:00:00+00:00",
        "portfolio_value": 1493489.6890453128,
        "drawdown_pct": -21.37892841892625,
        "normalized_value": 14950.98528803535
      },
      {
        "timestamp": "2023-11-29T06:00:00+00:00",
        "portfolio_value": 1570866.4424841057,
        "drawdown_pct": -17.30562057124291,
        "normalized_value": 15725.586352096818
      },
      {
        "timestamp": "2023-12-03T01:00:00+00:00",
        "portfolio_value": 1736342.9258348208,
        "drawdown_pct": -8.594520295205978,
        "normalized_value": 17382.133756634877
      },
      {
        "timestamp": "2023-12-06T20:00:00+00:00",
        "portfolio_value": 1557438.3013872665,
        "drawdown_pct": -18.012511854201957,
        "normalized_value": 15591.16028845734
      },
      {
        "timestamp": "2023-12-10T15:00:00+00:00",
        "portfolio_value": 1821026.459355858,
        "drawdown_pct": -4.653286320650753,
        "normalized_value": 18229.881332730438
      },
      {
        "timestamp": "2023-12-14T10:00:00+00:00",
        "portfolio_value": 1621615.626633896,
        "drawdown_pct": -15.094193136923447,
        "normalized_value": 16233.624881702153
      },
      {
        "timestamp": "2023-12-18T05:00:00+00:00",
        "portfolio_value": 1563432.7122648754,
        "drawdown_pct": -18.140579228059877,
        "normalized_value": 15651.168971141224
      },
      {
        "timestamp": "2023-12-22T00:00:00+00:00",
        "portfolio_value": 2047512.7121786,
        "drawdown_pct": -0.7294322086700974,
        "normalized_value": 20497.1836507395
      },
      {
        "timestamp": "2023-12-25T19:00:00+00:00",
        "portfolio_value": 2680125.2649230277,
        "drawdown_pct": -2.063517606852126,
        "normalized_value": 26830.123903681186
      },
      {
        "timestamp": "2023-12-29T14:00:00+00:00",
        "portfolio_value": 2280300.703097789,
        "drawdown_pct": -16.673809025701214,
        "normalized_value": 22827.571234258
      },
      {
        "timestamp": "2024-01-02T09:00:00+00:00",
        "portfolio_value": 2353899.632230157,
        "drawdown_pct": -13.984374945337919,
        "normalized_value": 23564.35335919961
      },
      {
        "timestamp": "2024-01-06T04:00:00+00:00",
        "portfolio_value": 1861193.8633046264,
        "drawdown_pct": -31.98870873335713,
        "normalized_value": 18631.988069657757
      },
      {
        "timestamp": "2024-01-09T23:00:00+00:00",
        "portfolio_value": 1826287.912388496,
        "drawdown_pct": -33.26423346052389,
        "normalized_value": 18282.55254128424
      },
      {
        "timestamp": "2024-01-13T18:00:00+00:00",
        "portfolio_value": 1677572.8790396252,
        "drawdown_pct": -38.69854186236883,
        "normalized_value": 16793.80019701466
      },
      {
        "timestamp": "2024-01-17T13:00:00+00:00",
        "portfolio_value": 1724307.664669403,
        "drawdown_pct": -36.990770748129606,
        "normalized_value": 17261.65149690341
      },
      {
        "timestamp": "2024-01-21T08:00:00+00:00",
        "portfolio_value": 1627338.6392435424,
        "drawdown_pct": -40.534189175466906,
        "normalized_value": 16290.916658108563
      },
      {
        "timestamp": "2024-01-25T03:00:00+00:00",
        "portfolio_value": 1627338.6392435424,
        "drawdown_pct": -40.534189175466906,
        "normalized_value": 16290.916658108563
      },
      {
        "timestamp": "2024-01-28T22:00:00+00:00",
        "portfolio_value": 1647400.0183867365,
        "drawdown_pct": -39.80111116193006,
        "normalized_value": 16491.746557790913
      },
      {
        "timestamp": "2024-02-01T17:00:00+00:00",
        "portfolio_value": 1755041.603599759,
        "drawdown_pct": -35.86769866328385,
        "normalized_value": 17569.32196303489
      },
      {
        "timestamp": "2024-02-05T12:00:00+00:00",
        "portfolio_value": 1755041.603599759,
        "drawdown_pct": -35.86769866328385,
        "normalized_value": 17569.32196303489
      },
      {
        "timestamp": "2024-02-09T07:00:00+00:00",
        "portfolio_value": 1810376.708973673,
        "drawdown_pct": -33.84565676692087,
        "normalized_value": 18123.26910604203
      },
      {
        "timestamp": "2024-02-13T02:00:00+00:00",
        "portfolio_value": 1937567.0379972789,
        "drawdown_pct": -29.19789884977112,
        "normalized_value": 19396.542535353674
      },
      {
        "timestamp": "2024-02-16T21:00:00+00:00",
        "portfolio_value": 1862184.8614424365,
        "drawdown_pct": -31.95249592161079,
        "normalized_value": 18641.908726417256
      },
      {
        "timestamp": "2024-02-20T16:00:00+00:00",
        "portfolio_value": 1795003.3138551996,
        "drawdown_pct": -34.407427613996,
        "normalized_value": 17969.369547223934
      },
      {
        "timestamp": "2024-02-24T11:00:00+00:00",
        "portfolio_value": 1795003.3138551996,
        "drawdown_pct": -34.407427613996,
        "normalized_value": 17969.369547223934
      },
      {
        "timestamp": "2024-02-28T06:00:00+00:00",
        "portfolio_value": 1754319.645651715,
        "drawdown_pct": -35.8940802741707,
        "normalized_value": 17562.094606368846
      },
      {
        "timestamp": "2024-03-03T01:00:00+00:00",
        "portfolio_value": 1922375.3153070875,
        "drawdown_pct": -29.753031067373737,
        "normalized_value": 19244.461657857857
      },
      {
        "timestamp": "2024-03-06T20:00:00+00:00",
        "portfolio_value": 1687866.4414435653,
        "drawdown_pct": -38.322397020808516,
        "normalized_value": 16896.846706938115
      },
      {
        "timestamp": "2024-03-10T15:00:00+00:00",
        "portfolio_value": 1704790.4926172548,
        "drawdown_pct": -37.70396247915273,
        "normalized_value": 17066.269530522208
      },
      {
        "timestamp": "2024-03-14T10:00:00+00:00",
        "portfolio_value": 2059954.3627032896,
        "drawdown_pct": -24.7256511425137,
        "normalized_value": 20621.734181833166
      },
      {
        "timestamp": "2024-03-18T05:00:00+00:00",
        "portfolio_value": 2231283.657545267,
        "drawdown_pct": -18.464977924240202,
        "normalized_value": 22336.87274983312
      },
      {
        "timestamp": "2024-03-22T00:00:00+00:00",
        "portfolio_value": 1847656.1696323736,
        "drawdown_pct": -32.48339982684022,
        "normalized_value": 18496.46530012517
      },
      {
        "timestamp": "2024-03-25T19:00:00+00:00",
        "portfolio_value": 1954047.6608725365,
        "drawdown_pct": -28.595667956620265,
        "normalized_value": 19561.526299187455
      },
      {
        "timestamp": "2024-03-29T14:00:00+00:00",
        "portfolio_value": 1912221.5364306413,
        "drawdown_pct": -30.124067973436237,
        "normalized_value": 19142.814489006796
      },
      {
        "timestamp": "2024-04-02T09:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-06T04:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-09T23:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-13T18:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-17T13:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-21T08:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-25T03:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-04-28T22:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-05-02T17:00:00+00:00",
        "portfolio_value": 1921663.9075128285,
        "drawdown_pct": -29.779027157119486,
        "normalized_value": 19237.339916379646
      },
      {
        "timestamp": "2024-05-06T12:00:00+00:00",
        "portfolio_value": 2062345.985393907,
        "drawdown_pct": -24.638257050679183,
        "normalized_value": 20645.676172142325
      },
      {
        "timestamp": "2024-05-10T07:00:00+00:00",
        "portfolio_value": 2035553.4787980567,
        "drawdown_pct": -25.61730227846563,
        "normalized_value": 20377.462487854886
      },
      {
        "timestamp": "2024-05-14T02:00:00+00:00",
        "portfolio_value": 1947330.6727279369,
        "drawdown_pct": -28.84111849573059,
        "normalized_value": 19494.28405997654
      },
      {
        "timestamp": "2024-05-17T21:00:00+00:00",
        "portfolio_value": 2130276.722595435,
        "drawdown_pct": -22.155948654514688,
        "normalized_value": 21325.715318013274
      },
      {
        "timestamp": "2024-05-21T16:00:00+00:00",
        "portfolio_value": 2204399.2663958054,
        "drawdown_pct": -19.44738077492762,
        "normalized_value": 22067.739230196752
      },
      {
        "timestamp": "2024-05-25T11:00:00+00:00",
        "portfolio_value": 2025226.931797844,
        "drawdown_pct": -25.994652435081317,
        "normalized_value": 20274.085776647
      },
      {
        "timestamp": "2024-05-29T06:00:00+00:00",
        "portfolio_value": 1976045.794764666,
        "drawdown_pct": -27.79181752440156,
        "normalized_value": 19781.74460976427
      },
      {
        "timestamp": "2024-06-02T01:00:00+00:00",
        "portfolio_value": 1864844.7033852437,
        "drawdown_pct": -31.855300627417037,
        "normalized_value": 18668.535798600682
      },
      {
        "timestamp": "2024-06-05T20:00:00+00:00",
        "portfolio_value": 1950790.8327703075,
        "drawdown_pct": -28.71467817314483,
        "normalized_value": 19528.922934464383
      },
      {
        "timestamp": "2024-06-09T15:00:00+00:00",
        "portfolio_value": 1824533.3068292765,
        "drawdown_pct": -33.32834982803344,
        "normalized_value": 18264.98758446223
      },
      {
        "timestamp": "2024-06-13T10:00:00+00:00",
        "portfolio_value": 1824533.3068292765,
        "drawdown_pct": -33.32834982803344,
        "normalized_value": 18264.98758446223
      },
      {
        "timestamp": "2024-06-17T05:00:00+00:00",
        "portfolio_value": 1824533.3068292765,
        "drawdown_pct": -33.32834982803344,
        "normalized_value": 18264.98758446223
      },
      {
        "timestamp": "2024-06-21T00:00:00+00:00",
        "portfolio_value": 1824533.3068292765,
        "drawdown_pct": -33.32834982803344,
        "normalized_value": 18264.98758446223
      },
      {
        "timestamp": "2024-06-24T19:00:00+00:00",
        "portfolio_value": 1824533.3068292765,
        "drawdown_pct": -33.32834982803344,
        "normalized_value": 18264.98758446223
      },
      {
        "timestamp": "2024-06-28T14:00:00+00:00",
        "portfolio_value": 1919172.4337364016,
        "drawdown_pct": -29.87006998292749,
        "normalized_value": 19212.398339581283
      },
      {
        "timestamp": "2024-07-02T09:00:00+00:00",
        "portfolio_value": 2017094.615952978,
        "drawdown_pct": -26.291821533100734,
        "normalized_value": 20192.67501402435
      },
      {
        "timestamp": "2024-07-06T04:00:00+00:00",
        "portfolio_value": 1930903.6311577607,
        "drawdown_pct": -29.441391433926732,
        "normalized_value": 19329.83668639029
      },
      {
        "timestamp": "2024-07-09T23:00:00+00:00",
        "portfolio_value": 1873196.3918127324,
        "drawdown_pct": -31.55011527009931,
        "normalized_value": 18752.142650208378
      },
      {
        "timestamp": "2024-07-13T18:00:00+00:00",
        "portfolio_value": 1821801.079619741,
        "drawdown_pct": -33.42819020695261,
        "normalized_value": 18237.635879851907
      },
      {
        "timestamp": "2024-07-17T13:00:00+00:00",
        "portfolio_value": 2127843.8594469503,
        "drawdown_pct": -22.244849745081204,
        "normalized_value": 21301.360478868686
      },
      {
        "timestamp": "2024-07-21T08:00:00+00:00",
        "portfolio_value": 2249830.279152513,
        "drawdown_pct": -17.787251810366723,
        "normalized_value": 22522.53875665366
      },
      {
        "timestamp": "2024-07-25T03:00:00+00:00",
        "portfolio_value": 2232164.341727158,
        "drawdown_pct": -18.432796178109704,
        "normalized_value": 22345.689078692594
      },
      {
        "timestamp": "2024-07-28T22:00:00+00:00",
        "portfolio_value": 2295028.357760262,
        "drawdown_pct": -16.135634668546647,
        "normalized_value": 22975.006432372193
      },
      {
        "timestamp": "2024-08-01T17:00:00+00:00",
        "portfolio_value": 2242884.852687254,
        "drawdown_pct": -18.041049886758973,
        "normalized_value": 22453.009673418008
      },
      {
        "timestamp": "2024-08-05T12:00:00+00:00",
        "portfolio_value": 2242884.852687254,
        "drawdown_pct": -18.041049886758973,
        "normalized_value": 22453.009673418008
      },
      {
        "timestamp": "2024-08-09T07:00:00+00:00",
        "portfolio_value": 2231745.431708763,
        "drawdown_pct": -18.44810388561794,
        "normalized_value": 22341.495465862205
      },
      {
        "timestamp": "2024-08-13T02:00:00+00:00",
        "portfolio_value": 2151776.462603085,
        "drawdown_pct": -21.370310409812365,
        "normalized_value": 21540.94432087063
      },
      {
        "timestamp": "2024-08-16T21:00:00+00:00",
        "portfolio_value": 2151776.462603085,
        "drawdown_pct": -21.370310409812365,
        "normalized_value": 21540.94432087063
      },
      {
        "timestamp": "2024-08-20T16:00:00+00:00",
        "portfolio_value": 2151776.462603085,
        "drawdown_pct": -21.370310409812365,
        "normalized_value": 21540.94432087063
      },
      {
        "timestamp": "2024-08-24T11:00:00+00:00",
        "portfolio_value": 2241661.904277197,
        "drawdown_pct": -18.085738568663814,
        "normalized_value": 22440.767015285917
      },
      {
        "timestamp": "2024-08-28T06:00:00+00:00",
        "portfolio_value": 2237142.962350169,
        "drawdown_pct": -18.250868640106475,
        "normalized_value": 22395.528916379677
      },
      {
        "timestamp": "2024-09-01T01:00:00+00:00",
        "portfolio_value": 2237142.962350169,
        "drawdown_pct": -18.250868640106475,
        "normalized_value": 22395.528916379677
      },
      {
        "timestamp": "2024-09-04T20:00:00+00:00",
        "portfolio_value": 2237142.962350169,
        "drawdown_pct": -18.250868640106475,
        "normalized_value": 22395.528916379677
      },
      {
        "timestamp": "2024-09-08T15:00:00+00:00",
        "portfolio_value": 2237142.9623501687,
        "drawdown_pct": -18.250868640106493,
        "normalized_value": 22395.528916379673
      },
      {
        "timestamp": "2024-09-12T10:00:00+00:00",
        "portfolio_value": 2237142.962350169,
        "drawdown_pct": -18.250868640106475,
        "normalized_value": 22395.528916379677
      },
      {
        "timestamp": "2024-09-16T05:00:00+00:00",
        "portfolio_value": 2237142.962350169,
        "drawdown_pct": -18.250868640106475,
        "normalized_value": 22395.528916379677
      },
      {
        "timestamp": "2024-09-20T00:00:00+00:00",
        "portfolio_value": 2212433.475417996,
        "drawdown_pct": -19.153796672444553,
        "normalized_value": 22148.167867751363
      },
      {
        "timestamp": "2024-09-23T19:00:00+00:00",
        "portfolio_value": 2216523.8172379485,
        "drawdown_pct": -19.004328401362358,
        "normalized_value": 22189.11534855536
      },
      {
        "timestamp": "2024-09-27T14:00:00+00:00",
        "portfolio_value": 2468154.4558427907,
        "drawdown_pct": -9.809303105405737,
        "normalized_value": 24708.132388575774
      },
      {
        "timestamp": "2024-10-01T09:00:00+00:00",
        "portfolio_value": 2369080.693085898,
        "drawdown_pct": -13.429632329883972,
        "normalized_value": 23716.32750349788
      },
      {
        "timestamp": "2024-10-05T04:00:00+00:00",
        "portfolio_value": 2369080.693085898,
        "drawdown_pct": -13.429632329883972,
        "normalized_value": 23716.32750349788
      },
      {
        "timestamp": "2024-10-08T23:00:00+00:00",
        "portfolio_value": 2325764.0035831737,
        "drawdown_pct": -15.012500210850298,
        "normalized_value": 23282.693985816473
      },
      {
        "timestamp": "2024-10-12T18:00:00+00:00",
        "portfolio_value": 2328685.533714968,
        "drawdown_pct": -14.905742370812655,
        "normalized_value": 23311.940758887216
      },
      {
        "timestamp": "2024-10-16T13:00:00+00:00",
        "portfolio_value": 2457641.946403986,
        "drawdown_pct": -10.193448656002186,
        "normalized_value": 24602.894049729006
      },
      {
        "timestamp": "2024-10-20T08:00:00+00:00",
        "portfolio_value": 2530687.9564700765,
        "drawdown_pct": -7.524219209022273,
        "normalized_value": 25334.141027768685
      },
      {
        "timestamp": "2024-10-24T03:00:00+00:00",
        "portfolio_value": 2793101.398701107,
        "drawdown_pct": -0.2503569979486011,
        "normalized_value": 27961.102260214004
      },
      {
        "timestamp": "2024-10-27T22:00:00+00:00",
        "portfolio_value": 2728261.258769219,
        "drawdown_pct": -4.994992200439688,
        "normalized_value": 27312.002380043083
      },
      {
        "timestamp": "2024-10-31T17:00:00+00:00",
        "portfolio_value": 2608124.108233041,
        "drawdown_pct": -9.178473854559375,
        "normalized_value": 26109.33671493156
      },
      {
        "timestamp": "2024-11-04T12:00:00+00:00",
        "portfolio_value": 2594304.648551814,
        "drawdown_pct": -9.65970264838574,
        "normalized_value": 25970.993250026426
      },
      {
        "timestamp": "2024-11-08T07:00:00+00:00",
        "portfolio_value": 2823463.937245775,
        "drawdown_pct": -1.6797923887894868,
        "normalized_value": 28265.05472163268
      },
      {
        "timestamp": "2024-11-12T02:00:00+00:00",
        "portfolio_value": 3082194.798546084,
        "drawdown_pct": -2.1738436334508906,
        "normalized_value": 30855.150474710405
      },
      {
        "timestamp": "2024-11-15T21:00:00+00:00",
        "portfolio_value": 3100096.68870045,
        "drawdown_pct": -2.871914832305528,
        "normalized_value": 31034.36222172758
      },
      {
        "timestamp": "2024-11-19T16:00:00+00:00",
        "portfolio_value": 3450191.724031611,
        "drawdown_pct": -2.2572208254163932,
        "normalized_value": 34539.08392221439
      },
      {
        "timestamp": "2024-11-23T11:00:00+00:00",
        "portfolio_value": 3662111.080195934,
        "drawdown_pct": -2.5411166146068367,
        "normalized_value": 36660.560353891706
      },
      {
        "timestamp": "2024-11-27T06:00:00+00:00",
        "portfolio_value": 3414108.624169706,
        "drawdown_pct": -9.26686249574776,
        "normalized_value": 34177.86422371958
      },
      {
        "timestamp": "2024-12-01T01:00:00+00:00",
        "portfolio_value": 3414108.624169706,
        "drawdown_pct": -9.26686249574776,
        "normalized_value": 34177.86422371958
      },
      {
        "timestamp": "2024-12-04T20:00:00+00:00",
        "portfolio_value": 3414108.624169706,
        "drawdown_pct": -9.26686249574776,
        "normalized_value": 34177.86422371958
      },
      {
        "timestamp": "2024-12-08T15:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2024-12-12T10:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2024-12-16T05:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2024-12-20T00:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2024-12-23T19:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2024-12-27T14:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2024-12-31T09:00:00+00:00",
        "portfolio_value": 3320601.7300371416,
        "drawdown_pct": -11.751895872501795,
        "normalized_value": 33241.787993156846
      },
      {
        "timestamp": "2025-01-04T04:00:00+00:00",
        "portfolio_value": 3478974.521795188,
        "drawdown_pct": -7.542990452859569,
        "normalized_value": 34827.22195829739
      },
      {
        "timestamp": "2025-01-07T23:00:00+00:00",
        "portfolio_value": 3306673.5568655995,
        "drawdown_pct": -12.122050132565475,
        "normalized_value": 33102.356222248505
      },
      {
        "timestamp": "2025-01-11T18:00:00+00:00",
        "portfolio_value": 3306673.5568655995,
        "drawdown_pct": -12.122050132565475,
        "normalized_value": 33102.356222248505
      },
      {
        "timestamp": "2025-01-15T13:00:00+00:00",
        "portfolio_value": 3306673.5568655995,
        "drawdown_pct": -12.122050132565475,
        "normalized_value": 33102.356222248505
      },
      {
        "timestamp": "2025-01-19T08:00:00+00:00",
        "portfolio_value": 4645926.28048107,
        "drawdown_pct": -0.06952184887668807,
        "normalized_value": 46509.31036100497
      },
      {
        "timestamp": "2025-01-23T03:00:00+00:00",
        "portfolio_value": 4102468.590363424,
        "drawdown_pct": -15.524501885356134,
        "normalized_value": 41068.87914195015
      },
      {
        "timestamp": "2025-01-26T22:00:00+00:00",
        "portfolio_value": 4073756.168204407,
        "drawdown_pct": -16.115730339769723,
        "normalized_value": 40781.44562002358
      },
      {
        "timestamp": "2025-01-30T17:00:00+00:00",
        "portfolio_value": 3642196.818867493,
        "drawdown_pct": -25.002133781566194,
        "normalized_value": 36461.20321716186
      },
      {
        "timestamp": "2025-02-03T12:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-02-07T07:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-02-11T02:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-02-14T21:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-02-18T16:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-02-22T11:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-02-26T06:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-03-02T01:00:00+00:00",
        "portfolio_value": 3479319.747271121,
        "drawdown_pct": -28.35599779087946,
        "normalized_value": 34830.677931947284
      },
      {
        "timestamp": "2025-03-05T20:00:00+00:00",
        "portfolio_value": 3225141.150902139,
        "drawdown_pct": -33.58988637903039,
        "normalized_value": 32286.153866785953
      },
      {
        "timestamp": "2025-03-09T15:00:00+00:00",
        "portfolio_value": 3225141.150902139,
        "drawdown_pct": -33.58988637903039,
        "normalized_value": 32286.153866785953
      },
      {
        "timestamp": "2025-03-13T10:00:00+00:00",
        "portfolio_value": 3225141.150902139,
        "drawdown_pct": -33.58988637903039,
        "normalized_value": 32286.153866785953
      },
      {
        "timestamp": "2025-03-17T05:00:00+00:00",
        "portfolio_value": 3225141.150902139,
        "drawdown_pct": -33.58988637903039,
        "normalized_value": 32286.153866785953
      },
      {
        "timestamp": "2025-03-21T00:00:00+00:00",
        "portfolio_value": 3225141.150902139,
        "drawdown_pct": -33.58988637903039,
        "normalized_value": 32286.153866785953
      },
      {
        "timestamp": "2025-03-24T19:00:00+00:00",
        "portfolio_value": 3279339.478949056,
        "drawdown_pct": -32.473867899449885,
        "normalized_value": 32828.72099075689
      },
      {
        "timestamp": "2025-03-28T14:00:00+00:00",
        "portfolio_value": 3197353.9247304383,
        "drawdown_pct": -34.162063769393,
        "normalized_value": 32007.982271269964
      },
      {
        "timestamp": "2025-04-01T09:00:00+00:00",
        "portfolio_value": 3197353.9247304383,
        "drawdown_pct": -34.162063769393,
        "normalized_value": 32007.982271269964
      },
      {
        "timestamp": "2025-04-05T04:00:00+00:00",
        "portfolio_value": 3197353.9247304383,
        "drawdown_pct": -34.162063769393,
        "normalized_value": 32007.982271269964
      },
      {
        "timestamp": "2025-04-08T23:00:00+00:00",
        "portfolio_value": 3197353.9247304383,
        "drawdown_pct": -34.162063769393,
        "normalized_value": 32007.982271269964
      },
      {
        "timestamp": "2025-04-12T18:00:00+00:00",
        "portfolio_value": 3485986.814353812,
        "drawdown_pct": -28.218713665390478,
        "normalized_value": 34897.42042277183
      },
      {
        "timestamp": "2025-04-16T13:00:00+00:00",
        "portfolio_value": 3409084.219809479,
        "drawdown_pct": -29.802244370707964,
        "normalized_value": 34127.56605546178
      },
      {
        "timestamp": "2025-04-20T08:00:00+00:00",
        "portfolio_value": 3595501.683450585,
        "drawdown_pct": -25.963651155066568,
        "normalized_value": 35993.748846527895
      },
      {
        "timestamp": "2025-04-24T03:00:00+00:00",
        "portfolio_value": 3818435.852713953,
        "drawdown_pct": -21.3731285025509,
        "normalized_value": 38225.492064646525
      },
      {
        "timestamp": "2025-04-27T22:00:00+00:00",
        "portfolio_value": 3839809.467550975,
        "drawdown_pct": -20.933016233535813,
        "normalized_value": 38439.45845713279
      },
      {
        "timestamp": "2025-05-01T17:00:00+00:00",
        "portfolio_value": 3770371.300225073,
        "drawdown_pct": -22.36284406617351,
        "normalized_value": 37744.32877144924
      },
      {
        "timestamp": "2025-05-05T12:00:00+00:00",
        "portfolio_value": 3691906.304474989,
        "drawdown_pct": -23.978546771642502,
        "normalized_value": 36958.83356134493
      },
      {
        "timestamp": "2025-05-09T07:00:00+00:00",
        "portfolio_value": 4171630.382848948,
        "drawdown_pct": -14.100364992647178,
        "normalized_value": 41761.24210202268
      },
      {
        "timestamp": "2025-05-13T02:00:00+00:00",
        "portfolio_value": 4277478.993619652,
        "drawdown_pct": -11.920795808229306,
        "normalized_value": 42820.86844829053
      },
      {
        "timestamp": "2025-05-16T21:00:00+00:00",
        "portfolio_value": 4278322.0254009785,
        "drawdown_pct": -11.903436618735137,
        "normalized_value": 42829.307847539414
      },
      {
        "timestamp": "2025-05-20T16:00:00+00:00",
        "portfolio_value": 3945394.146265249,
        "drawdown_pct": -18.75888177492859,
        "normalized_value": 39496.44263966715
      },
      {
        "timestamp": "2025-05-24T11:00:00+00:00",
        "portfolio_value": 4215772.5998390615,
        "drawdown_pct": -13.191415737829306,
        "normalized_value": 42203.13978745119
      },
      {
        "timestamp": "2025-05-28T06:00:00+00:00",
        "portfolio_value": 4137312.526972252,
        "drawdown_pct": -14.807017074327982,
        "normalized_value": 41417.69385921066
      },
      {
        "timestamp": "2025-06-01T01:00:00+00:00",
        "portfolio_value": 3968216.3982661576,
        "drawdown_pct": -18.288940064612188,
        "normalized_value": 39724.91100902263
      },
      {
        "timestamp": "2025-06-04T20:00:00+00:00",
        "portfolio_value": 3968216.3982661576,
        "drawdown_pct": -18.288940064612188,
        "normalized_value": 39724.91100902263
      },
      {
        "timestamp": "2025-06-08T15:00:00+00:00",
        "portfolio_value": 3968216.3982661576,
        "drawdown_pct": -18.288940064612188,
        "normalized_value": 39724.91100902263
      },
      {
        "timestamp": "2025-06-12T10:00:00+00:00",
        "portfolio_value": 3852020.294164809,
        "drawdown_pct": -20.68157843751754,
        "normalized_value": 38561.69826260134
      },
      {
        "timestamp": "2025-06-16T05:00:00+00:00",
        "portfolio_value": 3852020.294164809,
        "drawdown_pct": -20.68157843751754,
        "normalized_value": 38561.69826260134
      },
      {
        "timestamp": "2025-06-20T00:00:00+00:00",
        "portfolio_value": 3852020.294164809,
        "drawdown_pct": -20.68157843751754,
        "normalized_value": 38561.69826260134
      },
      {
        "timestamp": "2025-06-23T19:00:00+00:00",
        "portfolio_value": 3852020.294164809,
        "drawdown_pct": -20.68157843751754,
        "normalized_value": 38561.69826260134
      },
      {
        "timestamp": "2025-06-27T14:00:00+00:00",
        "portfolio_value": 3815324.273752585,
        "drawdown_pct": -21.43720021374012,
        "normalized_value": 38194.34275600704
      },
      {
        "timestamp": "2025-07-01T09:00:00+00:00",
        "portfolio_value": 3889856.236550248,
        "drawdown_pct": -19.90248409243108,
        "normalized_value": 38940.46526856926
      },
      {
        "timestamp": "2025-07-05T04:00:00+00:00",
        "portfolio_value": 3889856.236550248,
        "drawdown_pct": -19.90248409243108,
        "normalized_value": 38940.46526856926
      },
      {
        "timestamp": "2025-07-08T23:00:00+00:00",
        "portfolio_value": 3759789.457110802,
        "drawdown_pct": -22.580738840590257,
        "normalized_value": 37638.39634896101
      },
      {
        "timestamp": "2025-07-12T18:00:00+00:00",
        "portfolio_value": 3966982.8656490464,
        "drawdown_pct": -18.314340206008982,
        "normalized_value": 39712.562394803186
      },
      {
        "timestamp": "2025-07-16T13:00:00+00:00",
        "portfolio_value": 4144961.89936213,
        "drawdown_pct": -14.649505924963616,
        "normalized_value": 41494.269984846236
      },
      {
        "timestamp": "2025-07-20T08:00:00+00:00",
        "portfolio_value": 4502080.555546409,
        "drawdown_pct": -7.29593923634534,
        "normalized_value": 45069.303554784805
      },
      {
        "timestamp": "2025-07-24T03:00:00+00:00",
        "portfolio_value": 4745307.929891641,
        "drawdown_pct": -8.164025008504426,
        "normalized_value": 47504.19742928326
      },
      {
        "timestamp": "2025-07-27T22:00:00+00:00",
        "portfolio_value": 4529390.418693242,
        "drawdown_pct": -12.342677996171547,
        "normalized_value": 45342.69637772927
      },
      {
        "timestamp": "2025-07-31T17:00:00+00:00",
        "portfolio_value": 4529390.418693242,
        "drawdown_pct": -12.342677996171547,
        "normalized_value": 45342.69637772927
      },
      {
        "timestamp": "2025-08-04T12:00:00+00:00",
        "portfolio_value": 4529390.418693242,
        "drawdown_pct": -12.342677996171547,
        "normalized_value": 45342.69637772927
      },
      {
        "timestamp": "2025-08-08T07:00:00+00:00",
        "portfolio_value": 4529390.418693242,
        "drawdown_pct": -12.342677996171547,
        "normalized_value": 45342.69637772927
      },
      {
        "timestamp": "2025-08-12T02:00:00+00:00",
        "portfolio_value": 4608490.068666028,
        "drawdown_pct": -10.811861959770992,
        "normalized_value": 46134.544966779686
      },
      {
        "timestamp": "2025-08-15T21:00:00+00:00",
        "portfolio_value": 4689800.55179045,
        "drawdown_pct": -11.536656635798424,
        "normalized_value": 46948.525703220774
      },
      {
        "timestamp": "2025-08-19T16:00:00+00:00",
        "portfolio_value": 4493316.092937012,
        "drawdown_pct": -15.24292771435103,
        "normalized_value": 44981.56451480898
      },
      {
        "timestamp": "2025-08-23T11:00:00+00:00",
        "portfolio_value": 4657466.812288428,
        "drawdown_pct": -12.146565451371405,
        "normalized_value": 46624.83999775674
      },
      {
        "timestamp": "2025-08-27T06:00:00+00:00",
        "portfolio_value": 4375293.95462501,
        "drawdown_pct": -17.469170137834475,
        "normalized_value": 43800.071755596706
      },
      {
        "timestamp": "2025-08-31T01:00:00+00:00",
        "portfolio_value": 4463706.630205613,
        "drawdown_pct": -15.801448708903202,
        "normalized_value": 44685.15092392121
      },
      {
        "timestamp": "2025-09-03T20:00:00+00:00",
        "portfolio_value": 4526475.997423264,
        "drawdown_pct": -14.617435012881765,
        "normalized_value": 45313.52076985586
      },
      {
        "timestamp": "2025-09-07T15:00:00+00:00",
        "portfolio_value": 4422733.611767449,
        "drawdown_pct": -16.574319571690577,
        "normalized_value": 44274.97936373657
      },
      {
        "timestamp": "2025-09-11T10:00:00+00:00",
        "portfolio_value": 4898620.492088812,
        "drawdown_pct": -7.597702329339992,
        "normalized_value": 49038.97458823783
      },
      {
        "timestamp": "2025-09-15T05:00:00+00:00",
        "portfolio_value": 5320518.848392823,
        "drawdown_pct": -2.0114626540264062,
        "normalized_value": 53262.50298914678
      },
      {
        "timestamp": "2025-09-19T00:00:00+00:00",
        "portfolio_value": 5088520.433302867,
        "drawdown_pct": -6.2842010859400235,
        "normalized_value": 50940.01966950988
      },
      {
        "timestamp": "2025-09-22T19:00:00+00:00",
        "portfolio_value": 4733714.164599103,
        "drawdown_pct": -12.818704261690723,
        "normalized_value": 47388.13488422982
      },
      {
        "timestamp": "2025-09-26T14:00:00+00:00",
        "portfolio_value": 4733714.164599103,
        "drawdown_pct": -12.818704261690723,
        "normalized_value": 47388.13488422982
      },
      {
        "timestamp": "2025-09-30T09:00:00+00:00",
        "portfolio_value": 4733714.164599103,
        "drawdown_pct": -12.818704261690723,
        "normalized_value": 47388.13488422982
      },
      {
        "timestamp": "2025-10-04T04:00:00+00:00",
        "portfolio_value": 4955187.557653244,
        "drawdown_pct": -8.739806232228677,
        "normalized_value": 49605.254604259746
      },
      {
        "timestamp": "2025-10-07T23:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-10-11T18:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-10-15T13:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-10-19T08:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-10-23T03:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-10-26T22:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-10-30T17:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-03T12:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-07T07:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-11T02:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-14T21:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-18T16:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-22T11:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-26T06:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-11-30T01:00:00+00:00",
        "portfolio_value": 4624335.986729587,
        "drawdown_pct": -14.83313329998632,
        "normalized_value": 46293.17484523244
      },
      {
        "timestamp": "2025-12-03T20:00:00+00:00",
        "portfolio_value": 4696944.399155219,
        "drawdown_pct": -13.495896775629296,
        "normalized_value": 47020.04113291143
      },
      {
        "timestamp": "2025-12-07T15:00:00+00:00",
        "portfolio_value": 4615754.218196526,
        "drawdown_pct": -14.991184604822951,
        "normalized_value": 46207.26471406498
      },
      {
        "timestamp": "2025-12-11T10:00:00+00:00",
        "portfolio_value": 4277637.61903823,
        "drawdown_pct": -21.218312437275685,
        "normalized_value": 42822.45641124525
      },
      {
        "timestamp": "2025-12-15T05:00:00+00:00",
        "portfolio_value": 4277637.61903823,
        "drawdown_pct": -21.218312437275685,
        "normalized_value": 42822.45641124525
      },
      {
        "timestamp": "2025-12-19T00:00:00+00:00",
        "portfolio_value": 4277637.619038231,
        "drawdown_pct": -21.218312437275667,
        "normalized_value": 42822.45641124526
      },
      {
        "timestamp": "2025-12-22T19:00:00+00:00",
        "portfolio_value": 4277637.619038231,
        "drawdown_pct": -21.218312437275667,
        "normalized_value": 42822.45641124526
      },
      {
        "timestamp": "2025-12-26T14:00:00+00:00",
        "portfolio_value": 4277637.619038231,
        "drawdown_pct": -21.218312437275667,
        "normalized_value": 42822.45641124526
      },
      {
        "timestamp": "2025-12-30T09:00:00+00:00",
        "portfolio_value": 4277637.619038231,
        "drawdown_pct": -21.218312437275667,
        "normalized_value": 42822.45641124526
      },
      {
        "timestamp": "2026-01-03T04:00:00+00:00",
        "portfolio_value": 4268200.865769701,
        "drawdown_pct": -21.392110083036734,
        "normalized_value": 42727.987222526055
      },
      {
        "timestamp": "2026-01-06T23:00:00+00:00",
        "portfolio_value": 4550881.45673378,
        "drawdown_pct": -16.185952857781768,
        "normalized_value": 45557.83826717485
      },
      {
        "timestamp": "2026-01-10T18:00:00+00:00",
        "portfolio_value": 4382642.672318864,
        "drawdown_pct": -19.28442367978854,
        "normalized_value": 43873.63809551364
      },
      {
        "timestamp": "2026-01-14T13:00:00+00:00",
        "portfolio_value": 4616221.693802424,
        "drawdown_pct": -14.982575058992536,
        "normalized_value": 46211.94450593603
      },
      {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "portfolio_value": 4552307.881913853,
        "drawdown_pct": -16.159682240007143,
        "normalized_value": 45572.11788493062
      },
      {
        "timestamp": "2026-01-22T03:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-01-25T22:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-01-29T17:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-02-02T12:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-02-06T07:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-02-10T02:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-02-13T21:00:00+00:00",
        "portfolio_value": 4264179.776750127,
        "drawdown_pct": -21.466166888922324,
        "normalized_value": 42687.73301574143
      },
      {
        "timestamp": "2026-02-17T16:00:00+00:00",
        "portfolio_value": 4163414.392776437,
        "drawdown_pct": -23.32197322511637,
        "normalized_value": 41678.9936957553
      },
      {
        "timestamp": "2026-02-21T11:00:00+00:00",
        "portfolio_value": 4163414.392776437,
        "drawdown_pct": -23.32197322511637,
        "normalized_value": 41678.9936957553
      },
      {
        "timestamp": "2026-02-25T06:00:00+00:00",
        "portfolio_value": 4163414.392776437,
        "drawdown_pct": -23.32197322511637,
        "normalized_value": 41678.9936957553
      },
      {
        "timestamp": "2026-03-01T01:00:00+00:00",
        "portfolio_value": 4029323.673977248,
        "drawdown_pct": -25.791535645851297,
        "normalized_value": 40336.64203525586
      },
      {
        "timestamp": "2026-03-04T20:00:00+00:00",
        "portfolio_value": 4083332.455627141,
        "drawdown_pct": -24.79685041523517,
        "normalized_value": 40877.31165339593
      },
      {
        "timestamp": "2026-03-08T15:00:00+00:00",
        "portfolio_value": 3633459.766257488,
        "drawdown_pct": -33.08220153969324,
        "normalized_value": 36373.73857245014
      },
      {
        "timestamp": "2026-03-12T10:00:00+00:00",
        "portfolio_value": 3786008.325095,
        "drawdown_pct": -30.27269919966499,
        "normalized_value": 37900.867467694516
      },
      {
        "timestamp": "2026-03-16T05:00:00+00:00",
        "portfolio_value": 4123844.429669646,
        "drawdown_pct": -24.050737264518247,
        "normalized_value": 41282.867803091176
      },
      {
        "timestamp": "2026-03-20T00:00:00+00:00",
        "portfolio_value": 3883990.6223744177,
        "drawdown_pct": -28.46814925448273,
        "normalized_value": 38881.74594034665
      },
      {
        "timestamp": "2026-03-23T19:00:00+00:00",
        "portfolio_value": 3883990.6223744177,
        "drawdown_pct": -28.46814925448273,
        "normalized_value": 38881.74594034665
      },
      {
        "timestamp": "2026-03-27T14:00:00+00:00",
        "portfolio_value": 3644171.275230979,
        "drawdown_pct": -32.88492658832309,
        "normalized_value": 36480.96905033693
      },
      {
        "timestamp": "2026-03-31T09:00:00+00:00",
        "portfolio_value": 3644171.275230979,
        "drawdown_pct": -32.88492658832309,
        "normalized_value": 36480.96905033693
      },
      {
        "timestamp": "2026-04-04T04:00:00+00:00",
        "portfolio_value": 3644171.275230979,
        "drawdown_pct": -32.88492658832309,
        "normalized_value": 36480.96905033693
      },
      {
        "timestamp": "2026-04-07T23:00:00+00:00",
        "portfolio_value": 3644171.275230979,
        "drawdown_pct": -32.88492658832309,
        "normalized_value": 36480.96905033693
      },
      {
        "timestamp": "2026-04-11T18:00:00+00:00",
        "portfolio_value": 3649224.9478998873,
        "drawdown_pct": -32.79185258423395,
        "normalized_value": 36531.56021697009
      },
      {
        "timestamp": "2026-04-15T13:00:00+00:00",
        "portfolio_value": 3438408.767194192,
        "drawdown_pct": -36.67447564879759,
        "normalized_value": 34421.1274236741
      },
      {
        "timestamp": "2026-04-19T08:00:00+00:00",
        "portfolio_value": 3492070.52957278,
        "drawdown_pct": -35.686181507433275,
        "normalized_value": 34958.323110887155
      },
      {
        "timestamp": "2026-04-23T03:00:00+00:00",
        "portfolio_value": 3317802.6017274484,
        "drawdown_pct": -38.8956916778626,
        "normalized_value": 33213.7665568627
      },
      {
        "timestamp": "2026-04-26T22:00:00+00:00",
        "portfolio_value": 3273121.5457775346,
        "drawdown_pct": -39.71858723454844,
        "normalized_value": 32766.474677272912
      },
      {
        "timestamp": "2026-04-30T17:00:00+00:00",
        "portfolio_value": 3214804.2791910744,
        "drawdown_pct": -40.792622270914976,
        "normalized_value": 32182.673797248124
      },
      {
        "timestamp": "2026-05-04T12:00:00+00:00",
        "portfolio_value": 3214804.2791910744,
        "drawdown_pct": -40.792622270914976,
        "normalized_value": 32182.673797248124
      },
      {
        "timestamp": "2026-05-08T07:00:00+00:00",
        "portfolio_value": 3182727.7406956456,
        "drawdown_pct": -41.38337914629722,
        "normalized_value": 31861.562872509956
      },
      {
        "timestamp": "2026-05-12T02:00:00+00:00",
        "portfolio_value": 3493843.2902134107,
        "drawdown_pct": -35.65353239421899,
        "normalized_value": 34976.06981409623
      },
      {
        "timestamp": "2026-05-15T21:00:00+00:00",
        "portfolio_value": 3272506.669119119,
        "drawdown_pct": -39.72991148057089,
        "normalized_value": 32760.31928702057
      },
      {
        "timestamp": "2026-05-19T16:00:00+00:00",
        "portfolio_value": 3272506.669119119,
        "drawdown_pct": -39.72991148057089,
        "normalized_value": 32760.31928702057
      },
      {
        "timestamp": "2026-05-23T11:00:00+00:00",
        "portfolio_value": 3272506.669119119,
        "drawdown_pct": -39.72991148057089,
        "normalized_value": 32760.31928702057
      },
      {
        "timestamp": "2026-05-27T06:00:00+00:00",
        "portfolio_value": 3272506.669119119,
        "drawdown_pct": -39.72991148057089,
        "normalized_value": 32760.31928702057
      },
      {
        "timestamp": "2026-05-31T01:00:00+00:00",
        "portfolio_value": 3272506.669119119,
        "drawdown_pct": -39.72991148057089,
        "normalized_value": 32760.31928702057
      }
    ],
    "btcOnlyEquity": [
      {
        "timestamp": "2021-01-01T00:00:00+00:00",
        "portfolio_value": 9989.23923923924,
        "drawdown_pct": 0.0,
        "normalized_value": 100.0
      },
      {
        "timestamp": "2021-01-04T19:00:00+00:00",
        "portfolio_value": 10968.257272081017,
        "drawdown_pct": -8.690140588394323,
        "normalized_value": 109.80072665590035
      },
      {
        "timestamp": "2021-01-08T14:00:00+00:00",
        "portfolio_value": 14580.655902540988,
        "drawdown_pct": 0.0,
        "normalized_value": 145.9636269923937
      },
      {
        "timestamp": "2021-01-12T09:00:00+00:00",
        "portfolio_value": 12362.979486932452,
        "drawdown_pct": -15.209716424499522,
        "normalized_value": 123.76297324393637
      },
      {
        "timestamp": "2021-01-16T04:00:00+00:00",
        "portfolio_value": 12362.979486932452,
        "drawdown_pct": -15.209716424499522,
        "normalized_value": 123.76297324393637
      },
      {
        "timestamp": "2021-01-19T23:00:00+00:00",
        "portfolio_value": 12362.979486932452,
        "drawdown_pct": -15.209716424499522,
        "normalized_value": 123.76297324393637
      },
      {
        "timestamp": "2021-01-23T18:00:00+00:00",
        "portfolio_value": 12362.979486932452,
        "drawdown_pct": -15.209716424499522,
        "normalized_value": 123.76297324393637
      },
      {
        "timestamp": "2021-01-27T13:00:00+00:00",
        "portfolio_value": 11811.202900989902,
        "drawdown_pct": -18.994022080093465,
        "normalized_value": 118.23926345255316
      },
      {
        "timestamp": "2021-01-31T08:00:00+00:00",
        "portfolio_value": 11693.882145390404,
        "drawdown_pct": -19.798654988130558,
        "normalized_value": 117.06479207600783
      },
      {
        "timestamp": "2021-02-04T03:00:00+00:00",
        "portfolio_value": 13156.3219997724,
        "drawdown_pct": -9.768654526168241,
        "normalized_value": 131.7049445376419
      },
      {
        "timestamp": "2021-02-07T22:00:00+00:00",
        "portfolio_value": 13643.121301883311,
        "drawdown_pct": -6.429989205727647,
        "normalized_value": 136.57818153249423
      },
      {
        "timestamp": "2021-02-11T18:00:00+00:00",
        "portfolio_value": 16968.373809488643,
        "drawdown_pct": -1.006496747081457,
        "normalized_value": 169.86652740114891
      },
      {
        "timestamp": "2021-02-15T13:00:00+00:00",
        "portfolio_value": 17083.121161835617,
        "drawdown_pct": -3.1443460377978156,
        "normalized_value": 171.0152370235617
      },
      {
        "timestamp": "2021-02-19T08:00:00+00:00",
        "portfolio_value": 18688.094824612584,
        "drawdown_pct": -0.2295173040086374,
        "normalized_value": 187.0822629935914
      },
      {
        "timestamp": "2021-02-23T03:00:00+00:00",
        "portfolio_value": 18564.46471720027,
        "drawdown_pct": -11.264874494743577,
        "normalized_value": 185.84463013235532
      },
      {
        "timestamp": "2021-02-26T22:00:00+00:00",
        "portfolio_value": 18123.54151787326,
        "drawdown_pct": -13.372415758468314,
        "normalized_value": 181.43064835889857
      },
      {
        "timestamp": "2021-03-02T17:00:00+00:00",
        "portfolio_value": 18123.54151787326,
        "drawdown_pct": -13.372415758468314,
        "normalized_value": 181.43064835889857
      },
      {
        "timestamp": "2021-03-06T13:00:00+00:00",
        "portfolio_value": 18123.54151787326,
        "drawdown_pct": -13.372415758468314,
        "normalized_value": 181.43064835889857
      },
      {
        "timestamp": "2021-03-10T08:00:00+00:00",
        "portfolio_value": 17961.37316219894,
        "drawdown_pct": -14.147554153943254,
        "normalized_value": 179.80721786744235
      },
      {
        "timestamp": "2021-03-14T03:00:00+00:00",
        "portfolio_value": 18868.50352701206,
        "drawdown_pct": -9.811618375695788,
        "normalized_value": 188.88829344374625
      },
      {
        "timestamp": "2021-03-17T22:00:00+00:00",
        "portfolio_value": 16780.576578021348,
        "drawdown_pct": -19.791570003002345,
        "normalized_value": 167.98653206847536
      },
      {
        "timestamp": "2021-03-21T17:00:00+00:00",
        "portfolio_value": 16268.712083926062,
        "drawdown_pct": -22.238199131131587,
        "normalized_value": 162.86237314269246
      },
      {
        "timestamp": "2021-03-25T12:00:00+00:00",
        "portfolio_value": 16268.712083926062,
        "drawdown_pct": -22.238199131131587,
        "normalized_value": 162.86237314269246
      },
      {
        "timestamp": "2021-03-29T07:00:00+00:00",
        "portfolio_value": 15908.562311967966,
        "drawdown_pct": -23.959656533874885,
        "normalized_value": 159.25699576276773
      },
      {
        "timestamp": "2021-04-02T02:00:00+00:00",
        "portfolio_value": 16064.310376514768,
        "drawdown_pct": -23.21520608699739,
        "normalized_value": 160.81615418130875
      },
      {
        "timestamp": "2021-04-05T21:00:00+00:00",
        "portfolio_value": 15745.10539657019,
        "drawdown_pct": -24.74095403549825,
        "normalized_value": 157.6206657932572
      },
      {
        "timestamp": "2021-04-09T16:00:00+00:00",
        "portfolio_value": 15220.479984142392,
        "drawdown_pct": -27.248578280214332,
        "normalized_value": 152.368760219037
      },
      {
        "timestamp": "2021-04-13T11:00:00+00:00",
        "portfolio_value": 16413.04946680815,
        "drawdown_pct": -21.54828988891813,
        "normalized_value": 164.30730182469966
      },
      {
        "timestamp": "2021-04-17T06:00:00+00:00",
        "portfolio_value": 15790.029108269253,
        "drawdown_pct": -24.526225991544205,
        "normalized_value": 158.07038684431177
      },
      {
        "timestamp": "2021-04-21T03:00:00+00:00",
        "portfolio_value": 15790.029108269253,
        "drawdown_pct": -24.526225991544205,
        "normalized_value": 158.07038684431177
      },
      {
        "timestamp": "2021-04-24T22:00:00+00:00",
        "portfolio_value": 15790.029108269253,
        "drawdown_pct": -24.526225991544205,
        "normalized_value": 158.07038684431177
      },
      {
        "timestamp": "2021-04-28T20:00:00+00:00",
        "portfolio_value": 15790.029108269253,
        "drawdown_pct": -24.526225991544205,
        "normalized_value": 158.07038684431177
      },
      {
        "timestamp": "2021-05-02T15:00:00+00:00",
        "portfolio_value": 15813.59849178002,
        "drawdown_pct": -24.413568167267073,
        "normalized_value": 158.30633457712995
      },
      {
        "timestamp": "2021-05-06T10:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-05-10T05:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-05-14T00:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-05-17T19:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-05-21T14:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-05-25T09:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-05-29T04:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-06-01T23:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-06-05T18:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-06-09T13:00:00+00:00",
        "portfolio_value": 15523.695795439731,
        "drawdown_pct": -25.799255960367486,
        "normalized_value": 155.4041846796532
      },
      {
        "timestamp": "2021-06-13T08:00:00+00:00",
        "portfolio_value": 15002.885540319763,
        "drawdown_pct": -28.288644372933824,
        "normalized_value": 150.19047177673116
      },
      {
        "timestamp": "2021-06-17T03:00:00+00:00",
        "portfolio_value": 15535.119524758566,
        "drawdown_pct": -25.74465239003639,
        "normalized_value": 155.51854503328212
      },
      {
        "timestamp": "2021-06-20T22:00:00+00:00",
        "portfolio_value": 15535.119524758566,
        "drawdown_pct": -25.74465239003639,
        "normalized_value": 155.51854503328212
      },
      {
        "timestamp": "2021-06-24T17:00:00+00:00",
        "portfolio_value": 15535.119524758566,
        "drawdown_pct": -25.74465239003639,
        "normalized_value": 155.51854503328212
      },
      {
        "timestamp": "2021-06-28T12:00:00+00:00",
        "portfolio_value": 15535.119524758566,
        "drawdown_pct": -25.74465239003639,
        "normalized_value": 155.51854503328212
      },
      {
        "timestamp": "2021-07-02T07:00:00+00:00",
        "portfolio_value": 14837.480164887553,
        "drawdown_pct": -29.079255196989664,
        "normalized_value": 148.5346362173777
      },
      {
        "timestamp": "2021-07-06T02:00:00+00:00",
        "portfolio_value": 14837.480164887553,
        "drawdown_pct": -29.079255196989664,
        "normalized_value": 148.5346362173777
      },
      {
        "timestamp": "2021-07-09T21:00:00+00:00",
        "portfolio_value": 14837.480164887553,
        "drawdown_pct": -29.079255196989664,
        "normalized_value": 148.5346362173777
      },
      {
        "timestamp": "2021-07-13T16:00:00+00:00",
        "portfolio_value": 14837.480164887553,
        "drawdown_pct": -29.079255196989664,
        "normalized_value": 148.5346362173777
      },
      {
        "timestamp": "2021-07-17T11:00:00+00:00",
        "portfolio_value": 14837.480164887553,
        "drawdown_pct": -29.079255196989664,
        "normalized_value": 148.5346362173777
      },
      {
        "timestamp": "2021-07-21T06:00:00+00:00",
        "portfolio_value": 14837.480164887553,
        "drawdown_pct": -29.079255196989664,
        "normalized_value": 148.5346362173777
      },
      {
        "timestamp": "2021-07-25T01:00:00+00:00",
        "portfolio_value": 15688.17601507288,
        "drawdown_pct": -25.013067230737324,
        "normalized_value": 157.05075871491152
      },
      {
        "timestamp": "2021-07-28T20:00:00+00:00",
        "portfolio_value": 18962.016968076965,
        "drawdown_pct": -9.364639318894085,
        "normalized_value": 189.82443521415826
      },
      {
        "timestamp": "2021-08-01T15:00:00+00:00",
        "portfolio_value": 19601.8804964786,
        "drawdown_pct": -6.306195600538489,
        "normalized_value": 196.2299633337387
      },
      {
        "timestamp": "2021-08-05T10:00:00+00:00",
        "portfolio_value": 17623.852856776524,
        "drawdown_pct": -15.760846382858501,
        "normalized_value": 176.4283789254678
      },
      {
        "timestamp": "2021-08-09T05:00:00+00:00",
        "portfolio_value": 19000.580527867365,
        "drawdown_pct": -9.180311767841124,
        "normalized_value": 190.21048623232707
      },
      {
        "timestamp": "2021-08-13T00:00:00+00:00",
        "portfolio_value": 19444.24284810561,
        "drawdown_pct": -7.059677951137627,
        "normalized_value": 194.6518887216725
      },
      {
        "timestamp": "2021-08-16T23:00:00+00:00",
        "portfolio_value": 20102.69973558675,
        "drawdown_pct": -4.652278334811534,
        "normalized_value": 201.24355072626864
      },
      {
        "timestamp": "2021-08-20T18:00:00+00:00",
        "portfolio_value": 20768.005569439032,
        "drawdown_pct": -1.496712351001051,
        "normalized_value": 207.90377597384168
      },
      {
        "timestamp": "2021-08-24T13:00:00+00:00",
        "portfolio_value": 20819.458725446704,
        "drawdown_pct": -3.7465924829105304,
        "normalized_value": 208.4188618054589
      },
      {
        "timestamp": "2021-08-28T08:00:00+00:00",
        "portfolio_value": 19993.20828914314,
        "drawdown_pct": -7.566548659748827,
        "normalized_value": 200.14745678136126
      },
      {
        "timestamp": "2021-09-01T03:00:00+00:00",
        "portfolio_value": 19657.526551711573,
        "drawdown_pct": -9.118486752623133,
        "normalized_value": 196.787023324998
      },
      {
        "timestamp": "2021-09-04T22:00:00+00:00",
        "portfolio_value": 20087.431216936715,
        "drawdown_pct": -7.130933210472835,
        "normalized_value": 201.09070106190123
      },
      {
        "timestamp": "2021-09-08T17:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-09-12T12:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-09-16T07:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-09-20T02:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-09-23T21:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-09-27T16:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-10-01T13:00:00+00:00",
        "portfolio_value": 19007.9631791985,
        "drawdown_pct": -12.121575777519682,
        "normalized_value": 190.28439227416192
      },
      {
        "timestamp": "2021-10-05T08:00:00+00:00",
        "portfolio_value": 19277.945351667047,
        "drawdown_pct": -10.873382709118642,
        "normalized_value": 192.98712234201346
      },
      {
        "timestamp": "2021-10-09T03:00:00+00:00",
        "portfolio_value": 21202.275718320125,
        "drawdown_pct": -2.3692573169174382,
        "normalized_value": 212.25115557383373
      },
      {
        "timestamp": "2021-10-12T22:00:00+00:00",
        "portfolio_value": 21961.388108173487,
        "drawdown_pct": -2.032743436516275,
        "normalized_value": 219.850456898718
      },
      {
        "timestamp": "2021-10-16T17:00:00+00:00",
        "portfolio_value": 23702.92702223877,
        "drawdown_pct": -3.3258360042522734,
        "normalized_value": 237.284606510675
      },
      {
        "timestamp": "2021-10-20T12:00:00+00:00",
        "portfolio_value": 25270.700220058658,
        "drawdown_pct": 0.0,
        "normalized_value": 252.97922709460732
      },
      {
        "timestamp": "2021-10-24T07:00:00+00:00",
        "portfolio_value": 24586.479109605552,
        "drawdown_pct": -6.54594279593331,
        "normalized_value": 246.12964531899638
      },
      {
        "timestamp": "2021-10-28T02:00:00+00:00",
        "portfolio_value": 24112.824219447943,
        "drawdown_pct": -8.346321410636282,
        "normalized_value": 241.38799403991777
      },
      {
        "timestamp": "2021-10-31T21:00:00+00:00",
        "portfolio_value": 23408.238534100376,
        "drawdown_pct": -11.024475962579139,
        "normalized_value": 234.3345471409803
      },
      {
        "timestamp": "2021-11-04T16:00:00+00:00",
        "portfolio_value": 23048.24795595377,
        "drawdown_pct": -12.39281259724165,
        "normalized_value": 230.73076341406232
      },
      {
        "timestamp": "2021-11-08T11:00:00+00:00",
        "portfolio_value": 25001.073242825954,
        "drawdown_pct": -4.970056160448662,
        "normalized_value": 250.2800528054025
      },
      {
        "timestamp": "2021-11-12T06:00:00+00:00",
        "portfolio_value": 24377.65072016612,
        "drawdown_pct": -7.339706724696961,
        "normalized_value": 244.03911185155147
      },
      {
        "timestamp": "2021-11-16T01:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-11-19T20:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-11-23T15:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-11-27T10:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-01T05:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-05T00:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-08T19:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-12T14:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-16T09:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-20T04:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-23T23:00:00+00:00",
        "portfolio_value": 24072.74055041379,
        "drawdown_pct": -8.498680822578397,
        "normalized_value": 240.98672555415862
      },
      {
        "timestamp": "2021-12-27T18:00:00+00:00",
        "portfolio_value": 23763.29803512405,
        "drawdown_pct": -9.674882522559075,
        "normalized_value": 237.8889669773673
      },
      {
        "timestamp": "2021-12-31T13:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-04T08:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-08T03:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-11T22:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-15T17:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-19T12:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-23T07:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-27T02:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-01-30T21:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-02-03T16:00:00+00:00",
        "portfolio_value": 22879.769982547736,
        "drawdown_pct": -13.03320319949594,
        "normalized_value": 229.0441687758618
      },
      {
        "timestamp": "2022-02-07T11:00:00+00:00",
        "portfolio_value": 24284.30676805079,
        "drawdown_pct": -7.6945103141719455,
        "normalized_value": 243.10466679642997
      },
      {
        "timestamp": "2022-02-11T06:00:00+00:00",
        "portfolio_value": 24605.428333633117,
        "drawdown_pct": -6.473916140211269,
        "normalized_value": 246.31934168699536
      },
      {
        "timestamp": "2022-02-15T01:00:00+00:00",
        "portfolio_value": 24070.16770348985,
        "drawdown_pct": -8.508460302695713,
        "normalized_value": 240.96096936930488
      },
      {
        "timestamp": "2022-02-18T20:00:00+00:00",
        "portfolio_value": 23095.644111185928,
        "drawdown_pct": -12.212658172422511,
        "normalized_value": 231.20523553448146
      },
      {
        "timestamp": "2022-02-22T15:00:00+00:00",
        "portfolio_value": 23095.644111185928,
        "drawdown_pct": -12.212658172422511,
        "normalized_value": 231.20523553448146
      },
      {
        "timestamp": "2022-02-26T10:00:00+00:00",
        "portfolio_value": 23095.644111185928,
        "drawdown_pct": -12.212658172422511,
        "normalized_value": 231.20523553448146
      },
      {
        "timestamp": "2022-03-02T05:00:00+00:00",
        "portfolio_value": 24933.73109536613,
        "drawdown_pct": -5.2260259913829525,
        "normalized_value": 249.6059058974448
      },
      {
        "timestamp": "2022-03-06T00:00:00+00:00",
        "portfolio_value": 22857.260318947312,
        "drawdown_pct": -13.118763209141099,
        "normalized_value": 228.8188296578236
      },
      {
        "timestamp": "2022-03-09T19:00:00+00:00",
        "portfolio_value": 23148.62369978164,
        "drawdown_pct": -12.011280924334255,
        "normalized_value": 231.73560213524925
      },
      {
        "timestamp": "2022-03-13T14:00:00+00:00",
        "portfolio_value": 20810.082819087995,
        "drawdown_pct": -20.900155669844683,
        "normalized_value": 208.32500174130226
      },
      {
        "timestamp": "2022-03-17T09:00:00+00:00",
        "portfolio_value": 20408.460518637585,
        "drawdown_pct": -22.42673592045315,
        "normalized_value": 204.30445231975293
      },
      {
        "timestamp": "2022-03-21T04:00:00+00:00",
        "portfolio_value": 20430.44722983835,
        "drawdown_pct": -22.343163670960827,
        "normalized_value": 204.5245562803669
      },
      {
        "timestamp": "2022-03-24T23:00:00+00:00",
        "portfolio_value": 22109.39895728607,
        "drawdown_pct": -15.961410103063203,
        "normalized_value": 221.33215981490375
      },
      {
        "timestamp": "2022-03-28T18:00:00+00:00",
        "portfolio_value": 24246.313581525403,
        "drawdown_pct": -7.838923729817172,
        "normalized_value": 242.72432565517326
      },
      {
        "timestamp": "2022-04-01T13:00:00+00:00",
        "portfolio_value": 22915.40088252989,
        "drawdown_pct": -12.89776891668056,
        "normalized_value": 229.40086160430252
      },
      {
        "timestamp": "2022-04-05T08:00:00+00:00",
        "portfolio_value": 22117.31344773912,
        "drawdown_pct": -15.931326851197685,
        "normalized_value": 221.4113899771163
      },
      {
        "timestamp": "2022-04-09T03:00:00+00:00",
        "portfolio_value": 21668.039958839167,
        "drawdown_pct": -17.63903091670268,
        "normalized_value": 216.91381535566626
      },
      {
        "timestamp": "2022-04-12T22:00:00+00:00",
        "portfolio_value": 21668.039958839167,
        "drawdown_pct": -17.63903091670268,
        "normalized_value": 216.91381535566626
      },
      {
        "timestamp": "2022-04-16T17:00:00+00:00",
        "portfolio_value": 21668.039958839167,
        "drawdown_pct": -17.63903091670268,
        "normalized_value": 216.91381535566626
      },
      {
        "timestamp": "2022-04-20T12:00:00+00:00",
        "portfolio_value": 22044.46222728736,
        "drawdown_pct": -16.208236859057457,
        "normalized_value": 220.68209299356235
      },
      {
        "timestamp": "2022-04-24T07:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-04-28T02:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-01T21:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-05T16:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-09T11:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-13T06:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-17T01:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-20T20:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-24T15:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-05-28T10:00:00+00:00",
        "portfolio_value": 21099.303683016788,
        "drawdown_pct": -19.800817165876,
        "normalized_value": 211.22032596972485
      },
      {
        "timestamp": "2022-06-01T05:00:00+00:00",
        "portfolio_value": 21110.615716943128,
        "drawdown_pct": -19.757819733794175,
        "normalized_value": 211.33356816620673
      },
      {
        "timestamp": "2022-06-05T00:00:00+00:00",
        "portfolio_value": 20502.82296627268,
        "drawdown_pct": -22.068060994291773,
        "normalized_value": 205.24909330166503
      },
      {
        "timestamp": "2022-06-08T19:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-06-12T14:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-06-16T09:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-06-20T04:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-06-23T23:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-06-27T18:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-07-01T13:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-07-05T08:00:00+00:00",
        "portfolio_value": 19323.119797357267,
        "drawdown_pct": -26.552153529060917,
        "normalized_value": 193.43935343397462
      },
      {
        "timestamp": "2022-07-09T03:00:00+00:00",
        "portfolio_value": 19204.17985168813,
        "drawdown_pct": -27.004248375047606,
        "normalized_value": 192.2486727142465
      },
      {
        "timestamp": "2022-07-12T22:00:00+00:00",
        "portfolio_value": 18521.891558151707,
        "drawdown_pct": -29.59764976975593,
        "normalized_value": 185.4184399287878
      },
      {
        "timestamp": "2022-07-16T17:00:00+00:00",
        "portfolio_value": 18521.891558151707,
        "drawdown_pct": -29.59764976975593,
        "normalized_value": 185.4184399287878
      },
      {
        "timestamp": "2022-07-20T12:00:00+00:00",
        "portfolio_value": 19447.20265600696,
        "drawdown_pct": -26.080510293012445,
        "normalized_value": 194.68151868478046
      },
      {
        "timestamp": "2022-07-24T07:00:00+00:00",
        "portfolio_value": 18461.97744907953,
        "drawdown_pct": -29.825385369944986,
        "normalized_value": 184.81865342215548
      },
      {
        "timestamp": "2022-07-28T02:00:00+00:00",
        "portfolio_value": 18138.40073928702,
        "drawdown_pct": -31.05531163193827,
        "normalized_value": 181.57940064180906
      },
      {
        "timestamp": "2022-07-31T21:00:00+00:00",
        "portfolio_value": 18394.043850658403,
        "drawdown_pct": -30.08360332643335,
        "normalized_value": 184.13858563326647
      },
      {
        "timestamp": "2022-08-04T16:00:00+00:00",
        "portfolio_value": 17861.961932510243,
        "drawdown_pct": -32.10606509471642,
        "normalized_value": 178.81203467773364
      },
      {
        "timestamp": "2022-08-08T11:00:00+00:00",
        "portfolio_value": 18300.68096331557,
        "drawdown_pct": -30.438478889362774,
        "normalized_value": 183.203951021893
      },
      {
        "timestamp": "2022-08-12T06:00:00+00:00",
        "portfolio_value": 17332.792913055113,
        "drawdown_pct": -34.117454834350156,
        "normalized_value": 173.51464408790298
      },
      {
        "timestamp": "2022-08-16T01:00:00+00:00",
        "portfolio_value": 17442.78796107679,
        "drawdown_pct": -33.69935985360256,
        "normalized_value": 174.61577947356477
      },
      {
        "timestamp": "2022-08-19T20:00:00+00:00",
        "portfolio_value": 16912.885092537752,
        "drawdown_pct": -35.713538978977965,
        "normalized_value": 169.31104248761392
      },
      {
        "timestamp": "2022-08-23T15:00:00+00:00",
        "portfolio_value": 16912.885092537752,
        "drawdown_pct": -35.713538978977965,
        "normalized_value": 169.31104248761392
      },
      {
        "timestamp": "2022-08-27T10:00:00+00:00",
        "portfolio_value": 16912.885092537752,
        "drawdown_pct": -35.713538978977965,
        "normalized_value": 169.31104248761392
      },
      {
        "timestamp": "2022-08-31T05:00:00+00:00",
        "portfolio_value": 16912.885092537752,
        "drawdown_pct": -35.713538978977965,
        "normalized_value": 169.31104248761392
      },
      {
        "timestamp": "2022-09-04T00:00:00+00:00",
        "portfolio_value": 16912.885092537752,
        "drawdown_pct": -35.713538978977965,
        "normalized_value": 169.31104248761392
      },
      {
        "timestamp": "2022-09-07T19:00:00+00:00",
        "portfolio_value": 16912.885092537752,
        "drawdown_pct": -35.713538978977965,
        "normalized_value": 169.31104248761392
      },
      {
        "timestamp": "2022-09-11T14:00:00+00:00",
        "portfolio_value": 17504.653699019833,
        "drawdown_pct": -33.4642060445953,
        "normalized_value": 175.23510329254015
      },
      {
        "timestamp": "2022-09-15T09:00:00+00:00",
        "portfolio_value": 16852.299744634507,
        "drawdown_pct": -35.94382598117285,
        "normalized_value": 168.70453636184956
      },
      {
        "timestamp": "2022-09-19T04:00:00+00:00",
        "portfolio_value": 16852.299744634507,
        "drawdown_pct": -35.94382598117285,
        "normalized_value": 168.70453636184956
      },
      {
        "timestamp": "2022-09-22T23:00:00+00:00",
        "portfolio_value": 16852.299744634507,
        "drawdown_pct": -35.94382598117285,
        "normalized_value": 168.70453636184956
      },
      {
        "timestamp": "2022-09-26T18:00:00+00:00",
        "portfolio_value": 16852.299744634507,
        "drawdown_pct": -35.94382598117285,
        "normalized_value": 168.70453636184956
      },
      {
        "timestamp": "2022-09-30T13:00:00+00:00",
        "portfolio_value": 15826.307870369275,
        "drawdown_pct": -39.8436565702152,
        "normalized_value": 158.4335652729304
      },
      {
        "timestamp": "2022-10-04T08:00:00+00:00",
        "portfolio_value": 15810.481562498906,
        "drawdown_pct": -39.903812913644984,
        "normalized_value": 158.27513170765747
      },
      {
        "timestamp": "2022-10-08T03:00:00+00:00",
        "portfolio_value": 15544.633190444349,
        "drawdown_pct": -40.91431176786638,
        "normalized_value": 155.61378417470155
      },
      {
        "timestamp": "2022-10-11T22:00:00+00:00",
        "portfolio_value": 15544.633190444349,
        "drawdown_pct": -40.91431176786638,
        "normalized_value": 155.61378417470155
      },
      {
        "timestamp": "2022-10-15T17:00:00+00:00",
        "portfolio_value": 15265.657015706149,
        "drawdown_pct": -41.9747098539986,
        "normalized_value": 152.8210171975894
      },
      {
        "timestamp": "2022-10-19T12:00:00+00:00",
        "portfolio_value": 15128.916683706197,
        "drawdown_pct": -42.49446458389947,
        "normalized_value": 151.45214086251463
      },
      {
        "timestamp": "2022-10-23T07:00:00+00:00",
        "portfolio_value": 14941.510541983633,
        "drawdown_pct": -43.20680180839013,
        "normalized_value": 149.576060640245
      },
      {
        "timestamp": "2022-10-27T02:00:00+00:00",
        "portfolio_value": 15587.484541907084,
        "drawdown_pct": -40.751432299317116,
        "normalized_value": 156.0427592991976
      },
      {
        "timestamp": "2022-10-30T21:00:00+00:00",
        "portfolio_value": 15542.101328361492,
        "drawdown_pct": -40.923935463185636,
        "normalized_value": 155.58843827976182
      },
      {
        "timestamp": "2022-11-03T16:00:00+00:00",
        "portfolio_value": 15092.998477387691,
        "drawdown_pct": -42.6309909280333,
        "normalized_value": 151.0925718757452
      },
      {
        "timestamp": "2022-11-07T11:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-11-11T06:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-11-15T01:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-11-18T20:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-11-22T15:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-11-26T10:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-11-30T05:00:00+00:00",
        "portfolio_value": 15095.794141728758,
        "drawdown_pct": -42.62036451120929,
        "normalized_value": 151.12055863503798
      },
      {
        "timestamp": "2022-12-04T00:00:00+00:00",
        "portfolio_value": 15105.56938444519,
        "drawdown_pct": -42.58320847565281,
        "normalized_value": 151.21841636456392
      },
      {
        "timestamp": "2022-12-07T19:00:00+00:00",
        "portfolio_value": 14947.001120540985,
        "drawdown_pct": -43.18593192944997,
        "normalized_value": 149.63102557226688
      },
      {
        "timestamp": "2022-12-11T14:00:00+00:00",
        "portfolio_value": 14858.98051206265,
        "drawdown_pct": -43.52050130569943,
        "normalized_value": 148.74987129845013
      },
      {
        "timestamp": "2022-12-15T09:00:00+00:00",
        "portfolio_value": 15194.013935794062,
        "drawdown_pct": -42.24702767789408,
        "normalized_value": 152.10381463394813
      },
      {
        "timestamp": "2022-12-19T04:00:00+00:00",
        "portfolio_value": 14699.231492559711,
        "drawdown_pct": -44.12771285235348,
        "normalized_value": 147.15066023065012
      },
      {
        "timestamp": "2022-12-22T23:00:00+00:00",
        "portfolio_value": 14699.231492559711,
        "drawdown_pct": -44.12771285235348,
        "normalized_value": 147.15066023065012
      },
      {
        "timestamp": "2022-12-26T18:00:00+00:00",
        "portfolio_value": 14699.231492559711,
        "drawdown_pct": -44.12771285235348,
        "normalized_value": 147.15066023065012
      },
      {
        "timestamp": "2022-12-30T13:00:00+00:00",
        "portfolio_value": 14699.231492559711,
        "drawdown_pct": -44.12771285235348,
        "normalized_value": 147.15066023065012
      },
      {
        "timestamp": "2023-01-03T08:00:00+00:00",
        "portfolio_value": 14699.231492559711,
        "drawdown_pct": -44.12771285235348,
        "normalized_value": 147.15066023065012
      },
      {
        "timestamp": "2023-01-07T03:00:00+00:00",
        "portfolio_value": 14667.79713322771,
        "drawdown_pct": -44.24719593905696,
        "normalized_value": 146.83597801532662
      },
      {
        "timestamp": "2023-01-10T22:00:00+00:00",
        "portfolio_value": 15017.808807630832,
        "drawdown_pct": -42.91678946255657,
        "normalized_value": 150.33986520853972
      },
      {
        "timestamp": "2023-01-14T17:00:00+00:00",
        "portfolio_value": 17931.033425882048,
        "drawdown_pct": -31.84352196017639,
        "normalized_value": 179.50349367393508
      },
      {
        "timestamp": "2023-01-18T12:00:00+00:00",
        "portfolio_value": 18398.06037389701,
        "drawdown_pct": -30.06833638272706,
        "normalized_value": 184.1787941330572
      },
      {
        "timestamp": "2023-01-22T07:00:00+00:00",
        "portfolio_value": 19894.71727529552,
        "drawdown_pct": -24.3794917517155,
        "normalized_value": 199.16148566295286
      },
      {
        "timestamp": "2023-01-26T02:00:00+00:00",
        "portfolio_value": 19625.576959270405,
        "drawdown_pct": -25.40250339879208,
        "normalized_value": 196.4671832283101
      },
      {
        "timestamp": "2023-01-29T21:00:00+00:00",
        "portfolio_value": 20184.336202994848,
        "drawdown_pct": -23.278640193591457,
        "normalized_value": 202.06079481716412
      },
      {
        "timestamp": "2023-02-02T16:00:00+00:00",
        "portfolio_value": 19673.39348562663,
        "drawdown_pct": -25.22075112879549,
        "normalized_value": 196.94586358835588
      },
      {
        "timestamp": "2023-02-06T11:00:00+00:00",
        "portfolio_value": 18876.99181584489,
        "drawdown_pct": -28.247901412225858,
        "normalized_value": 188.97326777091507
      },
      {
        "timestamp": "2023-02-10T06:00:00+00:00",
        "portfolio_value": 18867.20471582482,
        "drawdown_pct": -28.28510251780344,
        "normalized_value": 188.87529134062174
      },
      {
        "timestamp": "2023-02-14T01:00:00+00:00",
        "portfolio_value": 18867.20471582482,
        "drawdown_pct": -28.28510251780344,
        "normalized_value": 188.87529134062174
      },
      {
        "timestamp": "2023-02-17T20:00:00+00:00",
        "portfolio_value": 20951.149938167066,
        "drawdown_pct": -20.363954672651968,
        "normalized_value": 209.73719255684443
      },
      {
        "timestamp": "2023-02-21T15:00:00+00:00",
        "portfolio_value": 20683.39118339349,
        "drawdown_pct": -21.381714957642235,
        "normalized_value": 207.05672061738204
      },
      {
        "timestamp": "2023-02-25T10:00:00+00:00",
        "portfolio_value": 19664.591986613188,
        "drawdown_pct": -25.254205930868494,
        "normalized_value": 196.8577537853704
      },
      {
        "timestamp": "2023-03-01T05:00:00+00:00",
        "portfolio_value": 19664.591986613188,
        "drawdown_pct": -25.254205930868494,
        "normalized_value": 196.8577537853704
      },
      {
        "timestamp": "2023-03-05T00:00:00+00:00",
        "portfolio_value": 19664.591986613188,
        "drawdown_pct": -25.254205930868494,
        "normalized_value": 196.8577537853704
      },
      {
        "timestamp": "2023-03-08T19:00:00+00:00",
        "portfolio_value": 19664.591986613188,
        "drawdown_pct": -25.254205930868494,
        "normalized_value": 196.8577537853704
      },
      {
        "timestamp": "2023-03-12T14:00:00+00:00",
        "portfolio_value": 19664.591986613188,
        "drawdown_pct": -25.254205930868494,
        "normalized_value": 196.8577537853704
      },
      {
        "timestamp": "2023-03-16T09:00:00+00:00",
        "portfolio_value": 21774.8135091951,
        "drawdown_pct": -17.2331809599699,
        "normalized_value": 217.9827010615618
      },
      {
        "timestamp": "2023-03-20T04:00:00+00:00",
        "portfolio_value": 24306.046142995045,
        "drawdown_pct": -7.6118781983425805,
        "normalized_value": 243.3222947300854
      },
      {
        "timestamp": "2023-03-23T23:00:00+00:00",
        "portfolio_value": 23894.719931364794,
        "drawdown_pct": -9.17534335087056,
        "normalized_value": 239.2046016627846
      },
      {
        "timestamp": "2023-03-27T19:00:00+00:00",
        "portfolio_value": 22505.485271997317,
        "drawdown_pct": -14.45587232566297,
        "normalized_value": 225.29728974347088
      },
      {
        "timestamp": "2023-03-31T14:00:00+00:00",
        "portfolio_value": 23356.543011928326,
        "drawdown_pct": -11.220972429792832,
        "normalized_value": 233.8170350368655
      },
      {
        "timestamp": "2023-04-04T09:00:00+00:00",
        "portfolio_value": 23038.482096261632,
        "drawdown_pct": -12.42993296759836,
        "normalized_value": 230.63299961585662
      },
      {
        "timestamp": "2023-04-08T04:00:00+00:00",
        "portfolio_value": 22834.170223452133,
        "drawdown_pct": -13.206529460487385,
        "normalized_value": 228.58767996822084
      },
      {
        "timestamp": "2023-04-11T23:00:00+00:00",
        "portfolio_value": 24769.85386534641,
        "drawdown_pct": -5.848929008949366,
        "normalized_value": 247.96536825393756
      },
      {
        "timestamp": "2023-04-15T18:00:00+00:00",
        "portfolio_value": 24856.45224187188,
        "drawdown_pct": -5.5197655855282015,
        "normalized_value": 248.83228488744155
      },
      {
        "timestamp": "2023-04-19T13:00:00+00:00",
        "portfolio_value": 24036.595003848448,
        "drawdown_pct": -8.636071294851021,
        "normalized_value": 240.62488071593157
      },
      {
        "timestamp": "2023-04-23T08:00:00+00:00",
        "portfolio_value": 23665.38558293976,
        "drawdown_pct": -10.047051138757812,
        "normalized_value": 236.90878770806245
      },
      {
        "timestamp": "2023-04-27T03:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-04-30T22:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-04T17:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-08T12:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-12T07:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-16T02:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-19T21:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-23T16:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-27T11:00:00+00:00",
        "portfolio_value": 22520.72049475545,
        "drawdown_pct": -14.397962717182308,
        "normalized_value": 225.4498060902442
      },
      {
        "timestamp": "2023-05-31T06:00:00+00:00",
        "portfolio_value": 21595.265682254296,
        "drawdown_pct": -17.91564845826397,
        "normalized_value": 216.18528863964767
      },
      {
        "timestamp": "2023-06-04T01:00:00+00:00",
        "portfolio_value": 21595.265682254296,
        "drawdown_pct": -17.91564845826397,
        "normalized_value": 216.18528863964767
      },
      {
        "timestamp": "2023-06-07T20:00:00+00:00",
        "portfolio_value": 20877.585691028144,
        "drawdown_pct": -20.643574919603434,
        "normalized_value": 209.00075762544395
      },
      {
        "timestamp": "2023-06-11T15:00:00+00:00",
        "portfolio_value": 20877.585691028144,
        "drawdown_pct": -20.643574919603434,
        "normalized_value": 209.00075762544395
      },
      {
        "timestamp": "2023-06-15T10:00:00+00:00",
        "portfolio_value": 20877.585691028144,
        "drawdown_pct": -20.643574919603434,
        "normalized_value": 209.00075762544395
      },
      {
        "timestamp": "2023-06-19T05:00:00+00:00",
        "portfolio_value": 20877.81115666497,
        "drawdown_pct": -20.642717916922095,
        "normalized_value": 209.00301471060757
      },
      {
        "timestamp": "2023-06-23T00:00:00+00:00",
        "portfolio_value": 23714.86811523637,
        "drawdown_pct": -9.858966322577572,
        "normalized_value": 237.40414607431558
      },
      {
        "timestamp": "2023-06-26T19:00:00+00:00",
        "portfolio_value": 23911.459401682896,
        "drawdown_pct": -9.111716045403204,
        "normalized_value": 239.37217668944274
      },
      {
        "timestamp": "2023-06-30T14:00:00+00:00",
        "portfolio_value": 23781.886007252477,
        "drawdown_pct": -9.604228997796401,
        "normalized_value": 238.07504693484205
      },
      {
        "timestamp": "2023-07-04T09:00:00+00:00",
        "portfolio_value": 24562.732806968263,
        "drawdown_pct": -6.636203313315946,
        "normalized_value": 245.89192648907778
      },
      {
        "timestamp": "2023-07-08T04:00:00+00:00",
        "portfolio_value": 23720.32314454511,
        "drawdown_pct": -9.838231567560557,
        "normalized_value": 237.45875513090223
      },
      {
        "timestamp": "2023-07-11T23:00:00+00:00",
        "portfolio_value": 23340.921101442924,
        "drawdown_pct": -11.28035185169506,
        "normalized_value": 233.66064764728293
      },
      {
        "timestamp": "2023-07-15T18:00:00+00:00",
        "portfolio_value": 22709.599195238483,
        "drawdown_pct": -13.680028245927534,
        "normalized_value": 227.34062776304077
      },
      {
        "timestamp": "2023-07-19T13:00:00+00:00",
        "portfolio_value": 22597.88108246154,
        "drawdown_pct": -14.104672655387038,
        "normalized_value": 226.22224316836514
      },
      {
        "timestamp": "2023-07-23T08:00:00+00:00",
        "portfolio_value": 22597.88108246154,
        "drawdown_pct": -14.104672655387038,
        "normalized_value": 226.22224316836514
      },
      {
        "timestamp": "2023-07-27T03:00:00+00:00",
        "portfolio_value": 22597.88108246154,
        "drawdown_pct": -14.104672655387038,
        "normalized_value": 226.22224316836514
      },
      {
        "timestamp": "2023-07-30T22:00:00+00:00",
        "portfolio_value": 22597.88108246154,
        "drawdown_pct": -14.104672655387038,
        "normalized_value": 226.22224316836514
      },
      {
        "timestamp": "2023-08-03T17:00:00+00:00",
        "portfolio_value": 22093.580313750583,
        "drawdown_pct": -16.0215373140911,
        "normalized_value": 221.17380297554257
      },
      {
        "timestamp": "2023-08-07T12:00:00+00:00",
        "portfolio_value": 21976.131822515887,
        "drawdown_pct": -16.467963090203565,
        "normalized_value": 219.99805286663195
      },
      {
        "timestamp": "2023-08-11T07:00:00+00:00",
        "portfolio_value": 21978.715314939174,
        "drawdown_pct": -16.458143146175143,
        "normalized_value": 220.02391562115628
      },
      {
        "timestamp": "2023-08-15T02:00:00+00:00",
        "portfolio_value": 21939.508361793327,
        "drawdown_pct": -16.607170130712255,
        "normalized_value": 219.63142373857286
      },
      {
        "timestamp": "2023-08-18T21:00:00+00:00",
        "portfolio_value": 21775.498380272864,
        "drawdown_pct": -17.230577741323046,
        "normalized_value": 217.98955715001222
      },
      {
        "timestamp": "2023-08-22T16:00:00+00:00",
        "portfolio_value": 21775.498380272864,
        "drawdown_pct": -17.230577741323046,
        "normalized_value": 217.98955715001222
      },
      {
        "timestamp": "2023-08-26T11:00:00+00:00",
        "portfolio_value": 21775.498380272864,
        "drawdown_pct": -17.230577741323046,
        "normalized_value": 217.98955715001222
      },
      {
        "timestamp": "2023-08-30T06:00:00+00:00",
        "portfolio_value": 21258.072190302773,
        "drawdown_pct": -19.19733258006103,
        "normalized_value": 212.80972135293203
      },
      {
        "timestamp": "2023-09-03T01:00:00+00:00",
        "portfolio_value": 20109.19662551052,
        "drawdown_pct": -23.56424832565483,
        "normalized_value": 201.3085896122956
      },
      {
        "timestamp": "2023-09-06T20:00:00+00:00",
        "portfolio_value": 20109.19662551052,
        "drawdown_pct": -23.56424832565483,
        "normalized_value": 201.3085896122956
      },
      {
        "timestamp": "2023-09-10T15:00:00+00:00",
        "portfolio_value": 19673.413614361307,
        "drawdown_pct": -25.220674618779015,
        "normalized_value": 196.94606509253646
      },
      {
        "timestamp": "2023-09-14T10:00:00+00:00",
        "portfolio_value": 19680.768227439192,
        "drawdown_pct": -25.192719480174976,
        "normalized_value": 197.01969044980086
      },
      {
        "timestamp": "2023-09-18T05:00:00+00:00",
        "portfolio_value": 19531.76825592359,
        "drawdown_pct": -25.759073523767295,
        "normalized_value": 195.52808565440958
      },
      {
        "timestamp": "2023-09-22T00:00:00+00:00",
        "portfolio_value": 19062.79321994672,
        "drawdown_pct": -27.541663850917423,
        "normalized_value": 190.83328333017786
      },
      {
        "timestamp": "2023-09-25T19:00:00+00:00",
        "portfolio_value": 18941.62447312036,
        "drawdown_pct": -28.002230447166337,
        "normalized_value": 189.62029058944546
      },
      {
        "timestamp": "2023-09-29T14:00:00+00:00",
        "portfolio_value": 18404.480879976145,
        "drawdown_pct": -30.043931817123195,
        "normalized_value": 184.24306835780413
      },
      {
        "timestamp": "2023-10-03T09:00:00+00:00",
        "portfolio_value": 18856.60997496866,
        "drawdown_pct": -28.32537349412421,
        "normalized_value": 188.76922980177557
      },
      {
        "timestamp": "2023-10-07T04:00:00+00:00",
        "portfolio_value": 18615.13357615553,
        "drawdown_pct": -29.243233635363282,
        "normalized_value": 186.35186454472407
      },
      {
        "timestamp": "2023-10-10T23:00:00+00:00",
        "portfolio_value": 18527.64839465198,
        "drawdown_pct": -29.57576783516852,
        "normalized_value": 185.47607030846333
      },
      {
        "timestamp": "2023-10-14T18:00:00+00:00",
        "portfolio_value": 18527.64839465198,
        "drawdown_pct": -29.57576783516852,
        "normalized_value": 185.47607030846333
      },
      {
        "timestamp": "2023-10-18T13:00:00+00:00",
        "portfolio_value": 19263.061810937408,
        "drawdown_pct": -26.780436012023262,
        "normalized_value": 192.83812660396794
      },
      {
        "timestamp": "2023-10-22T08:00:00+00:00",
        "portfolio_value": 20361.816069683704,
        "drawdown_pct": -22.604033083717372,
        "normalized_value": 203.83750535976168
      },
      {
        "timestamp": "2023-10-26T03:00:00+00:00",
        "portfolio_value": 23819.94625600527,
        "drawdown_pct": -9.459560676306035,
        "normalized_value": 238.45605942079078
      },
      {
        "timestamp": "2023-10-29T22:00:00+00:00",
        "portfolio_value": 23751.864061225613,
        "drawdown_pct": -9.718343452691075,
        "normalized_value": 237.7745040675841
      },
      {
        "timestamp": "2023-11-02T17:00:00+00:00",
        "portfolio_value": 23801.468844678988,
        "drawdown_pct": -9.529793955636984,
        "normalized_value": 238.27108626233743
      },
      {
        "timestamp": "2023-11-06T12:00:00+00:00",
        "portfolio_value": 24118.795032802664,
        "drawdown_pct": -8.323626142626601,
        "normalized_value": 241.4477664931719
      },
      {
        "timestamp": "2023-11-10T07:00:00+00:00",
        "portfolio_value": 25022.60157466289,
        "drawdown_pct": -4.8882262267673005,
        "normalized_value": 250.49556803455397
      },
      {
        "timestamp": "2023-11-14T02:00:00+00:00",
        "portfolio_value": 25011.641610962197,
        "drawdown_pct": -4.929885427748175,
        "normalized_value": 250.38585033295325
      },
      {
        "timestamp": "2023-11-17T21:00:00+00:00",
        "portfolio_value": 24088.502028206425,
        "drawdown_pct": -8.43877090051566,
        "normalized_value": 241.14451012028178
      },
      {
        "timestamp": "2023-11-21T16:00:00+00:00",
        "portfolio_value": 24639.926808587374,
        "drawdown_pct": -6.342786244081953,
        "normalized_value": 246.6646980662754
      },
      {
        "timestamp": "2023-11-25T11:00:00+00:00",
        "portfolio_value": 23794.53548625692,
        "drawdown_pct": -9.556147890728864,
        "normalized_value": 238.2016779895349
      },
      {
        "timestamp": "2023-11-29T06:00:00+00:00",
        "portfolio_value": 23535.77716953224,
        "drawdown_pct": -10.539697199493798,
        "normalized_value": 235.611307386454
      },
      {
        "timestamp": "2023-12-03T01:00:00+00:00",
        "portfolio_value": 24080.9804242157,
        "drawdown_pct": -8.467360777353328,
        "normalized_value": 241.06921305500398
      },
      {
        "timestamp": "2023-12-06T20:00:00+00:00",
        "portfolio_value": 26875.81277519424,
        "drawdown_pct": -1.2752370734636223,
        "normalized_value": 269.04764348442063
      },
      {
        "timestamp": "2023-12-10T15:00:00+00:00",
        "portfolio_value": 26971.21027618989,
        "drawdown_pct": -1.4939079069426673,
        "normalized_value": 270.002646149898
      },
      {
        "timestamp": "2023-12-14T10:00:00+00:00",
        "portfolio_value": 25804.174945693376,
        "drawdown_pct": -5.756233867273326,
        "normalized_value": 258.31972112881914
      },
      {
        "timestamp": "2023-12-18T05:00:00+00:00",
        "portfolio_value": 25455.104788138877,
        "drawdown_pct": -7.031131683681549,
        "normalized_value": 254.82525924644372
      },
      {
        "timestamp": "2023-12-22T00:00:00+00:00",
        "portfolio_value": 24849.735710407665,
        "drawdown_pct": -9.2421018815567,
        "normalized_value": 248.76504721995397
      },
      {
        "timestamp": "2023-12-25T19:00:00+00:00",
        "portfolio_value": 24610.165458540087,
        "drawdown_pct": -10.117076680668994,
        "normalized_value": 246.366763966045
      },
      {
        "timestamp": "2023-12-29T14:00:00+00:00",
        "portfolio_value": 23747.21340921826,
        "drawdown_pct": -13.268808960085146,
        "normalized_value": 237.7279474490472
      },
      {
        "timestamp": "2024-01-02T09:00:00+00:00",
        "portfolio_value": 25574.689109687737,
        "drawdown_pct": -6.594377675583626,
        "normalized_value": 256.0223906664133
      },
      {
        "timestamp": "2024-01-06T04:00:00+00:00",
        "portfolio_value": 23930.003248673347,
        "drawdown_pct": -12.60121145242219,
        "normalized_value": 239.55781492020617
      },
      {
        "timestamp": "2024-01-09T23:00:00+00:00",
        "portfolio_value": 23545.6573856692,
        "drawdown_pct": -14.004945604096319,
        "normalized_value": 235.71021598099588
      },
      {
        "timestamp": "2024-01-13T18:00:00+00:00",
        "portfolio_value": 22247.943635488857,
        "drawdown_pct": -18.744544193728103,
        "normalized_value": 222.71909904906047
      },
      {
        "timestamp": "2024-01-17T13:00:00+00:00",
        "portfolio_value": 22247.943635488857,
        "drawdown_pct": -18.744544193728103,
        "normalized_value": 222.71909904906047
      },
      {
        "timestamp": "2024-01-21T08:00:00+00:00",
        "portfolio_value": 22247.943635488857,
        "drawdown_pct": -18.744544193728103,
        "normalized_value": 222.71909904906047
      },
      {
        "timestamp": "2024-01-25T03:00:00+00:00",
        "portfolio_value": 22247.943635488857,
        "drawdown_pct": -18.744544193728103,
        "normalized_value": 222.71909904906047
      },
      {
        "timestamp": "2024-01-28T22:00:00+00:00",
        "portfolio_value": 22110.753449072297,
        "drawdown_pct": -19.245599541227413,
        "normalized_value": 221.3457193238292
      },
      {
        "timestamp": "2024-02-01T17:00:00+00:00",
        "portfolio_value": 21110.97397344567,
        "drawdown_pct": -22.897062270942836,
        "normalized_value": 211.33715459049753
      },
      {
        "timestamp": "2024-02-05T12:00:00+00:00",
        "portfolio_value": 21076.566404827485,
        "drawdown_pct": -23.022727937714688,
        "normalized_value": 210.99270825385332
      },
      {
        "timestamp": "2024-02-09T07:00:00+00:00",
        "portfolio_value": 22051.78241055137,
        "drawdown_pct": -19.4609775866277,
        "normalized_value": 220.7553736817979
      },
      {
        "timestamp": "2024-02-13T02:00:00+00:00",
        "portfolio_value": 23804.2443298332,
        "drawdown_pct": -13.060516745508238,
        "normalized_value": 238.29887101238438
      },
      {
        "timestamp": "2024-02-16T21:00:00+00:00",
        "portfolio_value": 24772.57859834296,
        "drawdown_pct": -9.523900344022802,
        "normalized_value": 247.99264493568774
      },
      {
        "timestamp": "2024-02-20T16:00:00+00:00",
        "portfolio_value": 24435.40718532306,
        "drawdown_pct": -10.75534075481539,
        "normalized_value": 244.6172986761303
      },
      {
        "timestamp": "2024-02-24T11:00:00+00:00",
        "portfolio_value": 24160.060599887034,
        "drawdown_pct": -11.760980317323803,
        "normalized_value": 241.86086669125584
      },
      {
        "timestamp": "2024-02-28T06:00:00+00:00",
        "portfolio_value": 25947.87896837715,
        "drawdown_pct": -5.231388243080951,
        "normalized_value": 259.7583093860638
      },
      {
        "timestamp": "2024-03-03T01:00:00+00:00",
        "portfolio_value": 28183.96781698248,
        "drawdown_pct": -2.0360547530065003,
        "normalized_value": 282.1432858097101
      },
      {
        "timestamp": "2024-03-06T20:00:00+00:00",
        "portfolio_value": 28569.603672093,
        "drawdown_pct": -9.416419759464459,
        "normalized_value": 286.00379856623397
      },
      {
        "timestamp": "2024-03-10T15:00:00+00:00",
        "portfolio_value": 29554.80469000008,
        "drawdown_pct": -6.292714002712336,
        "normalized_value": 295.8664216780828
      },
      {
        "timestamp": "2024-03-14T10:00:00+00:00",
        "portfolio_value": 29850.359016661136,
        "drawdown_pct": -5.355621231954352,
        "normalized_value": 298.825148760122
      },
      {
        "timestamp": "2024-03-18T05:00:00+00:00",
        "portfolio_value": 28854.44098897273,
        "drawdown_pct": -8.513306638078143,
        "normalized_value": 288.8552401030514
      },
      {
        "timestamp": "2024-03-22T00:00:00+00:00",
        "portfolio_value": 28854.44098897273,
        "drawdown_pct": -8.513306638078143,
        "normalized_value": 288.8552401030514
      },
      {
        "timestamp": "2024-03-25T19:00:00+00:00",
        "portfolio_value": 29179.90907925969,
        "drawdown_pct": -7.48136845613405,
        "normalized_value": 292.11342706295994
      },
      {
        "timestamp": "2024-03-29T14:00:00+00:00",
        "portfolio_value": 28761.12394779987,
        "drawdown_pct": -8.809180245003526,
        "normalized_value": 287.92106444724874
      },
      {
        "timestamp": "2024-04-02T09:00:00+00:00",
        "portfolio_value": 28067.376511960472,
        "drawdown_pct": -11.008795165892634,
        "normalized_value": 280.97611679683854
      },
      {
        "timestamp": "2024-04-06T04:00:00+00:00",
        "portfolio_value": 28067.376511960472,
        "drawdown_pct": -11.008795165892634,
        "normalized_value": 280.97611679683854
      },
      {
        "timestamp": "2024-04-09T23:00:00+00:00",
        "portfolio_value": 27913.659844186503,
        "drawdown_pct": -11.496173509303551,
        "normalized_value": 279.4372942289482
      },
      {
        "timestamp": "2024-04-13T18:00:00+00:00",
        "portfolio_value": 28237.460862844844,
        "drawdown_pct": -10.469520991044929,
        "normalized_value": 282.6787925142871
      },
      {
        "timestamp": "2024-04-17T13:00:00+00:00",
        "portfolio_value": 28237.460862844844,
        "drawdown_pct": -10.469520991044929,
        "normalized_value": 282.6787925142871
      },
      {
        "timestamp": "2024-04-21T08:00:00+00:00",
        "portfolio_value": 28237.460862844844,
        "drawdown_pct": -10.469520991044929,
        "normalized_value": 282.6787925142871
      },
      {
        "timestamp": "2024-04-25T03:00:00+00:00",
        "portfolio_value": 28237.460862844844,
        "drawdown_pct": -10.469520991044929,
        "normalized_value": 282.6787925142871
      },
      {
        "timestamp": "2024-04-28T22:00:00+00:00",
        "portfolio_value": 28237.460862844844,
        "drawdown_pct": -10.469520991044929,
        "normalized_value": 282.6787925142871
      },
      {
        "timestamp": "2024-05-02T17:00:00+00:00",
        "portfolio_value": 28237.460862844844,
        "drawdown_pct": -10.469520991044929,
        "normalized_value": 282.6787925142871
      },
      {
        "timestamp": "2024-05-06T12:00:00+00:00",
        "portfolio_value": 28437.27974298089,
        "drawdown_pct": -9.835969690504562,
        "normalized_value": 284.6791338350869
      },
      {
        "timestamp": "2024-05-10T07:00:00+00:00",
        "portfolio_value": 28532.142237450516,
        "drawdown_pct": -9.535196026360229,
        "normalized_value": 285.62878067202513
      },
      {
        "timestamp": "2024-05-14T02:00:00+00:00",
        "portfolio_value": 28366.61197368461,
        "drawdown_pct": -10.060030885890418,
        "normalized_value": 283.9716948839935
      },
      {
        "timestamp": "2024-05-17T21:00:00+00:00",
        "portfolio_value": 29209.7793729314,
        "drawdown_pct": -7.386660871994106,
        "normalized_value": 292.4124517730137
      },
      {
        "timestamp": "2024-05-21T16:00:00+00:00",
        "portfolio_value": 30511.236408145654,
        "drawdown_pct": -3.2602249881776157,
        "normalized_value": 305.44104187927456
      },
      {
        "timestamp": "2024-05-25T11:00:00+00:00",
        "portfolio_value": 29606.75161977676,
        "drawdown_pct": -6.12800962194167,
        "normalized_value": 296.3864505664953
      },
      {
        "timestamp": "2024-05-29T06:00:00+00:00",
        "portfolio_value": 28923.769444651567,
        "drawdown_pct": -8.293491907708928,
        "normalized_value": 289.54927149041174
      },
      {
        "timestamp": "2024-06-02T01:00:00+00:00",
        "portfolio_value": 28923.769444651567,
        "drawdown_pct": -8.293491907708928,
        "normalized_value": 289.54927149041174
      },
      {
        "timestamp": "2024-06-05T20:00:00+00:00",
        "portfolio_value": 29762.618935198123,
        "drawdown_pct": -5.633812375266551,
        "normalized_value": 297.94680277839444
      },
      {
        "timestamp": "2024-06-09T15:00:00+00:00",
        "portfolio_value": 28853.91311086699,
        "drawdown_pct": -8.514980343086348,
        "normalized_value": 288.8499556355049
      },
      {
        "timestamp": "2024-06-13T10:00:00+00:00",
        "portfolio_value": 27499.838723079018,
        "drawdown_pct": -12.808246268845688,
        "normalized_value": 275.29462519083035
      },
      {
        "timestamp": "2024-06-17T05:00:00+00:00",
        "portfolio_value": 26535.383185810777,
        "drawdown_pct": -15.866175827522211,
        "normalized_value": 265.63968036300287
      },
      {
        "timestamp": "2024-06-21T00:00:00+00:00",
        "portfolio_value": 25899.785933090025,
        "drawdown_pct": -17.881418160013286,
        "normalized_value": 259.2768609580573
      },
      {
        "timestamp": "2024-06-24T19:00:00+00:00",
        "portfolio_value": 25899.785933090025,
        "drawdown_pct": -17.881418160013286,
        "normalized_value": 259.2768609580573
      },
      {
        "timestamp": "2024-06-28T14:00:00+00:00",
        "portfolio_value": 25899.785933090025,
        "drawdown_pct": -17.881418160013286,
        "normalized_value": 259.2768609580573
      },
      {
        "timestamp": "2024-07-02T09:00:00+00:00",
        "portfolio_value": 25757.95276548928,
        "drawdown_pct": -18.331118347162924,
        "normalized_value": 257.8570014051536
      },
      {
        "timestamp": "2024-07-06T04:00:00+00:00",
        "portfolio_value": 25443.891594497723,
        "drawdown_pct": -19.326889433435966,
        "normalized_value": 254.7130065175562
      },
      {
        "timestamp": "2024-07-09T23:00:00+00:00",
        "portfolio_value": 25443.891594497723,
        "drawdown_pct": -19.326889433435966,
        "normalized_value": 254.7130065175562
      },
      {
        "timestamp": "2024-07-13T18:00:00+00:00",
        "portfolio_value": 25443.891594497723,
        "drawdown_pct": -19.326889433435966,
        "normalized_value": 254.7130065175562
      },
      {
        "timestamp": "2024-07-17T13:00:00+00:00",
        "portfolio_value": 27026.372640016103,
        "drawdown_pct": -14.309431004152154,
        "normalized_value": 270.5548640165953
      },
      {
        "timestamp": "2024-07-21T08:00:00+00:00",
        "portfolio_value": 27729.17547799733,
        "drawdown_pct": -12.081104773300918,
        "normalized_value": 277.5904632364089
      },
      {
        "timestamp": "2024-07-25T03:00:00+00:00",
        "portfolio_value": 27264.51204668949,
        "drawdown_pct": -13.554379576053291,
        "normalized_value": 272.9388234049934
      },
      {
        "timestamp": "2024-07-28T22:00:00+00:00",
        "portfolio_value": 28161.558574257542,
        "drawdown_pct": -10.710178898924983,
        "normalized_value": 281.9189519821959
      },
      {
        "timestamp": "2024-08-01T17:00:00+00:00",
        "portfolio_value": 27599.51873376088,
        "drawdown_pct": -12.492198053770242,
        "normalized_value": 276.2924990858744
      },
      {
        "timestamp": "2024-08-05T12:00:00+00:00",
        "portfolio_value": 27599.51873376088,
        "drawdown_pct": -12.492198053770242,
        "normalized_value": 276.2924990858744
      },
      {
        "timestamp": "2024-08-09T07:00:00+00:00",
        "portfolio_value": 27414.07905345424,
        "drawdown_pct": -13.080158263286346,
        "normalized_value": 274.4361046611798
      },
      {
        "timestamp": "2024-08-13T02:00:00+00:00",
        "portfolio_value": 27203.964426609127,
        "drawdown_pct": -13.746353544781478,
        "normalized_value": 272.3326949638752
      },
      {
        "timestamp": "2024-08-16T21:00:00+00:00",
        "portfolio_value": 25349.64778586841,
        "drawdown_pct": -19.625701475042487,
        "normalized_value": 253.76955320372315
      },
      {
        "timestamp": "2024-08-20T16:00:00+00:00",
        "portfolio_value": 24451.00845273493,
        "drawdown_pct": -22.474952345810138,
        "normalized_value": 244.77347941260314
      },
      {
        "timestamp": "2024-08-24T11:00:00+00:00",
        "portfolio_value": 25352.92184087101,
        "drawdown_pct": -19.61532066516342,
        "normalized_value": 253.80232902302416
      },
      {
        "timestamp": "2024-08-28T06:00:00+00:00",
        "portfolio_value": 24831.55289395477,
        "drawdown_pct": -21.26838755331361,
        "normalized_value": 248.58302318371437
      },
      {
        "timestamp": "2024-09-01T01:00:00+00:00",
        "portfolio_value": 24831.55289395477,
        "drawdown_pct": -21.26838755331361,
        "normalized_value": 248.58302318371437
      },
      {
        "timestamp": "2024-09-04T20:00:00+00:00",
        "portfolio_value": 24831.55289395477,
        "drawdown_pct": -21.26838755331361,
        "normalized_value": 248.58302318371437
      },
      {
        "timestamp": "2024-09-08T15:00:00+00:00",
        "portfolio_value": 24831.55289395477,
        "drawdown_pct": -21.26838755331361,
        "normalized_value": 248.58302318371437
      },
      {
        "timestamp": "2024-09-12T10:00:00+00:00",
        "portfolio_value": 24831.55289395477,
        "drawdown_pct": -21.26838755331361,
        "normalized_value": 248.58302318371437
      },
      {
        "timestamp": "2024-09-16T05:00:00+00:00",
        "portfolio_value": 24865.02355420656,
        "drawdown_pct": -21.1622645467287,
        "normalized_value": 248.91809034398727
      },
      {
        "timestamp": "2024-09-20T00:00:00+00:00",
        "portfolio_value": 24421.906675205104,
        "drawdown_pct": -22.56722325128967,
        "normalized_value": 244.48214814269508
      },
      {
        "timestamp": "2024-09-23T19:00:00+00:00",
        "portfolio_value": 24626.7938215015,
        "drawdown_pct": -21.917602365056556,
        "normalized_value": 246.53322672224866
      },
      {
        "timestamp": "2024-09-27T14:00:00+00:00",
        "portfolio_value": 25830.11580358114,
        "drawdown_pct": -18.10231621093379,
        "normalized_value": 258.57940915176545
      },
      {
        "timestamp": "2024-10-01T09:00:00+00:00",
        "portfolio_value": 25047.441224771213,
        "drawdown_pct": -20.583886005376176,
        "normalized_value": 250.74423211710743
      },
      {
        "timestamp": "2024-10-05T04:00:00+00:00",
        "portfolio_value": 25047.441224771213,
        "drawdown_pct": -20.583886005376176,
        "normalized_value": 250.74423211710743
      },
      {
        "timestamp": "2024-10-08T23:00:00+00:00",
        "portfolio_value": 25047.441224771213,
        "drawdown_pct": -20.583886005376176,
        "normalized_value": 250.74423211710743
      },
      {
        "timestamp": "2024-10-12T18:00:00+00:00",
        "portfolio_value": 25047.441224771213,
        "drawdown_pct": -20.583886005376176,
        "normalized_value": 250.74423211710743
      },
      {
        "timestamp": "2024-10-16T13:00:00+00:00",
        "portfolio_value": 25360.652712851486,
        "drawdown_pct": -19.590808947380765,
        "normalized_value": 253.87972102250808
      },
      {
        "timestamp": "2024-10-20T08:00:00+00:00",
        "portfolio_value": 25767.052354438478,
        "drawdown_pct": -18.302266937280027,
        "normalized_value": 257.94809531862654
      },
      {
        "timestamp": "2024-10-24T03:00:00+00:00",
        "portfolio_value": 25307.119799535834,
        "drawdown_pct": -19.7605418916839,
        "normalized_value": 253.3438152139319
      },
      {
        "timestamp": "2024-10-27T22:00:00+00:00",
        "portfolio_value": 24955.41145043092,
        "drawdown_pct": -20.87567816826906,
        "normalized_value": 249.82294299652264
      },
      {
        "timestamp": "2024-10-31T17:00:00+00:00",
        "portfolio_value": 25949.883701376966,
        "drawdown_pct": -17.72257678210769,
        "normalized_value": 259.7783783117528
      },
      {
        "timestamp": "2024-11-04T12:00:00+00:00",
        "portfolio_value": 25087.775895868774,
        "drawdown_pct": -20.455999775039334,
        "normalized_value": 251.14801332738335
      },
      {
        "timestamp": "2024-11-08T07:00:00+00:00",
        "portfolio_value": 25765.172890462683,
        "drawdown_pct": -18.308226018050437,
        "normalized_value": 257.9292804326199
      },
      {
        "timestamp": "2024-11-12T02:00:00+00:00",
        "portfolio_value": 29879.032651972877,
        "drawdown_pct": -5.264707806102873,
        "normalized_value": 299.1121939957502
      },
      {
        "timestamp": "2024-11-15T21:00:00+00:00",
        "portfolio_value": 31328.93960602681,
        "drawdown_pct": -1.317337456755255,
        "normalized_value": 313.6268824452818
      },
      {
        "timestamp": "2024-11-19T16:00:00+00:00",
        "portfolio_value": 31646.01857378488,
        "drawdown_pct": -0.31857410350053705,
        "normalized_value": 316.8010878093153
      },
      {
        "timestamp": "2024-11-23T11:00:00+00:00",
        "portfolio_value": 33785.2394085958,
        "drawdown_pct": -0.9013917801268817,
        "normalized_value": 338.2163405986142
      },
      {
        "timestamp": "2024-11-27T06:00:00+00:00",
        "portfolio_value": 32483.801217231143,
        "drawdown_pct": -4.718760421167594,
        "normalized_value": 325.18793913384184
      },
      {
        "timestamp": "2024-12-01T01:00:00+00:00",
        "portfolio_value": 32483.801217231143,
        "drawdown_pct": -4.718760421167594,
        "normalized_value": 325.18793913384184
      },
      {
        "timestamp": "2024-12-04T20:00:00+00:00",
        "portfolio_value": 32483.801217231143,
        "drawdown_pct": -4.718760421167594,
        "normalized_value": 325.18793913384184
      },
      {
        "timestamp": "2024-12-08T15:00:00+00:00",
        "portfolio_value": 31244.674215712446,
        "drawdown_pct": -8.608511135333693,
        "normalized_value": 312.783320805639
      },
      {
        "timestamp": "2024-12-12T10:00:00+00:00",
        "portfolio_value": 31040.42094976875,
        "drawdown_pct": -9.205957277712656,
        "normalized_value": 310.7385878579952
      },
      {
        "timestamp": "2024-12-16T05:00:00+00:00",
        "portfolio_value": 32464.64998096068,
        "drawdown_pct": -5.040050129944895,
        "normalized_value": 324.99622046726677
      },
      {
        "timestamp": "2024-12-20T00:00:00+00:00",
        "portfolio_value": 31038.916427499265,
        "drawdown_pct": -9.210358044681422,
        "normalized_value": 310.72352642805583
      },
      {
        "timestamp": "2024-12-23T19:00:00+00:00",
        "portfolio_value": 31038.916427499265,
        "drawdown_pct": -9.210358044681422,
        "normalized_value": 310.72352642805583
      },
      {
        "timestamp": "2024-12-27T14:00:00+00:00",
        "portfolio_value": 31038.916427499265,
        "drawdown_pct": -9.210358044681422,
        "normalized_value": 310.72352642805583
      },
      {
        "timestamp": "2024-12-31T09:00:00+00:00",
        "portfolio_value": 31038.916427499265,
        "drawdown_pct": -9.210358044681422,
        "normalized_value": 310.72352642805583
      },
      {
        "timestamp": "2025-01-04T04:00:00+00:00",
        "portfolio_value": 31384.221728899836,
        "drawdown_pct": -8.200331011274542,
        "normalized_value": 314.1802991925339
      },
      {
        "timestamp": "2025-01-07T23:00:00+00:00",
        "portfolio_value": 31612.15978926169,
        "drawdown_pct": -7.533606226065613,
        "normalized_value": 316.4621352253168
      },
      {
        "timestamp": "2025-01-11T18:00:00+00:00",
        "portfolio_value": 31612.15978926169,
        "drawdown_pct": -7.533606226065613,
        "normalized_value": 316.4621352253168
      },
      {
        "timestamp": "2025-01-15T13:00:00+00:00",
        "portfolio_value": 32152.928497785026,
        "drawdown_pct": -5.951843617111374,
        "normalized_value": 321.8756476617706
      },
      {
        "timestamp": "2025-01-19T08:00:00+00:00",
        "portfolio_value": 33247.57832986105,
        "drawdown_pct": -2.749964242463414,
        "normalized_value": 332.83393793653016
      },
      {
        "timestamp": "2025-01-23T03:00:00+00:00",
        "portfolio_value": 32406.33552557551,
        "drawdown_pct": -6.009933040344928,
        "normalized_value": 324.41244772953814
      },
      {
        "timestamp": "2025-01-26T22:00:00+00:00",
        "portfolio_value": 32994.155413292014,
        "drawdown_pct": -4.305043249177771,
        "normalized_value": 330.2969788098176
      },
      {
        "timestamp": "2025-01-30T17:00:00+00:00",
        "portfolio_value": 31923.13788000416,
        "drawdown_pct": -7.411380576002144,
        "normalized_value": 319.57526609839573
      },
      {
        "timestamp": "2025-02-03T12:00:00+00:00",
        "portfolio_value": 30614.82555123685,
        "drawdown_pct": -11.205958438344329,
        "normalized_value": 306.4780492089647
      },
      {
        "timestamp": "2025-02-07T07:00:00+00:00",
        "portfolio_value": 29528.411687407905,
        "drawdown_pct": -14.356950679555675,
        "normalized_value": 295.6022073374301
      },
      {
        "timestamp": "2025-02-11T02:00:00+00:00",
        "portfolio_value": 28403.11219826131,
        "drawdown_pct": -17.620725266197265,
        "normalized_value": 284.33709032305086
      },
      {
        "timestamp": "2025-02-14T21:00:00+00:00",
        "portfolio_value": 27401.10628417353,
        "drawdown_pct": -20.526903994265858,
        "normalized_value": 274.3062372211274
      },
      {
        "timestamp": "2025-02-18T16:00:00+00:00",
        "portfolio_value": 27236.85742310778,
        "drawdown_pct": -21.003284961112083,
        "normalized_value": 272.66197926381915
      },
      {
        "timestamp": "2025-02-22T11:00:00+00:00",
        "portfolio_value": 26198.09830261812,
        "drawdown_pct": -24.016061250264624,
        "normalized_value": 262.26319817937724
      },
      {
        "timestamp": "2025-02-26T06:00:00+00:00",
        "portfolio_value": 26198.09830261812,
        "drawdown_pct": -24.016061250264624,
        "normalized_value": 262.26319817937724
      },
      {
        "timestamp": "2025-03-02T01:00:00+00:00",
        "portfolio_value": 26198.09830261812,
        "drawdown_pct": -24.016061250264624,
        "normalized_value": 262.26319817937724
      },
      {
        "timestamp": "2025-03-05T20:00:00+00:00",
        "portfolio_value": 24618.087632467013,
        "drawdown_pct": -28.598662345882786,
        "normalized_value": 246.44607104576545
      },
      {
        "timestamp": "2025-03-09T15:00:00+00:00",
        "portfolio_value": 24404.425178593265,
        "drawdown_pct": -29.218360562933853,
        "normalized_value": 244.30714485973067
      },
      {
        "timestamp": "2025-03-13T10:00:00+00:00",
        "portfolio_value": 24404.425178593265,
        "drawdown_pct": -29.218360562933853,
        "normalized_value": 244.30714485973067
      },
      {
        "timestamp": "2025-03-17T05:00:00+00:00",
        "portfolio_value": 24404.425178593265,
        "drawdown_pct": -29.218360562933853,
        "normalized_value": 244.30714485973067
      },
      {
        "timestamp": "2025-03-21T00:00:00+00:00",
        "portfolio_value": 23810.79528135609,
        "drawdown_pct": -30.94010147827248,
        "normalized_value": 238.36445109677317
      },
      {
        "timestamp": "2025-03-24T19:00:00+00:00",
        "portfolio_value": 24886.0826625073,
        "drawdown_pct": -27.821380051855378,
        "normalized_value": 249.12890828313544
      },
      {
        "timestamp": "2025-03-28T14:00:00+00:00",
        "portfolio_value": 23566.200574259907,
        "drawdown_pct": -31.64951439167653,
        "normalized_value": 235.9158691653746
      },
      {
        "timestamp": "2025-04-01T09:00:00+00:00",
        "portfolio_value": 23566.200574259907,
        "drawdown_pct": -31.64951439167653,
        "normalized_value": 235.9158691653746
      },
      {
        "timestamp": "2025-04-05T04:00:00+00:00",
        "portfolio_value": 22290.290957851004,
        "drawdown_pct": -35.350112695551275,
        "normalized_value": 223.14302845297144
      },
      {
        "timestamp": "2025-04-08T23:00:00+00:00",
        "portfolio_value": 22290.290957851004,
        "drawdown_pct": -35.350112695551275,
        "normalized_value": 223.14302845297144
      },
      {
        "timestamp": "2025-04-12T18:00:00+00:00",
        "portfolio_value": 22687.030697945003,
        "drawdown_pct": -34.1994242844052,
        "normalized_value": 227.11469967429477
      },
      {
        "timestamp": "2025-04-16T13:00:00+00:00",
        "portfolio_value": 22356.15384545045,
        "drawdown_pct": -35.15908655466772,
        "normalized_value": 223.80236682721647
      },
      {
        "timestamp": "2025-04-20T08:00:00+00:00",
        "portfolio_value": 22228.429212968316,
        "drawdown_pct": -35.52953407873038,
        "normalized_value": 222.52374460761425
      },
      {
        "timestamp": "2025-04-24T03:00:00+00:00",
        "portfolio_value": 24150.06724531628,
        "drawdown_pct": -29.956090355355318,
        "normalized_value": 241.76082549360888
      },
      {
        "timestamp": "2025-04-27T22:00:00+00:00",
        "portfolio_value": 24481.70579645612,
        "drawdown_pct": -28.994218884159935,
        "normalized_value": 245.0807835324264
      },
      {
        "timestamp": "2025-05-01T17:00:00+00:00",
        "portfolio_value": 25244.415107848745,
        "drawdown_pct": -26.78208665489357,
        "normalized_value": 252.71609282001046
      },
      {
        "timestamp": "2025-05-05T12:00:00+00:00",
        "portfolio_value": 24652.755567412016,
        "drawdown_pct": -28.49811282450101,
        "normalized_value": 246.79312385043568
      },
      {
        "timestamp": "2025-05-09T07:00:00+00:00",
        "portfolio_value": 26993.37844723809,
        "drawdown_pct": -21.70946185134392,
        "normalized_value": 270.2245666637358
      },
      {
        "timestamp": "2025-05-13T02:00:00+00:00",
        "portfolio_value": 26556.665369244183,
        "drawdown_pct": -22.976087366913546,
        "normalized_value": 265.85273145652167
      },
      {
        "timestamp": "2025-05-16T21:00:00+00:00",
        "portfolio_value": 26985.546380737323,
        "drawdown_pct": -21.73217767042376,
        "normalized_value": 270.1461616289459
      },
      {
        "timestamp": "2025-05-20T16:00:00+00:00",
        "portfolio_value": 27430.17233870877,
        "drawdown_pct": -20.442601947529838,
        "normalized_value": 274.5972108762689
      },
      {
        "timestamp": "2025-05-24T11:00:00+00:00",
        "portfolio_value": 28542.66921608577,
        "drawdown_pct": -17.215959554885703,
        "normalized_value": 285.73416385870365
      },
      {
        "timestamp": "2025-05-28T06:00:00+00:00",
        "portfolio_value": 28406.905725438508,
        "drawdown_pct": -17.60972266847632,
        "normalized_value": 284.37506646003527
      },
      {
        "timestamp": "2025-06-01T01:00:00+00:00",
        "portfolio_value": 27666.47960127676,
        "drawdown_pct": -19.75722561380987,
        "normalized_value": 276.9628290871106
      },
      {
        "timestamp": "2025-06-04T20:00:00+00:00",
        "portfolio_value": 27408.05463436766,
        "drawdown_pct": -20.506751271367268,
        "normalized_value": 274.37579557314723
      },
      {
        "timestamp": "2025-06-08T15:00:00+00:00",
        "portfolio_value": 27016.48635347241,
        "drawdown_pct": -21.64244058469848,
        "normalized_value": 270.4558946525935
      },
      {
        "timestamp": "2025-06-12T10:00:00+00:00",
        "portfolio_value": 27265.14061127766,
        "drawdown_pct": -20.921253509335575,
        "normalized_value": 272.9451158219945
      },
      {
        "timestamp": "2025-06-16T05:00:00+00:00",
        "portfolio_value": 26788.91342292849,
        "drawdown_pct": -22.302484203735474,
        "normalized_value": 268.1777138512971
      },
      {
        "timestamp": "2025-06-20T00:00:00+00:00",
        "portfolio_value": 26788.91342292849,
        "drawdown_pct": -22.302484203735474,
        "normalized_value": 268.1777138512971
      },
      {
        "timestamp": "2025-06-23T19:00:00+00:00",
        "portfolio_value": 26788.91342292849,
        "drawdown_pct": -22.302484203735474,
        "normalized_value": 268.1777138512971
      },
      {
        "timestamp": "2025-06-27T14:00:00+00:00",
        "portfolio_value": 26915.12770687698,
        "drawdown_pct": -21.936417235434746,
        "normalized_value": 269.44121631555583
      },
      {
        "timestamp": "2025-07-01T09:00:00+00:00",
        "portfolio_value": 26496.37864834993,
        "drawdown_pct": -23.150940612173468,
        "normalized_value": 265.24921481776266
      },
      {
        "timestamp": "2025-07-05T04:00:00+00:00",
        "portfolio_value": 26055.987573407918,
        "drawdown_pct": -24.428233646110208,
        "normalized_value": 260.8405600203874
      },
      {
        "timestamp": "2025-07-08T23:00:00+00:00",
        "portfolio_value": 25715.428758617545,
        "drawdown_pct": -25.415977100799363,
        "normalized_value": 257.4313032528389
      },
      {
        "timestamp": "2025-07-12T18:00:00+00:00",
        "portfolio_value": 27717.289334510275,
        "drawdown_pct": -19.609859052569046,
        "normalized_value": 277.4714737598092
      },
      {
        "timestamp": "2025-07-16T13:00:00+00:00",
        "portfolio_value": 27677.429930103684,
        "drawdown_pct": -19.72546570874869,
        "normalized_value": 277.0724503361834
      },
      {
        "timestamp": "2025-07-20T08:00:00+00:00",
        "portfolio_value": 27269.516189646176,
        "drawdown_pct": -20.908562753124667,
        "normalized_value": 272.9889187409528
      },
      {
        "timestamp": "2025-07-24T03:00:00+00:00",
        "portfolio_value": 26648.672835555848,
        "drawdown_pct": -22.709232513405144,
        "normalized_value": 266.77379725651014
      },
      {
        "timestamp": "2025-07-27T22:00:00+00:00",
        "portfolio_value": 26071.671417588623,
        "drawdown_pct": -24.382744838418088,
        "normalized_value": 260.9975674140946
      },
      {
        "timestamp": "2025-07-31T17:00:00+00:00",
        "portfolio_value": 26071.671417588623,
        "drawdown_pct": -24.382744838418088,
        "normalized_value": 260.9975674140946
      },
      {
        "timestamp": "2025-08-04T12:00:00+00:00",
        "portfolio_value": 26071.671417588623,
        "drawdown_pct": -24.382744838418088,
        "normalized_value": 260.9975674140946
      },
      {
        "timestamp": "2025-08-08T07:00:00+00:00",
        "portfolio_value": 26071.671417588623,
        "drawdown_pct": -24.382744838418088,
        "normalized_value": 260.9975674140946
      },
      {
        "timestamp": "2025-08-12T02:00:00+00:00",
        "portfolio_value": 26393.68646650228,
        "drawdown_pct": -23.448784996351872,
        "normalized_value": 264.2211867628908
      },
      {
        "timestamp": "2025-08-15T21:00:00+00:00",
        "portfolio_value": 25880.92357596941,
        "drawdown_pct": -24.93598241111588,
        "normalized_value": 259.0880341948888
      },
      {
        "timestamp": "2025-08-19T16:00:00+00:00",
        "portfolio_value": 25673.81411486468,
        "drawdown_pct": -25.536674584467246,
        "normalized_value": 257.01470852769313
      },
      {
        "timestamp": "2025-08-23T11:00:00+00:00",
        "portfolio_value": 25673.81411486468,
        "drawdown_pct": -25.536674584467246,
        "normalized_value": 257.01470852769313
      },
      {
        "timestamp": "2025-08-27T06:00:00+00:00",
        "portfolio_value": 25673.81411486468,
        "drawdown_pct": -25.536674584467246,
        "normalized_value": 257.01470852769313
      },
      {
        "timestamp": "2025-08-31T01:00:00+00:00",
        "portfolio_value": 25673.81411486468,
        "drawdown_pct": -25.536674584467246,
        "normalized_value": 257.01470852769313
      },
      {
        "timestamp": "2025-09-03T20:00:00+00:00",
        "portfolio_value": 25673.81411486468,
        "drawdown_pct": -25.536674584467246,
        "normalized_value": 257.01470852769313
      },
      {
        "timestamp": "2025-09-07T15:00:00+00:00",
        "portfolio_value": 25213.10501471816,
        "drawdown_pct": -26.872897223324987,
        "normalized_value": 252.40265460534047
      },
      {
        "timestamp": "2025-09-11T10:00:00+00:00",
        "portfolio_value": 25120.98449916354,
        "drawdown_pct": -27.140079960432157,
        "normalized_value": 251.48045709511612
      },
      {
        "timestamp": "2025-09-15T05:00:00+00:00",
        "portfolio_value": 25427.973208214757,
        "drawdown_pct": -26.249701926272444,
        "normalized_value": 254.55365117625615
      },
      {
        "timestamp": "2025-09-19T00:00:00+00:00",
        "portfolio_value": 24876.0668330476,
        "drawdown_pct": -27.850429571534345,
        "normalized_value": 249.0286420944916
      },
      {
        "timestamp": "2025-09-22T19:00:00+00:00",
        "portfolio_value": 24662.5130184258,
        "drawdown_pct": -28.469812695551923,
        "normalized_value": 246.89080347127663
      },
      {
        "timestamp": "2025-09-26T14:00:00+00:00",
        "portfolio_value": 24662.5130184258,
        "drawdown_pct": -28.469812695551923,
        "normalized_value": 246.89080347127663
      },
      {
        "timestamp": "2025-09-30T09:00:00+00:00",
        "portfolio_value": 24454.80478362894,
        "drawdown_pct": -29.072241528679974,
        "normalized_value": 244.81148361695827
      },
      {
        "timestamp": "2025-10-04T04:00:00+00:00",
        "portfolio_value": 26176.23867441585,
        "drawdown_pct": -24.07946206017178,
        "normalized_value": 262.0443664177311
      },
      {
        "timestamp": "2025-10-07T23:00:00+00:00",
        "portfolio_value": 25908.351411184314,
        "drawdown_pct": -24.856431791565797,
        "normalized_value": 259.3626080093507
      },
      {
        "timestamp": "2025-10-11T18:00:00+00:00",
        "portfolio_value": 25299.075706185875,
        "drawdown_pct": -26.623551195254603,
        "normalized_value": 253.26328762662214
      },
      {
        "timestamp": "2025-10-15T13:00:00+00:00",
        "portfolio_value": 25299.075706185875,
        "drawdown_pct": -26.623551195254603,
        "normalized_value": 253.26328762662214
      },
      {
        "timestamp": "2025-10-19T08:00:00+00:00",
        "portfolio_value": 25299.075706185875,
        "drawdown_pct": -26.623551195254603,
        "normalized_value": 253.26328762662214
      },
      {
        "timestamp": "2025-10-23T03:00:00+00:00",
        "portfolio_value": 24291.14426201101,
        "drawdown_pct": -29.546916098823257,
        "normalized_value": 243.17311539192823
      },
      {
        "timestamp": "2025-10-26T22:00:00+00:00",
        "portfolio_value": 25468.633086129867,
        "drawdown_pct": -26.131773608064528,
        "normalized_value": 254.9606879579501
      },
      {
        "timestamp": "2025-10-30T17:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-03T12:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-07T07:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-11T02:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-14T21:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-18T16:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-22T11:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-26T06:00:00+00:00",
        "portfolio_value": 24811.65925090137,
        "drawdown_pct": -28.03723480145168,
        "normalized_value": 248.38387245184225
      },
      {
        "timestamp": "2025-11-30T01:00:00+00:00",
        "portfolio_value": 24780.60019667049,
        "drawdown_pct": -28.12731726648573,
        "normalized_value": 248.07294733044887
      },
      {
        "timestamp": "2025-12-03T20:00:00+00:00",
        "portfolio_value": 24245.754895252925,
        "drawdown_pct": -29.6785616413278,
        "normalized_value": 242.71873277408295
      },
      {
        "timestamp": "2025-12-07T15:00:00+00:00",
        "portfolio_value": 23776.68258300693,
        "drawdown_pct": -31.03904061316732,
        "normalized_value": 238.02295663926566
      },
      {
        "timestamp": "2025-12-11T10:00:00+00:00",
        "portfolio_value": 22556.091420128472,
        "drawdown_pct": -34.57919543995322,
        "normalized_value": 225.80389637205545
      },
      {
        "timestamp": "2025-12-15T05:00:00+00:00",
        "portfolio_value": 21829.34491355742,
        "drawdown_pct": -36.687022557937595,
        "normalized_value": 218.52860253670227
      },
      {
        "timestamp": "2025-12-19T00:00:00+00:00",
        "portfolio_value": 21829.34491355742,
        "drawdown_pct": -36.687022557937595,
        "normalized_value": 218.52860253670227
      },
      {
        "timestamp": "2025-12-22T19:00:00+00:00",
        "portfolio_value": 21829.34491355742,
        "drawdown_pct": -36.687022557937595,
        "normalized_value": 218.52860253670227
      },
      {
        "timestamp": "2025-12-26T14:00:00+00:00",
        "portfolio_value": 21829.34491355742,
        "drawdown_pct": -36.687022557937595,
        "normalized_value": 218.52860253670227
      },
      {
        "timestamp": "2025-12-30T09:00:00+00:00",
        "portfolio_value": 21829.34491355742,
        "drawdown_pct": -36.687022557937595,
        "normalized_value": 218.52860253670227
      },
      {
        "timestamp": "2026-01-03T04:00:00+00:00",
        "portfolio_value": 21829.34491355742,
        "drawdown_pct": -36.687022557937595,
        "normalized_value": 218.52860253670227
      },
      {
        "timestamp": "2026-01-06T23:00:00+00:00",
        "portfolio_value": 22350.941040989703,
        "drawdown_pct": -35.17420556866197,
        "normalized_value": 223.75018262844117
      },
      {
        "timestamp": "2026-01-10T18:00:00+00:00",
        "portfolio_value": 21504.064097254457,
        "drawdown_pct": -37.630454303895974,
        "normalized_value": 215.27229033401508
      },
      {
        "timestamp": "2026-01-14T13:00:00+00:00",
        "portfolio_value": 22246.64986370667,
        "drawdown_pct": -35.47668761660587,
        "normalized_value": 222.70614739427273
      },
      {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "portfolio_value": 22291.950001557787,
        "drawdown_pct": -35.34530087013009,
        "normalized_value": 223.1596367618431
      },
      {
        "timestamp": "2026-01-22T03:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-01-25T22:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-01-29T17:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-02T12:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-06T07:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-10T02:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-13T21:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-17T16:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-21T11:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-02-25T06:00:00+00:00",
        "portfolio_value": 21647.55514736363,
        "drawdown_pct": -37.21427847934992,
        "normalized_value": 216.7087465712981
      },
      {
        "timestamp": "2026-03-01T01:00:00+00:00",
        "portfolio_value": 21139.10731942066,
        "drawdown_pct": -38.68896065549833,
        "normalized_value": 211.61879111257096
      },
      {
        "timestamp": "2026-03-04T20:00:00+00:00",
        "portfolio_value": 20909.40680236121,
        "drawdown_pct": -39.35517504317713,
        "normalized_value": 209.31931152700702
      },
      {
        "timestamp": "2026-03-08T15:00:00+00:00",
        "portfolio_value": 19437.290943465872,
        "drawdown_pct": -43.624842251946966,
        "normalized_value": 194.58229478690689
      },
      {
        "timestamp": "2026-03-12T10:00:00+00:00",
        "portfolio_value": 19500.721724126382,
        "drawdown_pct": -43.44087009882032,
        "normalized_value": 195.21728589224898
      },
      {
        "timestamp": "2026-03-16T05:00:00+00:00",
        "portfolio_value": 20309.529928605814,
        "drawdown_pct": -41.095034444661145,
        "normalized_value": 203.3140807042334
      },
      {
        "timestamp": "2026-03-20T00:00:00+00:00",
        "portfolio_value": 19514.93304937183,
        "drawdown_pct": -43.39965212740387,
        "normalized_value": 195.35955223411034
      },
      {
        "timestamp": "2026-03-23T19:00:00+00:00",
        "portfolio_value": 19514.93304937183,
        "drawdown_pct": -43.39965212740387,
        "normalized_value": 195.35955223411034
      },
      {
        "timestamp": "2026-03-27T14:00:00+00:00",
        "portfolio_value": 18970.994457042827,
        "drawdown_pct": -44.97727032723382,
        "normalized_value": 189.91430681249378
      },
      {
        "timestamp": "2026-03-31T09:00:00+00:00",
        "portfolio_value": 18970.994457042827,
        "drawdown_pct": -44.97727032723382,
        "normalized_value": 189.91430681249378
      },
      {
        "timestamp": "2026-04-04T04:00:00+00:00",
        "portfolio_value": 18970.994457042827,
        "drawdown_pct": -44.97727032723382,
        "normalized_value": 189.91430681249378
      },
      {
        "timestamp": "2026-04-07T23:00:00+00:00",
        "portfolio_value": 19116.486040056505,
        "drawdown_pct": -44.55529223536565,
        "normalized_value": 191.37078992926772
      },
      {
        "timestamp": "2026-04-11T18:00:00+00:00",
        "portfolio_value": 19184.84933882909,
        "drawdown_pct": -44.35701399979819,
        "normalized_value": 192.05515935055502
      },
      {
        "timestamp": "2026-04-15T13:00:00+00:00",
        "portfolio_value": 18954.757268210124,
        "drawdown_pct": -45.02436403409302,
        "normalized_value": 189.75176001144288
      },
      {
        "timestamp": "2026-04-19T08:00:00+00:00",
        "portfolio_value": 19215.126299198757,
        "drawdown_pct": -44.26919988918288,
        "normalized_value": 192.3582551083454
      },
      {
        "timestamp": "2026-04-23T03:00:00+00:00",
        "portfolio_value": 19061.137328834404,
        "drawdown_pct": -44.71582347067889,
        "normalized_value": 190.8167065812117
      },
      {
        "timestamp": "2026-04-26T22:00:00+00:00",
        "portfolio_value": 19179.461493156035,
        "drawdown_pct": -44.37264069647234,
        "normalized_value": 192.00122285405095
      },
      {
        "timestamp": "2026-04-30T17:00:00+00:00",
        "portfolio_value": 18470.957351483583,
        "drawdown_pct": -46.427558373431864,
        "normalized_value": 184.9085491808713
      },
      {
        "timestamp": "2026-05-04T12:00:00+00:00",
        "portfolio_value": 18426.678894835397,
        "drawdown_pct": -46.55598187574205,
        "normalized_value": 184.46528763124044
      },
      {
        "timestamp": "2026-05-08T07:00:00+00:00",
        "portfolio_value": 18590.73228588761,
        "drawdown_pct": -46.080167842471084,
        "normalized_value": 186.1075887827414
      },
      {
        "timestamp": "2026-05-12T02:00:00+00:00",
        "portfolio_value": 18902.356594649023,
        "drawdown_pct": -45.176344896380016,
        "normalized_value": 189.22718879729814
      },
      {
        "timestamp": "2026-05-15T21:00:00+00:00",
        "portfolio_value": 18460.892622963707,
        "drawdown_pct": -46.45674971803074,
        "normalized_value": 184.80779347486776
      },
      {
        "timestamp": "2026-05-19T16:00:00+00:00",
        "portfolio_value": 18460.892622963707,
        "drawdown_pct": -46.45674971803074,
        "normalized_value": 184.80779347486776
      },
      {
        "timestamp": "2026-05-23T11:00:00+00:00",
        "portfolio_value": 18460.892622963707,
        "drawdown_pct": -46.45674971803074,
        "normalized_value": 184.80779347486776
      },
      {
        "timestamp": "2026-05-27T06:00:00+00:00",
        "portfolio_value": 18460.892622963707,
        "drawdown_pct": -46.45674971803074,
        "normalized_value": 184.80779347486776
      },
      {
        "timestamp": "2026-05-31T01:00:00+00:00",
        "portfolio_value": 18460.892622963707,
        "drawdown_pct": -46.45674971803074,
        "normalized_value": 184.80779347486776
      }
    ],
    "ethOnlyEquity": [
      {
        "timestamp": "2021-01-01T00:00:00+00:00",
        "portfolio_value": 9989.23923923924,
        "drawdown_pct": 0.0,
        "normalized_value": 100.0
      },
      {
        "timestamp": "2021-01-04T19:00:00+00:00",
        "portfolio_value": 14465.680214602597,
        "drawdown_pct": -6.193327740670838,
        "normalized_value": 144.81263155435522
      },
      {
        "timestamp": "2021-01-08T14:00:00+00:00",
        "portfolio_value": 17491.574967776258,
        "drawdown_pct": -1.7337340042682607,
        "normalized_value": 175.10417509139947
      },
      {
        "timestamp": "2021-01-12T09:00:00+00:00",
        "portfolio_value": 15391.91082140079,
        "drawdown_pct": -18.045009188022977,
        "normalized_value": 154.08491530505188
      },
      {
        "timestamp": "2021-01-16T04:00:00+00:00",
        "portfolio_value": 15391.91082140079,
        "drawdown_pct": -18.045009188022977,
        "normalized_value": 154.08491530505188
      },
      {
        "timestamp": "2021-01-19T23:00:00+00:00",
        "portfolio_value": 14465.61985408226,
        "drawdown_pct": -22.977091279497735,
        "normalized_value": 144.81202729892706
      },
      {
        "timestamp": "2021-01-23T18:00:00+00:00",
        "portfolio_value": 12781.501950981636,
        "drawdown_pct": -31.944260390365585,
        "normalized_value": 127.95270635599523
      },
      {
        "timestamp": "2021-01-27T13:00:00+00:00",
        "portfolio_value": 11538.280847836011,
        "drawdown_pct": -38.56385267281898,
        "normalized_value": 115.5071029084167
      },
      {
        "timestamp": "2021-01-31T08:00:00+00:00",
        "portfolio_value": 11538.280847836011,
        "drawdown_pct": -38.56385267281898,
        "normalized_value": 115.5071029084167
      },
      {
        "timestamp": "2021-02-04T03:00:00+00:00",
        "portfolio_value": 12974.504573686118,
        "drawdown_pct": -30.916608375357757,
        "normalized_value": 129.88481167535068
      },
      {
        "timestamp": "2021-02-07T22:00:00+00:00",
        "portfolio_value": 12649.632041622855,
        "drawdown_pct": -32.6464082481123,
        "normalized_value": 126.63258671324232
      },
      {
        "timestamp": "2021-02-11T18:00:00+00:00",
        "portfolio_value": 13856.864232569917,
        "drawdown_pct": -26.218440709512368,
        "normalized_value": 138.71791335358213
      },
      {
        "timestamp": "2021-02-15T13:00:00+00:00",
        "portfolio_value": 13906.05848527613,
        "drawdown_pct": -25.956503476681338,
        "normalized_value": 139.21038581848188
      },
      {
        "timestamp": "2021-02-19T08:00:00+00:00",
        "portfolio_value": 14863.04365922802,
        "drawdown_pct": -20.86098856313484,
        "normalized_value": 148.7905465397579
      },
      {
        "timestamp": "2021-02-23T03:00:00+00:00",
        "portfolio_value": 14373.50631023434,
        "drawdown_pct": -23.46755440180343,
        "normalized_value": 143.88989958087134
      },
      {
        "timestamp": "2021-02-26T22:00:00+00:00",
        "portfolio_value": 14373.50631023434,
        "drawdown_pct": -23.46755440180343,
        "normalized_value": 143.88989958087134
      },
      {
        "timestamp": "2021-03-02T17:00:00+00:00",
        "portfolio_value": 14373.50631023434,
        "drawdown_pct": -23.46755440180343,
        "normalized_value": 143.88989958087134
      },
      {
        "timestamp": "2021-03-06T13:00:00+00:00",
        "portfolio_value": 14373.50631023434,
        "drawdown_pct": -23.46755440180343,
        "normalized_value": 143.88989958087134
      },
      {
        "timestamp": "2021-03-10T08:00:00+00:00",
        "portfolio_value": 15228.299255200782,
        "drawdown_pct": -18.91616706830457,
        "normalized_value": 152.44703716156607
      },
      {
        "timestamp": "2021-03-14T03:00:00+00:00",
        "portfolio_value": 14781.48067489347,
        "drawdown_pct": -21.29527470923448,
        "normalized_value": 147.97403807118346
      },
      {
        "timestamp": "2021-03-17T22:00:00+00:00",
        "portfolio_value": 14320.993694730074,
        "drawdown_pct": -23.747160421556426,
        "normalized_value": 143.36420774141686
      },
      {
        "timestamp": "2021-03-21T17:00:00+00:00",
        "portfolio_value": 14320.993694730074,
        "drawdown_pct": -23.747160421556426,
        "normalized_value": 143.36420774141686
      },
      {
        "timestamp": "2021-03-25T12:00:00+00:00",
        "portfolio_value": 14320.993694730074,
        "drawdown_pct": -23.747160421556426,
        "normalized_value": 143.36420774141686
      },
      {
        "timestamp": "2021-03-29T07:00:00+00:00",
        "portfolio_value": 14320.993694730074,
        "drawdown_pct": -23.747160421556426,
        "normalized_value": 143.36420774141686
      },
      {
        "timestamp": "2021-04-02T02:00:00+00:00",
        "portfolio_value": 15018.22046438274,
        "drawdown_pct": -20.034741985419178,
        "normalized_value": 150.3439862105705
      },
      {
        "timestamp": "2021-04-05T21:00:00+00:00",
        "portfolio_value": 15898.928117908405,
        "drawdown_pct": -15.345370503849951,
        "normalized_value": 159.16055003923637
      },
      {
        "timestamp": "2021-04-09T16:00:00+00:00",
        "portfolio_value": 14878.282909295443,
        "drawdown_pct": -20.77984642206158,
        "normalized_value": 148.94310320300772
      },
      {
        "timestamp": "2021-04-13T11:00:00+00:00",
        "portfolio_value": 15962.902682870566,
        "drawdown_pct": -15.004734767032641,
        "normalized_value": 159.8009848454312
      },
      {
        "timestamp": "2021-04-17T06:00:00+00:00",
        "portfolio_value": 17793.866819425526,
        "drawdown_pct": -5.255675619692494,
        "normalized_value": 178.1303500023158
      },
      {
        "timestamp": "2021-04-21T03:00:00+00:00",
        "portfolio_value": 15344.900465210074,
        "drawdown_pct": -18.295318155790717,
        "normalized_value": 153.61430533101048
      },
      {
        "timestamp": "2021-04-24T22:00:00+00:00",
        "portfolio_value": 13879.377351349094,
        "drawdown_pct": -26.09856849454793,
        "normalized_value": 138.94328706062825
      },
      {
        "timestamp": "2021-04-28T20:00:00+00:00",
        "portfolio_value": 14777.814519731604,
        "drawdown_pct": -21.3147953338075,
        "normalized_value": 147.9373370264486
      },
      {
        "timestamp": "2021-05-02T15:00:00+00:00",
        "portfolio_value": 16097.923712354088,
        "drawdown_pct": -14.285808614253733,
        "normalized_value": 161.15264963440873
      },
      {
        "timestamp": "2021-05-06T10:00:00+00:00",
        "portfolio_value": 19356.34042438272,
        "drawdown_pct": -1.2095337903318248,
        "normalized_value": 193.7719175685381
      },
      {
        "timestamp": "2021-05-10T05:00:00+00:00",
        "portfolio_value": 23138.151892784022,
        "drawdown_pct": 0.0,
        "normalized_value": 231.63077125927538
      },
      {
        "timestamp": "2021-05-14T00:00:00+00:00",
        "portfolio_value": 21490.75230500404,
        "drawdown_pct": -11.713218436704873,
        "normalized_value": 215.13902901219066
      },
      {
        "timestamp": "2021-05-17T19:00:00+00:00",
        "portfolio_value": 19958.129372294952,
        "drawdown_pct": -18.009431066147236,
        "normalized_value": 199.7962897304172
      },
      {
        "timestamp": "2021-05-21T14:00:00+00:00",
        "portfolio_value": 19958.129372294952,
        "drawdown_pct": -18.009431066147236,
        "normalized_value": 199.7962897304172
      },
      {
        "timestamp": "2021-05-25T09:00:00+00:00",
        "portfolio_value": 19958.129372294952,
        "drawdown_pct": -18.009431066147236,
        "normalized_value": 199.7962897304172
      },
      {
        "timestamp": "2021-05-29T04:00:00+00:00",
        "portfolio_value": 18587.439628648644,
        "drawdown_pct": -23.64040127467609,
        "normalized_value": 186.07462674068688
      },
      {
        "timestamp": "2021-06-01T23:00:00+00:00",
        "portfolio_value": 17671.329284893833,
        "drawdown_pct": -27.403900693359706,
        "normalized_value": 176.90365463946625
      },
      {
        "timestamp": "2021-06-05T18:00:00+00:00",
        "portfolio_value": 17048.673221106048,
        "drawdown_pct": -29.961852090896084,
        "normalized_value": 170.67038653090103
      },
      {
        "timestamp": "2021-06-09T13:00:00+00:00",
        "portfolio_value": 17062.211886298486,
        "drawdown_pct": -29.90623350856172,
        "normalized_value": 170.80591902610104
      },
      {
        "timestamp": "2021-06-13T08:00:00+00:00",
        "portfolio_value": 17062.211886298486,
        "drawdown_pct": -29.90623350856172,
        "normalized_value": 170.80591902610104
      },
      {
        "timestamp": "2021-06-17T03:00:00+00:00",
        "portfolio_value": 17062.211886298486,
        "drawdown_pct": -29.90623350856172,
        "normalized_value": 170.80591902610104
      },
      {
        "timestamp": "2021-06-20T22:00:00+00:00",
        "portfolio_value": 17062.211886298486,
        "drawdown_pct": -29.90623350856172,
        "normalized_value": 170.80591902610104
      },
      {
        "timestamp": "2021-06-24T17:00:00+00:00",
        "portfolio_value": 17062.211886298486,
        "drawdown_pct": -29.90623350856172,
        "normalized_value": 170.80591902610104
      },
      {
        "timestamp": "2021-06-28T12:00:00+00:00",
        "portfolio_value": 17062.211886298486,
        "drawdown_pct": -29.90623350856172,
        "normalized_value": 170.80591902610104
      },
      {
        "timestamp": "2021-07-02T07:00:00+00:00",
        "portfolio_value": 16172.435649009789,
        "drawdown_pct": -33.561549022268814,
        "normalized_value": 161.89857166982267
      },
      {
        "timestamp": "2021-07-06T02:00:00+00:00",
        "portfolio_value": 16921.214009402367,
        "drawdown_pct": -30.48547096762068,
        "normalized_value": 169.3944213782896
      },
      {
        "timestamp": "2021-07-09T21:00:00+00:00",
        "portfolio_value": 16921.214009402367,
        "drawdown_pct": -30.48547096762068,
        "normalized_value": 169.3944213782896
      },
      {
        "timestamp": "2021-07-13T16:00:00+00:00",
        "portfolio_value": 16921.214009402367,
        "drawdown_pct": -30.48547096762068,
        "normalized_value": 169.3944213782896
      },
      {
        "timestamp": "2021-07-17T11:00:00+00:00",
        "portfolio_value": 16921.214009402367,
        "drawdown_pct": -30.48547096762068,
        "normalized_value": 169.3944213782896
      },
      {
        "timestamp": "2021-07-21T06:00:00+00:00",
        "portfolio_value": 16921.214009402367,
        "drawdown_pct": -30.48547096762068,
        "normalized_value": 169.3944213782896
      },
      {
        "timestamp": "2021-07-25T01:00:00+00:00",
        "portfolio_value": 18205.195266508355,
        "drawdown_pct": -25.210710402301316,
        "normalized_value": 182.24806544822354
      },
      {
        "timestamp": "2021-07-28T20:00:00+00:00",
        "portfolio_value": 18772.301375835523,
        "drawdown_pct": -22.88096538049863,
        "normalized_value": 187.92523560848446
      },
      {
        "timestamp": "2021-08-01T15:00:00+00:00",
        "portfolio_value": 20427.320874517816,
        "drawdown_pct": -16.081931875788086,
        "normalized_value": 204.49325904895957
      },
      {
        "timestamp": "2021-08-05T10:00:00+00:00",
        "portfolio_value": 20761.784865532063,
        "drawdown_pct": -14.707910673720297,
        "normalized_value": 207.84150192315587
      },
      {
        "timestamp": "2021-08-09T05:00:00+00:00",
        "portfolio_value": 24133.10915119963,
        "drawdown_pct": -7.855129460737221,
        "normalized_value": 241.59106187387258
      },
      {
        "timestamp": "2021-08-13T00:00:00+00:00",
        "portfolio_value": 25195.13666592912,
        "drawdown_pct": -6.281442202214669,
        "normalized_value": 252.22277755606072
      },
      {
        "timestamp": "2021-08-16T23:00:00+00:00",
        "portfolio_value": 25860.792051543223,
        "drawdown_pct": -5.7213936874646825,
        "normalized_value": 258.8865020867468
      },
      {
        "timestamp": "2021-08-20T18:00:00+00:00",
        "portfolio_value": 25413.015787146265,
        "drawdown_pct": -7.353815543030369,
        "normalized_value": 254.403915838957
      },
      {
        "timestamp": "2021-08-24T13:00:00+00:00",
        "portfolio_value": 25513.105340350903,
        "drawdown_pct": -6.988927121833907,
        "normalized_value": 255.40588957096526
      },
      {
        "timestamp": "2021-08-28T08:00:00+00:00",
        "portfolio_value": 24192.478950147557,
        "drawdown_pct": -11.803428366798524,
        "normalized_value": 242.1853994157618
      },
      {
        "timestamp": "2021-09-01T03:00:00+00:00",
        "portfolio_value": 24674.12235955502,
        "drawdown_pct": -10.047539788906201,
        "normalized_value": 247.00702194248527
      },
      {
        "timestamp": "2021-09-04T22:00:00+00:00",
        "portfolio_value": 28077.33278205304,
        "drawdown_pct": -2.5346106747864168,
        "normalized_value": 281.07578675021654
      },
      {
        "timestamp": "2021-09-08T17:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-09-12T12:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-09-16T07:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-09-20T02:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-09-23T21:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-09-27T16:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-10-01T13:00:00+00:00",
        "portfolio_value": 27034.133247765465,
        "drawdown_pct": -6.155889431656473,
        "normalized_value": 270.6325536940922
      },
      {
        "timestamp": "2021-10-05T08:00:00+00:00",
        "portfolio_value": 27463.939417953145,
        "drawdown_pct": -4.663894944968694,
        "normalized_value": 274.9352454195976
      },
      {
        "timestamp": "2021-10-09T03:00:00+00:00",
        "portfolio_value": 26649.73441831277,
        "drawdown_pct": -7.4902605366180905,
        "normalized_value": 266.78442451982323
      },
      {
        "timestamp": "2021-10-12T22:00:00+00:00",
        "portfolio_value": 26228.66365603932,
        "drawdown_pct": -8.951931632552066,
        "normalized_value": 262.5691809743546
      },
      {
        "timestamp": "2021-10-16T17:00:00+00:00",
        "portfolio_value": 27201.502029654308,
        "drawdown_pct": -5.574898936073865,
        "normalized_value": 272.30804446851874
      },
      {
        "timestamp": "2021-10-20T12:00:00+00:00",
        "portfolio_value": 27579.483082758503,
        "drawdown_pct": -4.262805982504476,
        "normalized_value": 276.0919267447528
      },
      {
        "timestamp": "2021-10-24T07:00:00+00:00",
        "portfolio_value": 28787.27302156661,
        "drawdown_pct": -5.277727970967161,
        "normalized_value": 288.1828368719597
      },
      {
        "timestamp": "2021-10-28T02:00:00+00:00",
        "portfolio_value": 28144.667424752857,
        "drawdown_pct": -7.39217146490873,
        "normalized_value": 281.7498585297302
      },
      {
        "timestamp": "2021-10-31T21:00:00+00:00",
        "portfolio_value": 29094.43826889563,
        "drawdown_pct": -4.267024730908017,
        "normalized_value": 291.25779823761036
      },
      {
        "timestamp": "2021-11-04T16:00:00+00:00",
        "portfolio_value": 30251.49386745126,
        "drawdown_pct": -4.233026350008391,
        "normalized_value": 302.8408184340888
      },
      {
        "timestamp": "2021-11-08T11:00:00+00:00",
        "portfolio_value": 32102.65720713832,
        "drawdown_pct": -0.19925481764951639,
        "normalized_value": 321.37239321523344
      },
      {
        "timestamp": "2021-11-12T06:00:00+00:00",
        "portfolio_value": 32025.686622038626,
        "drawdown_pct": -2.8042690992071755,
        "normalized_value": 320.6018582099515
      },
      {
        "timestamp": "2021-11-16T01:00:00+00:00",
        "portfolio_value": 30320.20462179676,
        "drawdown_pct": -7.980288321152029,
        "normalized_value": 303.5286661540193
      },
      {
        "timestamp": "2021-11-19T20:00:00+00:00",
        "portfolio_value": 30320.20462179676,
        "drawdown_pct": -7.980288321152029,
        "normalized_value": 303.5286661540193
      },
      {
        "timestamp": "2021-11-23T15:00:00+00:00",
        "portfolio_value": 30320.20462179676,
        "drawdown_pct": -7.980288321152029,
        "normalized_value": 303.5286661540193
      },
      {
        "timestamp": "2021-11-27T10:00:00+00:00",
        "portfolio_value": 30320.20462179676,
        "drawdown_pct": -7.980288321152029,
        "normalized_value": 303.5286661540193
      },
      {
        "timestamp": "2021-12-01T05:00:00+00:00",
        "portfolio_value": 31249.78962976357,
        "drawdown_pct": -5.15906249233319,
        "normalized_value": 312.8345300511943
      },
      {
        "timestamp": "2021-12-05T00:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-08T19:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-12T14:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-16T09:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-20T04:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-23T23:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-27T18:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2021-12-31T13:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-04T08:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-08T03:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-11T22:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-15T17:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-19T12:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-23T07:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-27T02:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-01-30T21:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-02-03T16:00:00+00:00",
        "portfolio_value": 29939.769043203654,
        "drawdown_pct": -9.134883835634374,
        "normalized_value": 299.7202121818819
      },
      {
        "timestamp": "2022-02-07T11:00:00+00:00",
        "portfolio_value": 31307.90839850113,
        "drawdown_pct": -4.98267607248313,
        "normalized_value": 313.4163438144413
      },
      {
        "timestamp": "2022-02-11T06:00:00+00:00",
        "portfolio_value": 31338.238040949716,
        "drawdown_pct": -5.919183578631286,
        "normalized_value": 313.71996696053077
      },
      {
        "timestamp": "2022-02-15T01:00:00+00:00",
        "portfolio_value": 29663.265942798407,
        "drawdown_pct": -10.947632921289753,
        "normalized_value": 296.9522025889281
      },
      {
        "timestamp": "2022-02-18T20:00:00+00:00",
        "portfolio_value": 27696.90482401075,
        "drawdown_pct": -16.85086395786057,
        "normalized_value": 277.2674090656787
      },
      {
        "timestamp": "2022-02-22T15:00:00+00:00",
        "portfolio_value": 27696.90482401075,
        "drawdown_pct": -16.85086395786057,
        "normalized_value": 277.2674090656787
      },
      {
        "timestamp": "2022-02-26T10:00:00+00:00",
        "portfolio_value": 27696.90482401075,
        "drawdown_pct": -16.85086395786057,
        "normalized_value": 277.2674090656787
      },
      {
        "timestamp": "2022-03-02T05:00:00+00:00",
        "portfolio_value": 28166.80084372268,
        "drawdown_pct": -15.440184738758237,
        "normalized_value": 281.9714311484226
      },
      {
        "timestamp": "2022-03-06T00:00:00+00:00",
        "portfolio_value": 26432.41831779512,
        "drawdown_pct": -20.64699068020917,
        "normalized_value": 264.6089225089794
      },
      {
        "timestamp": "2022-03-09T19:00:00+00:00",
        "portfolio_value": 26432.41831779512,
        "drawdown_pct": -20.64699068020917,
        "normalized_value": 264.6089225089794
      },
      {
        "timestamp": "2022-03-13T14:00:00+00:00",
        "portfolio_value": 26432.41831779512,
        "drawdown_pct": -20.64699068020917,
        "normalized_value": 264.6089225089794
      },
      {
        "timestamp": "2022-03-17T09:00:00+00:00",
        "portfolio_value": 26407.13072969137,
        "drawdown_pct": -20.722906784076255,
        "normalized_value": 264.3557742211256
      },
      {
        "timestamp": "2022-03-21T04:00:00+00:00",
        "portfolio_value": 27553.00111738575,
        "drawdown_pct": -17.2828786163634,
        "normalized_value": 275.8268218179559
      },
      {
        "timestamp": "2022-03-24T23:00:00+00:00",
        "portfolio_value": 28313.41028614428,
        "drawdown_pct": -15.00004716560935,
        "normalized_value": 283.4391049012515
      },
      {
        "timestamp": "2022-03-28T18:00:00+00:00",
        "portfolio_value": 31300.399258528683,
        "drawdown_pct": -6.03277973990588,
        "normalized_value": 313.34117152361307
      },
      {
        "timestamp": "2022-04-01T13:00:00+00:00",
        "portfolio_value": 29612.758785324877,
        "drawdown_pct": -11.09926092934889,
        "normalized_value": 296.4465869332821
      },
      {
        "timestamp": "2022-04-05T08:00:00+00:00",
        "portfolio_value": 30476.87489751848,
        "drawdown_pct": -8.505089897403746,
        "normalized_value": 305.0970566186934
      },
      {
        "timestamp": "2022-04-09T03:00:00+00:00",
        "portfolio_value": 28953.381963201988,
        "drawdown_pct": -13.078782230884356,
        "normalized_value": 289.84571567240806
      },
      {
        "timestamp": "2022-04-12T22:00:00+00:00",
        "portfolio_value": 28953.381963201988,
        "drawdown_pct": -13.078782230884356,
        "normalized_value": 289.84571567240806
      },
      {
        "timestamp": "2022-04-16T17:00:00+00:00",
        "portfolio_value": 28953.381963201988,
        "drawdown_pct": -13.078782230884356,
        "normalized_value": 289.84571567240806
      },
      {
        "timestamp": "2022-04-20T12:00:00+00:00",
        "portfolio_value": 29425.106087721924,
        "drawdown_pct": -11.662614841304274,
        "normalized_value": 294.5680384962217
      },
      {
        "timestamp": "2022-04-24T07:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-04-28T02:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-01T21:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-05T16:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-09T11:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-13T06:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-17T01:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-20T20:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-24T15:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-05-28T10:00:00+00:00",
        "portfolio_value": 27779.7347514927,
        "drawdown_pct": -16.602199460787006,
        "normalized_value": 278.09660061368544
      },
      {
        "timestamp": "2022-06-01T05:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-05T00:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-08T19:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-12T14:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-16T09:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-20T04:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-23T23:00:00+00:00",
        "portfolio_value": 26908.27795430591,
        "drawdown_pct": -19.218408031548105,
        "normalized_value": 269.37264500189497
      },
      {
        "timestamp": "2022-06-27T18:00:00+00:00",
        "portfolio_value": 26905.84715278237,
        "drawdown_pct": -19.225705563451275,
        "normalized_value": 269.34831080120836
      },
      {
        "timestamp": "2022-07-01T13:00:00+00:00",
        "portfolio_value": 26521.554856961902,
        "drawdown_pct": -20.379393045433787,
        "normalized_value": 265.5012481108795
      },
      {
        "timestamp": "2022-07-05T08:00:00+00:00",
        "portfolio_value": 26521.554856961902,
        "drawdown_pct": -20.379393045433787,
        "normalized_value": 265.5012481108795
      },
      {
        "timestamp": "2022-07-09T03:00:00+00:00",
        "portfolio_value": 27417.22659174767,
        "drawdown_pct": -17.690488584880555,
        "normalized_value": 274.4676139505065
      },
      {
        "timestamp": "2022-07-12T22:00:00+00:00",
        "portfolio_value": 27417.22659174767,
        "drawdown_pct": -17.690488584880555,
        "normalized_value": 274.4676139505065
      },
      {
        "timestamp": "2022-07-16T17:00:00+00:00",
        "portfolio_value": 28987.013432809432,
        "drawdown_pct": -12.977816882609694,
        "normalized_value": 290.1823926585327
      },
      {
        "timestamp": "2022-07-20T12:00:00+00:00",
        "portfolio_value": 35350.76721582765,
        "drawdown_pct": -0.05975467725208439,
        "normalized_value": 353.88848308852687
      },
      {
        "timestamp": "2022-07-24T07:00:00+00:00",
        "portfolio_value": 35405.92140422828,
        "drawdown_pct": -2.2463790374973387,
        "normalized_value": 354.44061911289975
      },
      {
        "timestamp": "2022-07-28T02:00:00+00:00",
        "portfolio_value": 32084.10579492872,
        "drawdown_pct": -11.417712280642784,
        "normalized_value": 321.1866792507833
      },
      {
        "timestamp": "2022-07-31T21:00:00+00:00",
        "portfolio_value": 33002.572971342765,
        "drawdown_pct": -8.881879610038812,
        "normalized_value": 330.38124506722875
      },
      {
        "timestamp": "2022-08-04T16:00:00+00:00",
        "portfolio_value": 29785.941626262043,
        "drawdown_pct": -17.76280542166237,
        "normalized_value": 298.1802809292861
      },
      {
        "timestamp": "2022-08-08T11:00:00+00:00",
        "portfolio_value": 31423.63325624907,
        "drawdown_pct": -13.24123726295046,
        "normalized_value": 314.5748390208966
      },
      {
        "timestamp": "2022-08-12T06:00:00+00:00",
        "portfolio_value": 30561.386228388863,
        "drawdown_pct": -15.62184948244845,
        "normalized_value": 305.94308031325477
      },
      {
        "timestamp": "2022-08-16T01:00:00+00:00",
        "portfolio_value": 30271.380520367853,
        "drawdown_pct": -16.422537811815367,
        "normalized_value": 303.0398991892926
      },
      {
        "timestamp": "2022-08-19T20:00:00+00:00",
        "portfolio_value": 30206.712895688943,
        "drawdown_pct": -16.601081236776274,
        "normalized_value": 302.39252632004667
      },
      {
        "timestamp": "2022-08-23T15:00:00+00:00",
        "portfolio_value": 30206.712895688943,
        "drawdown_pct": -16.601081236776274,
        "normalized_value": 302.39252632004667
      },
      {
        "timestamp": "2022-08-27T10:00:00+00:00",
        "portfolio_value": 30206.712895688943,
        "drawdown_pct": -16.601081236776274,
        "normalized_value": 302.39252632004667
      },
      {
        "timestamp": "2022-08-31T05:00:00+00:00",
        "portfolio_value": 30206.712895688943,
        "drawdown_pct": -16.601081236776274,
        "normalized_value": 302.39252632004667
      },
      {
        "timestamp": "2022-09-04T00:00:00+00:00",
        "portfolio_value": 29370.697049066934,
        "drawdown_pct": -18.909270741472344,
        "normalized_value": 294.023361996321
      },
      {
        "timestamp": "2022-09-07T19:00:00+00:00",
        "portfolio_value": 29324.96284916353,
        "drawdown_pct": -19.035540118601073,
        "normalized_value": 293.56552733235833
      },
      {
        "timestamp": "2022-09-11T14:00:00+00:00",
        "portfolio_value": 30241.560971840798,
        "drawdown_pct": -16.50486779302689,
        "normalized_value": 302.74138247733003
      },
      {
        "timestamp": "2022-09-15T09:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-09-19T04:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-09-22T23:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-09-26T18:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-09-30T13:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-10-04T08:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-10-08T03:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-10-11T22:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-10-15T17:00:00+00:00",
        "portfolio_value": 29936.62272891825,
        "drawdown_pct": -17.346783953753114,
        "normalized_value": 299.6887151458209
      },
      {
        "timestamp": "2022-10-19T12:00:00+00:00",
        "portfolio_value": 29572.149708946406,
        "drawdown_pct": -18.353072055639128,
        "normalized_value": 296.04005871420657
      },
      {
        "timestamp": "2022-10-23T07:00:00+00:00",
        "portfolio_value": 29572.149708946406,
        "drawdown_pct": -18.353072055639128,
        "normalized_value": 296.04005871420657
      },
      {
        "timestamp": "2022-10-27T02:00:00+00:00",
        "portfolio_value": 34529.04964564843,
        "drawdown_pct": -4.667369259508638,
        "normalized_value": 345.6624555553051
      },
      {
        "timestamp": "2022-10-30T21:00:00+00:00",
        "portfolio_value": 35634.65265019375,
        "drawdown_pct": -2.5462479918008163,
        "normalized_value": 356.7303955461939
      },
      {
        "timestamp": "2022-11-03T16:00:00+00:00",
        "portfolio_value": 34071.86957197964,
        "drawdown_pct": -6.8201517124832165,
        "normalized_value": 341.08572991364736
      },
      {
        "timestamp": "2022-11-07T11:00:00+00:00",
        "portfolio_value": 33511.98001228394,
        "drawdown_pct": -8.351339313445227,
        "normalized_value": 335.4808029889185
      },
      {
        "timestamp": "2022-11-11T06:00:00+00:00",
        "portfolio_value": 33511.98001228394,
        "drawdown_pct": -8.351339313445227,
        "normalized_value": 335.4808029889185
      },
      {
        "timestamp": "2022-11-15T01:00:00+00:00",
        "portfolio_value": 33511.98001228394,
        "drawdown_pct": -8.351339313445227,
        "normalized_value": 335.4808029889185
      },
      {
        "timestamp": "2022-11-18T20:00:00+00:00",
        "portfolio_value": 33511.98001228394,
        "drawdown_pct": -8.351339313445227,
        "normalized_value": 335.4808029889185
      },
      {
        "timestamp": "2022-11-22T15:00:00+00:00",
        "portfolio_value": 33511.98001228394,
        "drawdown_pct": -8.351339313445227,
        "normalized_value": 335.4808029889185
      },
      {
        "timestamp": "2022-11-26T10:00:00+00:00",
        "portfolio_value": 32690.49529928769,
        "drawdown_pct": -10.597938102683816,
        "normalized_value": 327.2571065359461
      },
      {
        "timestamp": "2022-11-30T05:00:00+00:00",
        "portfolio_value": 31978.83288644828,
        "drawdown_pct": -12.544194545119813,
        "normalized_value": 320.13281612908617
      },
      {
        "timestamp": "2022-12-04T00:00:00+00:00",
        "portfolio_value": 32004.15016472793,
        "drawdown_pct": -12.474956778634269,
        "normalized_value": 320.38626163853195
      },
      {
        "timestamp": "2022-12-07T19:00:00+00:00",
        "portfolio_value": 31601.51631205857,
        "drawdown_pct": -13.576081013331475,
        "normalized_value": 316.35558579799596
      },
      {
        "timestamp": "2022-12-11T14:00:00+00:00",
        "portfolio_value": 31000.760150446447,
        "drawdown_pct": -15.219030716415707,
        "normalized_value": 310.3415526246562
      },
      {
        "timestamp": "2022-12-15T09:00:00+00:00",
        "portfolio_value": 30523.90828906588,
        "drawdown_pct": -16.523126577818118,
        "normalized_value": 305.56789719444663
      },
      {
        "timestamp": "2022-12-19T04:00:00+00:00",
        "portfolio_value": 30049.20927711439,
        "drawdown_pct": -17.82133482032237,
        "normalized_value": 300.81579344978104
      },
      {
        "timestamp": "2022-12-22T23:00:00+00:00",
        "portfolio_value": 30049.20927711439,
        "drawdown_pct": -17.82133482032237,
        "normalized_value": 300.81579344978104
      },
      {
        "timestamp": "2022-12-26T18:00:00+00:00",
        "portfolio_value": 30049.20927711439,
        "drawdown_pct": -17.82133482032237,
        "normalized_value": 300.81579344978104
      },
      {
        "timestamp": "2022-12-30T13:00:00+00:00",
        "portfolio_value": 30049.20927711439,
        "drawdown_pct": -17.82133482032237,
        "normalized_value": 300.81579344978104
      },
      {
        "timestamp": "2023-01-03T08:00:00+00:00",
        "portfolio_value": 30049.20927711439,
        "drawdown_pct": -17.82133482032237,
        "normalized_value": 300.81579344978104
      },
      {
        "timestamp": "2023-01-07T03:00:00+00:00",
        "portfolio_value": 30388.086533951733,
        "drawdown_pct": -16.894572309874114,
        "normalized_value": 304.208216523464
      },
      {
        "timestamp": "2023-01-10T22:00:00+00:00",
        "portfolio_value": 32138.600236912545,
        "drawdown_pct": -12.107262329054926,
        "normalized_value": 321.7322107039671
      },
      {
        "timestamp": "2023-01-14T17:00:00+00:00",
        "portfolio_value": 37162.76903741506,
        "drawdown_pct": -2.2893537629355305,
        "normalized_value": 372.0280208269925
      },
      {
        "timestamp": "2023-01-18T12:00:00+00:00",
        "portfolio_value": 38472.337007890135,
        "drawdown_pct": -1.0343611855710042,
        "normalized_value": 385.1378076597164
      },
      {
        "timestamp": "2023-01-22T07:00:00+00:00",
        "portfolio_value": 39682.82677188985,
        "drawdown_pct": -2.2994322378858296,
        "normalized_value": 397.2557451223084
      },
      {
        "timestamp": "2023-01-26T02:00:00+00:00",
        "portfolio_value": 37486.57563982972,
        "drawdown_pct": -7.706682678584246,
        "normalized_value": 375.2695750100448
      },
      {
        "timestamp": "2023-01-29T21:00:00+00:00",
        "portfolio_value": 36914.943513201964,
        "drawdown_pct": -9.114061836415775,
        "normalized_value": 369.54709592092354
      },
      {
        "timestamp": "2023-02-02T16:00:00+00:00",
        "portfolio_value": 36532.25338019674,
        "drawdown_pct": -10.05625891039174,
        "normalized_value": 365.71607211780986
      },
      {
        "timestamp": "2023-02-06T11:00:00+00:00",
        "portfolio_value": 35651.53862988685,
        "drawdown_pct": -12.224610767893424,
        "normalized_value": 356.89943724485266
      },
      {
        "timestamp": "2023-02-10T06:00:00+00:00",
        "portfolio_value": 34318.40575568798,
        "drawdown_pct": -15.50683255769375,
        "normalized_value": 343.55374752543815
      },
      {
        "timestamp": "2023-02-14T01:00:00+00:00",
        "portfolio_value": 34318.40575568798,
        "drawdown_pct": -15.50683255769375,
        "normalized_value": 343.55374752543815
      },
      {
        "timestamp": "2023-02-17T20:00:00+00:00",
        "portfolio_value": 37630.28201164631,
        "drawdown_pct": -7.352872346517383,
        "normalized_value": 376.7081867839233
      },
      {
        "timestamp": "2023-02-21T15:00:00+00:00",
        "portfolio_value": 36722.52001880343,
        "drawdown_pct": -9.587815502241423,
        "normalized_value": 367.62078812320186
      },
      {
        "timestamp": "2023-02-25T10:00:00+00:00",
        "portfolio_value": 34393.82120836496,
        "drawdown_pct": -15.321156960868635,
        "normalized_value": 344.30871445405813
      },
      {
        "timestamp": "2023-03-01T05:00:00+00:00",
        "portfolio_value": 34393.82120836496,
        "drawdown_pct": -15.321156960868635,
        "normalized_value": 344.30871445405813
      },
      {
        "timestamp": "2023-03-05T00:00:00+00:00",
        "portfolio_value": 33750.59852486118,
        "drawdown_pct": -16.904794682471326,
        "normalized_value": 337.8695585974529
      },
      {
        "timestamp": "2023-03-08T19:00:00+00:00",
        "portfolio_value": 33750.59852486118,
        "drawdown_pct": -16.904794682471326,
        "normalized_value": 337.8695585974529
      },
      {
        "timestamp": "2023-03-12T14:00:00+00:00",
        "portfolio_value": 33750.59852486118,
        "drawdown_pct": -16.904794682471326,
        "normalized_value": 337.8695585974529
      },
      {
        "timestamp": "2023-03-16T09:00:00+00:00",
        "portfolio_value": 34832.70406948284,
        "drawdown_pct": -14.240611339490375,
        "normalized_value": 348.70227086618087
      },
      {
        "timestamp": "2023-03-20T04:00:00+00:00",
        "portfolio_value": 37017.36513500407,
        "drawdown_pct": -8.861896065597492,
        "normalized_value": 370.5724154607717
      },
      {
        "timestamp": "2023-03-23T23:00:00+00:00",
        "portfolio_value": 38412.52448612524,
        "drawdown_pct": -5.42696282591916,
        "normalized_value": 384.53903812049117
      },
      {
        "timestamp": "2023-03-27T19:00:00+00:00",
        "portfolio_value": 35936.603724194145,
        "drawdown_pct": -11.522770102078947,
        "normalized_value": 359.7531590096445
      },
      {
        "timestamp": "2023-03-31T14:00:00+00:00",
        "portfolio_value": 38787.112112349656,
        "drawdown_pct": -4.5047144193436415,
        "normalized_value": 388.2889495727365
      },
      {
        "timestamp": "2023-04-04T09:00:00+00:00",
        "portfolio_value": 38597.06418624916,
        "drawdown_pct": -4.972619349321652,
        "normalized_value": 386.3864230484547
      },
      {
        "timestamp": "2023-04-08T04:00:00+00:00",
        "portfolio_value": 39433.72372373119,
        "drawdown_pct": -3.089843614634575,
        "normalized_value": 394.7620312148454
      },
      {
        "timestamp": "2023-04-11T23:00:00+00:00",
        "portfolio_value": 39964.41034664629,
        "drawdown_pct": -2.0723668547161,
        "normalized_value": 400.0746141874354
      },
      {
        "timestamp": "2023-04-15T18:00:00+00:00",
        "portfolio_value": 44634.7684195575,
        "drawdown_pct": -1.14108926461644,
        "normalized_value": 446.8285056606252
      },
      {
        "timestamp": "2023-04-19T13:00:00+00:00",
        "portfolio_value": 41908.63587754292,
        "drawdown_pct": -7.614075307093369,
        "normalized_value": 419.53781337941604
      },
      {
        "timestamp": "2023-04-23T08:00:00+00:00",
        "portfolio_value": 41908.63587754292,
        "drawdown_pct": -7.614075307093369,
        "normalized_value": 419.53781337941604
      },
      {
        "timestamp": "2023-04-27T03:00:00+00:00",
        "portfolio_value": 39951.08962724351,
        "drawdown_pct": -11.929408332761055,
        "normalized_value": 399.9412634979209
      },
      {
        "timestamp": "2023-04-30T22:00:00+00:00",
        "portfolio_value": 39951.08962724351,
        "drawdown_pct": -11.929408332761055,
        "normalized_value": 399.9412634979209
      },
      {
        "timestamp": "2023-05-04T17:00:00+00:00",
        "portfolio_value": 39951.08962724351,
        "drawdown_pct": -11.929408332761055,
        "normalized_value": 399.9412634979209
      },
      {
        "timestamp": "2023-05-08T12:00:00+00:00",
        "portfolio_value": 40510.42282415364,
        "drawdown_pct": -10.696380496708963,
        "normalized_value": 405.54062080145786
      },
      {
        "timestamp": "2023-05-12T07:00:00+00:00",
        "portfolio_value": 40510.42282415364,
        "drawdown_pct": -10.696380496708963,
        "normalized_value": 405.54062080145786
      },
      {
        "timestamp": "2023-05-16T02:00:00+00:00",
        "portfolio_value": 40510.42282415364,
        "drawdown_pct": -10.696380496708963,
        "normalized_value": 405.54062080145786
      },
      {
        "timestamp": "2023-05-19T21:00:00+00:00",
        "portfolio_value": 40510.42282415364,
        "drawdown_pct": -10.696380496708963,
        "normalized_value": 405.54062080145786
      },
      {
        "timestamp": "2023-05-23T16:00:00+00:00",
        "portfolio_value": 40333.85489628695,
        "drawdown_pct": -11.085617486782779,
        "normalized_value": 403.77303947080856
      },
      {
        "timestamp": "2023-05-27T11:00:00+00:00",
        "portfolio_value": 39803.201845225776,
        "drawdown_pct": -12.255421079451597,
        "normalized_value": 398.4607925784057
      },
      {
        "timestamp": "2023-05-31T06:00:00+00:00",
        "portfolio_value": 38682.00588806922,
        "drawdown_pct": -14.727053073547648,
        "normalized_value": 387.23675508862044
      },
      {
        "timestamp": "2023-06-04T01:00:00+00:00",
        "portfolio_value": 38563.29029246044,
        "drawdown_pct": -14.988756892966487,
        "normalized_value": 386.04832028627385
      },
      {
        "timestamp": "2023-06-07T20:00:00+00:00",
        "portfolio_value": 37376.25307001962,
        "drawdown_pct": -17.605533343541914,
        "normalized_value": 374.16516087831855
      },
      {
        "timestamp": "2023-06-11T15:00:00+00:00",
        "portfolio_value": 37376.25307001962,
        "drawdown_pct": -17.605533343541914,
        "normalized_value": 374.16516087831855
      },
      {
        "timestamp": "2023-06-15T10:00:00+00:00",
        "portfolio_value": 37376.25307001962,
        "drawdown_pct": -17.605533343541914,
        "normalized_value": 374.16516087831855
      },
      {
        "timestamp": "2023-06-19T05:00:00+00:00",
        "portfolio_value": 36802.20244113472,
        "drawdown_pct": -18.87100517433693,
        "normalized_value": 368.4184707136667
      },
      {
        "timestamp": "2023-06-23T00:00:00+00:00",
        "portfolio_value": 39537.88248692448,
        "drawdown_pct": -12.84030707591346,
        "normalized_value": 395.8047408817051
      },
      {
        "timestamp": "2023-06-26T19:00:00+00:00",
        "portfolio_value": 38970.53643002444,
        "drawdown_pct": -14.090998943831579,
        "normalized_value": 390.12516866091556
      },
      {
        "timestamp": "2023-06-30T14:00:00+00:00",
        "portfolio_value": 37374.38379798186,
        "drawdown_pct": -17.6096540795691,
        "normalized_value": 374.14644802148337
      },
      {
        "timestamp": "2023-07-04T09:00:00+00:00",
        "portfolio_value": 38033.580881192654,
        "drawdown_pct": -16.15648026915811,
        "normalized_value": 380.74551995702546
      },
      {
        "timestamp": "2023-07-08T04:00:00+00:00",
        "portfolio_value": 36564.51944225092,
        "drawdown_pct": -19.394967913182114,
        "normalized_value": 366.0390803197502
      },
      {
        "timestamp": "2023-07-11T23:00:00+00:00",
        "portfolio_value": 36012.050472083334,
        "drawdown_pct": -20.612863833779734,
        "normalized_value": 360.5084392275096
      },
      {
        "timestamp": "2023-07-15T18:00:00+00:00",
        "portfolio_value": 36720.1167290242,
        "drawdown_pct": -19.051959869201383,
        "normalized_value": 367.59672933632464
      },
      {
        "timestamp": "2023-07-19T13:00:00+00:00",
        "portfolio_value": 36720.1167290242,
        "drawdown_pct": -19.051959869201383,
        "normalized_value": 367.59672933632464
      },
      {
        "timestamp": "2023-07-23T08:00:00+00:00",
        "portfolio_value": 36720.1167290242,
        "drawdown_pct": -19.051959869201383,
        "normalized_value": 367.59672933632464
      },
      {
        "timestamp": "2023-07-27T03:00:00+00:00",
        "portfolio_value": 36720.1167290242,
        "drawdown_pct": -19.051959869201383,
        "normalized_value": 367.59672933632464
      },
      {
        "timestamp": "2023-07-30T22:00:00+00:00",
        "portfolio_value": 36720.1167290242,
        "drawdown_pct": -19.051959869201383,
        "normalized_value": 367.59672933632464
      },
      {
        "timestamp": "2023-08-03T17:00:00+00:00",
        "portfolio_value": 36093.098425619806,
        "drawdown_pct": -20.43419683650944,
        "normalized_value": 361.3197918400099
      },
      {
        "timestamp": "2023-08-07T12:00:00+00:00",
        "portfolio_value": 35840.98390620877,
        "drawdown_pct": -20.989973289657545,
        "normalized_value": 358.79593077939285
      },
      {
        "timestamp": "2023-08-11T07:00:00+00:00",
        "portfolio_value": 35207.29120160193,
        "drawdown_pct": -22.386923709549954,
        "normalized_value": 352.4521773720503
      },
      {
        "timestamp": "2023-08-15T02:00:00+00:00",
        "portfolio_value": 34497.87758756842,
        "drawdown_pct": -23.95079786937062,
        "normalized_value": 345.3503991781031
      },
      {
        "timestamp": "2023-08-18T21:00:00+00:00",
        "portfolio_value": 34497.87758756842,
        "drawdown_pct": -23.95079786937062,
        "normalized_value": 345.3503991781031
      },
      {
        "timestamp": "2023-08-22T16:00:00+00:00",
        "portfolio_value": 34497.87758756842,
        "drawdown_pct": -23.95079786937062,
        "normalized_value": 345.3503991781031
      },
      {
        "timestamp": "2023-08-26T11:00:00+00:00",
        "portfolio_value": 34497.87758756842,
        "drawdown_pct": -23.95079786937062,
        "normalized_value": 345.3503991781031
      },
      {
        "timestamp": "2023-08-30T06:00:00+00:00",
        "portfolio_value": 34121.11968381124,
        "drawdown_pct": -24.781345716971177,
        "normalized_value": 341.5787615715352
      },
      {
        "timestamp": "2023-09-03T01:00:00+00:00",
        "portfolio_value": 32610.0538525956,
        "drawdown_pct": -28.112430376863877,
        "normalized_value": 326.45182552539524
      },
      {
        "timestamp": "2023-09-06T20:00:00+00:00",
        "portfolio_value": 32610.0538525956,
        "drawdown_pct": -28.112430376863877,
        "normalized_value": 326.45182552539524
      },
      {
        "timestamp": "2023-09-10T15:00:00+00:00",
        "portfolio_value": 32610.0538525956,
        "drawdown_pct": -28.112430376863877,
        "normalized_value": 326.45182552539524
      },
      {
        "timestamp": "2023-09-14T10:00:00+00:00",
        "portfolio_value": 32627.3225226488,
        "drawdown_pct": -28.0743622790202,
        "normalized_value": 326.62469825013056
      },
      {
        "timestamp": "2023-09-18T05:00:00+00:00",
        "portfolio_value": 32059.880233209697,
        "drawdown_pct": -29.3252662877518,
        "normalized_value": 320.9441626672995
      },
      {
        "timestamp": "2023-09-22T00:00:00+00:00",
        "portfolio_value": 31591.8586947614,
        "drawdown_pct": -30.357001196329737,
        "normalized_value": 316.25890558976516
      },
      {
        "timestamp": "2023-09-25T19:00:00+00:00",
        "portfolio_value": 31591.8586947614,
        "drawdown_pct": -30.357001196329737,
        "normalized_value": 316.25890558976516
      },
      {
        "timestamp": "2023-09-29T14:00:00+00:00",
        "portfolio_value": 32525.13204869356,
        "drawdown_pct": -28.299637123535486,
        "normalized_value": 325.60169267875705
      },
      {
        "timestamp": "2023-10-03T09:00:00+00:00",
        "portfolio_value": 31061.552939524357,
        "drawdown_pct": -31.526039189136306,
        "normalized_value": 310.95013539679667
      },
      {
        "timestamp": "2023-10-07T04:00:00+00:00",
        "portfolio_value": 31061.552939524357,
        "drawdown_pct": -31.526039189136306,
        "normalized_value": 310.95013539679667
      },
      {
        "timestamp": "2023-10-10T23:00:00+00:00",
        "portfolio_value": 31061.552939524357,
        "drawdown_pct": -31.526039189136306,
        "normalized_value": 310.95013539679667
      },
      {
        "timestamp": "2023-10-14T18:00:00+00:00",
        "portfolio_value": 31061.552939524357,
        "drawdown_pct": -31.526039189136306,
        "normalized_value": 310.95013539679667
      },
      {
        "timestamp": "2023-10-18T13:00:00+00:00",
        "portfolio_value": 31061.552939524357,
        "drawdown_pct": -31.526039189136306,
        "normalized_value": 310.95013539679667
      },
      {
        "timestamp": "2023-10-22T08:00:00+00:00",
        "portfolio_value": 31466.658683665384,
        "drawdown_pct": -30.63299965236313,
        "normalized_value": 315.005556780136
      },
      {
        "timestamp": "2023-10-26T03:00:00+00:00",
        "portfolio_value": 34245.44743021367,
        "drawdown_pct": -24.50727013383989,
        "normalized_value": 342.8233783378857
      },
      {
        "timestamp": "2023-10-29T22:00:00+00:00",
        "portfolio_value": 33991.4688108546,
        "drawdown_pct": -25.06715592134864,
        "normalized_value": 340.2808561970463
      },
      {
        "timestamp": "2023-11-02T17:00:00+00:00",
        "portfolio_value": 33936.02780733223,
        "drawdown_pct": -25.189373413497112,
        "normalized_value": 339.7258489317824
      },
      {
        "timestamp": "2023-11-06T12:00:00+00:00",
        "portfolio_value": 35968.7657214151,
        "drawdown_pct": -20.70828334891909,
        "normalized_value": 360.0751254422294
      },
      {
        "timestamp": "2023-11-10T07:00:00+00:00",
        "portfolio_value": 40358.944749172624,
        "drawdown_pct": -11.030307901760143,
        "normalized_value": 404.024208276408
      },
      {
        "timestamp": "2023-11-14T02:00:00+00:00",
        "portfolio_value": 39361.97960777889,
        "drawdown_pct": -13.22807799247335,
        "normalized_value": 394.0438172024061
      },
      {
        "timestamp": "2023-11-17T21:00:00+00:00",
        "portfolio_value": 37592.6016722099,
        "drawdown_pct": -17.12860143557501,
        "normalized_value": 376.3309774836555
      },
      {
        "timestamp": "2023-11-21T16:00:00+00:00",
        "portfolio_value": 36975.00839357471,
        "drawdown_pct": -18.490061309801327,
        "normalized_value": 370.14839176472316
      },
      {
        "timestamp": "2023-11-25T11:00:00+00:00",
        "portfolio_value": 36340.7657150001,
        "drawdown_pct": -19.888224125480555,
        "normalized_value": 363.7991326931894
      },
      {
        "timestamp": "2023-11-29T06:00:00+00:00",
        "portfolio_value": 35150.21507277986,
        "drawdown_pct": -22.51274577053344,
        "normalized_value": 351.88080123964306
      },
      {
        "timestamp": "2023-12-03T01:00:00+00:00",
        "portfolio_value": 36032.72063085964,
        "drawdown_pct": -20.56729728903038,
        "normalized_value": 360.715363481512
      },
      {
        "timestamp": "2023-12-06T20:00:00+00:00",
        "portfolio_value": 37284.839421852135,
        "drawdown_pct": -17.807051103274837,
        "normalized_value": 373.2500396565903
      },
      {
        "timestamp": "2023-12-10T15:00:00+00:00",
        "portfolio_value": 39326.77415303759,
        "drawdown_pct": -13.30568702035994,
        "normalized_value": 393.6913834094201
      },
      {
        "timestamp": "2023-12-14T10:00:00+00:00",
        "portfolio_value": 37299.75265703591,
        "drawdown_pct": -17.7741754681279,
        "normalized_value": 373.3993326590563
      },
      {
        "timestamp": "2023-12-18T05:00:00+00:00",
        "portfolio_value": 35317.643818949175,
        "drawdown_pct": -22.143655743832884,
        "normalized_value": 353.55689230283065
      },
      {
        "timestamp": "2023-12-22T00:00:00+00:00",
        "portfolio_value": 32910.12318199709,
        "drawdown_pct": -27.450939448126867,
        "normalized_value": 329.45575127204046
      },
      {
        "timestamp": "2023-12-25T19:00:00+00:00",
        "portfolio_value": 32221.42362945574,
        "drawdown_pct": -28.969150281403316,
        "normalized_value": 322.5613368321897
      },
      {
        "timestamp": "2023-12-29T14:00:00+00:00",
        "portfolio_value": 32655.03029427548,
        "drawdown_pct": -28.0132815960223,
        "normalized_value": 326.9020744442839
      },
      {
        "timestamp": "2024-01-02T09:00:00+00:00",
        "portfolio_value": 32576.90027751748,
        "drawdown_pct": -28.185516117459624,
        "normalized_value": 326.1199326326123
      },
      {
        "timestamp": "2024-01-06T04:00:00+00:00",
        "portfolio_value": 30148.368821340577,
        "drawdown_pct": -33.53911733894297,
        "normalized_value": 301.80845707362016
      },
      {
        "timestamp": "2024-01-09T23:00:00+00:00",
        "portfolio_value": 29038.841094667077,
        "drawdown_pct": -35.9850271819803,
        "normalized_value": 290.7012275829588
      },
      {
        "timestamp": "2024-01-13T18:00:00+00:00",
        "portfolio_value": 31868.555921263905,
        "drawdown_pct": -29.747033140936768,
        "normalized_value": 319.0288585348863
      },
      {
        "timestamp": "2024-01-17T13:00:00+00:00",
        "portfolio_value": 30548.24958838128,
        "drawdown_pct": -32.65759605683986,
        "normalized_value": 305.81157240066034
      },
      {
        "timestamp": "2024-01-21T08:00:00+00:00",
        "portfolio_value": 30335.346385180877,
        "drawdown_pct": -33.12693271946053,
        "normalized_value": 303.680246900275
      },
      {
        "timestamp": "2024-01-25T03:00:00+00:00",
        "portfolio_value": 30335.346385180877,
        "drawdown_pct": -33.12693271946053,
        "normalized_value": 303.680246900275
      },
      {
        "timestamp": "2024-01-28T22:00:00+00:00",
        "portfolio_value": 30335.346385180877,
        "drawdown_pct": -33.12693271946053,
        "normalized_value": 303.680246900275
      },
      {
        "timestamp": "2024-02-01T17:00:00+00:00",
        "portfolio_value": 29153.35711424236,
        "drawdown_pct": -35.732581161271646,
        "normalized_value": 291.8476213856564
      },
      {
        "timestamp": "2024-02-05T12:00:00+00:00",
        "portfolio_value": 29153.35711424236,
        "drawdown_pct": -35.732581161271646,
        "normalized_value": 291.8476213856564
      },
      {
        "timestamp": "2024-02-09T07:00:00+00:00",
        "portfolio_value": 30513.021927884667,
        "drawdown_pct": -32.73525403643283,
        "normalized_value": 305.45891631091297
      },
      {
        "timestamp": "2024-02-13T02:00:00+00:00",
        "portfolio_value": 33195.21459274144,
        "drawdown_pct": -26.822466746686935,
        "normalized_value": 332.30973648469273
      },
      {
        "timestamp": "2024-02-16T21:00:00+00:00",
        "portfolio_value": 35005.89407544497,
        "drawdown_pct": -22.830895681934653,
        "normalized_value": 350.4360365896187
      },
      {
        "timestamp": "2024-02-20T16:00:00+00:00",
        "portfolio_value": 36424.3717712913,
        "drawdown_pct": -19.703917892203034,
        "normalized_value": 364.63609389002187
      },
      {
        "timestamp": "2024-02-24T11:00:00+00:00",
        "portfolio_value": 37281.36124364443,
        "drawdown_pct": -17.81471860910596,
        "normalized_value": 373.215220406351
      },
      {
        "timestamp": "2024-02-28T06:00:00+00:00",
        "portfolio_value": 41246.8696805016,
        "drawdown_pct": -9.072912626991238,
        "normalized_value": 412.9130226301686
      },
      {
        "timestamp": "2024-03-03T01:00:00+00:00",
        "portfolio_value": 43537.25461756322,
        "drawdown_pct": -4.023849924701517,
        "normalized_value": 435.84154483498924
      },
      {
        "timestamp": "2024-03-06T20:00:00+00:00",
        "portfolio_value": 44123.9420883761,
        "drawdown_pct": -9.0498425863914,
        "normalized_value": 441.7147395474381
      },
      {
        "timestamp": "2024-03-10T15:00:00+00:00",
        "portfolio_value": 44398.778238816994,
        "drawdown_pct": -8.483338553377205,
        "normalized_value": 444.4660616837756
      },
      {
        "timestamp": "2024-03-14T10:00:00+00:00",
        "portfolio_value": 45378.681610741165,
        "drawdown_pct": -6.463519794932517,
        "normalized_value": 454.2756512676847
      },
      {
        "timestamp": "2024-03-18T05:00:00+00:00",
        "portfolio_value": 43878.18543369111,
        "drawdown_pct": -9.556406718496211,
        "normalized_value": 439.2545256232425
      },
      {
        "timestamp": "2024-03-22T00:00:00+00:00",
        "portfolio_value": 43878.18543369111,
        "drawdown_pct": -9.556406718496211,
        "normalized_value": 439.2545256232425
      },
      {
        "timestamp": "2024-03-25T19:00:00+00:00",
        "portfolio_value": 43878.18543369111,
        "drawdown_pct": -9.556406718496211,
        "normalized_value": 439.2545256232425
      },
      {
        "timestamp": "2024-03-29T14:00:00+00:00",
        "portfolio_value": 43878.18543369111,
        "drawdown_pct": -9.556406718496211,
        "normalized_value": 439.2545256232425
      },
      {
        "timestamp": "2024-04-02T09:00:00+00:00",
        "portfolio_value": 42786.60683865736,
        "drawdown_pct": -11.80641522518853,
        "normalized_value": 428.32698080335405
      },
      {
        "timestamp": "2024-04-06T04:00:00+00:00",
        "portfolio_value": 42786.60683865736,
        "drawdown_pct": -11.80641522518853,
        "normalized_value": 428.32698080335405
      },
      {
        "timestamp": "2024-04-09T23:00:00+00:00",
        "portfolio_value": 41116.19624946589,
        "drawdown_pct": -15.249537005378377,
        "normalized_value": 411.60488065953274
      },
      {
        "timestamp": "2024-04-13T18:00:00+00:00",
        "portfolio_value": 41116.19624946589,
        "drawdown_pct": -15.249537005378377,
        "normalized_value": 411.60488065953274
      },
      {
        "timestamp": "2024-04-17T13:00:00+00:00",
        "portfolio_value": 41116.19624946589,
        "drawdown_pct": -15.249537005378377,
        "normalized_value": 411.60488065953274
      },
      {
        "timestamp": "2024-04-21T08:00:00+00:00",
        "portfolio_value": 41116.19624946589,
        "drawdown_pct": -15.249537005378377,
        "normalized_value": 411.60488065953274
      },
      {
        "timestamp": "2024-04-25T03:00:00+00:00",
        "portfolio_value": 41116.19624946589,
        "drawdown_pct": -15.249537005378377,
        "normalized_value": 411.60488065953274
      },
      {
        "timestamp": "2024-04-28T22:00:00+00:00",
        "portfolio_value": 40817.32117748574,
        "drawdown_pct": -15.865591092049813,
        "normalized_value": 408.61291035206307
      },
      {
        "timestamp": "2024-05-02T17:00:00+00:00",
        "portfolio_value": 40676.25438396193,
        "drawdown_pct": -16.156364002845635,
        "normalized_value": 407.20072279558053
      },
      {
        "timestamp": "2024-05-06T12:00:00+00:00",
        "portfolio_value": 40676.25438396193,
        "drawdown_pct": -16.156364002845635,
        "normalized_value": 407.20072279558053
      },
      {
        "timestamp": "2024-05-10T07:00:00+00:00",
        "portfolio_value": 40676.25438396193,
        "drawdown_pct": -16.156364002845635,
        "normalized_value": 407.20072279558053
      },
      {
        "timestamp": "2024-05-14T02:00:00+00:00",
        "portfolio_value": 40676.25438396193,
        "drawdown_pct": -16.156364002845635,
        "normalized_value": 407.20072279558053
      },
      {
        "timestamp": "2024-05-17T21:00:00+00:00",
        "portfolio_value": 40510.09121140991,
        "drawdown_pct": -16.49886664392158,
        "normalized_value": 405.53730110177116
      },
      {
        "timestamp": "2024-05-21T16:00:00+00:00",
        "portfolio_value": 49708.78989952147,
        "drawdown_pct": -0.5629934117074054,
        "normalized_value": 497.62337960890795
      },
      {
        "timestamp": "2024-05-25T11:00:00+00:00",
        "portfolio_value": 49169.042951152696,
        "drawdown_pct": -4.983773852392081,
        "normalized_value": 492.2200957807605
      },
      {
        "timestamp": "2024-05-29T06:00:00+00:00",
        "portfolio_value": 50763.40043270819,
        "drawdown_pct": -2.923618799662493,
        "normalized_value": 508.18084557732783
      },
      {
        "timestamp": "2024-06-02T01:00:00+00:00",
        "portfolio_value": 48127.3572726478,
        "drawdown_pct": -7.964603613234661,
        "normalized_value": 481.7920175902513
      },
      {
        "timestamp": "2024-06-05T20:00:00+00:00",
        "portfolio_value": 47772.39532648501,
        "drawdown_pct": -8.64340804523859,
        "normalized_value": 478.2385743533685
      },
      {
        "timestamp": "2024-06-09T15:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-06-13T10:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-06-17T05:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-06-21T00:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-06-24T19:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-06-28T14:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-07-02T09:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-07-06T04:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-07-09T23:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-07-13T18:00:00+00:00",
        "portfolio_value": 46939.93566138209,
        "drawdown_pct": -10.23534576206985,
        "normalized_value": 469.90501015327516
      },
      {
        "timestamp": "2024-07-17T13:00:00+00:00",
        "portfolio_value": 49600.4677068618,
        "drawdown_pct": -5.1475301145529,
        "normalized_value": 496.5389907974541
      },
      {
        "timestamp": "2024-07-21T08:00:00+00:00",
        "portfolio_value": 49871.11865513403,
        "drawdown_pct": -4.629956145852034,
        "normalized_value": 499.24841582763133
      },
      {
        "timestamp": "2024-07-25T03:00:00+00:00",
        "portfolio_value": 48067.77862250454,
        "drawdown_pct": -8.078537662242576,
        "normalized_value": 481.19558928659006
      },
      {
        "timestamp": "2024-07-28T22:00:00+00:00",
        "portfolio_value": 48067.77862250454,
        "drawdown_pct": -8.078537662242576,
        "normalized_value": 481.19558928659006
      },
      {
        "timestamp": "2024-08-01T17:00:00+00:00",
        "portfolio_value": 46928.38852031321,
        "drawdown_pct": -10.257427708090448,
        "normalized_value": 469.7894143527108
      },
      {
        "timestamp": "2024-08-05T12:00:00+00:00",
        "portfolio_value": 46928.38852031321,
        "drawdown_pct": -10.257427708090448,
        "normalized_value": 469.7894143527108
      },
      {
        "timestamp": "2024-08-09T07:00:00+00:00",
        "portfolio_value": 46928.38852031321,
        "drawdown_pct": -10.257427708090448,
        "normalized_value": 469.7894143527108
      },
      {
        "timestamp": "2024-08-13T02:00:00+00:00",
        "portfolio_value": 46928.38852031321,
        "drawdown_pct": -10.257427708090448,
        "normalized_value": 469.7894143527108
      },
      {
        "timestamp": "2024-08-16T21:00:00+00:00",
        "portfolio_value": 46928.38852031321,
        "drawdown_pct": -10.257427708090448,
        "normalized_value": 469.7894143527108
      },
      {
        "timestamp": "2024-08-20T16:00:00+00:00",
        "portfolio_value": 46928.38852031321,
        "drawdown_pct": -10.257427708090448,
        "normalized_value": 469.7894143527108
      },
      {
        "timestamp": "2024-08-24T11:00:00+00:00",
        "portfolio_value": 46853.92209949566,
        "drawdown_pct": -10.399832089835744,
        "normalized_value": 469.04394796599115
      },
      {
        "timestamp": "2024-08-28T06:00:00+00:00",
        "portfolio_value": 46603.79619199416,
        "drawdown_pct": -10.878155404226073,
        "normalized_value": 466.53999444649816
      },
      {
        "timestamp": "2024-09-01T01:00:00+00:00",
        "portfolio_value": 46603.79619199416,
        "drawdown_pct": -10.878155404226073,
        "normalized_value": 466.53999444649816
      },
      {
        "timestamp": "2024-09-04T20:00:00+00:00",
        "portfolio_value": 46603.79619199416,
        "drawdown_pct": -10.878155404226073,
        "normalized_value": 466.53999444649816
      },
      {
        "timestamp": "2024-09-08T15:00:00+00:00",
        "portfolio_value": 46603.79619199416,
        "drawdown_pct": -10.878155404226073,
        "normalized_value": 466.53999444649816
      },
      {
        "timestamp": "2024-09-12T10:00:00+00:00",
        "portfolio_value": 46603.79619199416,
        "drawdown_pct": -10.878155404226073,
        "normalized_value": 466.53999444649816
      },
      {
        "timestamp": "2024-09-16T05:00:00+00:00",
        "portfolio_value": 46603.79619199416,
        "drawdown_pct": -10.878155404226073,
        "normalized_value": 466.53999444649816
      },
      {
        "timestamp": "2024-09-20T00:00:00+00:00",
        "portfolio_value": 46866.053309158735,
        "drawdown_pct": -10.376633211832397,
        "normalized_value": 469.16539074429016
      },
      {
        "timestamp": "2024-09-23T19:00:00+00:00",
        "portfolio_value": 51182.36786626416,
        "drawdown_pct": -2.122414744305312,
        "normalized_value": 512.375033177823
      },
      {
        "timestamp": "2024-09-27T14:00:00+00:00",
        "portfolio_value": 51336.34426691313,
        "drawdown_pct": -1.8279610308455576,
        "normalized_value": 513.9164558723974
      },
      {
        "timestamp": "2024-10-01T09:00:00+00:00",
        "portfolio_value": 50374.61699369394,
        "drawdown_pct": -3.667101092188503,
        "normalized_value": 504.2888230748828
      },
      {
        "timestamp": "2024-10-05T04:00:00+00:00",
        "portfolio_value": 47867.97846056659,
        "drawdown_pct": -8.46062153603446,
        "normalized_value": 479.1954353494103
      },
      {
        "timestamp": "2024-10-08T23:00:00+00:00",
        "portfolio_value": 47544.34696716312,
        "drawdown_pct": -9.079511798591168,
        "normalized_value": 475.9556341428058
      },
      {
        "timestamp": "2024-10-12T18:00:00+00:00",
        "portfolio_value": 47544.34696716312,
        "drawdown_pct": -9.079511798591168,
        "normalized_value": 475.9556341428058
      },
      {
        "timestamp": "2024-10-16T13:00:00+00:00",
        "portfolio_value": 48638.12202428604,
        "drawdown_pct": -6.987852778753882,
        "normalized_value": 486.90516724465016
      },
      {
        "timestamp": "2024-10-20T08:00:00+00:00",
        "portfolio_value": 49394.9581121442,
        "drawdown_pct": -5.540532308792781,
        "normalized_value": 494.48168102845466
      },
      {
        "timestamp": "2024-10-24T03:00:00+00:00",
        "portfolio_value": 49215.0983300739,
        "drawdown_pct": -5.884483592947025,
        "normalized_value": 492.6811456947549
      },
      {
        "timestamp": "2024-10-27T22:00:00+00:00",
        "portfolio_value": 49215.0983300739,
        "drawdown_pct": -5.884483592947025,
        "normalized_value": 492.6811456947549
      },
      {
        "timestamp": "2024-10-31T17:00:00+00:00",
        "portfolio_value": 48225.31977233867,
        "drawdown_pct": -7.777266971436308,
        "normalized_value": 482.7726978737513
      },
      {
        "timestamp": "2024-11-04T12:00:00+00:00",
        "portfolio_value": 46851.64650305538,
        "drawdown_pct": -10.40418378151936,
        "normalized_value": 469.021167488061
      },
      {
        "timestamp": "2024-11-08T07:00:00+00:00",
        "portfolio_value": 52306.58954725389,
        "drawdown_pct": -0.4495771445516445,
        "normalized_value": 523.6293605001041
      },
      {
        "timestamp": "2024-11-12T02:00:00+00:00",
        "portfolio_value": 59912.61413090732,
        "drawdown_pct": -2.069048546355936,
        "normalized_value": 599.771541115579
      },
      {
        "timestamp": "2024-11-15T21:00:00+00:00",
        "portfolio_value": 55291.882837680416,
        "drawdown_pct": -10.999280143786407,
        "normalized_value": 553.5144520364029
      },
      {
        "timestamp": "2024-11-19T16:00:00+00:00",
        "portfolio_value": 52827.44124260616,
        "drawdown_pct": -14.966174826123572,
        "normalized_value": 528.843488251758
      },
      {
        "timestamp": "2024-11-23T11:00:00+00:00",
        "portfolio_value": 52803.13021173418,
        "drawdown_pct": -15.005307138806884,
        "normalized_value": 528.6001160560407
      },
      {
        "timestamp": "2024-11-27T06:00:00+00:00",
        "portfolio_value": 53826.92359080799,
        "drawdown_pct": -13.357355521946776,
        "normalized_value": 538.8490785100803
      },
      {
        "timestamp": "2024-12-01T01:00:00+00:00",
        "portfolio_value": 58595.92201260956,
        "drawdown_pct": -5.680925081342664,
        "normalized_value": 586.5904360607956
      },
      {
        "timestamp": "2024-12-04T20:00:00+00:00",
        "portfolio_value": 61453.18164202623,
        "drawdown_pct": -1.0817298508108815,
        "normalized_value": 615.1938117632508
      },
      {
        "timestamp": "2024-12-08T15:00:00+00:00",
        "portfolio_value": 63337.20018747478,
        "drawdown_pct": -2.0095318728737275,
        "normalized_value": 634.0542925298724
      },
      {
        "timestamp": "2024-12-12T10:00:00+00:00",
        "portfolio_value": 61134.57799310099,
        "drawdown_pct": -5.417260337267839,
        "normalized_value": 612.0043431631424
      },
      {
        "timestamp": "2024-12-16T05:00:00+00:00",
        "portfolio_value": 59673.52402201536,
        "drawdown_pct": -7.677691208255255,
        "normalized_value": 597.3780644636956
      },
      {
        "timestamp": "2024-12-20T00:00:00+00:00",
        "portfolio_value": 59270.16083894585,
        "drawdown_pct": -8.301743850576587,
        "normalized_value": 593.3400874625538
      },
      {
        "timestamp": "2024-12-23T19:00:00+00:00",
        "portfolio_value": 59270.16083894585,
        "drawdown_pct": -8.301743850576587,
        "normalized_value": 593.3400874625538
      },
      {
        "timestamp": "2024-12-27T14:00:00+00:00",
        "portfolio_value": 59270.16083894585,
        "drawdown_pct": -8.301743850576587,
        "normalized_value": 593.3400874625538
      },
      {
        "timestamp": "2024-12-31T09:00:00+00:00",
        "portfolio_value": 59270.16083894585,
        "drawdown_pct": -8.301743850576587,
        "normalized_value": 593.3400874625538
      },
      {
        "timestamp": "2025-01-04T04:00:00+00:00",
        "portfolio_value": 59329.66663979951,
        "drawdown_pct": -8.209681030232934,
        "normalized_value": 593.9357864885609
      },
      {
        "timestamp": "2025-01-07T23:00:00+00:00",
        "portfolio_value": 57144.24445052909,
        "drawdown_pct": -11.590798963266646,
        "normalized_value": 572.0580224573847
      },
      {
        "timestamp": "2025-01-11T18:00:00+00:00",
        "portfolio_value": 57144.24445052909,
        "drawdown_pct": -11.590798963266646,
        "normalized_value": 572.0580224573847
      },
      {
        "timestamp": "2025-01-15T13:00:00+00:00",
        "portfolio_value": 57144.24445052909,
        "drawdown_pct": -11.590798963266646,
        "normalized_value": 572.0580224573847
      },
      {
        "timestamp": "2025-01-19T08:00:00+00:00",
        "portfolio_value": 53758.6696299675,
        "drawdown_pct": -16.82870118446157,
        "normalized_value": 538.1658036459407
      },
      {
        "timestamp": "2025-01-23T03:00:00+00:00",
        "portfolio_value": 53758.6696299675,
        "drawdown_pct": -16.82870118446157,
        "normalized_value": 538.1658036459407
      },
      {
        "timestamp": "2025-01-26T22:00:00+00:00",
        "portfolio_value": 51299.42597310169,
        "drawdown_pct": -20.6334547331131,
        "normalized_value": 513.5468752374035
      },
      {
        "timestamp": "2025-01-30T17:00:00+00:00",
        "portfolio_value": 51299.42597310169,
        "drawdown_pct": -20.6334547331131,
        "normalized_value": 513.5468752374035
      },
      {
        "timestamp": "2025-02-03T12:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-02-07T07:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-02-11T02:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-02-14T21:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-02-18T16:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-02-22T11:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-02-26T06:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-03-02T01:00:00+00:00",
        "portfolio_value": 49564.57915472805,
        "drawdown_pct": -23.317476940569406,
        "normalized_value": 496.17971867197764
      },
      {
        "timestamp": "2025-03-05T20:00:00+00:00",
        "portfolio_value": 45892.623136949005,
        "drawdown_pct": -28.99844622969726,
        "normalized_value": 459.4206028891155
      },
      {
        "timestamp": "2025-03-09T15:00:00+00:00",
        "portfolio_value": 45892.623136949005,
        "drawdown_pct": -28.99844622969726,
        "normalized_value": 459.4206028891155
      },
      {
        "timestamp": "2025-03-13T10:00:00+00:00",
        "portfolio_value": 45892.623136949005,
        "drawdown_pct": -28.99844622969726,
        "normalized_value": 459.4206028891155
      },
      {
        "timestamp": "2025-03-17T05:00:00+00:00",
        "portfolio_value": 45892.623136949005,
        "drawdown_pct": -28.99844622969726,
        "normalized_value": 459.4206028891155
      },
      {
        "timestamp": "2025-03-21T00:00:00+00:00",
        "portfolio_value": 44068.878130987454,
        "drawdown_pct": -31.820004908478612,
        "normalized_value": 441.1635068051854
      },
      {
        "timestamp": "2025-03-24T19:00:00+00:00",
        "portfolio_value": 44829.18597707164,
        "drawdown_pct": -30.643714805064004,
        "normalized_value": 448.77477557025395
      },
      {
        "timestamp": "2025-03-28T14:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-01T09:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-05T04:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-08T23:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-12T18:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-16T13:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-20T08:00:00+00:00",
        "portfolio_value": 43270.39677126454,
        "drawdown_pct": -33.05535415028968,
        "normalized_value": 433.1700916851795
      },
      {
        "timestamp": "2025-04-24T03:00:00+00:00",
        "portfolio_value": 44930.725066420666,
        "drawdown_pct": -30.48662129810345,
        "normalized_value": 449.7912602786206
      },
      {
        "timestamp": "2025-04-27T22:00:00+00:00",
        "portfolio_value": 43814.80222294141,
        "drawdown_pct": -32.21309170574092,
        "normalized_value": 438.6200107294483
      },
      {
        "timestamp": "2025-05-01T17:00:00+00:00",
        "portfolio_value": 45229.60025613436,
        "drawdown_pct": -30.024224481304774,
        "normalized_value": 452.783231764694
      },
      {
        "timestamp": "2025-05-05T12:00:00+00:00",
        "portfolio_value": 43888.46140227734,
        "drawdown_pct": -32.09913186154789,
        "normalized_value": 439.35739600546196
      },
      {
        "timestamp": "2025-05-09T07:00:00+00:00",
        "portfolio_value": 53640.2957100188,
        "drawdown_pct": -17.01184025274186,
        "normalized_value": 536.9807892808456
      },
      {
        "timestamp": "2025-05-13T02:00:00+00:00",
        "portfolio_value": 54956.03222156409,
        "drawdown_pct": -14.976233432158532,
        "normalized_value": 550.1523279739713
      },
      {
        "timestamp": "2025-05-16T21:00:00+00:00",
        "portfolio_value": 58047.44276979783,
        "drawdown_pct": -10.193439656969433,
        "normalized_value": 581.0997352208636
      },
      {
        "timestamp": "2025-05-20T16:00:00+00:00",
        "portfolio_value": 53105.38159783363,
        "drawdown_pct": -17.839418423322197,
        "normalized_value": 531.6258858755498
      },
      {
        "timestamp": "2025-05-24T11:00:00+00:00",
        "portfolio_value": 49549.2358535935,
        "drawdown_pct": -23.34121492166646,
        "normalized_value": 496.0261203771817
      },
      {
        "timestamp": "2025-05-28T06:00:00+00:00",
        "portfolio_value": 50874.46380196213,
        "drawdown_pct": -21.29092367653066,
        "normalized_value": 509.29267568364526
      },
      {
        "timestamp": "2025-06-01T01:00:00+00:00",
        "portfolio_value": 48631.97476367752,
        "drawdown_pct": -24.76033107030606,
        "normalized_value": 486.843628418106
      },
      {
        "timestamp": "2025-06-04T20:00:00+00:00",
        "portfolio_value": 48282.41961195812,
        "drawdown_pct": -25.30113604513674,
        "normalized_value": 483.3443113695534
      },
      {
        "timestamp": "2025-06-08T15:00:00+00:00",
        "portfolio_value": 47933.514938567394,
        "drawdown_pct": -25.84093464968668,
        "normalized_value": 479.8515061114696
      },
      {
        "timestamp": "2025-06-12T10:00:00+00:00",
        "portfolio_value": 50740.76776797327,
        "drawdown_pct": -21.49776794685726,
        "normalized_value": 507.9542751229331
      },
      {
        "timestamp": "2025-06-16T05:00:00+00:00",
        "portfolio_value": 46111.50517613848,
        "drawdown_pct": -28.659808692494625,
        "normalized_value": 461.61178115552116
      },
      {
        "timestamp": "2025-06-20T00:00:00+00:00",
        "portfolio_value": 44512.58014258368,
        "drawdown_pct": -31.13354311830526,
        "normalized_value": 445.60530663568
      },
      {
        "timestamp": "2025-06-23T19:00:00+00:00",
        "portfolio_value": 44512.58014258368,
        "drawdown_pct": -31.13354311830526,
        "normalized_value": 445.60530663568
      },
      {
        "timestamp": "2025-06-27T14:00:00+00:00",
        "portfolio_value": 43324.08503898763,
        "drawdown_pct": -32.97229177191571,
        "normalized_value": 433.7075527113625
      },
      {
        "timestamp": "2025-07-01T09:00:00+00:00",
        "portfolio_value": 42657.46741452072,
        "drawdown_pct": -34.00363153575047,
        "normalized_value": 427.03419542657207
      },
      {
        "timestamp": "2025-07-05T04:00:00+00:00",
        "portfolio_value": 41594.62891811717,
        "drawdown_pct": -35.64797390481387,
        "normalized_value": 416.3943611914628
      },
      {
        "timestamp": "2025-07-08T23:00:00+00:00",
        "portfolio_value": 41475.052113965634,
        "drawdown_pct": -35.832973983460725,
        "normalized_value": 415.19730502644654
      },
      {
        "timestamp": "2025-07-12T18:00:00+00:00",
        "portfolio_value": 46805.50594481356,
        "drawdown_pct": -27.586103823921913,
        "normalized_value": 468.5592648632787
      },
      {
        "timestamp": "2025-07-16T13:00:00+00:00",
        "portfolio_value": 51083.714275031,
        "drawdown_pct": -20.967187361203344,
        "normalized_value": 511.38743453421813
      },
      {
        "timestamp": "2025-07-20T08:00:00+00:00",
        "portfolio_value": 59897.713412935496,
        "drawdown_pct": -7.330842542694188,
        "normalized_value": 599.6223734200722
      },
      {
        "timestamp": "2025-07-24T03:00:00+00:00",
        "portfolio_value": 58692.489742220765,
        "drawdown_pct": -9.19547235490027,
        "normalized_value": 587.5571536185439
      },
      {
        "timestamp": "2025-07-27T22:00:00+00:00",
        "portfolio_value": 62501.14805315867,
        "drawdown_pct": -3.3030077413653007,
        "normalized_value": 625.6847649383021
      },
      {
        "timestamp": "2025-07-31T17:00:00+00:00",
        "portfolio_value": 60988.51428429856,
        "drawdown_pct": -5.64323892741027,
        "normalized_value": 610.5421326253402
      },
      {
        "timestamp": "2025-08-04T12:00:00+00:00",
        "portfolio_value": 59322.825518959726,
        "drawdown_pct": -8.2202650887581,
        "normalized_value": 593.867301585197
      },
      {
        "timestamp": "2025-08-08T07:00:00+00:00",
        "portfolio_value": 61983.884635714014,
        "drawdown_pct": -4.10327810807568,
        "normalized_value": 620.5065586199194
      },
      {
        "timestamp": "2025-08-12T02:00:00+00:00",
        "portfolio_value": 68361.45806299316,
        "drawdown_pct": -0.9371458454233298,
        "normalized_value": 684.3509943626041
      },
      {
        "timestamp": "2025-08-15T21:00:00+00:00",
        "portfolio_value": 70387.93522308447,
        "drawdown_pct": -8.060511372243175,
        "normalized_value": 704.6375958900858
      },
      {
        "timestamp": "2025-08-19T16:00:00+00:00",
        "portfolio_value": 67887.71603021472,
        "drawdown_pct": -11.326253907824174,
        "normalized_value": 679.6084707185861
      },
      {
        "timestamp": "2025-08-23T11:00:00+00:00",
        "portfolio_value": 67182.84338004433,
        "drawdown_pct": -12.246946222479599,
        "normalized_value": 672.5521510801341
      },
      {
        "timestamp": "2025-08-27T06:00:00+00:00",
        "portfolio_value": 65583.65150505563,
        "drawdown_pct": -14.335782650741594,
        "normalized_value": 656.543005271444
      },
      {
        "timestamp": "2025-08-31T01:00:00+00:00",
        "portfolio_value": 64511.780858540296,
        "drawdown_pct": -15.735841322776679,
        "normalized_value": 645.8127522377108
      },
      {
        "timestamp": "2025-09-03T20:00:00+00:00",
        "portfolio_value": 64511.780858540296,
        "drawdown_pct": -15.735841322776679,
        "normalized_value": 645.8127522377108
      },
      {
        "timestamp": "2025-09-07T15:00:00+00:00",
        "portfolio_value": 64511.780858540296,
        "drawdown_pct": -15.735841322776679,
        "normalized_value": 645.8127522377108
      },
      {
        "timestamp": "2025-09-11T10:00:00+00:00",
        "portfolio_value": 63563.81698869988,
        "drawdown_pct": -16.97405513249408,
        "normalized_value": 636.3229017382185
      },
      {
        "timestamp": "2025-09-15T05:00:00+00:00",
        "portfolio_value": 66552.73944584846,
        "drawdown_pct": -13.06997694938898,
        "normalized_value": 666.2443240364016
      },
      {
        "timestamp": "2025-09-19T00:00:00+00:00",
        "portfolio_value": 62760.4859547509,
        "drawdown_pct": -18.023352064219598,
        "normalized_value": 628.280937633551
      },
      {
        "timestamp": "2025-09-22T19:00:00+00:00",
        "portfolio_value": 61735.72717334062,
        "drawdown_pct": -19.36187404289478,
        "normalized_value": 618.0223107564925
      },
      {
        "timestamp": "2025-09-26T14:00:00+00:00",
        "portfolio_value": 61735.72717334062,
        "drawdown_pct": -19.36187404289478,
        "normalized_value": 618.0223107564925
      },
      {
        "timestamp": "2025-09-30T09:00:00+00:00",
        "portfolio_value": 61735.72717334062,
        "drawdown_pct": -19.36187404289478,
        "normalized_value": 618.0223107564925
      },
      {
        "timestamp": "2025-10-04T04:00:00+00:00",
        "portfolio_value": 64084.20763300499,
        "drawdown_pct": -16.29432998396622,
        "normalized_value": 641.5324140127964
      },
      {
        "timestamp": "2025-10-07T23:00:00+00:00",
        "portfolio_value": 64134.14715058632,
        "drawdown_pct": -16.229099860448706,
        "normalized_value": 642.0323471547033
      },
      {
        "timestamp": "2025-10-11T18:00:00+00:00",
        "portfolio_value": 64134.14715058632,
        "drawdown_pct": -16.229099860448706,
        "normalized_value": 642.0323471547033
      },
      {
        "timestamp": "2025-10-15T13:00:00+00:00",
        "portfolio_value": 64134.14715058632,
        "drawdown_pct": -16.229099860448706,
        "normalized_value": 642.0323471547033
      },
      {
        "timestamp": "2025-10-19T08:00:00+00:00",
        "portfolio_value": 64134.14715058632,
        "drawdown_pct": -16.229099860448706,
        "normalized_value": 642.0323471547033
      },
      {
        "timestamp": "2025-10-23T03:00:00+00:00",
        "portfolio_value": 64134.14715058632,
        "drawdown_pct": -16.229099860448706,
        "normalized_value": 642.0323471547033
      },
      {
        "timestamp": "2025-10-26T22:00:00+00:00",
        "portfolio_value": 64134.14715058632,
        "drawdown_pct": -16.229099860448706,
        "normalized_value": 642.0323471547033
      },
      {
        "timestamp": "2025-10-30T17:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-03T12:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-07T07:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-11T02:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-14T21:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-18T16:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-22T11:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-26T06:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-11-30T01:00:00+00:00",
        "portfolio_value": 62892.12962961934,
        "drawdown_pct": -17.851401400940105,
        "normalized_value": 629.5987924943229
      },
      {
        "timestamp": "2025-12-03T20:00:00+00:00",
        "portfolio_value": 65821.80508506458,
        "drawdown_pct": -14.024710614156108,
        "normalized_value": 658.9271065458779
      },
      {
        "timestamp": "2025-12-07T15:00:00+00:00",
        "portfolio_value": 65027.86629190993,
        "drawdown_pct": -15.061739565396573,
        "normalized_value": 650.9791660256835
      },
      {
        "timestamp": "2025-12-11T10:00:00+00:00",
        "portfolio_value": 66196.34409084787,
        "drawdown_pct": -13.535494322277005,
        "normalized_value": 662.6765312699553
      },
      {
        "timestamp": "2025-12-15T05:00:00+00:00",
        "portfolio_value": 62710.98205851846,
        "drawdown_pct": -18.088013186758097,
        "normalized_value": 627.7853653977999
      },
      {
        "timestamp": "2025-12-19T00:00:00+00:00",
        "portfolio_value": 62710.98205851846,
        "drawdown_pct": -18.088013186758097,
        "normalized_value": 627.7853653977999
      },
      {
        "timestamp": "2025-12-22T19:00:00+00:00",
        "portfolio_value": 61852.60702903963,
        "drawdown_pct": -19.209207621727643,
        "normalized_value": 619.1923683845038
      },
      {
        "timestamp": "2025-12-26T14:00:00+00:00",
        "portfolio_value": 61705.13088462588,
        "drawdown_pct": -19.401838379530385,
        "normalized_value": 617.7160182753338
      },
      {
        "timestamp": "2025-12-30T09:00:00+00:00",
        "portfolio_value": 60088.25101247317,
        "drawdown_pct": -21.513778560004386,
        "normalized_value": 601.5298019536608
      },
      {
        "timestamp": "2026-01-03T04:00:00+00:00",
        "portfolio_value": 59701.16996730314,
        "drawdown_pct": -22.01937704414288,
        "normalized_value": 597.6548217284449
      },
      {
        "timestamp": "2026-01-06T23:00:00+00:00",
        "portfolio_value": 63414.20034526061,
        "drawdown_pct": -17.169481772647845,
        "normalized_value": 634.8251235805832
      },
      {
        "timestamp": "2026-01-10T18:00:00+00:00",
        "portfolio_value": 59790.44594731375,
        "drawdown_pct": -21.902766322107876,
        "normalized_value": 598.5485432408891
      },
      {
        "timestamp": "2026-01-14T13:00:00+00:00",
        "portfolio_value": 62238.82623464695,
        "drawdown_pct": -18.704734857336767,
        "normalized_value": 623.0587209300529
      },
      {
        "timestamp": "2026-01-18T08:00:00+00:00",
        "portfolio_value": 62482.64670788195,
        "drawdown_pct": -18.386260824036075,
        "normalized_value": 625.4995521825194
      },
      {
        "timestamp": "2026-01-22T03:00:00+00:00",
        "portfolio_value": 60438.38970640613,
        "drawdown_pct": -21.056433528260516,
        "normalized_value": 605.0349607104715
      },
      {
        "timestamp": "2026-01-25T22:00:00+00:00",
        "portfolio_value": 60438.38970640613,
        "drawdown_pct": -21.056433528260516,
        "normalized_value": 605.0349607104715
      },
      {
        "timestamp": "2026-01-29T17:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-02T12:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-06T07:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-10T02:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-13T21:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-17T16:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-21T11:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-02-25T06:00:00+00:00",
        "portfolio_value": 58886.74972150565,
        "drawdown_pct": -23.083158510233627,
        "normalized_value": 589.5018460483919
      },
      {
        "timestamp": "2026-03-01T01:00:00+00:00",
        "portfolio_value": 57106.19131880771,
        "drawdown_pct": -25.408892721601937,
        "normalized_value": 571.6770812184171
      },
      {
        "timestamp": "2026-03-04T20:00:00+00:00",
        "portfolio_value": 53011.23919629215,
        "drawdown_pct": -30.757647489105832,
        "normalized_value": 530.6834477249879
      },
      {
        "timestamp": "2026-03-08T15:00:00+00:00",
        "portfolio_value": 48314.58529737876,
        "drawdown_pct": -36.89233458227178,
        "normalized_value": 483.6663147238658
      },
      {
        "timestamp": "2026-03-12T10:00:00+00:00",
        "portfolio_value": 50400.94704448806,
        "drawdown_pct": -34.167165396475966,
        "normalized_value": 504.55240721941595
      },
      {
        "timestamp": "2026-03-16T05:00:00+00:00",
        "portfolio_value": 55903.03739565599,
        "drawdown_pct": -26.980431311055693,
        "normalized_value": 559.6325811885697
      },
      {
        "timestamp": "2026-03-20T00:00:00+00:00",
        "portfolio_value": 53243.9675500532,
        "drawdown_pct": -30.45366178805981,
        "normalized_value": 533.0132382944926
      },
      {
        "timestamp": "2026-03-23T19:00:00+00:00",
        "portfolio_value": 53243.9675500532,
        "drawdown_pct": -30.45366178805981,
        "normalized_value": 533.0132382944926
      },
      {
        "timestamp": "2026-03-27T14:00:00+00:00",
        "portfolio_value": 50628.980157255166,
        "drawdown_pct": -33.869312537011375,
        "normalized_value": 506.83519480018947
      },
      {
        "timestamp": "2026-03-31T09:00:00+00:00",
        "portfolio_value": 50628.980157255166,
        "drawdown_pct": -33.869312537011375,
        "normalized_value": 506.83519480018947
      },
      {
        "timestamp": "2026-04-04T04:00:00+00:00",
        "portfolio_value": 48621.300864370765,
        "drawdown_pct": -36.49170807868074,
        "normalized_value": 486.73677444203116
      },
      {
        "timestamp": "2026-04-07T23:00:00+00:00",
        "portfolio_value": 49248.16020783561,
        "drawdown_pct": -35.67291537937752,
        "normalized_value": 493.0121206265778
      },
      {
        "timestamp": "2026-04-11T18:00:00+00:00",
        "portfolio_value": 50748.45615619179,
        "drawdown_pct": -33.71325507900086,
        "normalized_value": 508.03124182714726
      },
      {
        "timestamp": "2026-04-15T13:00:00+00:00",
        "portfolio_value": 50664.39454720986,
        "drawdown_pct": -33.82305488089206,
        "normalized_value": 507.1897201960332
      },
      {
        "timestamp": "2026-04-19T08:00:00+00:00",
        "portfolio_value": 50368.23940498835,
        "drawdown_pct": -34.2098875425411,
        "normalized_value": 504.22497848619247
      },
      {
        "timestamp": "2026-04-23T03:00:00+00:00",
        "portfolio_value": 48390.087461300514,
        "drawdown_pct": -36.79371497765837,
        "normalized_value": 484.4221496990176
      },
      {
        "timestamp": "2026-04-26T22:00:00+00:00",
        "portfolio_value": 49003.54577071022,
        "drawdown_pct": -35.99242647441102,
        "normalized_value": 490.56334118234844
      },
      {
        "timestamp": "2026-04-30T17:00:00+00:00",
        "portfolio_value": 48107.02128553176,
        "drawdown_pct": -37.16345106049006,
        "normalized_value": 481.5884386526665
      },
      {
        "timestamp": "2026-05-04T12:00:00+00:00",
        "portfolio_value": 46961.25551502644,
        "drawdown_pct": -38.66003024971593,
        "normalized_value": 470.11843835470006
      },
      {
        "timestamp": "2026-05-08T07:00:00+00:00",
        "portfolio_value": 47387.74992725992,
        "drawdown_pct": -38.10295071557323,
        "normalized_value": 474.38797682523904
      },
      {
        "timestamp": "2026-05-12T02:00:00+00:00",
        "portfolio_value": 46182.74573806517,
        "drawdown_pct": -39.67690609013771,
        "normalized_value": 462.3249542032427
      },
      {
        "timestamp": "2026-05-15T21:00:00+00:00",
        "portfolio_value": 45724.38215113376,
        "drawdown_pct": -40.275612582301406,
        "normalized_value": 457.73638067973667
      },
      {
        "timestamp": "2026-05-19T16:00:00+00:00",
        "portfolio_value": 45724.38215113376,
        "drawdown_pct": -40.275612582301406,
        "normalized_value": 457.73638067973667
      },
      {
        "timestamp": "2026-05-23T11:00:00+00:00",
        "portfolio_value": 45724.38215113376,
        "drawdown_pct": -40.275612582301406,
        "normalized_value": 457.73638067973667
      },
      {
        "timestamp": "2026-05-27T06:00:00+00:00",
        "portfolio_value": 45724.38215113376,
        "drawdown_pct": -40.275612582301406,
        "normalized_value": 457.73638067973667
      },
      {
        "timestamp": "2026-05-31T01:00:00+00:00",
        "portfolio_value": 45724.38215113376,
        "drawdown_pct": -40.275612582301406,
        "normalized_value": 457.73638067973667
      }
    ]
  },
  "trafficStates": [
    {
      "state": "Green",
      "meaning": "Trend confirmation is broad enough to allow normal long participation.",
      "behavior": "Use traffic-light ranking to select the strongest qualifying long asset. Apply wider 5% rebalance threshold under the current turnover-control checkpoint."
    },
    {
      "state": "Yellow",
      "meaning": "Signals are still investable but less clean.",
      "behavior": "Remain eligible for long exposure, keep cooldown active, and avoid over-trading small target changes."
    },
    {
      "state": "Orange",
      "meaning": "Market structure is deteriorating or drawdown risk is rising.",
      "behavior": "Drawdown and volatility governors dominate sizing. The current top checkpoint does not actively short in this state."
    },
    {
      "state": "Red",
      "meaning": "Risk state is hostile.",
      "behavior": "Exposure is cut by drawdown/volatility tiers. New long participation requires renewed traffic-light confirmation."
    },
    {
      "state": "Recovery",
      "meaning": "Drawdown is improving and enough green confirmation has returned.",
      "behavior": "Allow re-risking up to the recovery long target of 1.85x after at least 12% drawdown, provided worsening has stopped."
    }
  ],
  "governors": [
    "Multi-timeframe supertrend traffic-light ranking across SOL and ETH.",
    "Drawdown exposure tiers: 1.075x base, 1.125x at 30% drawdown, 0.85x at 42%, 0.70x at 50%.",
    "Realized-volatility governor with 336-hour lookback and 1.8% target, floored at 0.70x long fraction.",
    "Recovery state can re-risk to 1.85x when drawdown is improving and signal quality returns.",
    "Green/yellow 12-hour rebalance cooldown and 5% rebalance threshold to reduce churn.",
    "No same-asset debt/collateral overlap in the current implementation."
  ],
  "useCases": [
    "Use when the mandate accepts a SOL-led risk premium and can tolerate drawdowns around the high-50% area.",
    "Use as an actively governed alternative to raw SOL buy-and-hold when reducing catastrophic drawdown matters.",
    "Use when hourly monitoring and rebalancing are operationally feasible.",
    "Use as a research checkpoint for SOL/ETH, not as a universal multi-asset policy."
  ],
  "nonUseCases": [
    "Do not use for investors with a hard drawdown limit below 50% without further risk reduction.",
    "Do not assume the BTC-only variant works; it reduced drawdown but underperformed BTC buy-and-hold.",
    "Do not deploy without borrow availability, oracle, slippage, and health-factor monitoring.",
    "Do not treat the backtest as proof of live Kamino execution capacity."
  ],
  "failureModes": [
    "Path dependency: sharp moves can de-risk the system before a rebound and reduce recovery capture.",
    "SOL concentration: the edge is strongly SOL-driven, so regime transfer to BTC is poor.",
    "Turnover burden: even the improved checkpoint still performs roughly 789 actions per year.",
    "Health factor proximity: minimum observed health factor is around 1.22, leaving little room for live execution slippage.",
    "Execution assumptions: fees, borrow rates, latency, oracle marks, and liquidity can materially change outcomes."
  ],
  "productionControls": [
    "Pre-trade health-factor projection and minimum post-trade HF guard.",
    "Borrow availability and rate checks before every rebalance.",
    "Same-asset collateral/debt conflict prevention.",
    "Latency and stale-price guardrails around volatile moves.",
    "Daily reconciliation against expected holdings and debt.",
    "Kill switch for oracle anomalies, missing market data, or excessive action frequency."
  ],
  "artifactLinks": [
    "reports/latest_strategy_presets_20260627_035218/report.md",
    "reports/latest_strategy_presets_20260627_035218/summary.csv",
    "reports/btc_eth_directional_best_mechanics_20260628_172432/report.md",
    "reports/btc_eth_directional_best_mechanics_20260628_172432/summary.csv",
    "reports/btc_eth_directional_best_mechanics_20260628_172432/benchmarks.csv",
    "reports/btc_eth_directional_best_mechanics_20260628_172432/regime_summary.csv"
  ]
} satisfies StrategySystemCardData;

# Add Drawdown-Driven Crisis Mode

The SOL Supertrend strategy will add crisis mode as a distinct protective regime that can enter before normal full-short confirmation when SOL drawdown, bearish 1d trend, and either bearish 3d trend or portfolio drawdown confirm market/account damage. It can also enter from SOL-equivalent drawdown when the SOL-relative objective has already been damaged and either the 3d or 1w trend is bearish.

While active, crisis mode enforces a minimum ETH hedge floor and prevents short-term bullish votes from covering below that floor until the 1w trend turns green or SOL-equivalent value recovers near its trailing high. SOL-equivalent recovery exit is only eligible after crisis mode has actually experienced meaningful SOL-equivalent drawdown; this prevents a SOL-price-driven crisis from exiting immediately merely because the account still holds roughly the same SOL-equivalent value on the entry bar.

Crisis mode does not replace the normal strategy target. The effective ETH hedge target while crisis mode is active is the greater of the normal strategy target and the crisis hedge floor, so crisis mode can only add protection relative to the normal hedge or full-short logic.

If health-factor constraints prevent the strategy from reaching the crisis hedge floor, the run will enter under-hedged crisis cleanup and record an under-hedged crisis diagnostic. This keeps later analysis honest: poor crisis-period performance may be caused by account capacity limits rather than by the crisis trigger or cover discipline alone.

Under-hedged crisis cleanup is intentionally simple: stop adding risk, use available USDC collateral to reduce debt exposure, and cover ETH debt only when it exceeds the maximum safe crisis hedge. SOL sales remain reserved for the existing emergency de-risk path.

When crisis mode exits, the crisis hedge floor is removed, but ETH short exposure is not forcibly unwound on the exit bar. Normal targets and the existing rebalance cooldown govern the step-down after crisis exit.

Grid search will compare crisis-mode-enabled and crisis-mode-disabled candidates using the same optimization objective. Crisis diagnostics remain explanatory report fields rather than a separate ranking system.

The initial UI will keep crisis controls minimal: enable/disable, entry drawdown thresholds, hedge floor levels, and exit recovery threshold. Other lookbacks and diagnostics remain defaults until the crisis thesis has evidence behind it.

Crisis mode will use a dedicated crisis state rather than scattered booleans in the main strategy loop. The state records activation, entry and exit reasons, the active hedge floor, and under-hedged status while remaining separate from full-short state and emergency de-risk handling.

Crisis behavior will be observable in both strategy events and history output. Events record crisis transitions and reasons; history fields include at least in-crisis state, crisis hedge floor, effective hedge target, and under-hedged crisis status for charting and quarterly forensic reviews.

## Consequences

Crisis mode may reduce upside during sharp V-shaped recoveries because it exits slower than the normal short-cover ladder. We accept that trade-off because the 2022 forensic review showed that fast cover during a continuing waterfall gave back too much of the strategy's accumulated SOL advantage; the harness will keep crisis mode explicitly switchable even though it is enabled in the current best-thesis default.

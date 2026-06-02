# Radical Drawdown Reduction Candidates

Date: 2026-06-02

Branch: `research/radical-drawdown-candidates`

## Objective

Find drawdown controls that can materially reduce the SOL Supertrend / Kamino strategy's full-period max drawdown and 2024/2025 peak giveback without destroying the central objective: compounding SOL-equivalent wealth.

The current best-in-class baseline is the fast-break v2 no-partial-fill candidate:

`v2_ret8%_vol2.50_floor1.00_hold72_addhf2.50`

| Metric | Baseline |
|---|---:|
| Final SOL-equivalent | 424.90 SOL |
| Final USD value | $53,091.48 |
| Sortino | 2.290 |
| Max drawdown | 81.34% |
| 2024+ peak-to-trough | 64.84% |
| Min HF | 1.598 |

That baseline is better than the stateful profit-lock incumbent, but it still leaves the key problem almost untouched: the late-cycle red-mark drawdown remains about 65%. The next candidates must therefore be more structural than another fast-break threshold tweak.

## Deeper Research Readout

The literature now points to five useful principles.

1. Trend-following can provide crisis alpha, but it usually works because it can go materially short or materially de-risk during persistent crisis trends. Hamill, Rattray, and Van Hemert find trend-following strategies perform particularly strongly in the worst equity and bond market environments, but restrictions that improve crisis alpha can reduce average return. Source: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2831926).

2. Crypto trend-following should combine trend signals with volatility normalization and volatility-aware exits. The volatility-adaptive crypto trend-following paper uses multi-horizon moving averages, RSI confirmation, ATR scaling, and EMA-slope exits; its abstract explicitly emphasizes dynamic volatility states. Source: [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5821842).

3. Long-only crypto trend strategies and volatility-targeted crypto overlays can reduce drawdowns, but they sacrifice raw upside. The Monash crypto trend-following paper shows drawdown-driven strategies can lower max drawdown and improve Sharpe while generating lower returns, and then studies momentum with volatility targeting. Source: [Monash PDF](https://www.monash.edu/__data/assets/pdf_file/0011/3744821/Trend-following-Strategies-for-Crypto-Investors.pdf).

4. Portfolio insurance gives the missing structural frame. Multi-period drawdown-control research defines drawdown against a high-water mark and discusses CPPI-style allocation based on the cushion above a floor. It also warns that reactive drawdown control can trap a portfolio near the limit if the floor is too tight. Source: [Boyd et al. PDF](https://web.stanford.edu/~boyd/papers/pdf/multiperiod_portfolio_drawdown.pdf).

5. TIPP-style portfolio insurance is especially relevant because it ratchets the protective floor upward with gains. TIPP's floor is tied to a percentage of the maximum past portfolio value, which directly maps to the user's desire to stop giving back frothy rally gains. Source: [MDPI](https://www.mdpi.com/2227-9091/11/6/105).

These principles explain the current failure pattern:

- Fast-break v2 improves the old crash window, but an ETH hedge cannot fully solve SOL-specific downside.
- Generic partial ETH borrowing overtrades because the fast-break signal is good enough to raise a target, not good enough to authorize extra debt.
- Weekly-bearish reserve sells too late.
- Profit-lock reserve sells earlier and improves the 2024+ drawdown, but its state machine churns: 13 initial sells, 149 escalations, and 661 rebuys in the best 2024+ drawdown row.
- Floor-only drawdown containment had the most promising broad metrics, but the raw winner violated the health-factor guardrail.

The radical answer is not one mechanism. It is an explicit risk budget: protect gains with a high-watermark floor, reduce SOL exposure before the weekly trend breaks, keep ETH hedge adds selective, and slow re-entry.

## Candidate 1: TIPP-Style High-Watermark SOL Reserve

### Thesis

The strategy needs a portfolio-insurance layer that ratchets up after large gains. The current profit-lock reserve was close to this idea, but it behaved like a trading signal instead of a high-watermark insurance policy. TIPP is the cleaner model: protect a percentage of the maximum achieved portfolio value by reducing risky exposure as the cushion shrinks.

### Trigger

Enter a reserve episode when all are true:

- Portfolio value is at least `2.0x` initial value.
- Portfolio is within `5%-10%` of its trailing high, or has just fallen `3%-7%` from that high.
- Trend is no longer fully strong: green votes `<= 3`, 1d bearish, or fast-break active.
- No active reserve episode has already fired for the current high-watermark.

### Action

Sell a fixed SOL collateral slice into USDC collateral:

- initial slice: `5%-10%` of SOL collateral above the core minimum;
- optional single escalation: another `5%-10%` only if 3d turns bearish, fast-break is active, or drawdown from the high exceeds `12%-18%`;
- hard cap: `20%-30%` of starting episode SOL collateral;
- no repeated sell/escalate loops.

This should not block surplus hedge-profit reinvestment globally. It should only prevent reusing reserve USDC until the reserve exit condition fires.

### Exit / Rebuy

Rebuy is slow and episode-aware:

- no rebuy for at least `7d-14d` after the last reserve sale;
- require 1w not bearish and either 3d green or 3/4 base votes green;
- rebuy in `25%-50%` reserve buckets;
- reset the episode only after a fresh portfolio high or after portfolio value recovers within `3%-5%` of the prior high.

### Expected Trade-Off

This is the best candidate for reducing the 2024/2025 red-mark drawdown. It will lower final SOL-equivalent if it fires too early or rebuys too slowly. The test is whether a one-shot high-watermark episode can capture the drawdown benefit of profit-lock reserve without the churn that destroyed compounding.

### First Sweep

| Axis | Values |
|---|---|
| Gain gate | `1.5x`, `2.0x`, `3.0x` initial value |
| Near-high / drawdown entry | within `5%`, within `10%`, or `5%` drawdown |
| Initial sale | `5%`, `10%` |
| Escalation sale | `0%`, `5%`, `10%` |
| Reserve cap | `20%`, `30%` |
| Rebuy cooldown | `7d`, `14d`, `30d` |
| Rebuy gate | `1w green`, `1w green + 3d green`, `1w green + 3 base green` |

## Candidate 2: CPPI/TIPP Cushion-Based SOL Exposure Cap

### Thesis

Instead of discrete reserve episodes, define a protected high-watermark floor and continuously cap SOL collateral exposure based on the cushion above that floor. This is more radical: the strategy stops pretending that all SOL exposure is sacred once a large profit has been made.

### Trigger

Always compute after portfolio has gained at least `1.5x` initial value:

- high-watermark `H = max(portfolio_value)`;
- protected floor `F = protect_pct * H`, with `protect_pct` in `0.55-0.75`;
- cushion `C = portfolio_value - F`;
- risky budget `B = multiplier * C`.

If SOL collateral value exceeds `B` plus a buffer, rotate enough SOL into USDC to respect the cap.

### Action

Use a CPPI-like cap:

- max allowed SOL collateral value = `min(current SOL collateral value, multiplier * cushion)`;
- multiplier grid: `1.0`, `1.5`, `2.0`, `3.0`;
- never sell below a core SOL floor, e.g. `100 SOL` or a percentage of initial/accumulated SOL;
- use ETH hedge rules as normal, but reserve USDC is explicitly classified as protected capital.

This changes the problem from "detect the top" to "control exposure as the cushion above the protected floor shrinks."

### Exit / Rebuy

Rebuy as cushion expands:

- if portfolio recovers and `B` rises above current SOL collateral value by a threshold, allow bucketed SOL rebuy;
- require trend confirmation for re-risking;
- do not rebuy simply because SOL is cheaper.

### Expected Trade-Off

This is the most structurally capable way to reduce drawdown, but it is also the most likely to cap upside. It can get trapped if the protected floor is too high, as drawdown-control literature warns. The point of the sweep is to find whether there is a loose enough floor that still radically improves max drawdown.

### First Sweep

| Axis | Values |
|---|---|
| Activation gain | `1.5x`, `2.0x`, `3.0x` |
| Protected high-watermark floor | `55%`, `65%`, `75%` |
| Cushion multiplier | `1.0`, `1.5`, `2.0`, `3.0` |
| Core SOL minimum | `100 SOL`, `50%` of current accumulated SOL |
| Rebuy gate | `3 base green`, `4 base green`, `1w green + 3d green` |

## Candidate 3: SOL/ETH Hedge-Failure Circuit Breaker

### Thesis

ETH is not a perfect SOL hedge. The strategy's largest problem can happen when SOL falls harder than ETH, because the ETH short does not offset the dominant SOL collateral beta. The circuit breaker should identify "hedge not working" states and reduce SOL exposure or pause SOL buying, not merely add ETH debt.

### Trigger

Enter when fast-break/profit-lock/crisis is active and at least one relative-damage condition is true:

- SOL 24h/72h return underperforms ETH by `8%-15%`;
- portfolio drawdown speed exceeds ETH short PnL contribution by a threshold;
- SOL/ETH ratio breaks a `7d` or `14d` low;
- effective hedge target is high but realized hedge PnL is not offsetting SOL collateral losses.

### Action

Use a two-stage response:

1. Stage A: pause surplus SOL reinvestment and route realized hedge profits to USDC reserve or debt cleanup for `72h-168h`.
2. Stage B: if relative damage persists, sell `5%-15%` SOL collateral into protected USDC reserve.

Do not add generic partial ETH fills from this state unless green votes are `<= 1` or 3d/1w is bearish. The lesson from partial-fill sweeps is that extra ETH debt must have a very high confirmation bar.

### Exit / Rebuy

Exit when:

- SOL/ETH relative return stabilizes above the short moving average;
- 4h/8h trend recovers;
- realized hedge PnL begins offsetting SOL losses again;
- or a fixed timeout expires.

Rebuy reserve only after trend recovery, not immediately after relative underperformance stops.

### Expected Trade-Off

This candidate attacks the "wrong hedge" problem directly. It may be noisy because SOL can underperform ETH before both assets rally. That is why Stage A should be low-cost and reversible, while Stage B requires persistence.

### First Sweep

| Axis | Values |
|---|---|
| Relative underperformance lookback | `24h`, `72h`, `7d` |
| SOL-vs-ETH underperformance | `8%`, `12%`, `15%` |
| Stage A duration | `72h`, `168h` |
| Stage B sale | `5%`, `10%`, `15%` |
| Stage B persistence | `24h`, `72h` |

## Candidate 4: Confirmed-Crash Partial Hedge Fill

### Thesis

Fast-break partial fill failed because it borrowed during still-constructive regimes. Crisis-gated partial fill also failed because crisis mode can remain active while trend votes are mixed. The salvage path is a much stricter execution gate: partial fills only when the market is already confirmed broken, not merely shaky.

### Trigger

Allow partial fill only if all are true:

- fast-break target wants a higher hedge than can be safely added atomically;
- current hedge ratio is below requested target by at least `0.25`;
- fast-break active;
- crisis active or under-hedged crisis is true;
- and one of:
  - green votes `<= 1`,
  - 3d bearish,
  - 1w bearish,
  - SOL/ETH hedge-failure breaker active.

### Action

Borrow only a tightly bounded ETH amount:

- per-episode budget: `5%-15%` of SOL collateral value;
- projected min HF: `1.75`, `2.00`, `2.25`;
- max one fill per `72h-168h`;
- stop fills after two per crisis/fast-break episode.

This is not meant to solve the whole drawdown alone. It is meant to fix the 2025 "signal active, no hedge added" bottleneck without recreating the generic partial-fill overtrading failure.

### Exit

Use the existing fast-break staged decay and normal hedge-down mechanics, but add a rule that partial-fill debt cannot be held above the active target after recovery. If fast-break exits and crisis no longer confirms, unwind the extra partial-fill layer first.

### Expected Trade-Off

This has the clearest implementation path because the code already has partial-fill plumbing. It is less radical than reserve/portfolio insurance and probably cannot cut drawdown enough by itself. It is still worth testing because it targets a known execution bottleneck.

### First Sweep

| Axis | Values |
|---|---|
| Max green gate | `0`, `1` |
| Higher-timeframe gate | `3d bearish`, `1w bearish`, `3d or 1w bearish` |
| Budget | `5%`, `10%`, `15%` |
| Projected min HF | `1.75`, `2.00`, `2.25` |
| Fill cooldown | `72h`, `168h` |

## Candidate 5: Two-Book Architecture: Accumulation Book And Protected Book

### Thesis

The current strategy mixes two goals in one balance sheet: accumulate SOL aggressively and protect accumulated gains. That creates confused behavior: reinvestment helps long-term SOL-equivalent, but it also keeps all gains exposed to future SOL drawdowns. Split the simulated account into an accumulation book and a protected book.

### Trigger

When realized hedge profits or portfolio gains exceed thresholds:

- keep baseline surplus reinvestment for the accumulation book;
- divert a fraction of realized hedge profit or post-high gains into a protected USDC book once the portfolio is up `2x` or more;
- optionally increase protected allocation after profit-lock or fast-break triggers.

### Action

Protected-book rules:

- protected USDC is not used for SOL rebuy until a strict recovery or deep-value rule fires;
- it can be used for debt cleanup or health-factor defense;
- it is excluded from routine surplus SOL reinvestment;
- it can be bucketed into SOL only at deep drawdowns (`35%`, `50%`, `65%`) or confirmed recovery.

This is different from selling existing SOL. It first changes the destination of future realized gains, then optionally adds reserve sales only in frothy states.

### Exit / Rebuy

Deploy protected book using one of two modes:

- value buckets: buy SOL once at each deep drawdown tier;
- recovery buckets: buy SOL only after 1w green plus 3d/base confirmation.

Never deploy continuously while drawdown is still worsening.

### Expected Trade-Off

This may be the least painful way to cut drawdown because it does not immediately sell existing SOL. It will reduce final SOL-equivalent if protected USDC sits idle through major recoveries. But it directly addresses the existing failure where all hedge profits eventually become exposed SOL beta again.

### First Sweep

| Axis | Values |
|---|---|
| Protected profit fraction | `25%`, `50%`, `75%` of realized hedge profits after gain gate |
| Activation gain | `1.5x`, `2.0x`, `3.0x` |
| Protected max fraction | `10%`, `20%`, `30%` of portfolio value |
| Deployment mode | deep drawdown buckets, recovery confirmation |
| Drawdown buckets | `35/50/65%`, `25/40/55%` |

## Recommended Test Order

1. **TIPP-style high-watermark SOL reserve.** It is the most direct successor to the only reserve family that reduced the 2024+ drawdown, and it fixes the observed churn failure.
2. **Two-book protected profit architecture.** It changes less existing SOL collateral and may reduce drawdown with a lower SOL-equivalent penalty.
3. **SOL/ETH hedge-failure circuit breaker.** It adds a necessary diagnostic and can decide whether to pause reinvestment, reserve USDC, or sell SOL.
4. **Confirmed-crash partial hedge fill.** It is easy to test because plumbing exists, but prior evidence says it needs very strict gates.
5. **CPPI/TIPP cushion cap.** It is the most radical and most likely to reduce drawdown, but it changes the strategy identity the most. Test after the discrete high-watermark reserve so we know whether a simpler ratchet is enough.

## Promotion Standard

A candidate is only interesting if it does at least two of these:

- reduces full-period max drawdown below `75%`;
- reduces 2024+ peak-to-trough below `55%`;
- preserves final SOL-equivalent above `350 SOL`;
- keeps Sortino at or above `2.25`;
- keeps min HF at or above `1.50`;
- reduces final ETH debt or avoids materially increasing it.

For a truly radical drawdown candidate, I would accept lower final SOL-equivalent only if the drawdown improvement is large and clean. A mild 2-5 percentage point drawdown improvement is not worth cutting SOL-equivalent in half.

## Anti-Patterns To Avoid

- Do not attach borrowing authority directly to fast-break without a second damage gate.
- Do not use 1w bearish as the first reserve sell trigger.
- Do not let reserve sell/rebuy fire repeatedly inside the same high-watermark episode.
- Do not block all surplus reinvestment during generic drawdown containment.
- Do not evaluate only full-period max drawdown; segment the 2021-2023 crash and 2024+ red-mark drawdown separately.
- Do not promote any rule that "wins" by leaving the account with low SOL participation into recoveries.

## Bottom Line

The next serious attempt should be a high-watermark insurance policy, not another hedge tweak. The strategy has already learned how to hedge better. It has not learned how to protect accumulated SOL-era gains. A one-shot, TIPP-style profit-lock reserve with slow rebuy is the cleanest first candidate for radical drawdown reduction.

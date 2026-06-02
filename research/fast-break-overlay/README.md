# Fast Break Overlay Research

Downloaded on 2026-05-29 for the SOL Supertrend / Kamino fast-break hedge research question.

## Research Question

Can we detect an early SOL market break quickly enough to increase ETH hedge exposure before portfolio drawdown becomes severe, while preserving the strategy's long-term goal of maximizing SOL-equivalent wealth?

## Downloaded Papers

| File | Source |
|---|---|
| `time-series-momentum-moskowitz-ooi-pedersen.pdf` | Tobias Moskowitz, Yao Hua Ooi, Lasse Heje Pedersen, "Time Series Momentum" |
| `momentum-has-its-moments-barroso-santa-clara.pdf` | Pedro Barroso, Pedro Santa-Clara, "Momentum Has Its Moments" |
| `momentum-crashes-daniel-moskowitz.pdf` | Kent Daniel, Tobias Moskowitz, "Momentum Crashes" |
| `volatility-managed-portfolios-moreira-muir-nber.pdf` | Alan Moreira, Tyler Muir, "Volatility Managed Portfolios" |
| `investors-guide-to-crypto-ssrn-id4124576.pdf` | "An Investor's Guide to Crypto" |
| `adaptive-trend-crypto-arxiv-2602-11708.pdf` | "Systematic Trend-Following with Adaptive Portfolio Construction" |
| `bayesian-hmm-crypto-predictability.pdf` | "Exploring the Predictability of Cryptocurrencies via Bayesian Hidden Markov Models" |

## Access Notes

- `volatility-adaptive-trend-following-crypto-karassavidis.pdf` could not be downloaded directly because SSRN returned `403`.
- `trend-following-strategies-for-crypto-investors-monash.pdf` could not be downloaded directly from Monash or SSRN because both returned `403`.
- `in-crypto-we-trend-man-ahl-download-page.html` is the Man Group download page returned by the site instead of the PDF. It likely requires an attestation flow before serving the file.

## Implementation-Relevant Themes

- Time-series momentum supports the idea that fast trend breaks can be persistent enough to trade.
- Risk-managed momentum and volatility-managed portfolios support raising or lowering exposure based on realized risk.
- Momentum crash research warns that crash states are forecastable but path-dependent, especially around rebounds.
- Crypto trend-following papers support using volatility scaling, adaptive trend filters, and regime-aware switching, but the rules need to remain simple enough to audit in the backtester.

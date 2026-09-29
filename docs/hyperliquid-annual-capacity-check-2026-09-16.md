# Annual capacity checkpoint — 2026-09-16

The temporary-ranking staging implementation has passed its postformat93-test
backend/API regression. This does **not** prove that the annual comparison fits
the approved16GiB shared derived cache. No real staging or feature expiry is enabled.

## Verified local candidate-growth diagnostic

Read-only scan63320 completed exit0 in241.675seconds; peak RSS928759808bytes.
DuckDB used one thread,256MB memory and zero disk spill. All350qualified files
(319594956rows) were hashed before and after the query; source physical identities,
qualification report and engine were rechecked. No cache payload or AWS request
was created. Counts include dormant and ineligible wallets; there is no sampling.

Source: annual job batch0049 qualification_zs1klb6y/manifest.json,
SHA3e87c397c58710fd2262db34e6c1626b32b295909f57a06a2573fb310e2e7331.
Known source interval2025-06-02 through2026-05-18 exclusive.

For each market, the query grouped users by their first exchange timestamp,
counted first-observation days and accumulated candidates strictly before midnight
decisions. Evaluation calendar:2025-09-01 through2026-09-01 exclusive,365daily
and53Monday decisions. Counts are held constant after the known source cutoff.

| Market | First observed day | Candidates at evaluation start | Candidates at source cutoff | Daily row scenario | Weekly row scenario |
| --- | --- | ---: | ---: | ---: | ---: |
| BTC | 2025-06-02 | 108185 | 430600 | 115710481 | 16752644 |
| xyz:GOLD | 2025-12-22 | 0 | 51008 | 9548949 | 1383103 |
| xyz:SP500 | 2026-03-18 | 0 | 34363 | 4794741 | 698424 |
| xyz:TSLA | 2025-11-13 | 0 | 31436 | 6434700 | 931792 |

These are all-markets-at-every-decision scenarios, **not** a validated expanding-
market strategy bound. Native qualification, lookback eligibility and market
selection can delay inclusion. New traders after May18 are absent. First observed
day is not itself proof of launch time or readiness for a90-day ranking window.
Do not sum weekly and daily storage blindly: exact Monday queries can share receipts.

## Current admission arithmetic

- Shared cap:17179869184bytes.
- Reserved metadata:33554432bytes.
- Existing retained objects:5286350121bytes across495allocations; no pending.
- Conservative staged admission:4831903744bytes.
- Remaining for **all** new retained data:7028060887bytes.

An existing full real ranking has108185rows/8323840bytes, with verified
SHA408f4ce50b71e4cefed22af7b4ba5ec7325170e83311a0d1bc678350411f3fbd:
76.94079585894532encoded bytes per row. Applying that compression to the earlier
BTC-only114133907row lower bound gives8.782GB illustrative ranking storage,
already1.753GB above admission headroom before new features/candidates/other markets.
Compression extrapolation is not an annual encoded-byte bound or impossibility proof.
Existing5.286GB is preserved; reuse under a different final source pin is not assumed.

## Decision and remaining gates

User approved up to64GiB in the **same local cache** on2026-09-16, conditional on
remaining checks. The transition is not yet implemented or applied; the effective
cap remains16GiB. AWS lifetime acquisition cap
remains300GiB; no deletion or replacement cache is proposed. Host currently reports
about830GiB free, but free disk is not authority to exceed the approved cache cap.
An increased ceiling would not allocate it immediately or establish annual capacity.

If approved, a separately verified receipt-backed transition must preserve the
existing8-to16GiB receipt and its pinned engine dependencies; do not edit those four
pinned modules in place. Measure final-source candidate evidence, rolling windows,
ranking growth and admission before real execution. Report/result ranking copies
remain outside this cache and must be counted separately in total disk footprint.

All67archive batches, final annual registration, actual saved annual weekly and
matched daily runs, independent accounting reconciliation and rendered annual UI
remain incomplete. No annual strategy result is claimed by this diagnostic.

# Historical market metadata: reconstruction feasibility

Investigation date: 2026-09-06. Scope: source/schema investigation and metadata
cost estimation, not bulk acquisition or implementation. The user approved this
investigation while excluding large transaction-log downloads.

## Conclusion

Historical configuration is represented in HyperCore actions. The official
explorer-block archive is a promising smaller source, verified by decoding one
small object. This is evidence of a reconstruction route, **not proof that a
complete historical catalogue can already be built**.

The remaining requirements are a verified baseline (or complete creation history),
successful-action semantics, full block coverage and dated instrument descriptions.
Do not download a week of logs and assume that it establishes all settings at
the beginning of that week. Do not treat current classifications as historical.

## Primary-source findings

The [official deployer action schema](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/hip-3-deployer-actions)
documents registration with venue collateral-token index, asset identifier,
size precision and margin configuration. Other actions modify funding settings,
trading halts, margin settings, fees and annotations; a venue can be disabled.
Annotation fields include category and description. Their presence in today's
schema does not establish their historical availability or complete population.

Consequences for replay:

- Resolve collateral-token indexes from historical token identity evidence.
- Preserve namespace and source identifiers; never infer an instrument from an
  unqualified ticker alone.
- Apply only proven successful changes in block/transaction/action order.
- Preserve halts, disables and re-registration rather than assuming a ticker
  denotes one immutable contract forever.
- Treat missing baseline fields, unknown actions and uncovered intervals as
  unknown evidence, not default values or implied eligibility.
- Current API state is a reconciliation check, not a historical starting state.
- Core-market configuration needs its own verified mapping and initial state;
  the HIP-3 registration schema alone does not solve the whole market universe.

[Official node documentation](https://github.com/hyperliquid-dex/node/blob/main/README.md)
distinguishes action-only from action-and-response logs and describes local
periodic state snapshots. No downloadable historical state baseline has been
located in the three authorized archive root listings.

[Quicknode's own block schema](https://www.quicknode.com/docs/hyperliquid/datasets/blocks)
documents action result records. This supports checking execution outcomes,
but does not establish that the official S3 objects use Quicknode's format or
that Quicknode offers the required historical export on an accessible plan.

## Measured archive sizes

Metadata-only LIST requests returned untruncated results for both August 1 raw
transaction-log partitions:

| Prefix below `hl-mainnet-node-data/` | Objects | Compressed bytes |
| --- | ---: | ---: |
| replica_cmds/2026-07-25T07:35:28Z/20260801/ | 46 | 27,789,209,259 |
| replica_cmds/2026-08-01T09:05:45Z/20260801/ | 76 | 50,243,201,307 |
| **Total listed** | **122** | **78,032,410,566** |

This is 72.67 GiB. At the previously used USD 0.09/GiB transfer assumption:
approximately USD 6.54 for these files, or USD 45.78 for seven identical-volume
days. The latter is an extrapolation, not an exact seven-day manifest or quote.
Requests, compute, storage, tax, allowances and initial-state reconstruction
are excluded. Partition presence does not prove gap-free event-time coverage.
**None of these raw-log objects was downloaded.**

One explorer partition was also enumerated completely:

`explorer_blocks/1000000000/1093400000/`

It contains 1,000 objects totaling 1,866,364,040 bytes, with final-height labels
from 1093400100 through 1093500000. This is 1.74 GiB per 100,000 labelled blocks
in this partition, approximately USD 0.16 transfer at the same assumption.
It is materially smaller but still requires a historical baseline and potentially
many months of scanning. No full-period explorer cost or download is approved.

## Small explorer sample: actual evidence

One small object was acquired under the existing sample budget with a 2 MiB
hard object-size cap, conditional GET against its HEAD ETag and no automatic
download retries:

`explorer_blocks/1000000000/1093400000/1093400100.rmp.lz4`

- Compressed bytes: 1,470,756; decoded MessagePack bytes: 7,274,733.
- One MessagePack array of 100 blocks, heights 1093400001–1093400100.
- Source timestamps: 2026-07-31T22:55:38.593625733 through
  2026-07-31T22:55:45.670161755 (no timezone suffix).
- Blocks contain `header` with block time/height/hash/proposer, and `txs`.
- Transactions contain `actions`, `user`, `raw_tx_hash`, and `error`.
- 27,688 transactions had null error; 102 had non-null error.
- Transactions can contain multiple actions (sample maximum: 100). A null
  transaction error must not be assumed to prove each nested action succeeded
  without validating the source's execution semantics.
- Eleven `perpDeploy` actions were observed, all `setOracle`; no registration,
  halt or annotation action was observed in this seven-second sample.
- Namespaced HIP-3 oracle identifiers are present, including gold, equity and
  index instruments. `SetGlobalAction` was also observed with positional price
  arrays, underscoring the need for a core instrument-index baseline.

Raw evidence and a source/ETag/size/SHA-256 manifest are retained, Git-ignored:

`.hyperliquid_cache/source_probes/explorer_schema_xlkmf1vl/`

The sample establishes a decodable action-bearing source. It does **not** prove
registration coverage, success semantics or catalogue completeness.

## Alternative to scanning bulk logs

[Dune's perpetuals dataset](https://dune.com/data/perpetuals-trading) documents
market reference data including classifications and contract descriptions, with
HyperCore and operator-deployed market coverage. It is an Enterprise add-on.
The public description does not establish bitemporal reference history or an
accessible price. No subscription, query purchase or provider contact was made.

A provider export is useful only if it includes historical versions and source
provenance, not merely today's enriched market table. Before acquiring anything,
request or locate evidence for:

1. A dated baseline before the ranking warmup, including inactive markets and
   historical token/index mappings.
2. All successful configuration/lifecycle changes through the backtest end,
   with block/time and transaction provenance and explicit gap reporting.
3. Dated classification/contract-description evidence and clear handling of
   markets that were unclassified at the time.
4. A small registration/change sample cross-checkable against the official
   explorer archive, plus historical core-market evidence.
5. Export access, license and total price before committing to acquisition.

## Recommendation and authorization boundary

Do not perform the raw-log bulk pull. Prefer a verified baseline plus an indexed
configuration-change export; use explorer replay as a fallback only after the
baseline and outcome semantics are qualified and its complete cost is approved.
This recommendation is not a claim that such an export is currently accessible.

The authorized investigation is complete. Real-data implementation and a saved
strategy run remain incomplete. No large raw logs, full-period explorer data,
subscriptions, orders or external messages were purchased or sent.

# Polyglot monorepo migration verification

Verified on 2026-09-04 in the isolated worktree
`/home/lshoge/code/cryptoLab/.worktrees/polyglot-monorepo`.

Status: **DONE_WITH_CONCERNS**. The migration checks pass. Integration remains
pending an explicit user choice, and no integration or original-tree mutation was
performed.

## Revision and tools

- Branch: `feature/polyglot-monorepo`
- Migration baseline: `aef10192e859099b61936023e5c1e0078262d4b3`
- Verified pre-document HEAD: `a35a22f5fc2fa331b6c2ef8363380f21b0f6c49c`
- `uv 0.10.0`; isolated project Python `3.12.3`; Node.js `v18.19.1`; npm `9.2.0`
- Ephemeral `just 1.53.0`:
  `/tmp/cryptolab-task11.qB9g3d/just-root/bin/just`
- Ephemeral `pnpm 10.30.3` shim: `/tmp/cryptolab-task13-bin/pnpm`, backed by
  `/tmp/cryptolab-tools-XeV1op/node_modules/pnpm/bin/pnpm.cjs`

The temporary tools and install environments are verification artifacts, not
repository dependencies or global installations.

## Canonical commands

All canonical commands were invoked through the actual root `justfile` recipes,
with the pinned temporary tools on `PATH` and `UV_CACHE_DIR` under `/tmp`.

| Command | Fresh result |
| --- | --- |
| `just test` | PASS, exit 0: 384 passed, 8 deselected, 90 warnings in 5.36s |
| `just typecheck` | PASS, exit 0: `tsc -b --pretty false` |
| `just build` | PASS, exit 0: Vite 5.4.21 built 34 modules in 840ms |
| `uv run python -m kamino_simulator.cli --help` | PASS, exit 0; complete argparse help shown |
| `uv run python -m kamino_simulator.cli` | PASS, exit 0; bundled offline sample produced baseline and after-action risk/liquidation output |

The test warnings are pandas/NumPy timedelta deprecations from the strategy
backtester tests. The build warning is the existing Vite advisory for the
707.65 kB minified JavaScript chunk exceeding 500 kB. Neither warning failed its
command.

## Application and path smoke checks

The focused offline command below passed 24 tests in 1.40s:

```text
uv run pytest -q \
  apps/kamino-simulator/tests/test_cli.py \
  apps/kamino-simulator/tests/test_paths.py \
  apps/kamino-simulator/tests/test_server_helpers.py \
  apps/strategy-backtester/tests/test_app_entrypoint.py \
  apps/report-explorer/tests/test_app_paths.py
```

This executes the Kamino and strategy-backtester entrypoints with Streamlit
`AppTest`, asserts no startup exceptions, and blocks socket/network access. The
report-explorer test executes its entrypoint in a cold subprocess from an
unrelated directory with socket access blocked and verifies repository report
and cache paths. Kamino also has a cold no-argument CLI subprocess check.

A supplementary bounded `streamlit run` attempt was made for all three
entrypoints with headless mode and usage telemetry disabled. The sandbox refused
even the localhost listening socket with `PermissionError: [Errno 1] Operation
not permitted`, before application serving began. This is an environment limit,
not an application exception; the network-blocked AppTest/cold-subprocess
coverage above is the authoritative startup evidence. It was not escalated
because browser/server functional testing was intentionally out of scope.

## Working-directory and import independence

From `/tmp/cryptolab-task13-cwd.hoVIwP`, outside the repository, `uv run
--project /home/lshoge/code/cryptoLab/.worktrees/polyglot-monorepo` produced:

- CLI help: exit 0.
- no-argument bundled sample: exit 0.
- the absolute-path focused CLI/path/app suite: 21 passed in 1.40s.
- `arblab`, `kamino_simulator`, `strategy_backtester`, and `report_explorer`
  imports resolved respectively beneath `packages/arblab/src` and each
  `apps/*/src` package.
- repository root resolved to the migration worktree; the fixture resolved to
  `data/fixtures/kamino_sample.json`; report root resolved to `reports`; and the
  price cache resolved to `notebooks/.price_cache`, all within that worktree.
- legacy top-level import probes for `backtest`, `strategies`, `kamino_app`, and
  `strategy_report_app` all returned `None`.

## Frozen lock verification

Both installs used new isolated locations and did not alter the repository's
existing environments:

- `UV_CACHE_DIR=/tmp/cryptolab-task13-uv.geMQYb/cache
  UV_PROJECT_ENVIRONMENT=/tmp/cryptolab-task13-uv.geMQYb/venv uv sync --locked
  --all-packages --all-groups`: PASS after network authorization; 88 packages
  resolved and 84 installed. The initial sandboxed attempt failed only because
  DNS access to PyPI was blocked.
- `/tmp/cryptolab-task13-bin/pnpm install --frozen-lockfile --store-dir
  /tmp/cryptolab-task13-pnpm.Ql117n/store`, run against a temporary copy of the
  workspace manifests and lockfile: PASS after network authorization; 70
  packages installed, pnpm 10.30.3.

Lock hashes before and after all verification were unchanged:

| Lockfile | SHA-256 |
| --- | --- |
| `uv.lock` | `4a1de9b37078d863aacc245113ef2a55834df875e6d81bd381e7351f95f7dc0c` |
| `pnpm-lock.yaml` | `9f67db4323cdb378bd7e3cab5c812655e2e943c5f671192b75a5183ca8cc1aae` |

## Structure and stale-reference audit

- The worktree was clean before this document was added.
- `git diff --check aef10192e859099b61936023e5c1e0078262d4b3..HEAD`
  passed.
- `rg --files` listed 221 nonignored files. Runtime Python code is under the
  package/application `src` trees, tests are colocated with owners, the React
  app is under `apps/strategy-system-card`, shared fixtures are under
  `data/fixtures`, and the old exchange experiment is isolated under `legacy/`.
- Searches found no obsolete root runtime entrypoints and no old helper-module
  imports. Current `*/src/*/app.py` references are canonical.
- The `strategy_report_app.py` reference and npm commands in the migration
  baseline are intentional historical evidence. `kamino_app.py` appears only in
  a source-provenance docstring. `pip install` appears only in the isolated
  legacy Windows setup script. Generated system-card data labels are intentional
  provenance.

The system-card generator was also invoked to document its known isolated-tree
limitation. It failed safely before writing because its source report CSVs are
among the original tree's preserved untracked files and therefore absent from
this worktree. The checked-in generated TypeScript/JSON assets remain available;
regeneration was not an acceptance requirement.

## Original-tree preservation audit

The original tree at `/home/lshoge/code/cryptoLab` was inspected read-only. A
fresh audit was written only to `/tmp/cryptolab-task13-original-audit.HqGvfh`.
Fresh `git status --porcelain=v1 -z`, unstaged and staged `--binary --full-index`
patches, the nonignored untracked NUL manifest, and SHA-256 list for those
untracked files are byte-identical (`cmp` exit 0) to
`/tmp/cryptolab-monorepo-snapshot.nNbywJ`:

| Artifact | SHA-256 |
| --- | --- |
| `status.z` | `fd25bfcd992b3324c8795d5f94deaabbdc095817ec183042861f7c1b34eba93f` |
| `unstaged.patch` | `7445a5126fff867449ab93822ab79e9a1acab1182836e028abc2ce4048acc2c6` |
| `staged.patch` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `untracked.z` | `f6d455ee10f7fcd521466f530abe66c75f37a26d6ab4165123ce85f9afab63f1` |
| `untracked.sha256` | `d29b27cedb7b91f87920e6c078b6e4e2d3ddf9f7a08806d289d9a98c53b30217` |

The original still has exactly the captured seven modified tracked paths, no
staged changes, and the same 23 nonignored untracked files. No new source edit or
untracked change appeared. The snapshot directory was preserved.

If integration is later approved, use the existing clean migration branch and
the preserved original-tree snapshot/checkpoint to choose an explicit merge or
worktree integration path. Do not discard or overwrite the original working-tree
changes. No such integration was performed here.

## Intentional exclusions

Chromium was not installed, the eight Playwright functional tests remained
deselected, and no live Solana RPC, market-data, wallet, or browser action was
run. These exclusions match the migration scope and leave integration pending
explicit user direction.

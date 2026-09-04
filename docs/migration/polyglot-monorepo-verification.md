# Polyglot monorepo migration verification

Verified on 2026-09-04. All required offline acceptance checks passed. The eight
browser-backed functional tests, live Solana RPC and market-data behavior, and
integration into the original working tree remain deliberately unverified or
pending.

No source, test, build, or configuration file was changed by this verification.
This report is the only repository change.

## Revision, provenance, and tools

- Branch: `feature/polyglot-monorepo`
- Migration baseline: `aef10192e859099b61936023e5c1e0078262d4b3`
- Verified implementation HEAD: `a35a22f5fc2fa331b6c2ef8363380f21b0f6c49c`
- Initial report commit: `d61bc51`
- `uv 0.10.0`; isolated project Python `3.12.3`; Node.js `v18.19.1`; npm `9.2.0`
- Ephemeral pinned tools: `just 1.53.0` and `pnpm 10.30.3`

Absolute paths used during the run are recorded without embedding a workstation
user identifier: `$WORKTREE` was the isolated `polyglot-monorepo` worktree,
`$ORIGINAL` was its parent repository's original working tree, and `$SNAPSHOT`
was `/tmp/cryptolab-monorepo-snapshot.nNbywJ`. Temporary verification directories
were created under `/tmp`. These are provenance records, not durable dependencies.

The reproducible templates below use:

```sh
WORKTREE=/absolute/path/to/polyglot-monorepo-worktree
ORIGINAL=/absolute/path/to/original-working-tree
SNAPSHOT=/tmp/cryptolab-monorepo-snapshot.nNbywJ
BASELINE=aef10192e859099b61936023e5c1e0078262d4b3
JUST=/absolute/path/to/just-1.53.0
PNPM=/absolute/path/to/pnpm-10.30.3
UV_CACHE_DIR=$(mktemp -d /tmp/cryptolab-verify-uv-cache.XXXXXX)
export WORKTREE ORIGINAL SNAPSHOT BASELINE JUST PNPM UV_CACHE_DIR
```

## Workspace, CLI, and application checks

The three workspace recipes below were invoked through the root `justfile`.
The CLI checks were direct `uv run` commands, not `just` recipes.

```sh
cd "$WORKTREE"
PATH="$(dirname "$PNPM"):$PATH" "$JUST" test
PATH="$(dirname "$PNPM"):$PATH" "$JUST" typecheck
PATH="$(dirname "$PNPM"):$PATH" "$JUST" build

uv run python -m kamino_simulator.cli --help
uv run python -m kamino_simulator.cli
```

| Command | Fresh result |
| --- | --- |
| `just test` | PASS, exit 0: 384 passed, 8 deselected, 90 warnings in 5.36s |
| `just typecheck` | PASS, exit 0: `tsc -b --pretty false` |
| `just build` | PASS, exit 0: Vite 5.4.21 built 34 modules in 840ms |
| CLI help | PASS, exit 0; complete argparse help shown |
| CLI without arguments | PASS, exit 0; bundled offline sample produced baseline and after-action risk output |

The 90 warnings are pandas/NumPy timedelta deprecations. The build emitted the
existing Vite advisory for a 707.65 kB minified JavaScript chunk exceeding
500 kB. Neither warning failed its command.

Each focused application smoke can be reproduced independently:

```sh
cd "$WORKTREE"
uv run pytest -q \
  apps/kamino-simulator/tests/test_cli.py \
  apps/kamino-simulator/tests/test_paths.py \
  apps/kamino-simulator/tests/test_server_helpers.py
uv run pytest -q apps/strategy-backtester/tests/test_app_entrypoint.py
uv run pytest -q apps/report-explorer/tests/test_app_paths.py
```

The combined invocation passed 24 tests in 1.40s. Kamino and strategy-backtester
execute with Streamlit `AppTest`, assert no startup exceptions, and block socket
access. Report explorer executes in a cold subprocess from an unrelated directory
with socket access blocked and verifies repository report/cache paths. Kamino also
has a cold no-argument CLI subprocess check.

The supplementary bounded server startup used this template for each entrypoint:

```sh
cd "$WORKTREE"
APP=apps/kamino-simulator/src/kamino_simulator/app.py
PORT=18501
timeout 6s uv run streamlit run "$APP" \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  --server.address=127.0.0.1 \
  --server.port="$PORT"
```

The same command was attempted with the strategy-backtester and report-explorer
entrypoints on ports 18502 and 18503. All three exited 1 because the sandbox
refused the localhost listening socket with `PermissionError: [Errno 1] Operation
not permitted`, before serving began. It was not escalated: browser/server
functional testing was outside scope, while the network-blocked AppTest and cold
subprocess tests provide the required offline startup evidence.

## Working-directory and import independence

The external-directory run used absolute project and test paths:

```sh
OUTSIDE=$(mktemp -d /tmp/cryptolab-verify-cwd.XXXXXX)
cd "$OUTSIDE"
uv run --project "$WORKTREE" python -m kamino_simulator.cli --help
uv run --project "$WORKTREE" python -m kamino_simulator.cli
uv run --project "$WORKTREE" pytest -q \
  "$WORKTREE/apps/kamino-simulator/tests/test_cli.py" \
  "$WORKTREE/apps/kamino-simulator/tests/test_paths.py" \
  "$WORKTREE/apps/strategy-backtester/tests/test_app_entrypoint.py" \
  "$WORKTREE/apps/report-explorer/tests/test_app_paths.py"
```

Both CLI invocations exited 0 and the focused suite passed 21 tests in 1.40s.
This probe recorded package locations, resolved resources, and absent legacy
top-level modules:

```sh
uv run --project "$WORKTREE" python - <<'PY'
import importlib.util
import arblab, kamino_simulator, report_explorer, strategy_backtester
from arblab.paths import fixture_path, notebook_price_cache_dir, repo_root

for package in (arblab, kamino_simulator, strategy_backtester, report_explorer):
    print(package.__name__, package.__file__)
print("root", repo_root())
print("fixture", fixture_path("kamino_sample.json"))
print("reports", repo_root() / "reports")
print("cache", notebook_price_cache_dir())
for name in ("backtest", "strategies", "kamino_app", "strategy_report_app"):
    print(name, importlib.util.find_spec(name))
PY
```

The four packages resolved under their `packages/*/src` or `apps/*/src` trees.
Root, fixture, reports, and cache resolved within `$WORKTREE`; every legacy probe
printed `None`.

## Frozen lock verification

The uv install used a new cache and environment, leaving the repository's
existing environment untouched:

```sh
UV_TMP=$(mktemp -d /tmp/cryptolab-verify-uv.XXXXXX)
UV_CACHE_DIR="$UV_TMP/cache" \
UV_PROJECT_ENVIRONMENT="$UV_TMP/venv" \
uv sync --project "$WORKTREE" --locked --all-packages --all-groups
```

It passed after network authorization: 88 packages resolved and 84 installed.
The first sandboxed attempt failed only because DNS access to PyPI was blocked.

For pnpm, only the root `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`,
and `apps/strategy-system-card/package.json` were copied into a fresh temporary
directory with the same relative layout. The frozen install and store remained
entirely there:

```sh
PNPM_TMP=$(mktemp -d /tmp/cryptolab-verify-pnpm.XXXXXX)
mkdir -p "$PNPM_TMP/apps/strategy-system-card"
cp "$WORKTREE/package.json" "$WORKTREE/pnpm-workspace.yaml" \
  "$WORKTREE/pnpm-lock.yaml" "$PNPM_TMP/"
cp "$WORKTREE/apps/strategy-system-card/package.json" \
  "$PNPM_TMP/apps/strategy-system-card/"
cd "$PNPM_TMP"
"$PNPM" install --frozen-lockfile --store-dir "$PNPM_TMP/store"
```

This passed after network authorization: 70 packages installed with pnpm
10.30.3. Lockfile bytes were checked before and after all verification:

```sh
cd "$WORKTREE"
sha256sum uv.lock pnpm-lock.yaml
```

| Lockfile | Unchanged SHA-256 |
| --- | --- |
| `uv.lock` | `4a1de9b37078d863aacc245113ef2a55834df875e6d81bd381e7351f95f7dc0c` |
| `pnpm-lock.yaml` | `9f67db4323cdb378bd7e3cab5c812655e2e943c5f671192b75a5183ca8cc1aae` |

## Structure and stale-reference audit

The durable audit commands were:

```sh
cd "$WORKTREE"
git status --short --branch
git diff --check "$BASELINE"..HEAD
rg --files
find . -maxdepth 1 -type f -print
rg -n 'streamlit run (app\.py|strategy_report_app\.py)|python (app\.py|strategy_report_app\.py)|npm (install|run|test)|pip(3)? install|strategy_report_app\.py|kamino_app\.py' \
  --glob '!uv.lock' --glob '!pnpm-lock.yaml' --glob '!**/*.ipynb' \
  --glob '!**/*.pdf' .
rg -n '(from|import) (backtest|strategies)(\.|$)' --glob '*.py' .
```

`git diff --check` passed. `rg --files` listed 221 nonignored files before this
report was created and 222 after it was committed; the count is informational,
not an invariant. Runtime Python code is under package/application `src` trees,
tests are colocated, the React app is under `apps/strategy-system-card`, fixtures
are under `data/fixtures`, and the old exchange experiment is under `legacy/`.

No obsolete root runtime entrypoint or helper import was found. Current
`*/src/*/app.py` references are canonical. The old entrypoint and npm commands in
the baseline are historical evidence; `kamino_app.py` is source provenance in a
docstring; `pip install` is inside the isolated legacy Windows script; generated
system-card labels are intentional provenance.

The generator's fail-before-write behavior was checked while hashing its outputs:

```sh
cd "$WORKTREE"
OUTPUTS=(
  apps/strategy-system-card/src/data/strategySystemCardData.ts
  apps/strategy-system-card/src/data/purePerpAssetDiagnostics.json
  apps/strategy-system-card/src/data/purePerpSignalSystemCardData.ts
)
sha256sum "${OUTPUTS[@]}" > /tmp/generator-before.sha256
if uv run --group tools python tools/generate_strategy_system_card_data.py; then
  GENERATOR_EXIT=0
else
  GENERATOR_EXIT=$?
fi
sha256sum "${OUTPUTS[@]}" > /tmp/generator-after.sha256
cmp /tmp/generator-before.sha256 /tmp/generator-after.sha256
printf 'generator_exit=%s\n' "$GENERATOR_EXIT"
```

The generator exited nonzero with its explicit missing-input list because the
source report CSVs are preserved untracked files in the original tree and are
absent from this isolated worktree. Output hashes were unchanged. Checked-in
generated assets remain available; regeneration was not an acceptance check.

## Original-tree preservation audit

The original tree was audited read-only; new artifacts were written only under
`/tmp`. This is the exact regeneration and comparison template:

```sh
AUDIT=$(mktemp -d /tmp/cryptolab-original-audit.XXXXXX)
cd "$ORIGINAL"
git status --porcelain=v1 -z > "$AUDIT/status.z"
git diff --binary --full-index > "$AUDIT/unstaged.patch"
git diff --cached --binary --full-index > "$AUDIT/staged.patch"
git ls-files --others --exclude-standard -z > "$AUDIT/untracked.z"
xargs -0 sha256sum < "$AUDIT/untracked.z" > "$AUDIT/untracked.sha256"

sha256sum "$AUDIT/status.z" "$AUDIT/unstaged.patch" \
  "$AUDIT/staged.patch" "$AUDIT/untracked.z" "$AUDIT/untracked.sha256"
for artifact in status.z unstaged.patch staged.patch untracked.z untracked.sha256; do
  cmp "$AUDIT/$artifact" "$SNAPSHOT/$artifact"
done
```

The hashes in this table are hashes of the five artifact files. In particular,
the `untracked.sha256` artifact contains one SHA-256 digest for each of the 23
untracked files; the table value is the digest of that complete hash-list file.

| Artifact file | SHA-256 |
| --- | --- |
| `status.z` | `fd25bfcd992b3324c8795d5f94deaabbdc095817ec183042861f7c1b34eba93f` |
| `unstaged.patch` | `7445a5126fff867449ab93822ab79e9a1acab1182836e028abc2ce4048acc2c6` |
| `staged.patch` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `untracked.z` | `f6d455ee10f7fcd521466f530abe66c75f37a26d6ab4165123ce85f9afab63f1` |
| `untracked.sha256` | `d29b27cedb7b91f87920e6c078b6e4e2d3ddf9f7a08806d289d9a98c53b30217` |

Every `cmp` exited 0. The original source tree is unchanged from capture: exactly
the same seven modified tracked paths, no staged changes, and the same 23
nonignored untracked files with identical contents. The snapshot remains present.

## Pending integration procedure

Because the source tree is unchanged, the applicable path is the plan's
unchanged-source integration path. It must not be run without explicit user
approval:

1. Preflight every migration target against the original untracked manifest and
   stop on any collision.
2. Create a second timestamped backup of the original status, full-index staged
   and unstaged patches, untracked manifest, and per-file untracked hashes.
3. Recompute and verify the exact seven tracked WIP paths and 23 untracked files
   against the preserved snapshot.
4. Obtain explicit user approval to integrate.
5. Treat the former staged/unstaged distinction as represented by the existing
   WIP checkpoint commit on the migration branch. In the original tree, restore
   only the seven verified tracked WIP paths, path by path, to its current `HEAD`
   so it is clean; do not use a broad checkout/reset and do not touch the verified
   untracked files. The fast-forward will then reintroduce their checkpointed
   content as committed history.
6. With the original tree in the expected clean/checkpointed state, run
   `git merge --ff-only feature/polyglot-monorepo`.
7. Re-verify the restored path set and untracked hashes after the fast-forward.

No backup, restore, merge, integration, staging, or commit was performed in the
original tree during verification.

## Intentional exclusions

Chromium was not installed, the eight Playwright functional tests remained
deselected, and no live Solana RPC, market-data, wallet, or browser action was
run. Integration remains pending explicit user direction.

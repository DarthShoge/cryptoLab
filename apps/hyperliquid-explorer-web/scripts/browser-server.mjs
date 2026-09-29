import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync, spawn } from "node:child_process";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const python = join(root, ".venv/bin/python");
const reports = mkdtempSync(join(tmpdir(), "hyperliquid-explorer-browser-"));
const lab = join(reports, "lab");
execFileSync(python, ["-c", "import sys; from pathlib import Path; sys.path.insert(0, 'apps/hyperliquid-explorer-api/tests'); from test_lab_proxy import write_proxy_fixture; write_proxy_fixture(Path(sys.argv[1]))", join(lab, "datasets", "proxy")], { cwd: root, stdio: "inherit" });
execFileSync(python, ["-c", "import sys; from pathlib import Path; sys.path.insert(0, 'apps/hyperliquid-explorer-api/tests'); from test_lab_proxy import write_scheduled_fixture; write_scheduled_fixture(Path(sys.argv[1]))", join(lab, "datasets", "scheduled")], { cwd: root, stdio: "inherit" });
execFileSync(python, ["-c", "import sys; from pathlib import Path; sys.path.insert(0, 'apps/hyperliquid-explorer-api/tests'); from test_lab_proxy import write_annual_fixture; write_annual_fixture(Path(sys.argv[1]))", join(lab, "datasets", "annual")], { cwd: root, stdio: "inherit" });
execFileSync(
  python,
  [
    "tools/generate_hyperliquid_lab_cross_class_dataset.py",
    "--output",
    join(lab, "datasets", "market_demo"),
  ],
  { cwd: root, stdio: "inherit" },
);
execFileSync(
  python,
  [
    "tools/generate_hyperliquid_lab_dataset.py",
    "--output",
    join(lab, "datasets", "demo"),
  ],
  { cwd: root, stdio: "inherit" },
);
execFileSync(
  python,
  [
    "tools/generate_hyperliquid_explorer_demo.py",
    "--output",
    join(reports, "hyperliquid_trader_ensemble_browser_demo"),
  ],
  { cwd: root, stdio: "inherit" },
);
const server = spawn(
  python,
  [
    "-m",
    "uvicorn",
    "hyperliquid_explorer_api.app:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8011",
    "--ws",
    "none",
  ],
  {
    cwd: root,
    stdio: "inherit",
    env: {
      ...process.env,
      HYPERLIQUID_REPORTS_ROOT: reports,
      HYPERLIQUID_LAB_ROOT: lab,
      HYPERLIQUID_WEB_ROOT: join(root, "apps/hyperliquid-explorer-web/dist"),
    },
  },
);
for (const signal of ["SIGTERM", "SIGINT"])
  process.on(signal, () => server.kill(signal));
server.on("exit", (code) => process.exit(code ?? 0));

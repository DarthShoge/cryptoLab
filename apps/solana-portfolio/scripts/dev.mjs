import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const python = resolve(root, ".venv/bin/python");
const api = existsSync(python)
  ? spawn(python, ["tools/run_solana_portfolio.py"], { cwd: root, stdio: "inherit" })
  : spawn("uv", ["run", "--package", "arblab", "python", "tools/run_solana_portfolio.py"], { cwd: root, stdio: "inherit" });
const ui = spawn(process.execPath, [resolve(root, "node_modules/vite/bin/vite.js"), "--host", "127.0.0.1"], {
  cwd: resolve(root, "apps/solana-portfolio"), stdio: "inherit",
});
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  api.kill("SIGTERM"); ui.kill("SIGTERM");
  process.exitCode = code;
}
for (const child of [api, ui]) {
  child.on("error", (error) => { console.error(error.message); stop(1); });
  child.on("exit", (code) => stop(code || 0));
}
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());

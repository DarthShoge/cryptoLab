import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString } from "openapi-typescript";
const root = fileURLToPath(new URL("../../../", import.meta.url));
const schema = JSON.parse(
  execFileSync(
    `${root}/.venv/bin/python`,
    ["-m", "hyperliquid_explorer_api.export_schema"],
    { cwd: root, encoding: "utf8" },
  ),
);
const result =
  "// Generated from Python OpenAPI models. Do not edit.\n" +
  astToString(await openapiTS(schema));
const target = new URL("../src/api.generated.ts", import.meta.url);
if (process.argv.includes("--check")) {
  if (readFileSync(target, "utf8") !== result)
    throw new Error("API types have drifted; run pnpm types");
} else writeFileSync(target, result);

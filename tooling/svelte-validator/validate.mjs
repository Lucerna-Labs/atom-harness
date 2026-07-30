import fs from "node:fs";
import path from "node:path";
import { compile } from "svelte/compiler";
import * as ts from "typescript";

const [componentPath, ...typescriptPaths] = process.argv.slice(2);

if (!componentPath || typescriptPaths.length === 0) {
  throw new Error("component and TypeScript paths are required");
}

const componentSource = fs.readFileSync(componentPath, "utf8");
const compiled = compile(componentSource, {
  filename: path.basename(componentPath),
  generate: "client",
  dev: false
});

const diagnostics = [];
for (const typescriptPath of typescriptPaths) {
  const source = fs.readFileSync(typescriptPath, "utf8");
  const result = ts.transpileModule(source, {
    fileName: typescriptPath,
    reportDiagnostics: true,
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      strict: true,
      noUncheckedIndexedAccess: true
    }
  });
  for (const diagnostic of result.diagnostics ?? []) {
    diagnostics.push(
      ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")
    );
  }
}

if (diagnostics.length > 0) {
  throw new Error(diagnostics.join("\n"));
}

process.stdout.write(
  JSON.stringify({
    svelteCompiled: typeof compiled.js.code === "string",
    typescriptPassed: true,
    warningCount: compiled.warnings.length,
    componentBytes: Buffer.byteLength(compiled.js.code)
  })
);

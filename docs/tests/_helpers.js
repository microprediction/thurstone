import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

export function loadFixture(name) {
  const p = resolve(__dirname, "..", "fixtures", `${name}.json`);
  return JSON.parse(readFileSync(p, "utf8"));
}

export function maxAbsDiff(a, b) {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

export function assertClose(actual, expected, tol, label) {
  const d = maxAbsDiff(Array.from(actual), Array.from(expected));
  if (d > tol) {
    throw new Error(`${label}: max abs diff ${d.toExponential(3)} exceeds tolerance ${tol.toExponential(3)}`);
  }
  return d;
}

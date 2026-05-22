import { test } from "node:test";
import { UniformLattice } from "../js/thurstone/lattice.js";
import { Density } from "../js/thurstone/density.js";
import { AbilityCalibrator } from "../js/thurstone/inference.js";
import { GlobalAbilityCalibrator } from "../js/thurstone/global_fit.js";
import { loadFixture, assertClose } from "./_helpers.js";

test("GlobalAbilityCalibrator.fitWithRebuild matches Python theta and biases", () => {
  const fx = loadFixture("global_fit");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const base = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  const cal = new AbilityCalibrator(base, { nIter: 3 });
  const ids = Object.keys(fx.theta);
  const gn = new GlobalAbilityCalibrator(ids, { l2: 1e-8, stepBias: 0.3, stepTheta: 0.3 });
  for (const r of fx.races) gn.addRace(cal, r.ids, r.prices);
  gn.fitWithRebuild({ outer: 3, inner: 10 });
  const gotTheta = ids.map(h => gn.theta[h]);
  const wantTheta = ids.map(h => fx.theta[h]);
  assertClose(gotTheta, wantTheta, fx.tolerance, "global_fit theta");
  assertClose(gn.biases, fx.biases, fx.tolerance, "global_fit biases");
});

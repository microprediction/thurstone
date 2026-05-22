import { test } from "node:test";
import { UniformLattice } from "../js/thurstone/lattice.js";
import { Density } from "../js/thurstone/density.js";
import { AbilityCalibrator } from "../js/thurstone/inference.js";
import { GlobalLSCalibrator } from "../js/thurstone/global_ls.js";
import { loadFixture, assertClose } from "./_helpers.js";

test("GlobalLSCalibrator.fit matches Python theta", () => {
  const fx = loadFixture("global_ls");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const base = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  const cal = new AbilityCalibrator(base, { nIter: 3 });
  const ids = Object.keys(fx.theta);
  const gl = new GlobalLSCalibrator(ids);
  for (const r of fx.races) gl.addRace(cal, r.ids, r.prices);
  gl.fit({ useSlopeWeights: true, ridge: 1e-6 });
  const got = ids.map(h => gl.theta[h]);
  const want = ids.map(h => fx.theta[h]);
  assertClose(got, want, fx.tolerance, "global_ls theta");
});

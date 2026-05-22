import { test } from "node:test";
import { UniformLattice } from "../js/thurstone/lattice.js";
import { Density } from "../js/thurstone/density.js";
import { loadFixture, assertClose } from "./_helpers.js";

test("skew-normal density matches Python on multiple parameter sets", () => {
  const fx = loadFixture("density_skewnormal");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  for (const c of fx.cases) {
    const d = Density.skewNormal(lat, { loc: c.params.loc, scale: c.params.scale, a: c.params.a });
    assertClose(d.p, c.p, fx.tolerance, `skewNormal ${JSON.stringify(c.params)} pdf`);
    if (Math.abs(d.mean() - c.mean) > 1e-9) throw new Error(`mean mismatch ${d.mean()} vs ${c.mean}`);
  }
});

test("shift and convolve match Python", () => {
  const fx = loadFixture("density_transforms");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const d = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  assertClose(d.shiftInteger(5).p, fx.shift_integer_5, fx.tolerance, "shiftInteger(5)");
  assertClose(d.shiftInteger(-7).p, fx.shift_integer_neg7, fx.tolerance, "shiftInteger(-7)");
  assertClose(d.shiftFractional(3.25).p, fx.shift_fractional_3p25, fx.tolerance, "shiftFractional(3.25)");
  assertClose(d.shiftFractional(-2.7).p, fx.shift_fractional_neg2p7, fx.tolerance, "shiftFractional(-2.7)");
  assertClose(d.convolve(d).p, fx.convolve_self, fx.tolerance, "convolve(self)");
  assertClose(d.dilate(2.0).p, fx.dilate_2, fx.tolerance, "dilate(2.0)");
  const meanShift = d.shiftFractional(3.25).mean();
  if (Math.abs(meanShift - fx.mean_after_shift) > 1e-9) throw new Error(`shifted mean mismatch ${meanShift} vs ${fx.mean_after_shift}`);
});

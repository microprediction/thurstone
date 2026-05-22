import { test } from "node:test";
import { UniformLattice } from "../js/thurstone/lattice.js";
import { Density } from "../js/thurstone/density.js";
import { Race } from "../js/thurstone/pricing.js";
import { winnerOfMany } from "../js/thurstone/order_stats.js";
import { loadFixture, assertClose } from "./_helpers.js";

test("winnerOfMany matches Python", () => {
  const fx = loadFixture("winner_of_many");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const base = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  const densities = fx.offsets.map(o => base.shiftFractional(o));
  const { density: d, mult } = winnerOfMany(densities);
  assertClose(d.p, fx.winner_pdf, fx.tolerance, "winnerOfMany.pdf");
  assertClose(mult, fx.multiplicity, fx.tolerance, "winnerOfMany.multiplicity");
});

test("Race.statePrices matches Python on multiple race shapes", () => {
  const fx = loadFixture("race_state_prices");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const base = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  for (const c of fx.cases) {
    const densities = c.offsets.map(o => base.shiftFractional(o));
    const prices = new Race(densities).statePrices();
    assertClose(prices, c.prices, fx.tolerance, `statePrices ${JSON.stringify(c.offsets)}`);
  }
});

import { test } from "node:test";
import { UniformLattice } from "../js/thurstone/lattice.js";
import { Density } from "../js/thurstone/density.js";
import { StatePricer } from "../js/thurstone/pricing.js";
import { AbilityCalibrator } from "../js/thurstone/inference.js";
import { loadFixture, assertClose, maxAbsDiff } from "./_helpers.js";

test("AbilityCalibrator.solveFromDividends matches Python", () => {
  const fx = loadFixture("inverse_roundtrip");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const base = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  for (const c of fx.cases) {
    const cal = new AbilityCalibrator(base, { nIter: 3 });
    const prices = StatePricer.pricesFromDividends(c.dividends);
    const dPrice = maxAbsDiff(Array.from(prices), c.prices);
    if (dPrice > 1e-12) throw new Error(`prices_from_dividends mismatch ${dPrice}`);
    const ability = cal.solveFromDividends(c.dividends);
    assertClose(ability, c.ability, fx.tolerance, `ability for div=${JSON.stringify(c.dividends)}`);
  }
});

test("AbilityCalibrator.statePricesFromAbility matches Python (incl. cluster splitter)", () => {
  const fx = loadFixture("state_prices_from_ability");
  const lat = new UniformLattice(fx.lattice.L, fx.lattice.unit);
  const base = Density.skewNormal(lat, { loc: 0.0, scale: 1.0, a: 0.0 });
  const cal = new AbilityCalibrator(base);
  for (const c of fx.cases) {
    const prices = cal.statePricesFromAbility(c.ability);
    assertClose(prices, c.prices, fx.tolerance, `prices for ability=${JSON.stringify(c.ability)}`);
  }
});

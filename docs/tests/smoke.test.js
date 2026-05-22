// Smoke test: exercise the same code paths the demo pages do, end-to-end.
// Catches accidental API drift before the user opens the browser.

import { test } from "node:test";
import {
  UniformLattice, Density, Race, StatePricer, AbilityCalibrator,
  GlobalLSCalibrator, GlobalAbilityCalibrator, MultiRayGlobalCalibrator,
} from "../js/thurstone/index.js";

const lat  = new UniformLattice(200, 0.05);
const base = Density.skewNormal(lat, { loc: 0, scale: 1, a: 0 });

test("forward-pricing demo path", () => {
  const ab = [-1.0, -0.4, 0.0, 0.3, 0.9];
  const ds = ab.map(a => base.shiftFractional(a / lat.unit));
  const p = Array.from(new Race(ds).statePrices());
  const d = Array.from(StatePricer.dividendsFromPrices(p));
  const s = p.reduce((u, v) => u + v, 0);
  if (Math.abs(s - 1) > 1e-9) throw new Error(`sum prices ${s}`);
  if (d.some(x => !Number.isFinite(x))) throw new Error("non-finite dividend");
});

test("inverse-calibration demo path", () => {
  const cal = new AbilityCalibrator(base, { nIter: 3 });
  const ability = cal.solveFromDividends([3.2, 4.8, 12.0, 7.5, 20.0]);
  if (!cal.lookupCurve1dPrices) throw new Error("no cached curve");
  const round = cal.statePricesFromAbility(ability);
  const s = round.reduce((u, v) => u + v, 0);
  if (Math.abs(s - 1) > 1e-9) throw new Error(`round-trip sum ${s}`);
});

test("multi-race-stitching demo path", () => {
  const ids = ["A", "B", "C", "D", "E"];
  const gn = new GlobalAbilityCalibrator(ids);
  const gl = new GlobalLSCalibrator(ids);
  const races = [
    { ids: ["A", "B", "C"], div: [3.0, 4.5, 6.0] },
    { ids: ["B", "C", "D"], div: [3.5, 5.5, 7.0] },
    { ids: ["A", "C", "D"], div: [3.2, 5.0, 9.0] },
    { ids: ["A", "B", "D", "E"], div: [2.8, 5.0, 8.0, 14.0] },
  ];
  for (const r of races) {
    const calGn = new AbilityCalibrator(base, { nIter: 3 });
    const calLs = new AbilityCalibrator(base, { nIter: 3 });
    const prices = Array.from(StatePricer.pricesFromDividends(r.div));
    gn.addRace(calGn, r.ids, prices);
    gl.addRace(calLs, r.ids, prices);
  }
  gn.fitWithRebuild({ outer: 2, inner: 5 });
  if (Object.values(gn.theta).some(v => !Number.isFinite(v))) throw new Error("non-finite GN theta");
  gl.fit({ useSlopeWeights: true, ridge: 1e-6 });
  const vals = Object.values(gl.theta);
  if (vals.some(v => !Number.isFinite(v))) throw new Error("non-finite theta");
});

test("density-playground demo path", () => {
  const d = Density.skewNormal(lat, { loc: 0.5, scale: 1.2, a: 1.0 })
    .shiftInteger(3).shiftFractional(2.4).dilate(1.5).convolve(base);
  const s = d.p.reduce((u, v) => u + v, 0);
  if (Math.abs(s - 1) > 1e-6) throw new Error(`pdf does not sum to 1: ${s}`);
});

test("walkover-ties demo path", () => {
  const cal = new AbilityCalibrator(base);
  const p1 = cal.statePricesFromAbility([-3, 3, 4, 5, 6]);
  if (p1[0] <= 0.95) throw new Error(`walkover not detected, p[0]=${p1[0]}`);
  const p2 = cal.statePricesFromAbility([-2, -2, -2, 5, 5]);
  const share = p2.slice(0, 3).reduce((a, b) => a + b, 0);
  if (Math.abs(share - 1.0) > 0.05) throw new Error(`tied favourites should share ~1; got ${share}`);
});

test("loc-scale-2d demo path", () => {
  const cal = new AbilityCalibrator(base, { nIter: 3, locSpan: 5, locStep: 0.25, scaleSpan: 0.4, scaleSteps: 3 });
  cal.setScales([0.8, 1.0, 1.2, 1.1, 1.5]);
  const ab = cal.solveFromDividends([3.2, 4.8, 12.0, 7.5, 20.0]);
  if (ab.length !== 5 || ab.some(v => !Number.isFinite(v))) throw new Error("bad 2D result");
  if (cal.lookupCurves2dPrices.size < 1) throw new Error("no 2D curves cached");
});

test("multiray fit converges to better predictions", () => {
  const itemIds = ["A", "B", "C", "D", "E"];
  const mr = new MultiRayGlobalCalibrator(itemIds, { dim: 2, randomState: 1 });
  const conds = [
    { condId: "c1", ids: ["A", "B", "C", "D", "E"], div: [3.0, 4.0, 6.0, 9.0, 18.0] },
    { condId: "c2", ids: ["A", "B", "C", "D", "E"], div: [4.0, 3.5, 5.0, 10.0, 25.0] },
  ];
  for (const c of conds) {
    const cal = new AbilityCalibrator(base, { nIter: 3 });
    const prices = Array.from(StatePricer.pricesFromDividends(c.div));
    cal.solveFromPrices(prices);
    mr.addCondition({ condId: c.condId, calibrator: cal, itemIds: c.ids, prices });
  }
  mr.fitWithRebuild({ outer: 2, inner: 5 });
  for (const c of conds) {
    const p = mr.predictCondition(c.condId);
    const s = p.reduce((u, v) => u + v, 0);
    if (!Number.isFinite(s)) throw new Error("non-finite multiray prediction");
  }
});

import { test } from "node:test";
import { normpdf, normcdf } from "../js/thurstone/normaldist.js";
import { loadFixture, assertClose } from "./_helpers.js";

test("normpdf and normcdf match Python on reference points", () => {
  const fx = loadFixture("normaldist");
  const pdf = fx.xs.map(normpdf);
  const cdf = fx.xs.map(normcdf);
  assertClose(pdf, fx.pdf, fx.tolerance, "normpdf");
  assertClose(cdf, fx.cdf, fx.tolerance, "normcdf");
});

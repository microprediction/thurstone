# thurstone — GitHub Pages site

Static, dependency-free site (vanilla HTML + ES modules) that ships an educational
introduction to Thurstone models plus six interactive in-browser demos powered by a
JavaScript port of the Python analytics in `thurstone/`.

## Local preview

```
python -m http.server 8765 --directory docs
open http://127.0.0.1:8765/
```

No build step. Each demo imports JS modules directly via `<script type="module">`.

## Enabling on GitHub Pages

In the repo settings (`Settings → Pages`):

1. Source: **Deploy from a branch**
2. Branch: **main** / folder: **/docs**

`docs/.nojekyll` is present so GitHub Pages serves files verbatim instead of running
Jekyll.

## Cross-verification (JS vs Python)

The JS port (`docs/js/thurstone/`) is verified against the Python implementation using
JSON golden fixtures committed to `docs/fixtures/`.

Regenerate the fixtures from Python (after any analytics change):

```
python scripts/generate_fixtures.py
```

Run the JS test suite (Node 18+ for the built-in test runner):

```
cd docs && node --test 'tests/*.test.js'
```

Tests cover:
- `normpdf` / `normcdf` (1e-6)
- Skew-normal density, integer/fractional shifts, convolution, dilation (1e-10)
- `winnerOfMany` + multiplicity, `Race.statePrices` (1e-8)
- `AbilityCalibrator.solveFromDividends` and `statePricesFromAbility` (1e-6)
- `GlobalLSCalibrator.fit` over overlapping races (5e-4)
- End-to-end demo paths (smoke tests, no Python comparison)

## Folder structure

```
docs/
  index.html                   # landing page
  pages/                       # three educational pages
  demos/                       # six interactive demos
  js/thurstone/                # JS port of the Python package
  styles/main.css
  fixtures/*.json              # Python-generated golden values
  tests/*.test.js              # node:test cross-tests
  package.json                 # `npm test` shortcut
  .nojekyll
  README.md                    # (you are here)
```

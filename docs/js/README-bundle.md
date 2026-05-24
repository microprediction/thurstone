# Thurstone Bundle

The `thurstone-bundle.js` file contains all the thurstone library functions needed by the demos, bundled into a single JavaScript file that works without ES module imports.

## Why This Bundle?

ES modules don't work with the `file://` protocol in browsers, which means demos fail when opened directly from the filesystem. This bundle solves that problem by:

1. Combining all necessary classes and functions into one file
2. Making everything available as global variables
3. Working with regular `<script src="">` tags
4. Supporting the `file://` protocol

## Usage

Instead of ES module imports:
```javascript
// ❌ This doesn't work with file:// protocol
import { UniformLattice, Density, Race } from "../js/thurstone/index.js";
```

Use the bundle:
```html
<!-- ✅ This works with file:// protocol -->
<script src="../js/thurstone-bundle.js"></script>
<script>
  // Classes and functions are available globally
  const lattice = new UniformLattice(300, 0.05);
  const density = Density.skewNormal(lattice);
  const race = new Race([density]);
</script>
```

## Available Classes

- `UniformLattice` - The index-grid abstraction
- `Density` - Lattice-aligned probability density functions  
- `Race` - Multi-entrant race simulation
- `AbilityCalibrator` - Inverse calibration from prices/dividends
- `StatePricer` - Price/dividend conversion utilities
- `ClusterSplitter` - Handles edge cases and walkovers

## Available Functions

- `normpdf(x)` - Normal probability density function
- `normcdf(x)` - Normal cumulative distribution function  
- `erf(x)` - Error function
- `erfc(x)` - Complementary error function
- Various helper functions: `sumArr`, `interp`, `median`, etc.

## Available Constants

- `STD_L`, `STD_UNIT`, `STD_SCALE`, `STD_A` - Standard lattice defaults
- `ALT_L`, `ALT_UNIT`, `ALT_SCALE`, `ALT_A` - Alternative lattice defaults
- `NAN_DIVIDEND` - Default value for NaN dividends

## Testing

Run `test-bundle.html` to verify the bundle is working correctly. It performs basic tests on all major classes and functions.

## Updated Demo Files

The following files have been updated to use the bundle:
- `pages/multi-dimensional.html` - Interactive 2D demo
- `pages/fast-transform.html` - Code examples updated
- `test-demo-import.html` - Import test updated

## File Structure

```
docs/
├── js/
│   ├── thurstone-bundle.js     # ← The bundle (this file)
│   └── thurstone/              # ← Original ES modules
│       ├── index.js
│       ├── lattice.js
│       ├── density.js
│       ├── pricing.js
│       ├── inference.js
│       └── ...
├── test-bundle.html            # ← Test the bundle
└── pages/
    └── multi-dimensional.html  # ← Updated demo
```
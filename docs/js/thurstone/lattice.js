// Port of thurstone/lattice.py
// UniformLattice is the index-grid abstraction shared by every Density.

export class UniformLattice {
  constructor(L, unit) {
    this.L = L;
    this.unit = unit;
  }

  get size() { return 2 * this.L + 1; }

  grid() {
    const n = this.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.unit * (i - this.L);
    return out;
  }

  indexGrid() {
    const n = this.size;
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = i - this.L;
    return out;
  }

  assertCompatible(arr) {
    if (arr.length !== this.size) {
      throw new Error(`Array length ${arr.length} incompatible with lattice size ${this.size}.`);
    }
  }
}

// Standard lattice defaults (mirrors thurstone/conventions.py).
export const STD_L = 500;
export const STD_UNIT = 0.1;
export const STD_SCALE = 1.0;
export const STD_A = 0.0;
export const ALT_L = 500;
export const ALT_UNIT = 0.1;
export const ALT_SCALE = 1.0;
export const ALT_A = 0.5;
export const NAN_DIVIDEND = 2000.0;

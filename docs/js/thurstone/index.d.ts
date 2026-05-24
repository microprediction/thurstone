/**
 * TypeScript definitions for thurstone JavaScript modules
 *
 * These definitions provide type safety for the JavaScript port of the
 * thurstone Python package, enabling TypeScript development and better
 * IDE support for JavaScript users.
 */

// ================================
// Lattice Module
// ================================

export class UniformLattice {
    readonly L: number;
    readonly unit: number;

    constructor(L: number, unit: number);

    get size(): number;

    grid(): Float64Array;
    indexGrid(): Int32Array;
    assertCompatible(arr: ArrayLike<number>): void;
}

// Lattice constants
export const STD_L: 500;
export const STD_UNIT: 0.1;
export const STD_SCALE: 1.0;
export const STD_A: 0.0;
export const ALT_L: 500;
export const ALT_UNIT: 0.1;
export const ALT_SCALE: 1.0;
export const ALT_A: 0.5;
export const NAN_DIVIDEND: 2000.0;

// ================================
// Normal Distribution Module
// ================================

export function normpdf(x: number, mu?: number, sigma?: number): number;
export function normcdf(x: number, mu?: number, sigma?: number): number;
export function norminv(p: number, mu?: number, sigma?: number): number;

// ================================
// Density Module
// ================================

export function cdfFromPdf(pdf: ArrayLike<number>): Float64Array;
export function pdfFromCdf(cdf: ArrayLike<number>): Float64Array;
export function sumArr(a: ArrayLike<number>): number;
export function normalizePdf(pdf: ArrayLike<number>): Float64Array;

export class Density {
    readonly lattice: UniformLattice;
    readonly p: Float64Array;

    constructor(lattice: UniformLattice, p: ArrayLike<number>);

    cdf(): Float64Array;
    mean(): number;
    approxSupport(tol?: number): number[];
    approxSupportWidth(tol?: number): number;

    // Transformations
    shiftInteger(k: number): Density;
    shiftFractional(offset: number): Density;
    dilate(scale: number): Density;
    convolve(other: Density): Density;

    // Static factory methods
    static standardNormal(lattice: UniformLattice): Density;
    static skewNormal(lattice: UniformLattice, a?: number, scale?: number): Density;
    static fromAbility(lattice: UniformLattice, ability: number, scale?: number, a?: number): Density;
}

// ================================
// Order Statistics Module
// ================================

export interface WinnerResult {
    density: Density;
    mult: Float64Array;
}

export function winnerOfMany(densities: Density[]): WinnerResult;
export function expectedPayoffWithMultiplicity(
    selfDensity: Density,
    allDensity: Density,
    allMult: Float64Array,
    tieModel?: any,
    allCdf?: Float64Array
): Float64Array;

// ================================
// Pricing Module
// ================================

export function cdfMin(densities: Density[]): Float64Array;
export function winnerDensity(densities: Density[]): Density;

export class Race {
    readonly densities: Density[];

    constructor(densities: Density[]);

    winnerDensity(): Density;
    statePrices(): Float64Array;
    dividends(): Float64Array;
}

export class StatePricer {
    readonly baseDensity: Density;
    readonly offsets: number[];

    constructor(baseDensity: Density, offsets: number[]);

    prices(): Float64Array;
    dividends(): Float64Array;
    densities(): Density[];
}

// ================================
// Clustering Module
// ================================

export class ClusterSplitter {
    constructor(tieModel?: any);

    cluster(densities: Density[], tolerance?: number): Density[];
    split(clusterDensity: Density, count: number): Density[];
}

// ================================
// Inference Module
// ================================

export interface CalibrationOptions {
    offsetGrid?: number[];
    nIter?: number;
    scales?: number[] | null;
    scaleSpan?: number;
    scaleSteps?: number;
    locSpan?: number;
    locStep?: number;
    skewA?: number;
}

export class AbilityCalibrator {
    readonly baseDensity: Density;
    readonly options: CalibrationOptions;

    constructor(baseDensity: Density, options?: CalibrationOptions);

    calibrate(statePrices: number[]): Float64Array;
    calibrate2D(statePrices: number[]): {
        locations: Float64Array;
        scales: Float64Array;
    };

    // Helper methods
    implicitStatePrices(offsets: number[]): number[];
    densitiesFromOffsets(offsets: number[]): Density[];
}

// ================================
// Global Calibration Modules
// ================================

export interface RaceData {
    contestantIds: number[];
    statePrices: number[];
    [key: string]: any;
}

export class GlobalLSCalibrator {
    readonly baseDensity: Density;

    constructor(baseDensity: Density, options?: any);

    fitRaces(raceData: RaceData[]): Float64Array;
    getAbilities(): Float64Array | null;
    reset(): void;
}

export class GlobalAbilityCalibrator {
    readonly baseDensity: Density;

    constructor(baseDensity: Density, options?: any);

    fitRaces(raceData: RaceData[]): Float64Array;
    addRace(race: RaceData): void;
    solve(): Float64Array;
    getAbilities(): Float64Array | null;
}

// ================================
// Multi-Ray Global Calibration
// ================================

export interface ConditionSpec {
    [key: string]: any;
}

export class MultiRayGlobalCalibrator {
    readonly baseDensities: { [condition: string]: Density };

    constructor(baseDensities: { [condition: string]: Density }, options?: any);

    fitRaces(raceData: Array<RaceData & { condition?: string }>): Float64Array;
    getAbilities(): Float64Array | null;
    getConditionEffects(): { [condition: string]: number } | null;
}

// ================================
// Utility Types
// ================================

export type ArrayLike<T> = T[] | Float64Array | Float32Array | Uint8Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Int32Array;

export interface LatticeConventions {
    L: number;
    unit: number;
    scale: number;
    a: number;
}

export const STANDARD_CONVENTIONS: LatticeConventions;
export const ALTERNATIVE_CONVENTIONS: LatticeConventions;

// ================================
// Main Export Types
// ================================

declare const thurstone: {
    // Classes
    UniformLattice: typeof UniformLattice;
    Density: typeof Density;
    Race: typeof Race;
    StatePricer: typeof StatePricer;
    ClusterSplitter: typeof ClusterSplitter;
    AbilityCalibrator: typeof AbilityCalibrator;
    GlobalLSCalibrator: typeof GlobalLSCalibrator;
    GlobalAbilityCalibrator: typeof GlobalAbilityCalibrator;
    MultiRayGlobalCalibrator: typeof MultiRayGlobalCalibrator;

    // Utility Functions
    normpdf: typeof normpdf;
    normcdf: typeof normcdf;
    norminv: typeof norminv;
    cdfFromPdf: typeof cdfFromPdf;
    pdfFromCdf: typeof pdfFromCdf;
    sumArr: typeof sumArr;
    normalizePdf: typeof normalizePdf;
    cdfMin: typeof cdfMin;
    winnerDensity: typeof winnerDensity;
    winnerOfMany: typeof winnerOfMany;
    expectedPayoffWithMultiplicity: typeof expectedPayoffWithMultiplicity;

    // Constants
    STD_L: typeof STD_L;
    STD_UNIT: typeof STD_UNIT;
    STD_SCALE: typeof STD_SCALE;
    STD_A: typeof STD_A;
    ALT_L: typeof ALT_L;
    ALT_UNIT: typeof ALT_UNIT;
    ALT_SCALE: typeof ALT_SCALE;
    ALT_A: typeof ALT_A;
    NAN_DIVIDEND: typeof NAN_DIVIDEND;
};

export default thurstone;
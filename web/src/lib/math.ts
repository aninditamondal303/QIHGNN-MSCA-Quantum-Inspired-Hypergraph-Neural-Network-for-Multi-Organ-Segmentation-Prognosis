/**
 * TypeScript ports of the formulas the verification lab runs in the browser.
 * They mirror ml/qihgnn/reference.py and ml/qihgnn/param_count.py and are tested
 * against fixtures written by Python (tests/math.test.ts), so both languages agree.
 */

const EPS_DICE = 1e-5;
const EPS_LOG = 1e-8;

export type Matrix = number[][]; // [channel][voxel]

export function mergeUnsupervised(p: Matrix, supervised: boolean[]): Matrix {
  const sup = supervised.map((s, i) => (i === 0 ? true : s));
  const v = p[0].length;
  const merged = p.map((row, c) => (sup[c] ? row.slice() : new Array<number>(v).fill(0)));
  for (let c = 0; c < p.length; c++) {
    if (sup[c]) continue;
    for (let i = 0; i < v; i++) merged[0][i] += p[c][i];
  }
  return merged;
}

function diceTerm(p: number[], labels: number[], c: number): number {
  let inter = 0;
  let ps = 0;
  let gs = 0;
  for (let i = 0; i < p.length; i++) {
    const g = labels[i] === c ? 1 : 0;
    inter += p[i] * g;
    ps += p[i];
    gs += g;
  }
  return (2 * inter + EPS_DICE) / (ps + gs + EPS_DICE);
}

function crossEntropy(p: Matrix, labels: number[]): number {
  let s = 0;
  for (let i = 0; i < labels.length; i++) s -= Math.log(Math.max(p[labels[i]][i], EPS_LOG));
  return s / labels.length;
}

/** Eq. 4.10 on the merged distribution, Dice over supervised channels plus background. */
export function marginalLoss(p: Matrix, labels: number[], supervised: boolean[]): number {
  const sup = supervised.map((s, i) => (i === 0 ? true : s));
  const y = labels.map((l) => (sup[l] ? l : 0));
  const m = mergeUnsupervised(p, sup);
  const channels = sup.flatMap((s, c) => (s ? [c] : []));
  const dice = channels.reduce((acc, c) => acc + diceTerm(m[c], y, c), 0) / channels.length;
  return 1 - dice + crossEntropy(m, y);
}

/** The same objective on the unmerged distribution over every channel. */
export function naiveLoss(p: Matrix, labels: number[]): number {
  const dice = p.reduce((acc, _row, c) => acc + diceTerm(p[c], labels, c), 0) / p.length;
  return 1 - dice + crossEntropy(p, labels);
}

/**
 * Move a fraction `t` of all unsupervised mass into channel `target`, voxel by voxel.
 * t = 0 returns p unchanged; t = 1 concentrates every unsupervised channel's mass into `target`.
 * Each column still sums to 1 and supervised channels are untouched.
 */
export function reallocate(p: Matrix, supervised: boolean[], target: number, t: number): Matrix {
  const v = p[0].length;
  const out = p.map((row) => row.slice());
  const unsup = supervised.flatMap((s, c) => (!s && c !== 0 ? [c] : []));
  for (let i = 0; i < v; i++) {
    let total = 0;
    for (const c of unsup) total += p[c][i];
    for (const c of unsup) {
      const goal = c === target ? total : 0;
      out[c][i] = (1 - t) * p[c][i] + t * goal;
    }
  }
  return out;
}

/**
 * On voxels labelled background, hand a fraction `t` of the background probability to `target`:
 * the model recognising an organ that this sample's dataset never annotated.
 */
export function shiftToChannel(p: Matrix, labels: number[], target: number, t: number): Matrix {
  const out = p.map((row) => row.slice());
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== 0) continue;
    const delta = t * p[0][i];
    out[0][i] -= delta;
    out[target][i] += delta;
  }
  return out;
}

// --------------------------------------------------------------------------- cost formulas
export function attentionCost(shape: [number, number, number], budgets: number[] = [256, 32, 4, 1]) {
  const voxels = shape[0] * shape[1] * shape[2];
  const multi = budgets.reduce((a, n) => a + n * n, 0);
  const naive = voxels * voxels; // exact below 2^53; above that a double is still accurate to ~1e-16 relative
  return { voxels, multi_scale: multi, naive, reduction: naive / multi };
}

export function mpoParameters(bond: number, inFactors: number[] = [16, 16], outFactors: number[] = [16, 16]): number {
  const d = inFactors.length;
  const ranks = [1, ...new Array(d - 1).fill(bond), 1];
  let total = 0;
  for (let k = 0; k < d; k++) total += ranks[k] * inFactors[k] * outFactors[k] * ranks[k + 1];
  return total;
}

export function mpoRow(bond: number) {
  const dense = 256 * 256;
  const parameters = mpoParameters(bond);
  return { bond, parameters, dense, compression: dense / parameters };
}

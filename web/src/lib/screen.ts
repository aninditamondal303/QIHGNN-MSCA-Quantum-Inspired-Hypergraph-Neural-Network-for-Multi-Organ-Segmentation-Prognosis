import { BUILD } from './build';
/* In-browser screening: everything runs on this device through ONNX Runtime Web (WebAssembly).
   The preprocessing below mirrors ml/qihgnn/screen/common.py exactly; tests/screen.test.ts checks it. */

export const CLASSES = ['no_tumour', 'glioma', 'meningioma', 'pituitary', 'not_brain_mri'] as const;
export type ClassName = (typeof CLASSES)[number];
export const CLASS_LABEL: Record<ClassName, string> = {
  no_tumour: 'No tumour',
  glioma: 'Glioma',
  meningioma: 'Meningioma',
  pituitary: 'Pituitary tumour',
  not_brain_mri: 'Not a brain MRI',
};
export const CLS_SIZE = 224;
export const SEG_SIZE = 192;

export interface Gray {
  data: Uint8Array; // row-major, 0..255
  w: number;
  h: number;
}

export interface Letterbox {
  scale: number;
  top: number;
  left: number;
  h: number;
  w: number;
}

// ---------------------------------------------------------------- image helpers
export function letterboxParams(w: number, h: number, size: number): Letterbox {
  const scale = size / Math.max(h, w);
  const nh = Math.max(1, Math.round(h * scale));
  const nw = Math.max(1, Math.round(w * scale));
  return { scale, top: Math.floor((size - nh) / 2), left: Math.floor((size - nw) / 2), h: nh, w: nw };
}

/** Letterbox a grey image to size x size with bilinear resampling (canvas), zero padding. */
export function letterbox(g: Gray, size: number): { pixels: Uint8Array; lb: Letterbox } {
  const lb = letterboxParams(g.w, g.h, size);
  const src = document.createElement('canvas');
  src.width = g.w;
  src.height = g.h;
  const sctx = src.getContext('2d')!;
  const img = sctx.createImageData(g.w, g.h);
  for (let i = 0; i < g.data.length; i++) {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g.data[i];
    img.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  const dst = document.createElement('canvas');
  dst.width = dst.height = size;
  const dctx = dst.getContext('2d')!;
  dctx.fillStyle = '#000';
  dctx.fillRect(0, 0, size, size);
  dctx.imageSmoothingEnabled = true;
  dctx.imageSmoothingQuality = 'high';
  dctx.drawImage(src, lb.left, lb.top, lb.w, lb.h);
  const out = dctx.getImageData(0, 0, size, size).data;
  const pixels = new Uint8Array(size * size);
  for (let i = 0; i < pixels.length; i++) pixels[i] = out[i * 4];
  return { pixels, lb };
}

export function percentile(sorted: Float32Array | number[], q: number): number {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Segmenter input: scale by the 1st-99th percentile of non-zero pixels (numpy 'linear' percentile). */
export function segNormalise(px: ArrayLike<number>): Float32Array {
  const nz: number[] = [];
  for (let i = 0; i < px.length; i++) if (px[i] > 0) nz.push(px[i]);
  const out = new Float32Array(px.length);
  if (nz.length < 16) return out;
  nz.sort((a, b) => a - b);
  const lo = percentile(nz, 0.01);
  const hi = percentile(nz, 0.99);
  const span = Math.max(hi - lo, 1);
  for (let i = 0; i < px.length; i++) out[i] = Math.min(1, Math.max(0, (px[i] - lo) / span));
  return out;
}

/** MRI slice of any range -> 0..255 by the 1st-99th percentile of its non-zero voxels; zeros stay zero. */
export function toUint8(slice: ArrayLike<number>): Uint8Array {
  const nz: number[] = [];
  for (let i = 0; i < slice.length; i++) if (slice[i] > 0) nz.push(slice[i]);
  const out = new Uint8Array(slice.length);
  if (nz.length < 16) return out;
  nz.sort((a, b) => a - b);
  const lo = percentile(nz, 0.01);
  let hi = percentile(nz, 0.99);
  if (hi <= lo) hi = lo + 1;
  for (let i = 0; i < slice.length; i++) {
    if (slice[i] > 0) out[i] = Math.floor(Math.min(1, Math.max(0, (slice[i] - lo) / (hi - lo))) * 255);
  }
  return out;
}

export async function fileToGray(file: Blob): Promise<Gray> {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const data = new Uint8Array(c.width * c.height);
  // ITU-R 601 luma, the same weights Pillow uses for convert("L").
  for (let i = 0; i < data.length; i++) data[i] = Math.round((d[i * 4] * 299 + d[i * 4 + 1] * 587 + d[i * 4 + 2] * 114) / 1000);
  return { data, w: c.width, h: c.height };
}

// ---------------------------------------------------------------- NIfTI-1 reader
export interface Volume {
  data: Float32Array; // x fastest, then y, then z
  nx: number;
  ny: number;
  nz: number;
  spacing: [number, number, number]; // mm
  name: string;
}

export async function readNifti(file: File): Promise<Volume> {
  let buf = await file.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    buf = await new Response(stream).arrayBuffer();
  }
  let v = new DataView(buf);
  let le = true;
  if (v.getInt32(0, true) !== 348) {
    if (v.getInt32(0, false) !== 348) throw new Error('Not a NIfTI-1 file (header size is not 348).');
    le = false;
  }
  const dim = Array.from({ length: 8 }, (_, i) => v.getInt16(40 + 2 * i, le));
  const datatype = v.getInt16(70, le);
  const pixdim = Array.from({ length: 8 }, (_, i) => v.getFloat32(76 + 4 * i, le));
  const voxOffset = Math.max(352, Math.round(v.getFloat32(108, le)));
  let slope = v.getFloat32(112, le);
  const inter = v.getFloat32(116, le);
  if (!slope) slope = 1;
  const [nx, ny, nz] = [dim[1], dim[2], Math.max(1, dim[3])];
  const n = nx * ny * nz;
  const data = new Float32Array(n);
  v = new DataView(buf, voxOffset);
  const readers: Record<number, [number, (o: number) => number]> = {
    2: [1, (o) => v.getUint8(o)],
    4: [2, (o) => v.getInt16(o, le)],
    8: [4, (o) => v.getInt32(o, le)],
    16: [4, (o) => v.getFloat32(o, le)],
    64: [8, (o) => v.getFloat64(o, le)],
    256: [1, (o) => v.getInt8(o)],
    512: [2, (o) => v.getUint16(o, le)],
    768: [4, (o) => v.getUint32(o, le)],
  };
  const r = readers[datatype];
  if (!r) throw new Error(`Unsupported NIfTI data type ${datatype}.`);
  if (dim[0] > 3 && dim[4] > 1) console.warn('4D volume: using the first frame only.');
  for (let i = 0; i < n; i++) data[i] = r[1](i * r[0]) * slope + inter;
  return { data, nx, ny, nz, spacing: [Math.abs(pixdim[1]) || 1, Math.abs(pixdim[2]) || 1, Math.abs(pixdim[3]) || 1], name: file.name };
}

/** Axial slice z as a grey image (rows = y, columns = x), scaled like a viewer export. */
export function volumeSlice(vol: Volume, z: number): Gray {
  const { nx, ny } = vol;
  const raw = vol.data.subarray(z * nx * ny, (z + 1) * nx * ny);
  // MRI volumes can have negative values after bias correction; shift so background stays the minimum.
  let min = Infinity;
  for (let i = 0; i < raw.length; i++) if (raw[i] < min) min = raw[i];
  const shifted = min < 0 ? raw.map((x) => x - min) : raw;
  return { data: toUint8(shifted), w: nx, h: ny };
}

// ---------------------------------------------------------------- ONNX Runtime
type Ort = typeof import('onnxruntime-web/wasm');
export type Session = Awaited<ReturnType<Ort['InferenceSession']['create']>>;
let ortPromise: Promise<Ort> | null = null;
const sessions: Record<string, Promise<Session>> = {};

export function loadOrt(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = import('onnxruntime-web/wasm').then((ort) => {
      ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;
      ort.env.wasm.numThreads = 1; // GitHub Pages cannot send the cross-origin isolation headers threads need
      return ort;
    });
  }
  return ortPromise;
}

type ModelName = 'classifier' | 'segmenter' | 'liver' | 'liver_tumour' | 'cervix' | 'cells' | 'net3d';
type Feed = { data: Float32Array; dims: readonly number[] };
/** What the call sites need from a session: run(feeds) -> outputs with .data. */
export interface ModelRunner {
  run(feeds: Record<string, Feed>): Promise<Record<string, { data: Float32Array; dims: readonly number[] }>>;
}

// Models run in a Web Worker so the page stays responsive; if a worker cannot start, they run on the page itself.
let worker: Worker | null | undefined;
let nextId = 1;
const pending = new Map<number, { resolve: (v: Record<string, { data: Float32Array; dims: number[] }>) => void; reject: (e: Error) => void }>();

function getWorker(): Worker | null {
  if (worker !== undefined) return worker;
  try {
    worker = new Worker(new URL(`assets/infer.js?v=${encodeURIComponent(BUILD.time)}`, document.baseURI), { type: 'module' });
    worker.onmessage = (e: MessageEvent<{ id: number; ok: boolean; out?: Record<string, { data: Float32Array; dims: number[] }>; error?: string }>) => {
      const p = pending.get(e.data.id);
      if (!p) return;
      pending.delete(e.data.id);
      if (e.data.ok && e.data.out) p.resolve(e.data.out);
      else p.reject(new Error(e.data.error || 'inference failed'));
    };
    worker.onerror = () => {
      // The worker could not load: fall back to the page thread for every waiting and future request.
      worker = null;
      for (const [, p] of pending) p.reject(new Error('worker-unavailable'));
      pending.clear();
    };
  } catch {
    worker = null;
  }
  return worker;
}

function directSession(name: ModelName): Promise<Session> {
  if (!sessions[name]) {
    sessions[name] = loadOrt().then((ort) => ort.InferenceSession.create(new URL(`screen/${name}.onnx`, document.baseURI).href, { executionProviders: ['wasm'] }));
  }
  return sessions[name];
}

async function runDirect(name: ModelName, feeds: Record<string, Feed>) {
  const ort = await loadOrt();
  const s = await directSession(name);
  const inputs: Record<string, InstanceType<Ort['Tensor']>> = {};
  for (const [k, f] of Object.entries(feeds)) inputs[k] = new ort.Tensor('float32', f.data, [...f.dims]);
  const out = await s.run(inputs);
  const res: Record<string, { data: Float32Array; dims: readonly number[] }> = {};
  for (const [k, t] of Object.entries(out)) res[k] = { data: t.data as Float32Array, dims: t.dims };
  return res;
}

export function session(name: ModelName): Promise<ModelRunner> {
  return Promise.resolve({
    run: async (feeds: Record<string, Feed>) => {
      const w = getWorker();
      if (w) {
        const plain: Record<string, { data: Float32Array; dims: number[] }> = {};
        for (const [k, f] of Object.entries(feeds)) plain[k] = { data: f.data, dims: [...f.dims] };
        try {
          return await new Promise<Record<string, { data: Float32Array; dims: number[] }>>((resolve, reject) => {
            const id = nextId++;
            pending.set(id, { resolve, reject });
            // Large inputs (the 3D network's 16 MB volume) are handed over, not copied; small ones are copied.
            const big = Object.values(plain).filter((f) => f.data.byteLength > 4_000_000).map((f) => f.data.buffer);
            w.postMessage({ id, base: new URL('.', document.baseURI).href, model: name, feeds: plain }, big);
          });
        } catch (e) {
          if ((e as Error).message !== 'worker-unavailable') throw e;
        }
      }
      return runDirect(name, feeds);
    },
  });
}

export interface Classification {
  probs: number[]; // calibrated, CLASSES order
  cams: Float32Array; // 5 x 7 x 7
  camSize: number;
  lb: Letterbox;
  input: Uint8Array; // letterboxed 224 x 224, for display
}

export async function classify(g: Gray): Promise<Classification> {
  const ort = await loadOrt();
  const s = await session('classifier');
  const { pixels, lb } = letterbox(g, CLS_SIZE);
  const x = new Float32Array(pixels.length);
  for (let i = 0; i < x.length; i++) x[i] = pixels[i] / 255;
  const out = await s.run({ image: new ort.Tensor('float32', x, [1, 1, CLS_SIZE, CLS_SIZE]) });
  const cams = out.cams.data as Float32Array;
  return { probs: Array.from(out.probs.data as Float32Array), cams, camSize: out.cams.dims[3] as number, lb, input: pixels };
}

/** Let the browser paint and handle input between inference batches. ONNX Runtime runs on the page's own thread, and
    awaiting it does not by itself give the browser a turn, so long 3D jobs would otherwise freeze the page. */
export function yieldToBrowser(): Promise<void> {
  // A frame lets the progress text paint; the timeout guarantees progress when no frames come (hidden or occluded tab).
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (!done) {
        done = true;
        resolve();
      }
    };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => setTimeout(finish, 0));
    setTimeout(finish, 50);
  });
}

export async function segmentBatch(slices: Uint8Array[]): Promise<Float32Array[]> {
  const ort = await loadOrt();
  const s = await session('segmenter');
  const n = slices.length;
  const x = new Float32Array(n * SEG_SIZE * SEG_SIZE);
  slices.forEach((px, i) => x.set(segNormalise(px), i * SEG_SIZE * SEG_SIZE));
  const out = await s.run({ slice: new ort.Tensor('float32', x, [n, 1, SEG_SIZE, SEG_SIZE]) });
  const p = out.prob.data as Float32Array;
  return Array.from({ length: n }, (_, i) => p.slice(i * SEG_SIZE * SEG_SIZE, (i + 1) * SEG_SIZE * SEG_SIZE));
}

export interface Segmentation {
  prob: Float32Array; // SEG_SIZE^2
  lb: Letterbox;
  input: Uint8Array;
  tumourPixels: number; // above 0.5, letterboxed space
  brainPixels: number;
}

export async function segment(g: Gray): Promise<Segmentation> {
  const { pixels, lb } = letterbox(g, SEG_SIZE);
  const [prob] = await segmentBatch([pixels]);
  let t = 0;
  let b = 0;
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] > 0.5) t++;
    if (pixels[i] > 10) b++;
  }
  return { prob, lb, input: pixels, tumourPixels: t, brainPixels: b };
}

// ---------------------------------------------------------------- decisions
export function tumourProbability(p: number[]): number {
  const brain = p[0] + p[1] + p[2] + p[3];
  return brain > 0 ? (p[1] + p[2] + p[3]) / brain : 0;
}

export type Verdict =
  | { kind: 'not-brain'; confidence: number }
  | { kind: 'uncertain'; best: ClassName; confidence: number }
  | { kind: 'no-tumour'; confidence: number }
  | { kind: 'tumour'; type: ClassName; confidence: number; typeConfidence: number };

export function verdict(p: number[], threshold: number): Verdict {
  const best = p.indexOf(Math.max(...p));
  const conf = p[best];
  if (CLASSES[best] === 'not_brain_mri') return { kind: 'not-brain', confidence: conf };
  if (conf < threshold) return { kind: 'uncertain', best: CLASSES[best], confidence: conf };
  if (best === 0) return { kind: 'no-tumour', confidence: conf };
  return { kind: 'tumour', type: CLASSES[best], confidence: tumourProbability(p), typeConfidence: conf };
}

// ---------------------------------------------------------------- prognosis (Cox model exported by survival.py)
export interface SurvivalModel {
  features: string[];
  n: number;
  events: number;
  model: {
    mu: number[];
    sd: number[];
    beta: number[];
    grid_days: number[];
    H0: number[];
    hazard_ratio_per_sd: number[];
    boot_beta: number[][];
    boot_H0: number[][];
    lp_tertiles?: number[];
  };
}

export type Resection = 'GTR' | 'STR' | 'unknown';

export function survivalDesign(age: number, wtCc: number, resection: Resection): number[] {
  return [age, Math.log1p(Math.max(0, wtCc)), resection === 'GTR' ? 1 : 0, resection === 'STR' ? 1 : 0];
}

function lp(x: number[], mu: number[], sd: number[], beta: number[]): number {
  return x.reduce((s, xi, i) => s + ((xi - mu[i]) / sd[i]) * beta[i], 0);
}

export interface SurvivalPrediction {
  days: number[];
  survival: number[];
  lower: number[];
  upper: number[];
  medianDays: number | null;
  medianLower: number | null;
  medianUpper: number | null;
  riskGroup: 'lower' | 'intermediate' | 'higher' | null;
  linearPredictor: number;
}

function medianOf(days: number[], s: number[]): number | null {
  const i = s.findIndex((v) => v <= 0.5);
  return i < 0 ? null : days[i];
}

export function predictSurvival(m: SurvivalModel, x: number[]): SurvivalPrediction {
  const { mu, sd, beta, grid_days: days, H0, boot_beta, boot_H0, lp_tertiles } = m.model;
  const eta = lp(x, mu, sd, beta);
  const survival = H0.map((h) => Math.exp(-h * Math.exp(eta)));
  const curves = boot_beta.map((b, k) => {
    const e = lp(x, mu, sd, b);
    return boot_H0[k].map((h) => Math.exp(-h * Math.exp(e)));
  });
  const q = (arr: number[], p: number) => {
    const s = [...arr].sort((a, b) => a - b);
    return percentile(s, p);
  };
  const lower = days.map((_, j) => q(curves.map((c) => c[j]), 0.05));
  const upper = days.map((_, j) => q(curves.map((c) => c[j]), 0.95));
  const medians = curves.map((c) => medianOf(days, c)).filter((v): v is number => v !== null);
  const riskGroup = lp_tertiles ? (eta < lp_tertiles[0] ? 'lower' : eta < lp_tertiles[1] ? 'intermediate' : 'higher') : null;
  return {
    days,
    survival,
    lower,
    upper,
    medianDays: medianOf(days, survival),
    medianLower: medians.length > 10 ? q(medians, 0.05) : null,
    medianUpper: medians.length > 10 ? q(medians, 0.95) : null,
    riskGroup,
    linearPredictor: eta,
  };
}

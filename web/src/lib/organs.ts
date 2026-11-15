/* Liver CT and cervical cytology analysis in the browser. Mirrors ml/qihgnn/screen/organs.py;
   web/tests/organs.test.ts checks the cell separation against the Python output. */
import { letterbox, loadOrt, session, toUint8, type Gray, type Letterbox, type Volume } from './screen';

export const LIVER = 256;
export const CERV_W = 512;
export const CERV_H = 384;
export const MIN_NUCLEUS_PX = 12;
export const MIN_LESION_VOXELS = 30;

// ---------------------------------------------------------------- liver
/** CT slice to grey: Hounsfield units get an abdominal window (-160 to 240 HU); anything else is scaled by percentiles. */
export function ctSlice(vol: Volume, z: number): Gray {
  const { nx, ny } = vol;
  const raw = vol.data.subarray(z * nx * ny, (z + 1) * nx * ny);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < raw.length; i += 7) {
    if (raw[i] < min) min = raw[i];
    if (raw[i] > max) max = raw[i];
  }
  const data = new Uint8Array(raw.length);
  if (min < -300 || max > 1200) {
    const lo = -160;
    const hi = 240;
    for (let i = 0; i < raw.length; i++) data[i] = Math.round(Math.min(1, Math.max(0, (raw[i] - lo) / (hi - lo))) * 255);
  } else {
    data.set(toUint8(min < 0 ? raw.map((v) => v - min) : raw));
  }
  return { data, w: nx, h: ny };
}

// Stage 2 (liver_cascade.py): a tumour network that sees only a square around the liver, resized to TUM px.
export const TUM = 192;
export const BOX_MARGIN = 8;
export const BOX_MIN = 48;
export const MIN_TUMOUR_PX = 8;

export interface Box { y0: number; x0: number; s: number }

/** Square box around the liver pixels (class >= 1) of a size x size map; null without liver. */
export function liverBox(classes: Uint8Array, size = LIVER, margin = BOX_MARGIN, minSide = BOX_MIN): Box | null {
  let y0 = size, y1 = -1, x0 = size, x1 = -1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (classes[y * size + x] >= 1) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
      }
    }
  }
  if (y1 < 0) return null;
  y1 += 1;
  x1 += 1;
  const side = Math.max(Math.max(y1 - y0, x1 - x0) + 2 * margin, minSide);
  const cy = Math.floor((y0 + y1) / 2);
  const cx = Math.floor((x0 + x1) / 2);
  return { y0: cy - Math.floor(side / 2), x0: cx - Math.floor(side / 2), s: side };
}

/** Half-pixel-centre bilinear resize with edge clamping (same formula as resize_bilinear in Python). */
export function resizeBilinear(a: Float32Array, h: number, w: number, oh: number, ow: number): Float32Array {
  const out = new Float32Array(oh * ow);
  const sy = h / oh;
  const sx = w / ow;
  const x0s = new Int32Array(ow), x1s = new Int32Array(ow), wxs = new Float32Array(ow);
  for (let x = 0; x < ow; x++) {
    const fx = Math.min(w - 1, Math.max(0, (x + 0.5) * sx - 0.5));
    x0s[x] = Math.floor(fx);
    x1s[x] = Math.min(x0s[x] + 1, w - 1);
    wxs[x] = fx - x0s[x];
  }
  for (let y = 0; y < oh; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, h - 1);
    const wy = fy - y0;
    for (let x = 0; x < ow; x++) {
      const wx = wxs[x];
      const top = a[y0 * w + x0s[x]] * (1 - wx) + a[y0 * w + x1s[x]] * wx;
      const bot = a[y1 * w + x0s[x]] * (1 - wx) + a[y1 * w + x1s[x]] * wx;
      out[y * ow + x] = top * (1 - wy) + bot * wy;
    }
  }
  return out;
}

/** A box on the 256 px grid mapped onto the original slice (letterbox scale and offsets undone). */
export function fullBox(box: Box, lb?: Letterbox): Box {
  if (!lb) return box;
  const r = (v: number) => Math.floor(v + 0.5);
  return { y0: r((box.y0 - lb.top) / lb.scale), x0: r((box.x0 - lb.left) / lb.scale), s: Math.max(1, r(box.s / lb.scale)) };
}

/** Crop (zero outside the image), scale to [0, 1] and resize to TUM x TUM. With lb, src is the original-resolution
    slice and the 256-grid box is mapped onto it, so small tumours keep their detail. */
export function stage2Input(src: Gray, box: Box, lb?: Letterbox): Float32Array {
  const { y0, x0, s } = fullBox(box, lb);
  const c = new Float32Array(s * s);
  for (let y = 0; y < s; y++) {
    const sy = y0 + y;
    if (sy < 0 || sy >= src.h) continue;
    for (let x = 0; x < s; x++) {
      const sx = x0 + x;
      if (sx >= 0 && sx < src.w) c[y * s + x] = src.data[sy * src.w + sx] / 255;
    }
  }
  return resizeBilinear(c, s, s, TUM, TUM);
}

/** Stage-2 probability (TUM x TUM) back onto the size x size slice; zero outside the box. */
export function uncrop(prob: Float32Array, box: Box, size = LIVER): Float32Array {
  const { y0, x0, s } = box;
  const big = resizeBilinear(prob, TUM, TUM, s, s);
  const out = new Float32Array(size * size);
  for (let y = Math.max(0, y0); y < Math.min(size, y0 + s); y++) {
    for (let x = Math.max(0, x0); x < Math.min(size, x0 + s); x++) out[y * size + x] = big[(y - y0) * s + (x - x0)];
  }
  return out;
}

/** Keep 8-connected components of mask with at least minPx pixels. */
export function dropSmall(mask: Uint8Array, w: number, h: number, minPx = MIN_TUMOUR_PX): Uint8Array {
  const out = new Uint8Array(mask.length);
  const seen = new Uint8Array(mask.length);
  const q = new Int32Array(mask.length);
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    let head = 0;
    let tail = 0;
    q[tail++] = i;
    seen[i] = 1;
    while (head < tail) {
      const cur = q[head++];
      const cy = Math.floor(cur / w);
      const cx = cur - cy * w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = cy + dy;
          const nx = cx + dx;
          if ((dy || dx) && ny >= 0 && nx >= 0 && ny < h && nx < w) {
            const n = ny * w + nx;
            if (mask[n] && !seen[n]) {
              seen[n] = 1;
              q[tail++] = n;
            }
          }
        }
      }
    }
    if (tail >= minPx) for (let k = 0; k < tail; k++) out[q[k]] = 1;
  }
  return out;
}

/** Final map: liver where stage 1 says liver or tumour; tumour where stage 2 > thr inside that liver. */
export function combine(stage1: Uint8Array, prob: Float32Array | null, thr: number, size = LIVER): Uint8Array {
  const out = new Uint8Array(stage1.length);
  const t = new Uint8Array(stage1.length);
  for (let i = 0; i < stage1.length; i++) {
    out[i] = stage1[i] >= 1 ? 1 : 0;
    if (prob && out[i] && prob[i] > thr) t[i] = 1;
  }
  if (prob) {
    const kept = dropSmall(t, size, size);
    for (let i = 0; i < out.length; i++) if (kept[i]) out[i] = 2;
  }
  return out;
}

async function stage1Batch(slices: Uint8Array[]): Promise<Uint8Array[]> {
  const ort = await loadOrt();
  const s = await session('liver');
  const n = slices.length;
  const px = LIVER * LIVER;
  const x = new Float32Array(n * px);
  slices.forEach((sl, i) => {
    for (let j = 0; j < px; j++) x[i * px + j] = sl[j] / 255;
  });
  const out = await s.run({ slice: new ort.Tensor('float32', x, [n, 1, LIVER, LIVER]) });
  const p = out.prob.data as Float32Array;
  return Array.from({ length: n }, (_, i) => argmax3(p, i * 3 * px, px));
}

export interface LiverInput {
  pixels: Uint8Array; // letterboxed LIVER x LIVER slice for stage 1
  full: Gray; // the original slice, for the stage-2 crop
  lb: Letterbox;
}

/** Both stages on a batch of slices -> 0/1/2 maps on the 256 px grid. thr: tumour threshold chosen on validation patients. */
export async function liverBatch(slices: LiverInput[], thr: number): Promise<Uint8Array[]> {
  const s1 = await stage1Batch(slices.map((x) => x.pixels));
  const boxes = s1.map((c) => liverBox(c));
  const idx = boxes.map((b, i) => (b ? i : -1)).filter((i) => i >= 0);
  const probs: (Float32Array | null)[] = slices.map(() => null);
  if (idx.length) {
    const ort = await loadOrt();
    const s = await session('liver_tumour');
    const tp = TUM * TUM;
    const x = new Float32Array(idx.length * tp);
    idx.forEach((i, k) => x.set(stage2Input(slices[i].full, boxes[i]!, slices[i].lb), k * tp));
    const out = await s.run({ crop: new ort.Tensor('float32', x, [idx.length, 1, TUM, TUM]) });
    const p = out.prob.data as Float32Array;
    idx.forEach((i, k) => (probs[i] = uncrop(p.slice((k * 2 + 1) * tp, (k * 2 + 2) * tp), boxes[i]!)));
  }
  return s1.map((c, i) => combine(c, probs[i], thr));
}

function argmax3(p: Float32Array, off: number, px: number): Uint8Array {
  const c = new Uint8Array(px);
  for (let j = 0; j < px; j++) {
    const a = p[off + j];
    const b = p[off + px + j];
    const d = p[off + 2 * px + j];
    c[j] = d > a && d > b ? 2 : b > a ? 1 : 0;
  }
  return c;
}

export interface LiverSlice {
  classes: Uint8Array; // LIVER^2, 0 background, 1 liver, 2 tumour
  lb: Letterbox;
  liverPx: number;
  tumourPx: number;
}

export async function liverSlice(g: Gray, thr: number): Promise<LiverSlice> {
  const { pixels, lb } = letterbox(g, LIVER);
  const [classes] = await liverBatch([{ pixels, full: g, lb }], thr);
  let l = 0;
  let t = 0;
  for (const v of classes) {
    if (v >= 1) l++;
    if (v === 2) t++;
  }
  return { classes, lb, liverPx: l, tumourPx: t };
}

/** 6-connected components of the tumour class across slices; returns sizes of components >= minVoxels. */
export function lesions3d(stack: Uint8Array[], w: number, h: number, minVoxels = MIN_LESION_VOXELS): number[] {
  const n = stack.length;
  const seen = new Uint8Array(n * w * h);
  const sizes: number[] = [];
  const queue = new Int32Array(n * w * h);
  for (let z = 0; z < n; z++) {
    for (let i = 0; i < w * h; i++) {
      const id = z * w * h + i;
      if (stack[z][i] !== 2 || seen[id]) continue;
      let head = 0;
      let tail = 0;
      queue[tail++] = id;
      seen[id] = 1;
      while (head < tail) {
        const cur = queue[head++];
        const cz = Math.floor(cur / (w * h));
        const r = cur - cz * w * h;
        const cy = Math.floor(r / w);
        const cx = r - cy * w;
        const nb: [number, number, number][] = [[cz, cy, cx - 1], [cz, cy, cx + 1], [cz, cy - 1, cx], [cz, cy + 1, cx], [cz - 1, cy, cx], [cz + 1, cy, cx]];
        for (const [nz, ny, nx] of nb) {
          if (nz < 0 || ny < 0 || nx < 0 || nz >= n || ny >= h || nx >= w) continue;
          const nid = nz * w * h + ny * w + nx;
          if (!seen[nid] && stack[nz][ny * w + nx] === 2) {
            seen[nid] = 1;
            queue[tail++] = nid;
          }
        }
      }
      if (tail >= minVoxels) sizes.push(tail);
    }
  }
  return sizes.sort((a, b) => b - a);
}

// ---------------------------------------------------------------- cervix
/** Resize to fill 512 x 384 (centre crop), like the 4:3 training images. */
export interface CervixCrop { sx: number; sy: number; sw: number; sh: number; w: number; h: number }
export async function cervixInput(file: Blob): Promise<{ rgb: Uint8ClampedArray; crop: CervixCrop; bitmap: ImageBitmap }> {
  const bmp = await createImageBitmap(file);
  const scale = Math.max(CERV_W / bmp.width, CERV_H / bmp.height);
  const sw = CERV_W / scale;
  const sh = CERV_H / scale;
  const sx = (bmp.width - sw) / 2;
  const sy = (bmp.height - sh) / 2;
  const c = document.createElement('canvas');
  c.width = CERV_W;
  c.height = CERV_H;
  const ctx = c.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, sx, sy, sw, sh, 0, 0, CERV_W, CERV_H);
  return { rgb: ctx.getImageData(0, 0, CERV_W, CERV_H).data, crop: { sx, sy, sw, sh, w: bmp.width, h: bmp.height }, bitmap: bmp };
}

export async function cervixClasses(rgba: Uint8ClampedArray): Promise<Uint8Array> {
  const ort = await loadOrt();
  const s = await session('cervix');
  const px = CERV_W * CERV_H;
  const x = new Float32Array(3 * px);
  for (let i = 0; i < px; i++) {
    x[i] = rgba[i * 4] / 255;
    x[px + i] = rgba[i * 4 + 1] / 255;
    x[2 * px + i] = rgba[i * 4 + 2] / 255;
  }
  const out = await s.run({ image: new ort.Tensor('float32', x, [1, 3, CERV_H, CERV_W]) });
  return argmax3(out.prob.data as Float32Array, 0, px);
}

export interface Cell {
  id: number;
  nucleusPx: number;
  cytoplasmPx: number;
  nc: number;
  cx: number;
  cy: number;
}

/** Same algorithm as organs.cells_from_classes: 8-connected nuclei >= MIN_NUCLEUS_PX, grown through cell pixels by
 *  4-connected breadth-first search seeded in raster order. Returns the owner map and per-cell measurements. */
export function cellsFromClasses(cls: Uint8Array, w: number, h: number): { owner: Int32Array; cells: Cell[] } {
  const n = w * h;
  const lab = new Int32Array(n);
  let next = 0;
  const sizes: number[] = [0];
  const stack = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (cls[i] !== 2 || lab[i]) continue;
    next++;
    let top = 0;
    stack[top++] = i;
    lab[i] = next;
    let size = 0;
    while (top) {
      const cur = stack[--top];
      size++;
      const y = Math.floor(cur / w);
      const x = cur - y * w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy;
          const xx = x + dx;
          if ((dy || dx) && yy >= 0 && xx >= 0 && yy < h && xx < w) {
            const j = yy * w + xx;
            if (cls[j] === 2 && !lab[j]) {
              lab[j] = next;
              stack[top++] = j;
            }
          }
        }
      }
    }
    sizes.push(size);
  }
  const owner = new Int32Array(n);
  const queue = new Int32Array(n);
  let head = 0;
  let tail = 0;
  for (let i = 0; i < n; i++) {
    if (cls[i] === 2 && sizes[lab[i]] >= MIN_NUCLEUS_PX) {
      owner[i] = lab[i];
      queue[tail++] = i;
    }
  }
  while (head < tail) {
    const cur = queue[head++];
    const o = owner[cur];
    const y = Math.floor(cur / w);
    const x = cur - y * w;
    const nb = [y + 1 < h ? cur + w : -1, y > 0 ? cur - w : -1, x + 1 < w ? cur + 1 : -1, x > 0 ? cur - 1 : -1];
    for (const j of nb) {
      if (j >= 0 && cls[j] > 0 && owner[j] === 0) {
        owner[j] = o;
        queue[tail++] = j;
      }
    }
  }
  const acc = new Map<number, { n: number; c: number; sx: number; sy: number; t: number }>();
  for (let i = 0; i < n; i++) {
    const o = owner[i];
    if (!o) continue;
    let a = acc.get(o);
    if (!a) acc.set(o, (a = { n: 0, c: 0, sx: 0, sy: 0, t: 0 }));
    if (cls[i] === 2) a.n++;
    a.t++;
    const y = Math.floor(i / w);
    a.sx += i - y * w;
    a.sy += y;
  }
  const cells: Cell[] = [];
  for (const id of [...acc.keys()].sort((p, q) => p - q)) {
    const a = acc.get(id)!;
    const cyto = a.t - a.n;
    if (cyto <= 0) continue;
    cells.push({ id, nucleusPx: a.n, cytoplasmPx: cyto, nc: a.n / cyto, cx: a.sx / a.t, cy: a.sy / a.t });
  }
  return { owner, cells };
}

// ---------------------------------------------------------------- cell grading
/** The five cell types cytologists graded in SIPaKMeD, and the group the website reports for each. */
export const CELL_CLASSES = ['superficial_intermediate', 'parabasal', 'metaplastic', 'koilocytotic', 'dyskeratotic'] as const;
export const CELL_GROUP: Record<(typeof CELL_CLASSES)[number], 'normal' | 'benign' | 'abnormal'> = {
  superficial_intermediate: 'normal', parabasal: 'normal', metaplastic: 'benign', koilocytotic: 'abnormal', dyskeratotic: 'abnormal',
};
export const CELL_LABEL: Record<(typeof CELL_CLASSES)[number], string> = {
  superficial_intermediate: 'Superficial / intermediate', parabasal: 'Parabasal', metaplastic: 'Metaplastic',
  koilocytotic: 'Koilocytotic', dyskeratotic: 'Dyskeratotic',
};
export const CELL_SIZE = 128;
const CELL_MARGIN = 0.1;

export interface CellGrade { id: number; probs: number[]; top: number; abnormal: number }

/** Bounding box of every cell in the 512 x 384 analysis grid. */
export function cellBoxes(owner: Int32Array, w: number, h: number): Map<number, { x0: number; y0: number; x1: number; y1: number }> {
  const boxes = new Map<number, { x0: number; y0: number; x1: number; y1: number }>();
  for (let i = 0; i < owner.length; i++) {
    const o = owner[i];
    if (!o) continue;
    const y = Math.floor(i / w);
    const x = i - y * w;
    const b = boxes.get(o);
    if (!b) boxes.set(o, { x0: x, y0: y, x1: x + 1, y1: y + 1 });
    else {
      if (x < b.x0) b.x0 = x;
      if (y < b.y0) b.y0 = y;
      if (x + 1 > b.x1) b.x1 = x + 1;
      if (y + 1 > b.y1) b.y1 = y + 1;
    }
  }
  void h;
  return boxes;
}

/** Same as cells.square in Python: pad the crop to a square with the median colour of its border, then resize. */
export function squareCrop(rgba: Uint8ClampedArray, w: number, h: number, size = CELL_SIZE): Float32Array {
  const border: number[][] = [[], [], []];
  const push = (x: number, y: number) => { for (let c = 0; c < 3; c++) border[c].push(rgba[(y * w + x) * 4 + c]); };
  for (let x = 0; x < w; x++) { push(x, 0); push(x, h - 1); }
  for (let y = 0; y < h; y++) { push(0, y); push(w - 1, y); }
  const med = border.map((v) => {
    v.sort((a, b) => a - b);
    const m = v.length >> 1;
    return Math.floor(v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2);
  });
  const s = Math.max(w, h);
  const sq = document.createElement('canvas');
  sq.width = s;
  sq.height = s;
  const sctx = sq.getContext('2d')!;
  sctx.fillStyle = `rgb(${med[0]},${med[1]},${med[2]})`;
  sctx.fillRect(0, 0, s, s);
  sctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), (s - w) >> 1, (s - h) >> 1);
  const out = document.createElement('canvas');
  out.width = size;
  out.height = size;
  const octx = out.getContext('2d')!;
  octx.imageSmoothingQuality = 'high';
  octx.drawImage(sq, 0, 0, size, size);
  const px = octx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    x[i] = px[i * 4] / 255;
    x[n + i] = px[i * 4 + 1] / 255;
    x[2 * n + i] = px[i * 4 + 2] / 255;
  }
  return x;
}

/** Grade every measured cell: crop it from the full-resolution image, run the SIPaKMeD classifier on the crops. */
export async function gradeCells(bitmap: ImageBitmap, crop: CervixCrop, owner: Int32Array, cells: Cell[], batch = 16): Promise<CellGrade[]> {
  const ort = await loadOrt();
  const s = await session('cells');
  const boxes = cellBoxes(owner, CERV_W, CERV_H);
  const k = crop.sw / CERV_W; // source pixels per analysis pixel
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  const per = 3 * CELL_SIZE * CELL_SIZE;
  const out: CellGrade[] = [];
  for (let i = 0; i < cells.length; i += batch) {
    const part = cells.slice(i, i + batch);
    const x = new Float32Array(part.length * per);
    part.forEach((cell, j) => {
      const b = boxes.get(cell.id)!;
      const mx = (b.x1 - b.x0) * CELL_MARGIN;
      const my = (b.y1 - b.y0) * CELL_MARGIN;
      const x0 = Math.max(0, Math.floor(crop.sx + (b.x0 - mx) * k));
      const y0 = Math.max(0, Math.floor(crop.sy + (b.y0 - my) * k));
      const x1 = Math.min(crop.w, Math.ceil(crop.sx + (b.x1 + mx) * k));
      const y1 = Math.min(crop.h, Math.ceil(crop.sy + (b.y1 + my) * k));
      // very large crops are shrunk first; the classifier only sees 128 px anyway
      const f = Math.min(1, 512 / Math.max(x1 - x0, y1 - y0));
      c.width = Math.max(1, Math.round((x1 - x0) * f));
      c.height = Math.max(1, Math.round((y1 - y0) * f));
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, c.width, c.height);
      x.set(squareCrop(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height), j * per);
    });
    const res = await s.run({ rgb: new ort.Tensor('float32', x, [part.length, 3, CELL_SIZE, CELL_SIZE]) });
    const p = res.probs.data as Float32Array;
    part.forEach((cell, j) => {
      const probs = Array.from(p.subarray(j * 5, j * 5 + 5));
      const top = probs.indexOf(Math.max(...probs));
      out.push({ id: cell.id, probs, top, abnormal: probs[3] + probs[4] });
    });
  }
  return out;
}

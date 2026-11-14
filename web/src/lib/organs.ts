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

export async function liverBatch(slices: Uint8Array[]): Promise<Uint8Array[]> {
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

export async function liverSlice(g: Gray): Promise<LiverSlice> {
  const { pixels, lb } = letterbox(g, LIVER);
  const [classes] = await liverBatch([pixels]);
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
export async function cervixInput(file: Blob): Promise<{ rgb: Uint8ClampedArray; crop: { sx: number; sy: number; sw: number; sh: number; w: number; h: number } }> {
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
  return { rgb: ctx.getImageData(0, 0, CERV_W, CERV_H).data, crop: { sx, sy, sw, sh, w: bmp.width, h: bmp.height } };
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

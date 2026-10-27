/**
 * Volumes arrive as lossless 8-bit PNG mosaics (see ml/qihgnn/export_web.py).
 * Slice k of an (H, W, D) volume sits at mosaic row floor(k / cols), column k % cols.
 * In memory a volume is a Uint8Array indexed (z * H + y) * W + x.
 */
export interface Volume {
  data: Uint8Array;
  h: number; // rows: anterior -> posterior
  w: number; // columns: patient right -> patient left
  d: number; // slices: inferior -> superior
}

export type Plane = 'axial' | 'coronal' | 'sagittal';

export interface Slice {
  data: Uint8Array;
  rows: number;
  cols: number;
}

async function decodePng(src: string | Blob): Promise<ImageData> {
  const blob = typeof src === 'string' ? await (await fetch(src)).blob() : src;
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  if (!ctx) throw new Error('2D canvas is not available');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

export function fromMosaic(img: { data: Uint8ClampedArray | Uint8Array; width: number }, shape: [number, number, number], cols: number): Volume {
  const [h, w, d] = shape;
  const out = new Uint8Array(h * w * d);
  for (let k = 0; k < d; k++) {
    const r0 = Math.floor(k / cols) * h;
    const c0 = (k % cols) * w;
    for (let y = 0; y < h; y++) {
      const src = ((r0 + y) * img.width + c0) * 4;
      const dst = (k * h + y) * w;
      for (let x = 0; x < w; x++) out[dst + x] = img.data[src + x * 4];
    }
  }
  return { data: out, h, w, d };
}

export async function loadVolume(src: string | Blob, shape: [number, number, number], cols: number): Promise<Volume> {
  return fromMosaic(await decodePng(src), shape, cols);
}

export function planeDepth(v: Volume, plane: Plane): number {
  return plane === 'axial' ? v.d : plane === 'coronal' ? v.h : v.w;
}

/** Extract a 2D slice. Coronal and sagittal slices put superior at the top, as radiology viewers do. */
export function slice(v: Volume, plane: Plane, index: number): Slice {
  const { h, w, d, data } = v;
  if (plane === 'axial') {
    const k = clamp(index, 0, d - 1);
    return { data: data.subarray(k * h * w, (k + 1) * h * w), rows: h, cols: w };
  }
  if (plane === 'coronal') {
    const y = clamp(index, 0, h - 1);
    const out = new Uint8Array(d * w);
    for (let z = 0; z < d; z++) {
      const row = d - 1 - z;
      const base = (z * h + y) * w;
      out.set(data.subarray(base, base + w), row * w);
    }
    return { data: out, rows: d, cols: w };
  }
  const x = clamp(index, 0, w - 1);
  const out = new Uint8Array(d * h);
  for (let z = 0; z < d; z++) {
    const row = d - 1 - z;
    for (let y = 0; y < h; y++) out[row * h + y] = data[(z * h + y) * w + x];
  }
  return { data: out, rows: d, cols: h };
}

/** Voxel coordinate (x, y, z) under a pixel of a plane view. */
export function voxelAt(v: Volume, plane: Plane, index: number, row: number, col: number): [number, number, number] {
  if (plane === 'axial') return [col, row, index];
  if (plane === 'coronal') return [col, index, v.d - 1 - row];
  return [index, col, v.d - 1 - row];
}

export function valueAt(v: Volume, x: number, y: number, z: number): number {
  return v.data[(z * v.h + y) * v.w + x];
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/** Slices that contain any voxel of `label`, so the viewer can open on an informative slice. */
export function labelExtent(labels: Volume, label: number): { first: number; last: number; peak: number } | null {
  const { h, w, d, data } = labels;
  let first = -1;
  let last = -1;
  let peak = -1;
  let best = 0;
  for (let z = 0; z < d; z++) {
    let n = 0;
    const base = z * h * w;
    for (let i = 0; i < h * w; i++) if (data[base + i] === label) n++;
    if (n > 0) {
      if (first < 0) first = z;
      last = z;
      if (n > best) {
        best = n;
        peak = z;
      }
    }
  }
  return first < 0 ? null : { first, last, peak };
}

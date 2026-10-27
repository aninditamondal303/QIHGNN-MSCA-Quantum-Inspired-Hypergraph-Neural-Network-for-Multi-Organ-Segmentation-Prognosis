import { useEffect, useMemo, useRef, useState } from 'react';
import type { Overlay } from '../lib/useCase';
import { slice, voxelAt, type Plane, type Volume } from '../lib/volume';

// Overlay colours on film black (brighter than the paper tokens so they read on #050505).
const COBALT = [84, 116, 255];
const VERMILION = [255, 74, 54];

interface Props {
  image: Volume | null;
  labels: Volume | null;
  plane: Plane;
  index: number;
  overlays: Overlay[];
  opacity?: number; // 0..1 fill strength of supervised overlays
  spacing?: [number, number, number];
  showOverlay?: boolean;
  className?: string;
  develop?: boolean; // play the one-time "film developing" entrance
  crosshair?: [number, number, number] | null; // voxel (x, y, z) to mark with thin guide lines
  onHover?: (v: [number, number, number] | null) => void;
  onPick?: (v: [number, number, number]) => void;
  ariaLabel: string;
}

function hatchPattern(ctx: CanvasRenderingContext2D, dpr: number): CanvasPattern | null {
  const size = Math.round(7 * dpr);
  const tile = document.createElement('canvas');
  tile.width = tile.height = size;
  const t = tile.getContext('2d');
  if (!t) return null;
  t.strokeStyle = `rgb(${VERMILION.join(',')})`;
  t.lineWidth = 1.5 * dpr;
  t.beginPath();
  for (const o of [-size, 0, size]) {
    t.moveTo(o, size);
    t.lineTo(o + size, 0);
  }
  t.stroke();
  return ctx.createPattern(tile, 'repeat');
}

export function FilmCanvas({
  image, labels, plane, index, overlays, opacity = 0.16, spacing = [1, 1, 2], showOverlay = true,
  className = '', develop = false, crosshair = null, onHover, onPick, ariaLabel,
}: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const s = useMemo(() => (image ? slice(image, plane, index) : null), [image, plane, index]);
  // Physical aspect: axial rows/cols are in-plane; coronal and sagittal rows run along z.
  const rowScale = plane === 'axial' ? spacing[0] : spacing[2];
  const colScale = plane === 'sagittal' ? spacing[0] : spacing[1];
  const aspect = s ? (s.rows * rowScale) / (s.cols * colScale) : 1;
  const height = Math.round(width * aspect);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !image || !s || width === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    const ctx = c.getContext('2d');
    if (!ctx) return;

    // 1. grayscale slice at native resolution, scaled up without smoothing
    const native = document.createElement('canvas');
    native.width = s.cols;
    native.height = s.rows;
    const nctx = native.getContext('2d');
    if (!nctx) return;
    const img = nctx.createImageData(s.cols, s.rows);
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      img.data[i * 4] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    nctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(native, 0, 0, c.width, c.height);

    const drawCrosshair = () => {
      if (!crosshair || !image) return;
      const [cx, cy, cz] = crosshair;
      const row = plane === 'axial' ? cy : image.d - 1 - cz;
      const col = plane === 'sagittal' ? cy : cx;
      const px = ((col + 0.5) / s.cols) * c.width;
      const py = ((row + 0.5) / s.rows) * c.height;
      ctx.save();
      ctx.strokeStyle = 'rgba(233,231,225,0.45)';
      ctx.lineWidth = Math.max(1, dpr);
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(c.width, py);
      ctx.moveTo(px, 0);
      ctx.lineTo(px, c.height);
      ctx.stroke();
      ctx.restore();
    };

    if (!labels || !showOverlay) {
      drawCrosshair();
      return;
    }
    const ls = slice(labels, plane, index);
    const pattern = hatchPattern(ctx, dpr);
    for (let label = 1; label < overlays.length; label++) {
      const mode = overlays[label];
      if (mode === 'hidden') continue;
      let any = false;
      const fill = nctx.createImageData(ls.cols, ls.rows);
      const edge = nctx.createImageData(ls.cols, ls.rows);
      const col = mode === 'supervised' ? COBALT : VERMILION;
      for (let r = 0; r < ls.rows; r++) {
        for (let q = 0; q < ls.cols; q++) {
          const i = r * ls.cols + q;
          if (ls.data[i] !== label) continue;
          any = true;
          fill.data[i * 4 + 3] = 255;
          const boundary =
            r === 0 || q === 0 || r === ls.rows - 1 || q === ls.cols - 1 ||
            ls.data[i - 1] !== label || ls.data[i + 1] !== label || ls.data[i - ls.cols] !== label || ls.data[i + ls.cols] !== label;
          if (boundary) {
            edge.data[i * 4] = col[0];
            edge.data[i * 4 + 1] = col[1];
            edge.data[i * 4 + 2] = col[2];
            edge.data[i * 4 + 3] = 255;
          }
        }
      }
      if (!any) continue;
      // 2. region fill: cobalt wash for supervised, vermilion hatching for silent channels
      nctx.putImageData(fill, 0, 0);
      const layer = document.createElement('canvas');
      layer.width = c.width;
      layer.height = c.height;
      const lctx = layer.getContext('2d');
      if (!lctx) continue;
      lctx.imageSmoothingEnabled = false;
      lctx.drawImage(native, 0, 0, c.width, c.height);
      lctx.globalCompositeOperation = 'source-in';
      if (mode === 'suppressed' && pattern) {
        lctx.fillStyle = pattern;
      } else {
        lctx.fillStyle = `rgba(${col.join(',')},${opacity})`;
      }
      lctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(layer, 0, 0);
      // 3. voxel-exact contour
      nctx.putImageData(edge, 0, 0);
      ctx.globalAlpha = 0.95;
      ctx.drawImage(native, 0, 0, c.width, c.height);
      ctx.globalAlpha = 1;
    }
    drawCrosshair();
  }, [image, labels, plane, index, overlays, opacity, width, height, showOverlay, s, crosshair]);

  const locate = (e: React.PointerEvent<HTMLCanvasElement>): [number, number, number] | null => {
    if (!image || !s) return null;
    const rect = e.currentTarget.getBoundingClientRect();
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * s.cols);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * s.rows);
    if (col < 0 || row < 0 || col >= s.cols || row >= s.rows) return null;
    return voxelAt(image, plane, index, row, col);
  };

  return (
    <div ref={wrap} className={`relative w-full ${className}`}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={ariaLabel}
        className={`block w-full ${develop ? 'animate-develop' : ''} ${onPick ? 'cursor-crosshair' : ''}`}
        style={{ height: height || undefined, aspectRatio: height ? undefined : '1 / 1' }}
        onPointerMove={(e) => onHover?.(locate(e))}
        onPointerLeave={() => onHover?.(null)}
        onPointerDown={(e) => {
          const v = locate(e);
          if (v && onPick) onPick(v);
        }}
      />
      {!image && (
        <div className="absolute inset-0 grid place-items-center">
          <span className="meta text-film-muted">Loading volume</span>
        </div>
      )}
    </div>
  );
}

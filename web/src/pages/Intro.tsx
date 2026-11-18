import { useCallback, useEffect, useRef, useState } from 'react';

/* Opening gateway. A point cloud morphs between a brain, a liver and a cluster of cells; points near the pointer join
   into hyperedges (several nodes tied to one centre, as in a hypergraph); pressing and holding sweeps a scan beam that
   lights up the lesion region, then the cloud rushes past the viewer into the site. Pure canvas, no libraries. */

const N = 2200;
const HOLD_MS = 1400;
const ORGANS = [
  { key: 'brain', label: 'Brain MRI', line: 'the brain.' },
  { key: 'liver', label: 'Liver CT', line: 'the liver.' },
  { key: 'cells', label: 'Cervical cells', line: 'the cell.' },
] as const;

type Shape = { pos: Float32Array; lesion: Uint8Array };

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function onSphere(r: () => number): [number, number, number] {
  const u = r() * 2 - 1;
  const t = r() * Math.PI * 2;
  const s = Math.sqrt(1 - u * u);
  return [s * Math.cos(t), u, s * Math.sin(t)];
}

function brain(): Shape {
  const r = rng(7);
  const pos = new Float32Array(N * 3);
  const lesion = new Uint8Array(N);
  const L: [number, number, number] = [0.32, 0.28, 0.35];
  for (let i = 0; i < N; i++) {
    const [x, y, z] = onSphere(r);
    let px: number, py: number, pz: number;
    if (i % 9 === 0) {
      // cerebellum: a smaller lobe low at the back
      px = x * 0.42;
      py = -0.42 + y * 0.2;
      pz = -0.62 + z * 0.3;
    } else {
      // two hemispheres (gap at the midline), longer front to back, folded surface
      const side = x >= 0 ? 1 : -1;
      const fold = 1 + 0.055 * Math.sin(11 * y + 5 * z) * Math.cos(9 * z - 4 * x);
      px = (Math.abs(x) * 0.44 + 0.035) * side * fold;
      py = y * 0.62 * fold;
      pz = z * 0.98 * fold;
      if (py < -0.28) py = -0.28 + (py + 0.28) * 0.35;
      if (pz > 0.55) py *= 1 - (pz - 0.55) * 0.5; // frontal lobe tapers
    }
    if (Math.hypot(px - L[0], py - L[1], pz - L[2]) < 0.24) lesion[i] = 1;
    pos.set([px, py, pz], i * 3);
  }
  return { pos, lesion };
}

function liver(): Shape {
  const r = rng(11);
  const pos = new Float32Array(N * 3);
  const lesion = new Uint8Array(N);
  const L: [number, number, number] = [-0.35, 0.05, 0.2];
  for (let i = 0; i < N; i++) {
    const [x, y, z] = onSphere(r);
    // wedge: large right lobe tapering to a thin left lobe, flat underside
    const t = (x + 1) / 2; // 0 left tip .. 1 right lobe
    const h = 0.18 + 0.62 * Math.pow(t, 0.8);
    let py = y * h;
    if (py < -0.12) py = -0.12 + (py + 0.12) * 0.35;
    const px = x * 1.15 + 0.1;
    const pz = z * (0.35 + 0.45 * t);
    const d = Math.hypot(px - L[0], py - L[1], pz - L[2]);
    if (d < 0.3) lesion[i] = 1;
    pos.set([px, py + 0.05, pz], i * 3);
  }
  return { pos, lesion };
}

function cells(): Shape {
  const r = rng(23);
  const pos = new Float32Array(N * 3);
  const lesion = new Uint8Array(N);
  const centres: [number, number, number, number][] = [];
  for (let k = 0; k < 7; k++) {
    const a = (k / 7) * Math.PI * 2;
    const rad = k === 0 ? 0 : 0.62;
    centres.push([Math.cos(a) * rad * (k === 0 ? 0 : 1), Math.sin(a) * rad * 0.7, Math.sin(a * 2) * 0.25, k === 0 ? 0.36 : 0.27]);
  }
  for (let i = 0; i < N; i++) {
    const k = i % 7;
    const [cx, cy, cz, R] = centres[k];
    const [x, y, z] = onSphere(r);
    const odd = k === 3; // one cell with an enlarged nucleus (high nucleus-to-cytoplasm ratio)
    const nucleus = r() < (odd ? 0.55 : 0.3);
    const s = nucleus ? R * (odd ? 0.62 : 0.32) : R * (0.92 + 0.08 * Math.sin(5 * x + 3 * y));
    pos.set([cx + x * s, cy + y * s, cz + z * s], i * 3);
    if (odd) lesion[i] = 1;
  }
  return { pos, lesion };
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function Intro({ onEnter }: { onEnter: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [organ, setOrgan] = useState(0);
  const [progress, setProgress] = useState(0);
  const [leaving, setLeaving] = useState(false);
  const state = useRef({
    organ: 0, from: 0, morphStart: -1e9, mx: 0, my: 0, px: -1, py: -1, holding: false, holdStart: 0,
    scanned: 0, warpStart: -1, auto: true, done: false,
  });
  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  const pick = useCallback((i: number, manual = true) => {
    const s = state.current;
    if (i === s.organ) return;
    s.from = s.organ;
    s.organ = i;
    s.morphStart = performance.now();
    s.scanned = 0;
    if (manual) s.auto = false;
    setOrgan(i);
  }, []);

  const finish = useCallback(() => {
    const s = state.current;
    if (s.done) return;
    s.done = true;
    setLeaving(true);
    window.setTimeout(onEnter, reduced ? 0 : 650);
  }, [onEnter, reduced]);

  useEffect(() => {
    const c = canvas.current!;
    const ctx = c.getContext('2d')!;
    const shapes = [brain(), liver(), cells()];
    const cur = new Float32Array(N * 3);
    const proj = new Float32Array(N * 3); // sx, sy, depth
    const order = new Int32Array(N);
    let raf = 0;
    let w = 0;
    let h = 0;
    let dpr = 1;
    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = c.clientWidth;
      h = c.clientHeight;
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    };
    resize();
    window.addEventListener('resize', resize);
    const t0 = performance.now();
    let lastAuto = t0;

    const frame = (now: number) => {
      const s = state.current;
      if (s.auto && now - lastAuto > 6500 && !s.holding && s.warpStart < 0) {
        lastAuto = now;
        pick((s.organ + 1) % 3, false);
      }
      if (!s.auto) lastAuto = now;
      // morph
      const m = reduced ? 1 : Math.min(1, (now - s.morphStart) / 1600);
      const e = ease(m);
      const A = shapes[s.from].pos;
      const B = shapes[s.organ].pos;
      for (let i = 0; i < N * 3; i++) cur[i] = A[i] + (B[i] - A[i]) * e;
      const lesion = shapes[s.organ].lesion;

      // hold / scan / warp
      let hold = 0;
      if (s.holding && s.warpStart < 0) {
        hold = Math.min(1, (now - s.holdStart) / HOLD_MS);
        s.scanned = Math.max(s.scanned, hold);
        if (hold >= 1) s.warpStart = now;
      }
      setProgress((p) => (Math.abs(p - hold) > 0.01 ? hold : p));
      const warp = s.warpStart > 0 ? Math.min(1, (now - s.warpStart) / 900) : 0;
      if (warp >= 1) finish();

      // camera
      const time = (now - t0) / 1000;
      const yaw = (reduced ? 0.6 : time * 0.18) + s.mx * 0.9;
      const pitch = -0.25 + s.my * 0.5;
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const scale = w < 900 ? Math.min(w * 0.42, h * 0.24) : Math.min(w * 0.24, h * 0.42);
      const ox = w < 900 ? w / 2 : w * 0.7;
      const oy = w < 900 ? h * 0.63 : h * 0.5;
      const camZ = 3.2 - warp * warp * 3.6;
      for (let i = 0; i < N; i++) {
        const x = cur[i * 3], y = cur[i * 3 + 1], z = cur[i * 3 + 2];
        const x1 = x * cy + z * sy;
        const z1 = -x * sy + z * cy;
        const y2 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const d = camZ - z2;
        const f = d > 0.05 ? 2.6 / d : 1e3;
        proj[i * 3] = ox + x1 * f * scale;
        proj[i * 3 + 1] = oy - y2 * f * scale;
        proj[i * 3 + 2] = z2;
        order[i] = i;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#050505';
      ctx.fillRect(0, 0, w, h);

      // faint grid
      ctx.strokeStyle = 'rgba(255,255,255,0.035)';
      ctx.lineWidth = 1;
      for (let gx = (ox % 48); gx < w; gx += 48) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke(); }
      for (let gy = (oy % 48); gy < h; gy += 48) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke(); }

      // halo and scanner frame
      const R = scale * 0.92;
      const halo = ctx.createRadialGradient(ox, oy, 0, ox, oy, R * 1.3);
      halo.addColorStop(0, 'rgba(84,116,255,0.16)');
      halo.addColorStop(1, 'rgba(84,116,255,0)');
      ctx.fillStyle = halo;
      ctx.fillRect(ox - R * 1.3, oy - R * 1.3, R * 2.6, R * 2.6);
      ctx.strokeStyle = 'rgba(242,241,236,0.45)';
      ctx.lineWidth = 1;
      const c = 18;
      for (const [bx, by, dx, dy] of [[-1, -1, 1, 1], [1, -1, -1, 1], [-1, 1, 1, -1], [1, 1, -1, -1]] as const) {
        const x0 = ox + bx * R, y0 = oy + by * R;
        ctx.beginPath(); ctx.moveTo(x0 + dx * c, y0); ctx.lineTo(x0, y0); ctx.lineTo(x0, y0 + dy * c); ctx.stroke();
      }
      ctx.font = '10px "JetBrains Mono", monospace';
      if (w >= 900) {
      ctx.fillStyle = 'rgba(242,241,236,0.5)';
      ctx.fillText(`NODES ${N}  ·  YAW ${(((yaw % 6.283) + 6.283) % 6.283 * 57.3).toFixed(0).padStart(3, '0')}°`, ox - R, oy - R - 10);
      ctx.fillText(`0${s.organ + 1}/03  ${ORGANS[s.organ].label.toUpperCase()}`, ox - R, oy + R + 20);
      }

      // scan beam position (screen y), sweeping top to bottom of the cloud while holding
      const top = oy - R;
      const bottom = oy + R;
      const beamY = top + (bottom - top) * s.scanned;
      const scanning = s.scanned > 0 && s.warpStart < 0;

      // points, back to front
      order.sort((a, b) => proj[a * 3 + 2] - proj[b * 3 + 2]);
      for (let k = 0; k < N; k++) {
        const i = order[k];
        const sx = proj[i * 3], sy2 = proj[i * 3 + 1], z = proj[i * 3 + 2];
        if (sx < -20 || sx > w + 20 || sy2 < -20 || sy2 > h + 20) continue;
        const depth = Math.max(0, Math.min(1, (z + 1.2) / 2.4));
        const passed = scanning && sy2 < beamY;
        const near = scanning && Math.abs(sy2 - beamY) < 10;
        const isL = lesion[i] === 1 && (passed || warp > 0);
        const size = (1.1 + depth * 1.7) * (1 + warp * 3);
        if (isL) ctx.fillStyle = `rgba(255,${90 + depth * 40},${60 + depth * 30},${0.55 + depth * 0.45})`;
        else if (near) ctx.fillStyle = 'rgba(255,255,255,0.95)';
        else if (passed) ctx.fillStyle = `rgba(120,150,255,${0.35 + depth * 0.55})`;
        else ctx.fillStyle = `rgba(${84 + depth * 60},${116 + depth * 60},255,${0.18 + depth * 0.6})`;
        ctx.fillRect(sx - size / 2, sy2 - size / 2, size, size);
      }

      // beam
      if (scanning) {
        const g = ctx.createLinearGradient(0, beamY - 40, 0, beamY + 2);
        g.addColorStop(0, 'rgba(84,116,255,0)');
        g.addColorStop(1, 'rgba(84,116,255,0.28)');
        ctx.fillStyle = g;
        ctx.fillRect(ox - R, beamY - 40, R * 2, 42);
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.fillRect(ox - R - 12, beamY, R * 2 + 24, 1);
      }

      // hyperedges around the pointer: nodes within reach, split into five angular sectors (five hyperedges, as in
      // the model); each hyperedge ties its nodes to one centre.
      if (s.px >= 0 && warp === 0) {
        const reach = Math.max(110, scale * 0.45);
        const sectors: number[][] = [[], [], [], [], []];
        for (let i = 0; i < N; i += 3) {
          const dx = proj[i * 3] - s.px, dy = proj[i * 3 + 1] - s.py;
          const d = Math.hypot(dx, dy);
          if (d < 22 || d > reach) continue;
          const a = (Math.atan2(dy, dx) + Math.PI + time * 0.25) % (Math.PI * 2);
          const k = Math.min(4, Math.floor((a / (Math.PI * 2)) * 5));
          if (sectors[k].length < 5) sectors[k].push(i);
        }
        sectors.forEach((g, gi) => {
          if (g.length < 3) return;
          let cx = 0, cyy = 0;
          g.forEach((i) => { cx += proj[i * 3]; cyy += proj[i * 3 + 1]; });
          cx /= g.length; cyy /= g.length;
          const hue = gi % 2 === 0 ? '84,116,255' : gi === 1 ? '230,59,43' : '242,241,236';
          const pts = g.map((i) => [proj[i * 3], proj[i * 3 + 1]] as const).sort((p1, p2) => Math.atan2(p1[1] - cyy, p1[0] - cx) - Math.atan2(p2[1] - cyy, p2[0] - cx));
          ctx.fillStyle = `rgba(${hue},0.09)`;
          ctx.beginPath();
          pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = `rgba(${hue},0.8)`;
          ctx.lineWidth = 1;
          pts.forEach(([x, y]) => { ctx.beginPath(); ctx.moveTo(cx, cyy); ctx.lineTo(x, y); ctx.stroke(); ctx.fillStyle = `rgba(${hue},1)`; ctx.fillRect(x - 2, y - 2, 4, 4); });
          ctx.strokeStyle = 'rgba(242,241,236,0.25)';
          ctx.beginPath(); ctx.moveTo(s.px, s.py); ctx.lineTo(cx, cyy); ctx.stroke();
          ctx.fillStyle = '#fff';
          ctx.beginPath(); ctx.arc(cx, cyy, 3.5, 0, Math.PI * 2); ctx.fill();
        });
        // reticle
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.beginPath();
        ctx.arc(s.px, s.py, 18, 0, Math.PI * 2);
        ctx.stroke();
        if (hold > 0) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(s.px, s.py, 26, -Math.PI / 2, -Math.PI / 2 + hold * Math.PI * 2);
          ctx.stroke();
        }
      }

      if (warp > 0) {
        ctx.fillStyle = `rgba(247,246,243,${Math.max(0, (warp - 0.55) / 0.45)})`;
        ctx.fillRect(0, 0, w, h);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, [pick, finish, reduced]);

  useEffect(() => {
    const s = state.current;
    const down = (e: KeyboardEvent) => {
      if (e.key === 'Escape') finish();
      if (['1', '2', '3'].includes(e.key)) pick(Number(e.key) - 1);
      if ((e.key === ' ' || e.key === 'Enter') && !e.repeat && !(e.target instanceof HTMLButtonElement)) {
        e.preventDefault();
        s.holding = true;
        s.holdStart = performance.now();
        if (s.px < 0) { s.px = window.innerWidth < 900 ? window.innerWidth / 2 : window.innerWidth * 0.62; s.py = window.innerHeight * 0.55; }
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === ' ' || e.key === 'Enter') s.holding = false;
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [pick, finish]);

  const move = (e: React.PointerEvent) => {
    const s = state.current;
    s.px = e.clientX;
    s.py = e.clientY;
    s.mx = e.clientX / window.innerWidth - 0.5;
    s.my = e.clientY / window.innerHeight - 0.5;
  };

  return (
    <div className={`fixed inset-0 z-[100] select-none overflow-hidden bg-[#050505] text-[#f2f1ec] transition-opacity duration-500 ${leaving ? 'opacity-0' : 'opacity-100'}`} role="dialog" aria-label="Introduction">
      <canvas
        ref={canvas}
        className="absolute inset-0 h-full w-full touch-none cursor-crosshair"
        aria-hidden="true"
        onPointerMove={move}
        onPointerDown={(e) => {
          move(e);
          (e.target as Element).setPointerCapture?.(e.pointerId);
          state.current.holding = true;
          state.current.holdStart = performance.now();
        }}
        onPointerUp={() => (state.current.holding = false)}
        onPointerCancel={() => (state.current.holding = false)}
        onPointerLeave={() => { state.current.px = -1; state.current.holding = false; }}
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between px-5 py-5 sm:px-10">
        <span className="font-display text-[0.95rem] font-bold tracking-tight">TriScan AI</span>
        <button type="button" onClick={finish} className="pointer-events-auto font-mono text-[0.72rem] uppercase tracking-[0.18em] text-[#f2f1ec]/70 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-white">
          Skip intro →
        </button>
      </div>

      <div className="pointer-events-none absolute left-5 top-[18%] max-w-[90vw] sm:left-10 sm:top-1/2 sm:-translate-y-1/2">
        <p className="font-mono text-[0.7rem] uppercase tracking-[0.22em] text-[#f2f1ec]/55">TriScan AI · medical scan analysis · by Anindita Mondal</p>
        <h1 className="mt-4 font-display font-extrabold uppercase leading-[0.86] tracking-tight" style={{ fontSize: 'clamp(2.9rem, 7.2vw, 7.4rem)' }}>
          Look<br />inside
          <span key={organ} className="intro-word block text-[#5474ff]">{ORGANS[organ].line}</span>
        </h1>
        <p className="mt-6 hidden max-w-[34ch] text-[0.95rem] text-[#f2f1ec]/70 sm:block">
          Brain tumours, liver tumours and cervical cells, found, outlined and measured privately in your browser.
        </p>
      </div>

      <div className="absolute inset-x-0 bottom-0 flex flex-col gap-5 px-5 pb-6 sm:flex-row sm:items-end sm:justify-between sm:px-10 sm:pb-8">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Choose an organ">
          {ORGANS.map((o, i) => (
            <button
              key={o.key}
              type="button"
              onClick={() => pick(i)}
              aria-pressed={organ === i}
              className={`border px-3 py-2 font-mono text-[0.72rem] uppercase tracking-[0.14em] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-white ${organ === i ? 'border-[#f2f1ec] bg-[#f2f1ec] text-[#050505]' : 'border-[#f2f1ec]/30 text-[#f2f1ec]/75 hover:border-[#f2f1ec]'}`}
            >
              0{i + 1} {o.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="font-display text-[1.05rem] font-bold uppercase tracking-tight">{progress > 0 ? 'Scanning…' : 'Press and hold to scan'}</p>
            <p className="font-mono text-[0.68rem] uppercase tracking-[0.16em] text-[#f2f1ec]/55">anywhere · or hold space · Esc to skip</p>
          </div>
          <button
            type="button"
            onClick={finish}
            className="grid size-14 shrink-0 place-items-center rounded-full border border-[#f2f1ec]/40 text-[1.2rem] hover:bg-[#f2f1ec] hover:text-[#050505] focus-visible:outline focus-visible:outline-2 focus-visible:outline-white"
            aria-label="Enter the site"
            style={{ background: progress > 0 ? `conic-gradient(#f2f1ec ${progress * 360}deg, transparent 0)` : undefined }}
          >
            <span className={progress > 0 ? 'text-[#050505] mix-blend-difference' : ''}>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}

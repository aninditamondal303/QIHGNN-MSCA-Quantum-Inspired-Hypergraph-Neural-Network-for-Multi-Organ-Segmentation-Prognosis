import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LuCircleAlert, LuLoader, LuPrinter, LuRotateCcw } from 'react-icons/lu';
import { Link } from '../lib/router';
import { cellsFromClasses, CERV_H, CERV_W, cervixClasses, cervixInput, ctSlice, LIVER, lesions3d, liverBatch, liverSlice, type Cell } from '../lib/organs';
import { fileToGray, letterbox, readNifti, type Gray, type Letterbox, type Volume } from '../lib/screen';
import { pct, useOrgans, type OrganMetrics, type OrganSample } from '../lib/screenMetrics';
import { Dropzone, isVolumeFile } from './Analyse';

function NotReady({ what }: { what: string }) {
  return (
    <div className="mb-8 flex gap-3 border border-vermilion/50 bg-vermilion/5 p-4 text-small" role="status">
      <LuCircleAlert className="mt-0.5 size-5 shrink-0 text-vermilion-text" aria-hidden="true" />
      <p>The {what} model is being trained and evaluated. This tool will work as soon as it is published here with its measured accuracy.</p>
    </div>
  );
}

function Busy({ text }: { text: string }) {
  return (
    <div className="mt-4 flex items-center gap-3 text-small" role="status" aria-live="polite">
      <LuLoader className="size-4 animate-spin" aria-hidden="true" />
      <span>{text}</span>
    </div>
  );
}

function Actions({ onReset }: { onReset: () => void }) {
  return (
    <div className="no-print mt-4 flex flex-wrap gap-2">
      <button type="button" onClick={onReset} className="inline-flex h-10 items-center gap-2 border border-rule-strong px-4 text-small hover:bg-raised">
        <LuRotateCcw className="size-4" aria-hidden="true" /> Analyse another
      </button>
      <button type="button" onClick={() => window.print()} className="inline-flex h-10 items-center gap-2 border border-rule-strong px-4 text-small hover:bg-raised">
        <LuPrinter className="size-4" aria-hidden="true" /> Print or save the report
      </button>
    </div>
  );
}

function SampleStrip({ samples, onPick, disabled }: { samples: OrganSample[]; onPick: (s: OrganSample) => void; disabled: boolean }) {
  if (!samples.length) return null;
  return (
    <section className="mt-8">
      <h2 className="font-semibold">Or try a held-out test image</h2>
      <p className="mt-1 text-small text-muted">From patients or slides that were never used for training.</p>
      <ul className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
        {samples.map((s) => (
          <li key={s.file}>
            <button type="button" disabled={disabled} onClick={() => onPick(s)} className="group block w-full text-left disabled:opacity-50">
              {s.file.endsWith('.nii.gz') ? (
                <span className="film grid aspect-square place-items-center text-center font-mono text-[0.7rem] text-film-muted">3D CT<br />volume</span>
              ) : (
                <img src={`screen/samples/${s.file}`} alt="" className="aspect-square w-full bg-film object-cover group-hover:opacity-80" loading="lazy" />
              )}
              <span className="meta mt-1 block text-muted">{s.truth.replace(/_/g, ' ')}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

async function fetchSample(file: string): Promise<File> {
  const res = await fetch(new URL(`screen/samples/${file}`, document.baseURI));
  const blob = await res.blob();
  return new File([blob], file, { type: blob.type });
}

const Disclaimer = () => (
  <p className="mt-8 border-l-2 border-vermilion pl-3 text-small">
    Research tool, not a medical device. Results can be wrong; they are not a diagnosis and must not guide anyone’s care.{' '}
    <Link to="/about#limitations">Limitations</Link>
  </p>
);

// ================================================================== liver
interface Liver2D { kind: '2d'; name: string; gray: Gray; classes: Uint8Array; lb: Letterbox; liverPx: number; tumourPx: number }
interface Liver3D { kind: '3d'; name: string; vol: Volume; stack: Uint8Array[]; scale: number; liverCc: number; tumourCc: number; lesions: number[]; best: number }

export function LiverTool() {
  const { metrics, samples, ready, loading } = useOrgans();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<Liver2D | Liver3D | null>(null);

  const run = useCallback(async (file: File) => {
    setError(null);
    setRes(null);
    try {
      if (isVolumeFile(file.name)) {
        setBusy('Reading the CT volume');
        const vol = await readNifti(file);
        // Two passes: every 4th slice to find where the liver is, then every slice in that range (plus a margin).
        // Slices outside the range hold no liver by construction, so they are left as background.
        const stack: Uint8Array[] = Array.from({ length: vol.nz }, () => new Uint8Array(LIVER * LIVER));
        let scale = 1;
        const runSlices = async (zs: number[], label: string) => {
          for (let i = 0; i < zs.length; i += 8) {
            setBusy(`${label}: ${Math.min(zs.length, i + 8)} of ${zs.length} slices`);
            const batch = zs.slice(i, i + 8).map((z) => {
              const lb = letterbox(ctSlice(vol, z), LIVER);
              scale = lb.lb.scale;
              return lb.pixels;
            });
            const out = await liverBatch(batch);
            out.forEach((c, k) => (stack[zs[i + k]] = c));
          }
        };
        const coarse = Array.from({ length: Math.ceil(vol.nz / 4) }, (_, i) => i * 4);
        await runSlices(coarse, 'Finding the liver');
        const withLiver = coarse.filter((z) => stack[z].some((v) => v >= 1));
        if (withLiver.length) {
          const lo = Math.max(0, withLiver[0] - 4);
          const hi = Math.min(vol.nz - 1, withLiver[withLiver.length - 1] + 4);
          const fine = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).filter((z) => z % 4 !== 0);
          await runSlices(fine, 'Outlining the liver');
        }
        const px = Math.max(vol.spacing[0], vol.spacing[1]) / scale;
        const voxel = (px * px * vol.spacing[2]) / 1000;
        let l = 0;
        let t = 0;
        let best = 0;
        let bestT = -1;
        stack.forEach((c, z) => {
          let tz = 0;
          for (const v of c) {
            if (v >= 1) l++;
            if (v === 2) { t++; tz++; }
          }
          if (tz > bestT) { bestT = tz; best = z; }
        });
        setRes({ kind: '3d', name: file.name, vol, stack, scale, liverCc: l * voxel, tumourCc: t * voxel, lesions: lesions3d(stack, LIVER, LIVER), best });
      } else {
        setBusy('Outlining the liver');
        const gray = await fileToGray(file);
        const r = await liverSlice(gray);
        setRes({ kind: '2d', name: file.name, gray, ...r });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, []);

  return (
    <>
      {!ready && !loading && <NotReady what="liver" />}
      <div className="grid grid-cols-12 gap-x-6 gap-y-10">
        <div className="col-span-12 lg:col-span-7">
          {res ? <LiverViewer res={res} /> : <Dropzone onFile={run} disabled={!!busy || !ready} hint="An axial abdominal CT slice as JPG or PNG, or a whole CT scan as NIfTI (.nii or .nii.gz) in Hounsfield units." />}
          {busy && <Busy text={busy} />}
          {error && <p className="mt-4 border-l-2 border-vermilion pl-3 text-small" role="alert">Could not analyse this file: {error}</p>}
          {res && <Actions onReset={() => setRes(null)} />}
          {!res && <SampleStrip samples={samples.filter((s) => s.kind !== 'cervix')} disabled={!!busy || !ready} onPick={async (s) => run(await fetchSample(s.file))} />}
        </div>
        <aside className="col-span-12 lg:col-span-5" aria-label="Report">
          {res ? <LiverReport res={res} metrics={metrics} /> : <LiverHow />}
        </aside>
      </div>
    </>
  );
}

function LiverHow() {
  return (
    <div className="border-t border-rule-strong pt-5">
      <h2 className="display text-h3">What you get</h2>
      <ul className="mt-5 space-y-4 text-small">
        <li><span className="font-medium">Outline.</span> Liver in blue, tumour in red, on every slice.</li>
        <li><span className="font-medium">Size.</span> On a 3D scan: liver and tumour volume in cm³ and tumour burden (tumour as a share of the liver).</li>
        <li><span className="font-medium">Lesions.</span> Separate tumours counted across slices.</li>
        <li><span className="font-medium">Verdict.</span> Whether the scan shows a liver tumour, using the threshold validated on held-out patients.</li>
      </ul>
      <Disclaimer />
    </div>
  );
}

function LiverViewer({ res }: { res: Liver2D | Liver3D }) {
  const [z, setZ] = useState(res.kind === '3d' ? res.best : 0);
  const [show, setShow] = useState(true);
  const canvas = useRef<HTMLCanvasElement>(null);
  const view = useMemo(() => {
    if (res.kind === '2d') return { gray: res.gray, classes: res.classes, lb: res.lb };
    const gray = ctSlice(res.vol, z);
    return { gray, classes: res.stack[z], lb: letterbox(gray, LIVER).lb };
  }, [res, z]);
  useEffect(() => {
    const c = canvas.current!;
    paintGray(c, view.gray);
    if (show) paintClasses(c.getContext('2d')!, view.classes, LIVER, LIVER, view.lb, view.gray.w, view.gray.h, LIVER_COLOURS);
  }, [view, show]);
  const areas = res.kind === '3d' ? res.stack.map((c) => c.reduce((s, v) => s + (v === 2 ? 1 : 0), 0)) : [];
  return (
    <figure>
      <div className="film">
        <div className="flex items-center justify-between gap-3 border-b border-film-rule px-4 py-2.5">
          <span className="truncate font-mono text-small">{res.name}</span>
          <button type="button" onClick={() => setShow((v) => !v)} className="meta px-2 py-1 text-film-muted hover:text-film-fg" aria-pressed={show}>{show ? 'Hide outline' : 'Show outline'}</button>
        </div>
        <canvas key={res.name} ref={canvas} className="develop-sweep block h-auto max-h-[70vh] w-full object-contain" aria-label="CT with liver and tumour outline" />
        {res.kind === '3d' && (
          <div className="border-t border-film-rule px-4 py-3">
            <label className="flex items-center gap-3 text-small text-film-muted">
              <span className="num w-16 font-mono">z {z}/{res.vol.nz - 1}</span>
              <input type="range" min={0} max={res.vol.nz - 1} value={z} onChange={(e) => setZ(Number(e.target.value))} className="w-full accent-[#5474ff]" aria-label="Slice" />
            </label>
            <div className="mt-2 flex h-8 items-end gap-px" aria-hidden="true">
              {areas.map((a, i) => <span key={i} className={`flex-1 ${i === z ? 'bg-film-fg' : a > 0 ? 'bg-[#e63b2b]' : 'bg-film-rule'}`} style={{ height: `${Math.max(6, (a / Math.max(1, ...areas)) * 100)}%` }} />)}
            </div>
          </div>
        )}
      </div>
      <figcaption className="mt-2 text-small text-muted">Blue: liver. Red: tumour.{res.kind === '3d' && ' Bars show tumour area per slice.'}</figcaption>
    </figure>
  );
}

function LiverReport({ res, metrics }: { res: Liver2D | Liver3D; metrics: OrganMetrics | null }) {
  const m = metrics?.liver;
  const thr = m?.patient_threshold_burden ?? 0.002;
  let title: string;
  let body: string;
  let tone = 'border-cobalt';
  if (res.kind === '2d') {
    if (res.liverPx < 400) {
      title = 'No liver visible on this slice';
      body = 'Upload an axial slice through the upper abdomen, where the liver is.';
      tone = 'border-muted';
    } else if (res.tumourPx >= 30) {
      title = 'Tumour visible on this slice';
      body = `Tumour covers ${pct(res.tumourPx / res.liverPx)} of the liver on this slice. A single slice can miss tumours elsewhere: a whole CT scan is more reliable.`;
      tone = 'border-vermilion';
    } else {
      title = 'No tumour visible on this slice';
      body = 'Nothing on this slice was outlined as tumour. Tumours elsewhere in the liver would not show on a single slice.';
    }
  } else {
    const burden = res.tumourCc / Math.max(1e-9, res.liverCc);
    if (burden > thr && res.lesions.length) {
      title = 'Liver tumour detected';
      body = `${res.lesions.length} separate lesion${res.lesions.length > 1 ? 's' : ''} found across the scan.`;
      tone = 'border-vermilion';
    } else {
      title = 'No liver tumour found';
      body = 'The outlined tumour volume is below the level validated to separate patients with and without tumours.';
    }
  }
  return (
    <div className="print-report">
      <p className="meta text-muted">Report · {new Date().toLocaleString()} · {res.name}</p>
      <section className={`mt-3 border-l-4 ${tone} bg-raised p-5`}>
        <h2 className="display text-h3">{title}</h2>
        <p className="mt-2 text-small">{body}</p>
      </section>
      {res.kind === '3d' && (
        <dl className="mt-6 grid grid-cols-2 gap-px bg-rule">
          {[
            ['Liver volume', `${res.liverCc.toFixed(0)} cm³`],
            ['Tumour volume', `${res.tumourCc.toFixed(1)} cm³`],
            ['Tumour burden', pct(res.tumourCc / Math.max(1e-9, res.liverCc))],
            ['Lesions', String(res.lesions.length)],
          ].map(([k, v]) => (
            <div key={k} className="bg-surface p-4"><dt className="text-small text-muted">{k}</dt><dd className="display num mt-1 text-h3">{v}</dd></div>
          ))}
        </dl>
      )}
      {m && (
        <p className="mt-4 text-small text-muted">
          On {m.test_patients} held-out patients the outline matched experts with Dice {m.volume_dice_liver.mean?.toFixed(2)} for the liver and {m.volume_dice_tumour.mean?.toFixed(2)} for tumours
          {m.patient_detection && `; tumour detection per patient: sensitivity ${pct(m.patient_detection.sensitivity)}, specificity ${pct(m.patient_detection.specificity)}`}.
          {res.kind === '3d' && ' Volumes use the voxel size stored in your file.'}
        </p>
      )}
      <Disclaimer />
    </div>
  );
}

// ================================================================== cervix
interface CervixResult { name: string; rgba: Uint8ClampedArray; classes: Uint8Array; owner: Int32Array; cells: Cell[] }

export function CervixTool() {
  const { metrics, samples, ready, loading } = useOrgans();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<CervixResult | null>(null);
  const run = useCallback(async (file: File) => {
    setError(null);
    setRes(null);
    try {
      setBusy('Outlining cells and nuclei');
      const { rgb } = await cervixInput(file);
      const classes = await cervixClasses(rgb);
      setBusy('Separating cells and measuring them');
      const { owner, cells } = cellsFromClasses(classes, CERV_W, CERV_H);
      setRes({ name: file.name, rgba: rgb, classes, owner, cells });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, []);
  return (
    <>
      {!ready && !loading && <NotReady what="cytology" />}
      <div className="grid grid-cols-12 gap-x-6 gap-y-10">
        <div className="col-span-12 lg:col-span-7">
          {res ? <CervixViewer res={res} threshold={metrics?.cervix?.unusual_nc_threshold ?? null} /> : <Dropzone onFile={run} disabled={!!busy || !ready} accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff" hint="A microscope image of a Papanicolaou-stained cervical smear (JPG or PNG), ideally at around 10x to 20x magnification." />}
          {busy && <Busy text={busy} />}
          {error && <p className="mt-4 border-l-2 border-vermilion pl-3 text-small" role="alert">Could not analyse this file: {error}</p>}
          {res && <Actions onReset={() => setRes(null)} />}
          {!res && <SampleStrip samples={samples.filter((s) => s.kind === 'cervix')} disabled={!!busy || !ready} onPick={async (s) => run(await fetchSample(s.file))} />}
        </div>
        <aside className="col-span-12 lg:col-span-5" aria-label="Report">
          {res ? <CervixReport res={res} metrics={metrics} /> : (
            <div className="border-t border-rule-strong pt-5">
              <h2 className="display text-h3">What you get</h2>
              <ul className="mt-5 space-y-4 text-small">
                <li><span className="font-medium">Cells and nuclei.</span> Every cell outlined, its nucleus marked, touching cells separated.</li>
                <li><span className="font-medium">Counts.</span> Number of cells and nuclei in the image.</li>
                <li><span className="font-medium">N:C ratio.</span> Nucleus-to-cytoplasm area ratio for each cell. Abnormal (dysplastic) cells typically have enlarged nuclei and a high ratio.</li>
                <li><span className="font-medium">Cells to review.</span> Cells whose ratio is higher than 99% of the cells in the training slides are ringed.</li>
              </ul>
              <p className="mt-6 text-small text-muted">The training data outlines cells but carries no diagnosis, so this tool measures; it does not classify a smear as normal or abnormal.</p>
              <Disclaimer />
            </div>
          )}
        </aside>
      </div>
    </>
  );
}

function CervixViewer({ res, threshold }: { res: CervixResult; threshold: number | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [show, setShow] = useState(true);
  useEffect(() => {
    const c = canvas.current!;
    c.width = CERV_W;
    c.height = CERV_H;
    const ctx = c.getContext('2d')!;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(res.rgba), CERV_W, CERV_H), 0, 0);
    if (!show) return;
    const img = ctx.getImageData(0, 0, CERV_W, CERV_H);
    for (let i = 0; i < res.classes.length; i++) {
      const k = res.classes[i];
      const o = res.owner[i];
      const y = Math.floor(i / CERV_W);
      const x = i - y * CERV_W;
      const edge = o && ((x + 1 < CERV_W && res.owner[i + 1] !== o) || (y + 1 < CERV_H && res.owner[i + CERV_W] !== o));
      if (edge) {
        img.data[i * 4] = 20; img.data[i * 4 + 1] = 40; img.data[i * 4 + 2] = 200;
      } else if (k === 2) {
        img.data[i * 4] = Math.round(img.data[i * 4] * 0.4 + 230 * 0.6);
        img.data[i * 4 + 1] = Math.round(img.data[i * 4 + 1] * 0.4 + 59 * 0.6);
        img.data[i * 4 + 2] = Math.round(img.data[i * 4 + 2] * 0.4 + 43 * 0.6);
      }
    }
    ctx.putImageData(img, 0, 0);
    if (threshold !== null) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffb000';
      for (const cell of res.cells) {
        if (cell.nc < threshold) continue;
        const r = Math.sqrt((cell.nucleusPx + cell.cytoplasmPx) / Math.PI) + 6;
        ctx.beginPath();
        ctx.arc(cell.cx, cell.cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }, [res, show, threshold]);
  return (
    <figure>
      <div className="film">
        <div className="flex items-center justify-between gap-3 border-b border-film-rule px-4 py-2.5">
          <span className="truncate font-mono text-small">{res.name}</span>
          <button type="button" onClick={() => setShow((v) => !v)} className="meta px-2 py-1 text-film-muted hover:text-film-fg" aria-pressed={show}>{show ? 'Hide outline' : 'Show outline'}</button>
        </div>
        <canvas key={res.name} ref={canvas} className="develop-sweep block h-auto w-full" aria-label="Cytology image with cells and nuclei outlined" />
      </div>
      <figcaption className="mt-2 text-small text-muted">Blue lines: cell boundaries. Red: nuclei. Amber rings: cells with an unusually high nucleus-to-cytoplasm ratio.</figcaption>
    </figure>
  );
}

function CervixReport({ res, metrics }: { res: CervixResult; metrics: OrganMetrics | null }) {
  const m = metrics?.cervix;
  const thr = m?.unusual_nc_threshold ?? null;
  const ncs = res.cells.map((c) => c.nc).sort((a, b) => a - b);
  const median = ncs.length ? ncs[Math.floor(ncs.length / 2)] : null;
  const unusual = thr === null ? [] : res.cells.filter((c) => c.nc >= thr).sort((a, b) => b.nc - a.nc);
  const bins = 12;
  const maxNc = Math.max(1, ...ncs);
  const hist = Array.from({ length: bins }, (_, i) => ncs.filter((v) => v >= (i * maxNc) / bins && v < ((i + 1) * maxNc) / bins + (i === bins - 1 ? 1e-9 : 0)).length);
  return (
    <div className="print-report">
      <p className="meta text-muted">Report · {new Date().toLocaleString()} · {res.name}</p>
      <section className={`mt-3 border-l-4 ${unusual.length ? 'border-[#ffb000]' : 'border-cobalt'} bg-raised p-5`}>
        <h2 className="display text-h3">{res.cells.length} cells measured</h2>
        <p className="mt-2 text-small">
          {unusual.length
            ? `${unusual.length} cell${unusual.length > 1 ? 's have' : ' has'} a nucleus-to-cytoplasm ratio above ${thr!.toFixed(2)}, higher than 99% of cells in the training slides. Worth a closer look by a cytologist.`
            : 'No cell has an unusually high nucleus-to-cytoplasm ratio compared with the training slides.'}
        </p>
      </section>
      <dl className="mt-6 grid grid-cols-2 gap-px bg-rule">
        <div className="bg-surface p-4"><dt className="text-small text-muted">Cells with a nucleus</dt><dd className="display num mt-1 text-h3">{res.cells.length}</dd></div>
        <div className="bg-surface p-4"><dt className="text-small text-muted">Median N:C ratio</dt><dd className="display num mt-1 text-h3">{median !== null ? median.toFixed(2) : '—'}</dd></div>
      </dl>
      <figure className="mt-6">
        <div className="flex h-24 items-end gap-0.5" aria-hidden="true">
          {hist.map((n, i) => <span key={i} className={`flex-1 ${thr !== null && ((i + 1) * maxNc) / bins > thr ? 'bg-[#ffb000]' : 'bg-cobalt'}`} style={{ height: `${(n / Math.max(1, ...hist)) * 100}%` }} />)}
        </div>
        <figcaption className="mt-2 text-small text-muted">Distribution of N:C ratios in this image (0 to {maxNc.toFixed(2)}).</figcaption>
      </figure>
      {m && (
        <p className="mt-4 text-small text-muted">
          On {m.test_images} held-out images: nucleus outline Dice {m.dice_nucleus.mean?.toFixed(2)}, cell outline Dice {m.dice_cell.mean?.toFixed(2)}; nuclei counted within {m.nucleus_count_abs_error.median?.toFixed(0)} of the expert count (median).
          Measurements are in pixels, so they do not depend on knowing the magnification.
        </p>
      )}
      <Disclaimer />
    </div>
  );
}

// ================================================================== drawing helpers
const LIVER_COLOURS: Record<number, [number, number, number, number, number]> = { 1: [84, 116, 255, 60, 255], 2: [230, 59, 43, 120, 255] };

function paintGray(c: HTMLCanvasElement, g: Gray) {
  c.width = g.w;
  c.height = g.h;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(g.w, g.h);
  for (let i = 0; i < g.data.length; i++) {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = g.data[i];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Classes drawn in letterboxed space, then mapped back over the original image: translucent fill, solid edges. */
function paintClasses(ctx: CanvasRenderingContext2D, cls: Uint8Array, w: number, h: number, lb: Letterbox, outW: number, outH: number, colours: typeof LIVER_COLOURS) {
  const off = document.createElement('canvas');
  off.width = w;
  off.height = h;
  const o = off.getContext('2d')!;
  const im = o.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const k = cls[i];
      const col = colours[k];
      if (!col) continue;
      const edge = (x > 0 && cls[i - 1] !== k) || (x + 1 < w && cls[i + 1] !== k) || (y > 0 && cls[i - w] !== k) || (y + 1 < h && cls[i + w] !== k);
      im.data[i * 4] = col[0];
      im.data[i * 4 + 1] = col[1];
      im.data[i * 4 + 2] = col[2];
      im.data[i * 4 + 3] = edge ? col[4] : col[3];
    }
  }
  o.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, lb.left, lb.top, lb.w, lb.h, 0, 0, outW, outH);
}

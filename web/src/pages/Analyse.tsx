import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import { LuCircleAlert, LuFileUp, LuLoader, LuPrinter, LuRotateCcw, LuShieldCheck } from 'react-icons/lu';
import { Container, PageHead } from '../components/chrome';
import { CervixTool, LiverTool } from './AnalyseOrgans';
import { Link, useTitle } from '../lib/router';
import {
  CLASS_LABEL, CLASSES, classify, fileToGray, predictSurvival, readNifti, SEG_SIZE, segment, segmentBatch, survivalDesign, tumourProbability, yieldToBrowser,
  verdict, volumeSlice, letterbox, type Classification, type Gray, type Resection, type Segmentation, type SurvivalModel, type Verdict, type Volume,
} from '../lib/screen';
import { best3d, pct, useScreen, type Sample, type ScreenMetrics } from '../lib/screenMetrics';

type Stage = 'idle' | 'engine' | 'classify' | 'segment' | 'volume' | 'done' | 'error';
const STAGE_TEXT: Record<Stage, string> = {
  idle: '',
  engine: 'Loading the analysis engine (about 30 MB, once per visit)',
  classify: 'Checking the image',
  segment: 'Outlining',
  volume: 'Outlining every slice',
  done: '',
  error: '',
};

interface Result2D {
  kind: '2d';
  name: string;
  gray: Gray;
  cls: Classification;
  seg: Segmentation;
}
interface Result3D {
  kind: '3d';
  name: string;
  vol: Volume;
  probs: Float32Array[]; // per slice, SEG_SIZE^2
  lbScale: number;
  areas: number[]; // tumour pixels per slice
  volumeCc: number;
  best: number; // slice with the largest tumour area
  cls: Classification | null;
  bestGray: Gray;
  bestSeg: Segmentation | null;
}
type Result = Result2D | Result3D;

export const isVolumeFile = (name: string) => /\.nii(\.gz)?$/i.test(name);

type Mode = 'brain' | 'liver' | 'cervix';
const MODES: Record<Mode, { tab: string; sub: string; title: string; lead: string }> = {
  brain: { tab: 'Brain MRI', sub: 'tumour check, outline, prognosis', title: 'Upload a brain MRI.', lead: 'Get a tumour check, an outline with its size, and a survival estimate. The models run inside this browser tab, so the scan is never uploaded anywhere.' },
  liver: { tab: 'Liver CT', sub: 'liver and tumour outline, burden', title: 'Upload a liver CT.', lead: 'Outline the liver and any tumours on a CT slice, or on a whole CT scan get liver and tumour volume, tumour burden and the number of lesions. Computed in this browser tab.' },
  cervix: { tab: 'Cervical cells', sub: 'cells, nuclei, N:C ratio', title: 'Upload a cervical cell image.', lead: 'Outline every cell and nucleus in a microscope image of cervical cells, count them, and measure each cell’s nucleus-to-cytoplasm ratio, flagging unusual cells for review. Computed in this browser tab.' },
};

function ModeTabs({ mode, onPick }: { mode: Mode; onPick: (m: Mode) => void }) {
  return (
    <div role="tablist" aria-label="Scan type" className="mb-10 grid grid-cols-1 gap-px border border-rule-strong bg-rule-strong sm:grid-cols-3">
      {(Object.keys(MODES) as Mode[]).map((m) => (
        <button
          key={m}
          type="button"
          role="tab"
          aria-selected={mode === m}
          onClick={() => onPick(m)}
          className={`flex items-baseline justify-between gap-3 px-5 py-4 text-left transition-colors ${mode === m ? 'bg-fg text-surface' : 'bg-surface hover:bg-raised'}`}
        >
          <span className="display text-lead">{MODES[m].tab}</span>
          <span className={`meta ${mode === m ? 'text-surface/70' : 'text-muted'}`}>{MODES[m].sub}</span>
        </button>
      ))}
    </div>
  );
}

export function Analyse() {
  useTitle('Analyse a scan');
  const [mode, setMode] = useState<Mode>(() => (new URLSearchParams(window.location.search).get('type') as Mode) || 'brain');
  const pick = (m: Mode) => {
    setMode(m);
    const u = new URL(window.location.href);
    u.searchParams.set('type', m);
    window.history.replaceState({}, '', u);
  };
  return (
    <>
      <PageHead eyebrow="Analyse a scan" title={MODES[mode].title}>{MODES[mode].lead}</PageHead>
      <Container className="pb-24">
        <ModeTabs mode={mode} onPick={pick} />
        {mode === 'brain' && <BrainTool />}
        {mode === 'liver' && <LiverTool />}
        {mode === 'cervix' && <CervixTool />}
      </Container>
    </>
  );
}

function BrainTool() {
  const { metrics, survival, samples, ready, loading } = useScreen();
  const [stage, setStage] = useState<Stage>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const threshold = metrics?.classifier?.test.abstention.threshold ?? 0.6;

  const run = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setProgress(0);
    try {
      setStage('engine');
      if (isVolumeFile(file.name)) {
        const vol = await readNifti(file);
        setStage('volume');
        const probs: Float32Array[] = [];
        const areas: number[] = [];
        let scale = 1;
        const B = 4;
        for (let z = 0; z < vol.nz; z += B) {
          await yieldToBrowser();
          const batch: Uint8Array[] = [];
          for (let k = z; k < Math.min(vol.nz, z + B); k++) {
            const lb = letterbox(volumeSlice(vol, k), SEG_SIZE);
            scale = lb.lb.scale;
            batch.push(lb.pixels);
          }
          const out = await segmentBatch(batch);
          for (const p of out) {
            probs.push(p);
            let a = 0;
            for (let i = 0; i < p.length; i++) if (p[i] > 0.5) a++;
            areas.push(a);
          }
          setProgress(Math.min(1, (z + B) / vol.nz));
        }
        const best = areas.indexOf(Math.max(...areas));
        // Voxel volume after letterboxing: in-plane spacing grows by 1/scale; slice thickness is unchanged.
        const px = Math.max(vol.spacing[0], vol.spacing[1]) / scale;
        const volumeCc = (areas.reduce((s, a) => s + a, 0) * px * px * vol.spacing[2]) / 1000;
        const bestGray = volumeSlice(vol, best);
        setStage('classify');
        const cls = await classify(bestGray);
        const bestSeg = await segment(bestGray);
        setResult({ kind: '3d', name: file.name, vol, probs, lbScale: scale, areas, volumeCc, best, cls, bestGray, bestSeg });
      } else {
        const gray = await fileToGray(file);
        setStage('classify');
        const cls = await classify(gray);
        setStage('segment');
        const seg = await segment(gray);
        setResult({ kind: '2d', name: file.name, gray, cls, seg });
      }
      setStage('done');
    } catch (e) {
      console.error(e);
      setError(e instanceof Error ? e.message : String(e));
      setStage('error');
    }
  }, []);

  const runSample = useCallback(async (s: Sample) => {
    const res = await fetch(new URL(`screen/samples/${s.file}`, document.baseURI));
    const blob = await res.blob();
    await run(new File([blob], s.file, { type: blob.type }));
  }, [run]);

  const busy = stage !== 'idle' && stage !== 'done' && stage !== 'error';

  return (
    <>
        {!ready && !loading && (
          <div className="mb-8 flex gap-3 border border-vermilion/50 bg-vermilion/5 p-4 text-small" role="status">
            <LuCircleAlert className="mt-0.5 size-5 shrink-0 text-vermilion-text" aria-hidden="true" />
            <p>
              The screening models are being trained and evaluated. This page will analyse scans as soon as they are published here,
              together with their measured accuracy. Nothing on this page is simulated.
            </p>
          </div>
        )}
        <div className="grid grid-cols-12 gap-x-6 gap-y-10">
          <div className="col-span-12 lg:col-span-7">
            {result ? <Viewer result={result} /> : <Dropzone disabled={busy || !ready} onFile={run} />}
            {busy && (
              <div className="mt-4 flex items-center gap-3 text-small" role="status" aria-live="polite">
                <LuLoader className="size-4 animate-spin" aria-hidden="true" />
                <span>{STAGE_TEXT[stage]}{stage === 'volume' ? ` · ${Math.round(progress * 100)}%` : ''}</span>
              </div>
            )}
            {error && (
              <p className="mt-4 border-l-2 border-vermilion pl-3 text-small" role="alert">
                Could not analyse this file: {error}
              </p>
            )}
            {result && (
              <div className="no-print mt-4 flex flex-wrap gap-2">
                <button type="button" onClick={() => { setResult(null); setStage('idle'); }} className="inline-flex h-10 items-center gap-2 border border-rule-strong px-4 text-small hover:bg-raised">
                  <LuRotateCcw className="size-4" aria-hidden="true" /> Analyse another scan
                </button>
                <button type="button" onClick={() => window.print()} className="inline-flex h-10 items-center gap-2 border border-rule-strong px-4 text-small hover:bg-raised">
                  <LuPrinter className="size-4" aria-hidden="true" /> Print or save the report
                </button>
              </div>
            )}
            {!result && samples.length > 0 && <Samples samples={samples} disabled={busy || !ready} onPick={runSample} />}
          </div>
          <aside className="col-span-12 lg:col-span-5" aria-label="Report">
            {result ? (
              <Report result={result} metrics={metrics} survival={survival} threshold={threshold} />
            ) : (
              <HowToUse />
            )}
          </aside>
        </div>
    </>
  );
}

// ------------------------------------------------------------------ input
export function Dropzone({ onFile, disabled, hint = 'A brain MRI slice as JPG or PNG, or a whole 3D scan as NIfTI (.nii or .nii.gz) to measure tumour volume.', accept = '.jpg,.jpeg,.png,.bmp,.nii,.gz' }: { onFile: (f: File) => void; disabled: boolean; hint?: string; accept?: string }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f && !disabled) onFile(f);
  };
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={`film flex aspect-[4/3] flex-col items-center justify-center gap-4 border-2 border-dashed p-8 text-center transition-colors ${over ? 'border-[#5474ff]' : 'border-film-rule'}`}
    >
      <LuFileUp className="size-10 text-film-muted" aria-hidden="true" />
      <p className="display text-h3 text-film-fg">Drop a scan here</p>
      <p className="max-w-[44ch] text-small text-film-muted">{hint}</p>
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        className="h-11 bg-film-fg px-5 text-small font-medium text-film hover:bg-white disabled:opacity-40"
      >
        Choose a file
      </button>
      <input ref={input} type="file" accept={accept} className="sr-only" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
      <p className="meta flex items-center gap-1.5 text-film-muted"><LuShieldCheck className="size-3.5" aria-hidden="true" /> Processed on this device. Nothing is uploaded.</p>
    </div>
  );
}

export function Samples({ samples, onPick, disabled }: { samples: Sample[]; onPick: (s: Sample) => void; disabled: boolean }) {
  return (
    <section className="mt-8" aria-labelledby="samples-title">
      <h2 id="samples-title" className="font-semibold">Or try a held-out test scan</h2>
      <p className="mt-1 text-small text-muted">These were never used for training. Their true diagnosis is shown so you can check the answer.</p>
      <ul className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
        {samples.map((s) => (
          <li key={s.file}>
            <button type="button" disabled={disabled} onClick={() => onPick(s)} className="group block w-full text-left disabled:opacity-50">
              {s.kind === 'volume' ? (
                <span className="film grid aspect-square place-items-center text-center font-mono text-[0.7rem] text-film-muted">3D scan<br />T2</span>
              ) : (
                <img src={`screen/samples/${s.file}`} alt="" className="aspect-square w-full bg-film object-cover group-hover:opacity-80" loading="lazy" />
              )}
              <span className="meta mt-1 block text-muted">{CLASS_LABEL[s.truth as keyof typeof CLASS_LABEL] ?? s.truth}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function HowToUse() {
  return (
    <div className="border-t border-rule-strong pt-5">
      <h2 className="display text-h3">What you get</h2>
      <ol className="mt-5 space-y-5">
        {[
          ['Tumour check', 'No tumour, glioma, meningioma or pituitary tumour, with a calibrated confidence. Images that are not brain MRIs, or that it is unsure about, are flagged instead of guessed.'],
          ['Outline and size', 'The tumour region drawn on the scan, its area on a single slice, or its volume in cm³ on a 3D scan.'],
          ['Why it decided', 'A heat map of the image regions that drove the decision.'],
          ['Prognosis', 'For glioma, an estimated survival curve from age, tumour volume and extent of surgery, with its uncertainty band.'],
        ].map(([t, d], i) => (
          <li key={t} className="grid grid-cols-[2rem_1fr] gap-x-3">
            <span className="font-mono text-small text-muted">0{i + 1}</span>
            <span><span className="block font-medium">{t}</span><span className="text-small text-muted">{d}</span></span>
          </li>
        ))}
      </ol>
      <p className="mt-8 border-l-2 border-vermilion pl-3 text-small">
        A research tool, not a medical device. It does not replace a radiologist and must not be used to make decisions about anyone’s care.
        <Link to="/about#limitations" className="ml-1 link-underline">Limitations</Link>
      </p>
    </div>
  );
}

// ------------------------------------------------------------------ viewer
type Layer = 'outline' | 'heatmap' | 'none';

function Viewer({ result }: { result: Result }) {
  const [layer, setLayer] = useState<Layer>('outline');
  const [z, setZ] = useState(result.kind === '3d' ? result.best : 0);
  useEffect(() => setZ(result.kind === '3d' ? result.best : 0), [result]);
  const canvas = useRef<HTMLCanvasElement>(null);

  const view = useMemo(() => {
    if (result.kind === '2d') return { gray: result.gray, prob: result.seg.prob, segLb: result.seg.lb, cls: result.cls };
    const gray = volumeSlice(result.vol, z);
    const segLb = letterbox(gray, SEG_SIZE).lb;
    return { gray, prob: result.probs[z], segLb, cls: z === result.best ? result.cls : null };
  }, [result, z]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const { gray } = view;
    c.width = gray.w;
    c.height = gray.h;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(gray.w, gray.h);
    for (let i = 0; i < gray.data.length; i++) {
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = gray.data[i];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    if (layer === 'outline') drawMask(ctx, view.prob, SEG_SIZE, view.segLb, gray);
    if (layer === 'heatmap' && view.cls) drawCam(ctx, view.cls, gray);
  }, [view, layer]);

  const options: [Layer, string][] = [['outline', 'Tumour outline'], ['heatmap', 'Decision heat map'], ['none', 'Scan only']];
  return (
    <figure>
      <div className="film">
        <div className="flex items-center justify-between gap-3 border-b border-film-rule px-4 py-2.5">
          <span className="truncate font-mono text-small">{result.name}</span>
          <div className="flex shrink-0" role="radiogroup" aria-label="Overlay">
            {options.map(([k, label]) => (
              <button
                key={k}
                type="button"
                role="radio"
                aria-checked={layer === k}
                disabled={k === 'heatmap' && !view.cls}
                onClick={() => setLayer(k)}
                className={`meta px-2 py-1 disabled:opacity-30 ${layer === k ? 'bg-film-fg text-film' : 'text-film-muted hover:text-film-fg'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <canvas key={result.name} ref={canvas} className="develop-sweep block h-auto max-h-[70vh] w-full object-contain [image-rendering:auto]" aria-label={`Scan ${result.name} with ${layer} overlay`} />
        {result.kind === '3d' && (
          <div className="border-t border-film-rule px-4 py-3">
            <label className="flex items-center gap-3 text-small text-film-muted">
              <span className="font-mono num w-16">z {z}/{result.vol.nz - 1}</span>
              <input type="range" min={0} max={result.vol.nz - 1} value={z} onChange={(e) => setZ(Number(e.target.value))} className="w-full accent-[#5474ff]" aria-label="Slice" />
            </label>
            <SliceStrip areas={result.areas} z={z} onPick={setZ} />
          </div>
        )}
      </div>
      <figcaption className="mt-2 text-small text-muted">
        {layer === 'outline' && 'Red: region the outliner marks as tumour (probability above 0.5).'}
        {layer === 'heatmap' && 'Warm colours: the image regions that pushed the tumour check towards its answer.'}
        {result.kind === '3d' && ' The bars under the slider show tumour area per slice; the heat map is computed on the slice with the largest area.'}
      </figcaption>
    </figure>
  );
}

function SliceStrip({ areas, z, onPick }: { areas: number[]; z: number; onPick: (z: number) => void }) {
  const max = Math.max(1, ...areas);
  return (
    <div className="mt-2 flex h-8 items-end gap-px" aria-hidden="true">
      {areas.map((a, i) => (
        <button key={i} type="button" tabIndex={-1} onClick={() => onPick(i)} className={`flex-1 ${i === z ? 'bg-film-fg' : a > 0 ? 'bg-[#e63b2b]' : 'bg-film-rule'}`} style={{ height: `${Math.max(6, (a / max) * 100)}%` }} />
      ))}
    </div>
  );
}

function drawMask(ctx: CanvasRenderingContext2D, prob: Float32Array, size: number, lb: { left: number; top: number; w: number; h: number }, gray: Gray) {
  const off = document.createElement('canvas');
  off.width = off.height = size;
  const o = off.getContext('2d')!;
  const im = o.createImageData(size, size);
  const on = (x: number, y: number) => x >= 0 && y >= 0 && x < size && y < size && prob[y * size + x] > 0.5;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!on(x, y)) continue;
      const edge = !on(x - 1, y) || !on(x + 1, y) || !on(x, y - 1) || !on(x, y + 1);
      const i = (y * size + x) * 4;
      im.data[i] = 230;
      im.data[i + 1] = 59;
      im.data[i + 2] = 43;
      im.data[i + 3] = edge ? 255 : 70;
    }
  }
  o.putImageData(im, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, lb.left, lb.top, lb.w, lb.h, 0, 0, gray.w, gray.h);
}

function drawCam(ctx: CanvasRenderingContext2D, cls: Classification, gray: Gray) {
  const k = cls.probs.indexOf(Math.max(...cls.probs.slice(0, 4)));
  const n = cls.camSize;
  const cam = cls.cams.subarray(k * n * n, (k + 1) * n * n);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of cam) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const small = document.createElement('canvas');
  small.width = small.height = n;
  const s = small.getContext('2d')!;
  const im = s.createImageData(n, n);
  for (let i = 0; i < cam.length; i++) {
    const t = hi > lo ? (cam[i] - lo) / (hi - lo) : 0;
    // cobalt (cold) -> vermilion (hot), transparent where the evidence is weak
    im.data[i * 4] = Math.round(27 + t * (230 - 27));
    im.data[i * 4 + 1] = Math.round(64 + t * (59 - 64));
    im.data[i * 4 + 2] = Math.round(216 + t * (43 - 216));
    im.data[i * 4 + 3] = Math.round(Math.max(0, t - 0.25) * 200);
  }
  s.putImageData(im, 0, 0);
  const lb = cls.lb;
  // CAM covers the 224 x 224 letterboxed input; crop to the image part and stretch over the original.
  const sx = (lb.left / 224) * n;
  const sy = (lb.top / 224) * n;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(small, sx, sy, (lb.w / 224) * n, (lb.h / 224) * n, 0, 0, gray.w, gray.h);
}

// ------------------------------------------------------------------ report
function Report({ result, metrics, survival, threshold }: { result: Result; metrics: ScreenMetrics | null; survival: SurvivalModel | null; threshold: number }) {
  const cls = result.kind === '2d' ? result.cls : result.cls;
  const v: Verdict | null = cls ? verdict(cls.probs, threshold) : null;
  const tumourCc = result.kind === '3d' ? result.volumeCc : null;
  const vol3d = best3d(metrics);
  return (
    <div className="print-report">
      <p className="meta text-muted">Report · {new Date().toLocaleString()} · {result.name}</p>
      {v && <VerdictCard v={v} />}
      {cls && <Probabilities probs={cls.probs} />}
      <section className="mt-6 border-t border-rule pt-4">
        <h3 className="font-semibold">Size</h3>
        {result.kind === '2d' ? (
          <p className="mt-1 text-small">
            {result.seg.tumourPixels > 0
              ? `The outlined region covers ${pct(result.seg.tumourPixels / Math.max(1, result.seg.brainPixels))} of the visible head on this slice.`
              : 'Nothing was outlined on this slice.'}{' '}
            <span className="text-muted">A single image has no physical scale; upload the 3D scan (NIfTI) to measure volume in cm³.</span>
          </p>
        ) : (
          <p className="mt-1 text-small">
            <span className="display num text-h3">{tumourCc!.toFixed(1)} cm³</span>
            <span className="mt-1 block text-muted">
              Whole-tumour volume, from {result.areas.filter((a) => a > 0).length} slices with tumour.
              {vol3d && ` On held-out BraTS ${vol3d.sequence} scans this estimate was off by a median ${vol3d.volume_abs_error_cc.median?.toFixed(1)} cm³ (${pct(vol3d.volume_rel_error.median, 0)}; r = ${vol3d.volume_pearson_r?.toFixed(2)}).`}
            </span>
          </p>
        )}
      </section>
      {survival && ((v?.kind === 'tumour') || (tumourCc !== null && tumourCc >= 1)) && (
        <Prognosis model={survival} metrics={metrics} initialVolume={tumourCc} glioma={!v || (v.kind === 'tumour' && v.type === 'glioma')} />
      )}
      <p className="mt-8 border-l-2 border-vermilion pl-3 text-small">
        Research tool, not a medical device. Results can be wrong; they are not a diagnosis and must not guide anyone’s care.
      </p>
    </div>
  );
}

function VerdictCard({ v }: { v: Verdict }) {
  const content = {
    'not-brain': { tone: 'border-vermilion', title: 'This does not look like a brain MRI', body: 'The result is withheld. Upload an axial brain MRI slice (T1, T1 with contrast, T2 or FLAIR).' },
    uncertain: { tone: 'border-vermilion', title: 'Not confident enough to answer', body: `The most likely answer would be “${CLASS_LABEL[(v as { best: (typeof CLASSES)[number] }).best ?? 'no_tumour']}”, but the confidence is below the level at which the model was validated. Try a clearer slice through the middle of the head.` },
    'no-tumour': { tone: 'border-cobalt', title: 'No tumour found', body: 'No tumour was detected on this image.' },
    tumour: { tone: 'border-vermilion', title: 'Tumour detected', body: v.kind === 'tumour' ? `Most consistent with ${CLASS_LABEL[v.type].toLowerCase()} (${(v.typeConfidence * 100).toFixed(0)}% for this type).` : '' },
  }[v.kind];
  return (
    <section className={`mt-3 border-l-4 ${content.tone} bg-raised p-5`} aria-live="polite">
      <h2 className="display text-h3">{content.title}</h2>
      <p className="mt-2 text-small">{content.body}</p>
      {(v.kind === 'tumour' || v.kind === 'no-tumour') && (
        <p className="mt-3 font-mono text-small num">
          {v.kind === 'tumour' ? 'Tumour probability' : 'Confidence'} {(v.confidence * 100).toFixed(1)}%
        </p>
      )}
    </section>
  );
}

function Probabilities({ probs }: { probs: number[] }) {
  return (
    <section className="mt-6" aria-labelledby="probs-title">
      <h3 id="probs-title" className="font-semibold">Calibrated probabilities</h3>
      <ul className="mt-2 space-y-1.5">
        {CLASSES.map((c, i) => (
          <li key={c} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3 text-small">
            <span>{CLASS_LABEL[c]}</span>
            <span className="h-2 bg-rule" aria-hidden="true"><span className={`block h-full ${i === 0 ? 'bg-cobalt' : i === 4 ? 'bg-muted' : 'bg-vermilion'}`} style={{ width: `${probs[i] * 100}%` }} /></span>
            <span className="text-right font-mono num">{(probs[i] * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-small text-muted">Tumour vs no tumour: {(tumourProbability(probs) * 100).toFixed(1)}% tumour.</p>
    </section>
  );
}

function Prognosis({ model, metrics, initialVolume, glioma }: { model: SurvivalModel; metrics: ScreenMetrics | null; initialVolume: number | null; glioma: boolean }) {
  const [age, setAge] = useState(60);
  const [volume, setVolume] = useState<number>(initialVolume ? Math.round(initialVolume * 10) / 10 : 60);
  const [resection, setResection] = useState<Resection>('unknown');
  useEffect(() => { if (initialVolume) setVolume(Math.round(initialVolume * 10) / 10); }, [initialVolume]);
  const pred = useMemo(() => predictSurvival(model, survivalDesign(age, volume, resection)), [model, age, volume, resection]);
  const c = metrics?.survival?.cv_c_index;
  const months = (d: number | null) => (d === null ? '—' : `${(d / 30.44).toFixed(1)} months`);
  return (
    <section className="mt-6 border-t border-rule pt-4" aria-labelledby="prog-title">
      <h3 id="prog-title" className="font-semibold">Prognosis estimate</h3>
      <p className="mt-1 text-small text-muted">
        {glioma ? 'For glioma.' : 'This model was fitted on glioma patients only; for other tumour types it does not apply.'} Fitted on
        {' '}{model.n} BraTS 2020 patients ({model.events} deaths observed).
      </p>
      <div className="no-print mt-4 grid grid-cols-3 gap-3 text-small">
        <label className="flex flex-col gap-1">Age
          <input type="number" min={18} max={95} value={age} onChange={(e) => setAge(Number(e.target.value))} className="h-10 border border-rule-strong bg-surface px-2 font-mono" />
        </label>
        <label className="flex flex-col gap-1">Tumour cm³
          <input type="number" min={0} max={400} step={0.1} value={volume} onChange={(e) => setVolume(Number(e.target.value))} className="h-10 border border-rule-strong bg-surface px-2 font-mono" />
        </label>
        <label className="flex flex-col gap-1">Surgery
          <select value={resection} onChange={(e) => setResection(e.target.value as Resection)} className="h-10 border border-rule-strong bg-surface px-2">
            <option value="unknown">Unknown / none</option>
            <option value="GTR">Gross total</option>
            <option value="STR">Subtotal</option>
          </select>
        </label>
      </div>
      <p className="mt-4 text-small">
        Estimated median survival <span className="font-mono num">{months(pred.medianDays)}</span>
        {pred.medianLower !== null && pred.medianUpper !== null && (
          <span className="text-muted"> (90% band {months(pred.medianLower)} to {months(pred.medianUpper)})</span>
        )}
        {pred.riskGroup && <span>, {pred.riskGroup} risk third of the training cohort</span>}.
      </p>
      <SurvivalChart days={pred.days} s={pred.survival} lo={pred.lower} hi={pred.upper} />
      {c && (
        <p className="mt-2 text-small text-muted">
          How good is this? Cross-validated concordance {c.full.mean.toFixed(2)} (0.5 is chance, 1 is perfect ordering); age alone gives {c.age_only.mean.toFixed(2)}.
          Survival depends on far more than these three numbers, so treat this as a population average.
        </p>
      )}
    </section>
  );
}

export function SurvivalChart({ days, s, lo, hi, height = 180 }: { days: number[]; s: number[]; lo?: number[]; hi?: number[]; height?: number }) {
  const W = 480;
  const H = height;
  const pad = { l: 34, r: 8, t: 8, b: 26 };
  const maxD = days[days.length - 1] || 1;
  const x = (d: number) => pad.l + (d / maxD) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const line = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${x(days[i]).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const band = lo && hi ? `${line(hi)}${[...lo].reverse().map((v, i) => `L${x(days[days.length - 1 - i]).toFixed(1)},${y(v).toFixed(1)}`).join('')}Z` : '';
  const ticks = [0, 6, 12, 18, 24, 36, 48].filter((m) => m * 30.44 <= maxD);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full" role="img" aria-label="Estimated survival curve with uncertainty band">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--rule)" />
          <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" className="fill-[var(--muted)] font-mono text-[10px]">{v * 100}%</text>
        </g>
      ))}
      {ticks.map((m) => (
        <text key={m} x={x(m * 30.44)} y={H - 8} textAnchor="middle" className="fill-[var(--muted)] font-mono text-[10px]">{m}m</text>
      ))}
      {band && <path d={band} fill="var(--cobalt)" opacity={0.15} />}
      <line x1={pad.l} x2={W - pad.r} y1={y(0.5)} y2={y(0.5)} stroke="var(--vermilion)" strokeDasharray="3 3" />
      <path d={line(s)} fill="none" stroke="var(--cobalt)" strokeWidth={2} />
    </svg>
  );
}

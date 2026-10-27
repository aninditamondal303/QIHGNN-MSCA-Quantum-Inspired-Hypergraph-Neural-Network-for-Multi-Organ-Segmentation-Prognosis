import { useMemo, useState } from 'react';
import { LuRefreshCw } from 'react-icons/lu';
import fixtures from '../lib/fixtures.json';
import { apiError, type ApiState } from '../lib/data';
import { dateTime, int, sci } from '../lib/format';
import { attentionCost, marginalLoss, mpoRow, naiveLoss, reallocate, shiftToChannel, type Matrix } from '../lib/math';
import type { Verification } from '../lib/types';
import { Container, SectionHead } from './chrome';

const CHANNELS = ['background', 'brain', 'liver', 'kidney', 'cervix', 'spleen'];

function Instrument({ title, note, className = '', children }: { title: string; note: string; className?: string; children: React.ReactNode }) {
  return (
    <article className={`flex flex-col border-t-2 border-fg pt-4 ${className}`}>
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-1 max-w-[60ch] text-small text-muted">{note}</p>
      <div className="mt-6 flex-1">{children}</div>
    </article>
  );
}

function Readout({ label, value, tone = 'fg' }: { label: string; value: string; tone?: 'fg' | 'cobalt' | 'vermilion' }) {
  const color = tone === 'cobalt' ? 'text-cobalt' : tone === 'vermilion' ? 'text-vermilion-text' : '';
  return (
    <div>
      <p className="text-small text-muted">{label}</p>
      <p className={`display num mt-1 text-[clamp(1.5rem,1.1rem+1vw,2.1rem)] leading-none tracking-[-0.03em] ${color}`}>{value}</p>
    </div>
  );
}

function Slider({ label, value, min, max, step = 1, onChange, suffix = '' }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; suffix?: string }) {
  return (
    <label className="grid grid-cols-[4.5rem_1fr_3.5rem] items-center gap-3 text-small">
      <span className="text-muted">{label}</span>
      <input className="range" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} aria-label={label} />
      <span className="text-right font-mono num">{value}{suffix}</span>
    </label>
  );
}

// ------------------------------------------------------------------------------------------- A
type Mode = 'predict' | 'shuffle';

function MarginalInstrument() {
  const probs = fixtures.marginal.probs as Matrix;
  const labels = fixtures.marginal.labels as number[];
  const supervised = fixtures.marginal.supervised as boolean[];
  const [mode, setMode] = useState<Mode>('predict');
  const [target, setTarget] = useState(3);
  const [t, setT] = useState(0.5);
  // -log(p) diverges as the background probability reaches 0, so the first experiment stops at 90%.
  const tMax = mode === 'predict' ? 0.9 : 1;
  const tt0 = Math.min(t, tMax);
  const apply = (tt: number) => (mode === 'predict' ? shiftToChannel(probs, labels, target, tt) : reallocate(probs, supervised, target, tt));
  const q = useMemo(() => apply(tt0), [mode, target, tt0]); // eslint-disable-line react-hooks/exhaustive-deps
  const curve = useMemo(
    () =>
      Array.from({ length: 41 }, (_, i) => {
        const tt = (i / 40) * tMax;
        const qq = apply(tt);
        return { t: tt, m: marginalLoss(qq, labels, supervised), n: naiveLoss(qq, labels) };
      }),
    [mode, target], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const m0 = curve[0].m;
  const n0 = curve[0].n;
  const m = marginalLoss(q, labels, supervised);
  const n = naiveLoss(q, labels);
  const lo = Math.min(...curve.map((c) => Math.min(c.m, c.n))) - 0.1;
  const hi = Math.max(...curve.map((c) => Math.max(c.m, c.n))) + 0.1;
  const W = 520;
  const H = 150;
  const x = (v: number) => 34 + (v / tMax) * (W - 44);
  const y = (v: number) => 10 + (1 - (v - lo) / (hi - lo)) * (H - 30);
  const path = (k: 'm' | 'n') => curve.map((c, i) => `${i ? 'L' : 'M'}${x(c.t).toFixed(1)},${y(c[k]).toFixed(1)}`).join('');
  const voxels = labels.length;
  const name = CHANNELS[target];

  return (
    <Instrument
      className="col-span-12 lg:col-span-8"
      title="What each loss charges for an organ nobody labelled"
      note={`${voxels} voxels from a liver-only sample: background and liver are labelled, the other four channels are not. The marginal loss (cobalt) and a naive loss (vermilion) score the same predictions.`}
    >
      <div className="grid grid-cols-12 gap-x-6 gap-y-6">
        <div className="col-span-12 space-y-4 md:col-span-5">
          <div role="radiogroup" aria-label="Experiment" className="grid grid-cols-1 border border-rule-strong">
            {([
              ['predict', 'Model recognises an unlabelled organ'],
              ['shuffle', 'Report test: shuffle unlabelled mass'],
            ] as [Mode, string][]).map(([v, label]) => (
              <button key={v} type="button" role="radio" aria-checked={mode === v} onClick={() => setMode(v)} className={`border-b border-rule-strong px-3 py-2 text-left text-small last:border-b-0 ${mode === v ? 'bg-fg text-surface' : 'hover:bg-raised'}`}>
                {label}
              </button>
            ))}
          </div>
          <div role="radiogroup" aria-label="Unlabelled channel" className="flex border border-rule-strong">
            {[1, 3, 4, 5].map((c) => (
              <button key={c} type="button" role="radio" aria-checked={target === c} onClick={() => setTarget(c)} className={`flex-1 border-r border-rule-strong py-2 text-small capitalize last:border-r-0 ${target === c ? 'bg-fg text-surface' : 'hover:bg-raised'}`}>
                {CHANNELS[c]}
              </button>
            ))}
          </div>
          <Slider label="Moved" value={Math.round(tt0 * 100)} min={0} max={tMax * 100} onChange={(v) => setT(v / 100)} suffix="%" />
          <p className="text-small text-muted">
            {mode === 'predict'
              ? `On background-labelled voxels, ${Math.round(tt0 * 100)}% of the background probability goes to ${name}: the model calls tissue ${name} where the dataset was silent.`
              : `${Math.round(tt0 * 100)}% of every voxel’s unlabelled mass is concentrated into ${name}, as in the report’s Table 5.2.`}
          </p>
          <div className="grid grid-cols-2 gap-6 border-t border-rule pt-4">
            <Readout label="Marginal loss" value={m.toFixed(6)} tone="cobalt" />
            <Readout label="Naive loss" value={n.toFixed(6)} tone="vermilion" />
            <p className="font-mono text-small text-muted num">Δ {Math.abs(m - m0) < 1e-12 ? '0' : (m - m0).toExponential(1)}</p>
            <p className="font-mono text-small text-muted num">Δ {(n - n0 >= 0 ? '+' : '') + (n - n0).toFixed(3)}</p>
          </div>
        </div>
        <div className="col-span-12 md:col-span-7">
          <div className="flex h-28 items-stretch gap-px" role="img" aria-label={`Per-voxel probabilities after moving ${Math.round(tt0 * 100)}% toward ${name}`}>
            {Array.from({ length: voxels }, (_, i) => (
              <div key={i} className="flex flex-1 flex-col-reverse">
                {q.map((row, c) => (
                  <div key={c} style={{ height: `${row[i] * 100}%` }} className={c === 0 ? 'bg-fg/15' : c === 2 ? 'bg-cobalt' : c === target ? 'bg-vermilion' : 'hatch-fine'} />
                ))}
              </div>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[0.72rem] text-muted">
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 bg-fg/15" />background</span>
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 bg-cobalt" />liver</span>
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 bg-vermilion" />{name}</span>
            <span className="inline-flex items-center gap-1.5"><span className="hatch-fine size-2.5" />other unlabelled</span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="mt-5 w-full" role="img" aria-label="Loss against share moved: marginal stays flat, naive rises">
            <line x1={34} x2={W - 10} y1={H - 20} y2={H - 20} stroke="var(--rule-strong)" strokeWidth="1" />
            {[0, tMax / 2, tMax].map((v) => (
              <text key={v} x={x(v)} y={H - 6} textAnchor="middle" className="fill-muted font-mono text-[10px]">{Math.round(v * 100)}%</text>
            ))}
            {[lo + 0.1, hi - 0.1].map((v) => (
              <text key={v} x={28} y={y(v) + 3} textAnchor="end" className="fill-muted font-mono text-[10px]">{v.toFixed(2)}</text>
            ))}
            <path d={path('n')} fill="none" stroke="var(--vermilion)" strokeWidth="1.75" />
            <path d={path('m')} fill="none" stroke="var(--cobalt)" strokeWidth="1.75" />
            <line x1={x(tt0)} x2={x(tt0)} y1={6} y2={H - 20} stroke="var(--fg)" strokeWidth="1" strokeDasharray="2 3" />
            <circle cx={x(tt0)} cy={y(m)} r="3.5" fill="var(--cobalt)" />
            <circle cx={x(tt0)} cy={y(n)} r="3.5" fill="var(--vermilion)" />
          </svg>
          {mode === 'shuffle' && (
            <p className="mt-2 text-small text-muted">
              The naive loss only moves at 100%, when three channels become exactly empty and their smoothed Dice jumps to 1. This test proves
              invariance of the marginal loss, but the first experiment shows the failure it prevents.
            </p>
          )}
        </div>
      </div>
    </Instrument>
  );
}

// ------------------------------------------------------------------------------------------- B
function AttentionInstrument() {
  const [h, setH] = useState(128);
  const [w, setW] = useState(128);
  const [d, setD] = useState(64);
  const c = attentionCost([h, w, d]);
  const MAX = 18;
  const bar = (v: number) => `${(Math.log10(Math.max(v, 1)) / MAX) * 100}%`;
  return (
    <Instrument className="col-span-12 md:col-span-6 lg:col-span-4" title="Attention cost by input size" note="Voxel attention grows with the square of the voxel count. The multi-scale budgets do not move.">
      <div className="space-y-2">
        <Slider label="Height" value={h} min={16} max={512} step={16} onChange={setH} />
        <Slider label="Width" value={w} min={16} max={512} step={16} onChange={setW} />
        <Slider label="Depth" value={d} min={16} max={256} step={16} onChange={setD} />
      </div>
      <div className="mt-6 space-y-3" role="img" aria-label={`Naive ${sci(c.naive)} entries, multi-scale ${int(c.multi_scale)} entries`}>
        <div>
          <div className="flex justify-between text-small"><span>Voxel attention</span><span className="font-mono num">{sci(c.naive)}</span></div>
          <div className="mt-1 h-3 bg-raised"><div className="hatch h-full" style={{ width: bar(c.naive) }} /></div>
        </div>
        <div>
          <div className="flex justify-between text-small"><span>Multi-scale</span><span className="font-mono num">{int(c.multi_scale)}</span></div>
          <div className="mt-1 h-3 bg-raised"><div className="h-full bg-cobalt" style={{ width: bar(c.multi_scale) }} /></div>
        </div>
        <p className="font-mono text-[0.68rem] text-muted">log scale, 10⁰ to 10¹⁸</p>
      </div>
      <div className="mt-6 grid grid-cols-2 gap-6 border-t border-rule pt-4">
        <Readout label="Voxels" value={sci(c.voxels, 2)} />
        <Readout label="Reduction" value={`${sci(c.reduction, 2)}×`} tone="cobalt" />
      </div>
    </Instrument>
  );
}

// ------------------------------------------------------------------------------------------- C
function MpoInstrument() {
  const [bond, setBond] = useState(32);
  const r = mpoRow(bond);
  const regime = r.compression > 1 ? 'smaller than the dense map' : r.compression === 1 ? 'break-even with the dense map' : 'larger than the dense map';
  const maxP = mpoRow(160).parameters;
  return (
    <Instrument className="col-span-12 md:col-span-6 lg:col-span-4" title="MPO compression by bond dimension" note="Two cores factorise the 256 → 256 channel map. The report runs at bond 32.">
      <Slider label="Bond r" value={bond} min={1} max={160} onChange={setBond} />
      <div className="relative mt-6 h-16 border-b border-rule-strong" role="img" aria-label={`${int(r.parameters)} parameters against 65,536 dense`}>
        <div className={`absolute bottom-0 left-0 h-10 ${r.compression >= 1 ? 'bg-cobalt' : 'hatch'}`} style={{ width: `${(r.parameters / maxP) * 100}%` }} />
        <div className="absolute bottom-0 top-0 border-l border-dashed border-fg" style={{ left: `${(65536 / maxP) * 100}%` }}>
          <span className="absolute -top-0.5 right-1.5 whitespace-nowrap font-mono text-[0.68rem] text-muted">dense 65,536</span>
        </div>
      </div>
      <div className="mt-6 grid grid-cols-2 gap-6">
        <Readout label="Weights" value={int(r.parameters)} />
        <Readout label="Compression" value={`${r.compression.toFixed(2)}×`} tone={r.compression >= 1 ? 'cobalt' : 'vermilion'} />
      </div>
      <p className="mt-4 text-small text-muted">At r = {bond} the factorised map is {regime}.</p>
    </Instrument>
  );
}

// ------------------------------------------------------------------------------------------- D
const CHECK_NAMES: Record<string, string> = {
  marginal_loss: 'Marginal loss unchanged when unlabelled mass is shuffled (report test)',
  unlabelled_prediction: 'Naive loss penalises recognising an unlabelled organ; marginal does not',
  cox: 'Cox loss unchanged by a +37.5 log-risk shift; zero when all censored',
  uncertainty_weighting: 'Uncertainty weights converge to s = log L',
  dice: 'Dice matches closed form on three constructed cases',
  horizon_labels: 'Censored horizons are skipped, not defaulted',
  exclusion: 'Exclusion term is zero under softmax (finding)',
  concordance: 'C-index is 1 for perfect and 0 for reversed ordering',
  parameter_budget: 'Parameter budget equals the report, stage by stage',
};

function RecordedRun({ verification, api }: { verification: Verification | null; api: ApiState }) {
  const [live, setLive] = useState<Verification | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const v = live ?? verification;
  const rerun = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const res = await fetch('api/verification/run', { method: 'POST' });
      if (!res.ok) setMsg(await apiError(res));
      else {
        setLive((await res.json()) as Verification);
        setMsg('Re-ran on the server just now.');
      }
    } catch {
      setMsg('The API is not running. Start it with: uvicorn api.app:app --port 8765');
    } finally {
      setBusy(false);
    }
  };
  return (
    <Instrument
      className="col-span-12 lg:col-span-8"
      title="Recorded verification run"
      note="Written by python -m qihgnn.verify. These are checks of the mathematics and the parameter budget, not segmentation scores."
    >
      {!v ? (
        <p className="text-small text-muted">No verification record found. Run python -m qihgnn.verify from ml/ and export again.</p>
      ) : (
        <>
          <ul className="border-t border-rule">
            {Object.entries(v.checks).map(([k, ok]) => (
              <li key={k} className="grid grid-cols-[3.25rem_1fr] items-baseline gap-3 border-b border-rule py-2.5 text-small">
                <span className={`font-mono text-[0.7rem] ${ok ? 'text-cobalt' : 'text-vermilion-text'}`}>{ok ? 'PASS' : 'FAIL'}</span>
                <span>{CHECK_NAMES[k] ?? k}</span>
              </li>
            ))}
          </ul>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
            <p className="font-mono text-[0.7rem] text-muted">
              {dateTime(v.generated_at)}, git {v.git ?? 'n/a'}, Python {v.environment.python}, NumPy {v.environment.numpy}
            </p>
            <button type="button" onClick={rerun} disabled={busy || !api.online} className="inline-flex items-center gap-2 border border-rule-strong px-3 py-2 text-small hover:bg-fg hover:text-surface disabled:cursor-not-allowed disabled:opacity-50" title={api.online ? 'Re-run on the server' : 'Needs the Python API'}>
              <LuRefreshCw className={`size-3.5 ${busy ? 'animate-spin' : ''}`} /> Re-run on server
            </button>
          </div>
          {msg && <p className="mt-2 text-small text-muted" role="status">{msg}</p>}
        </>
      )}
    </Instrument>
  );
}

export function Lab({ verification, api }: { verification: Verification | null; api: ApiState }) {
  return (
    <section id="verification" aria-labelledby="verification-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="04"
          id="verification"
          title="Verification lab"
          standfirst="Every property the report claims is computed here in the browser with the same formulas as the Python code, and tested to agree with it. Move the controls; the numbers are live."
        />
        <div className="mt-14 grid grid-cols-12 gap-x-6 gap-y-14 lg:gap-x-10">
          <MarginalInstrument />
          <AttentionInstrument />
          <MpoInstrument />
          <RecordedRun verification={verification} api={api} />
        </div>
      </Container>
    </section>
  );
}

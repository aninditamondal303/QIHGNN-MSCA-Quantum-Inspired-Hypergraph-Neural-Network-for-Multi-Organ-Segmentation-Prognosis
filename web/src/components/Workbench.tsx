import { useEffect, useMemo, useState } from 'react';
import { LuPlay } from 'react-icons/lu';
import { apiError, type ApiState } from '../lib/data';
import { int } from '../lib/format';
import type { CaseInfo, CasesFile, Label, PredictResponse } from '../lib/types';
import { defaultChannel, LABEL_ORDER, overlaysFor, useCase } from '../lib/useCase';
import { labelExtent, loadVolume, planeDepth, valueAt, type Plane, type Volume } from '../lib/volume';
import { Container, SectionHead } from './chrome';
import { FilmCanvas } from './FilmCanvas';
import { labelName, Swatch } from './Legend';
import { runNet3d } from '../lib/net3d';

const PLANES: Plane[] = ['axial', 'coronal', 'sagittal'];
type Source = 'reference' | 'network' | 'off';

function axisOf(plane: Plane): 0 | 1 | 2 {
  return plane === 'axial' ? 2 : plane === 'coronal' ? 1 : 0;
}

function Segmented<T extends string>({ value, options, onChange, label, disabled = [] }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; label: string; disabled?: T[] }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex border border-rule-strong">
      {options.map((o) => (
        <button
          key={o.v}
          type="button"
          role="radio"
          aria-checked={value === o.v}
          disabled={disabled.includes(o.v)}
          onClick={() => onChange(o.v)}
          className={`flex-1 border-r border-rule-strong px-2 py-2 text-small last:border-r-0 disabled:cursor-not-allowed disabled:text-muted/60 ${value === o.v ? 'bg-fg text-surface' : 'hover:bg-raised'}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-rule py-4">
      <p className="mb-2.5 text-small text-muted">{label}</p>
      {children}
    </div>
  );
}

export function Workbench({ cases, api }: { cases: CasesFile | null; api: ApiState }) {
  const list = cases?.cases ?? [];
  const [caseId, setCaseId] = useState<string | null>(null);
  const info: CaseInfo | null = list.find((c) => c.id === caseId) ?? list[0] ?? null;
  const [channel, setChannel] = useState(0);
  const [plane, setPlane] = useState<Plane>('axial');
  const [cursor, setCursor] = useState<[number, number, number] | null>(null);
  const [hover, setHover] = useState<[number, number, number] | null>(null);
  const [source, setSource] = useState<Source>('reference');
  const [opacity, setOpacity] = useState(0.16);
  const [prediction, setPrediction] = useState<{ caseId: string; volume: Volume; response: PredictResponse } | null>(null);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const { image, labels } = useCase(info, channel);

  // Reset view when the case changes; open on the slice with the most labelled anatomy.
  useEffect(() => {
    if (!info) return;
    setChannel(defaultChannel(info));
    setCursor(null);
    setMessage(null);
    setSource('reference');
  }, [info?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!labels || !info || cursor) return;
    // Centre the crosshair on the first labelled structure so every plane opens on anatomy.
    const first = info.present[0];
    const id = first ? LABEL_ORDER.indexOf(first) : -1;
    const ext = id > 0 ? labelExtent(labels, id) : null;
    const z = ext ? ext.peak : Math.floor(labels.d / 2);
    let sx = 0, sy = 0, n = 0;
    for (let yy = 0; yy < labels.h; yy++)
      for (let xx = 0; xx < labels.w; xx++)
        if (labels.data[(z * labels.h + yy) * labels.w + xx] === id) { sx += xx; sy += yy; n++; }
    setCursor(n ? [Math.round(sx / n), Math.round(sy / n), z] : [Math.floor(labels.w / 2), Math.floor(labels.h / 2), z]);
  }, [labels, info, cursor]);

  const pos = cursor ?? [64, 64, 32];
  const index = (p: Plane) => pos[axisOf(p)];
  const depth = image ? planeDepth(image, plane) : 1;
  const setIndex = (p: Plane, v: number) => {
    const next: [number, number, number] = [...pos];
    next[axisOf(p)] = v;
    setCursor(next);
  };

  const pred = prediction && info && prediction.caseId === info.id ? prediction : null;
  const overlayVolume = source === 'network' ? pred?.volume ?? null : source === 'reference' ? labels : null;
  const overlays = useMemo(() => (info ? overlaysFor(info) : []), [info]);

  const readout = hover && image ? hover : null;
  const readLabel = readout && labels ? LABEL_ORDER[valueAt(labels, ...readout)] : null;

  const run = async () => {
    if (!info) return;
    setRunning(true);
    setMessage(null);
    if (!api.online) {
      // No server: the trained network runs right here in the browser.
      try {
        const { volume, response } = await runNet3d(info, setStage);
        setPrediction({ caseId: info.id, volume, response });
        setSource('network');
      } catch (e) {
        setMessage(e instanceof Error ? e.message : String(e));
      } finally {
        setStage(null);
        setRunning(false);
      }
      return;
    }
    try {
      const res = await fetch('api/predict', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ case_id: info.id }) });
      if (!res.ok) {
        setMessage(res.status === 502 || (res.status === 404 && !api.online) ? 'The API is not running. Start it from the repository root with: uvicorn api.app:app --port 8765' : await apiError(res));
        return;
      }
      const body = (await res.json()) as PredictResponse;
      const bytes = Uint8Array.from(atob(body.label_png_base64), (ch) => ch.charCodeAt(0));
      const volume = await loadVolume(new Blob([bytes], { type: 'image/png' }), body.shape as [number, number, number], body.mosaic_columns);
      setPrediction({ caseId: info.id, volume, response: body });
      setSource('network');
    } catch {
      setMessage('The API could not be reached. This static preview shows the reference masks only.');
    } finally {
      setRunning(false);
    }
  };

  const others = PLANES.filter((p) => p !== plane);

  return (
    <section id="workbench" aria-labelledby="workbench-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="03"
          id="workbench"
          title="Workbench"
          standfirst="Scroll through a volume in three planes, switch MRI sequences or CT windows, and compare the reference mask with what the network returns. Hatched organs are present in the anatomy but no dataset supervises them."
        />
        <div className="mt-14 grid grid-cols-12 gap-x-6 gap-y-8">
          {/* Controls */}
          <div className="col-span-12 lg:col-span-3">
            <Field label="Case">
              <div role="radiogroup" aria-label="Case" className="flex flex-col">
                {list.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    role="radio"
                    aria-checked={info?.id === c.id}
                    onClick={() => setCaseId(c.id)}
                    className={`border-b border-rule px-2 py-2.5 text-left last:border-b-0 ${info?.id === c.id ? 'bg-fg text-surface' : 'hover:bg-raised'}`}
                  >
                    <span className="block font-mono text-small">{c.id}</span>
                    <span className={`block text-small ${info?.id === c.id ? 'text-surface/75' : 'text-muted'}`}>{c.modality}, {c.synthetic ? (c.analogOf.startsWith('no ') ? c.analogOf : `imitates ${c.analogOf}`) : `${c.analogOf}, held-out patient`}</span>
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Plane">
              <Segmented value={plane} onChange={setPlane} label="Plane" options={PLANES.map((p) => ({ v: p, label: p[0].toUpperCase() + p.slice(1) }))} />
            </Field>
            {info && (
              <Field label={info.modality === 'MRI' ? 'Sequence' : 'CT window'}>
                <div className="grid grid-cols-2 border-l border-t border-rule-strong" role="radiogroup" aria-label={info.modality === 'MRI' ? 'Sequence' : 'CT window'}>
                  {info.channels.map((c, i) => (
                    <button
                      key={c.name}
                      type="button"
                      role="radio"
                      aria-checked={channel === i}
                      onClick={() => setChannel(i)}
                      className={`border-b border-r border-rule-strong px-2 py-2 text-small ${channel === i ? 'bg-fg text-surface' : 'hover:bg-raised'}`}
                    >
                      {c.name.replace(' window', '')}
                    </button>
                  ))}
                </div>
              </Field>
            )}
            <Field label="Overlay">
              <Segmented
                value={source}
                onChange={setSource}
                label="Overlay source"
                disabled={pred ? [] : ['network']}
                options={[{ v: 'reference', label: 'Reference' }, { v: 'network', label: 'Network' }, { v: 'off', label: 'Off' }]}
              />
              <label className="mt-3 flex items-center gap-3 text-small text-muted">
                <span className="w-16 shrink-0">Fill {Math.round(opacity * 100)}%</span>
                <input className="range" type="range" min={0} max={0.6} step={0.02} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Overlay fill opacity" />
              </label>
            </Field>
          </div>

          {/* Viewer */}
          <div className="col-span-12 lg:col-span-9">
            <div className="film">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-film-rule px-4 py-2.5">
                <span className="font-mono text-small">{info?.id ?? 'Loading'}</span>
                <span className="meta num text-film-muted">
                  {plane} {index(plane)} / {depth - 1}
                  {info?.synthetic ? ', synthetic phantom' : ''}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                <div className="border-film-rule md:border-r">
                  <FilmCanvas
                    image={image}
                    labels={overlayVolume}
                    plane={plane}
                    index={index(plane)}
                    overlays={overlays}
                    opacity={opacity}
                    spacing={info?.spacing}
                    showOverlay={source !== 'off'}
                    crosshair={cursor}
                    onHover={setHover}
                    onPick={(v) => setCursor(v)}
                    ariaLabel={info ? `${info.title}, ${plane} slice ${index(plane)}. Click to move the crosshair.` : 'Loading'}
                  />
                  <div className="flex items-center gap-4 border-t border-film-rule px-4 py-2">
                    <label htmlFor="slice" className="meta shrink-0 text-film-muted">Slice</label>
                    <input
                      id="slice"
                      className="range text-film-fg"
                      type="range"
                      min={0}
                      max={depth - 1}
                      value={index(plane)}
                      onChange={(e) => setIndex(plane, Number(e.target.value))}
                    />
                    <span className="meta num w-10 shrink-0 text-right text-film-fg">{String(index(plane)).padStart(2, '0')}</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-1">
                  {others.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setPlane(p)}
                      className="group relative border-b border-film-rule text-left last:border-b-0 max-md:border-r max-md:last:border-r-0"
                      aria-label={`Switch main view to ${p}`}
                    >
                      <FilmCanvas image={image} labels={overlayVolume} plane={p} index={index(p)} overlays={overlays} opacity={opacity} spacing={info?.spacing} showOverlay={source !== 'off'} crosshair={cursor} ariaLabel={`${p} slice ${index(p)}`} />
                      <span className="meta num absolute left-2 top-2 bg-film/80 px-1 text-film-muted group-hover:text-film-fg">{p} {index(p)}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-px border-t border-film-rule bg-film-rule font-mono text-[0.72rem] sm:grid-cols-4" aria-live="polite">
                <p className="bg-film px-4 py-2.5 text-film-muted">x {readout ? readout[0] : '—'} y {readout ? readout[1] : '—'} z {readout ? readout[2] : '—'}</p>
                <p className="bg-film px-4 py-2.5 text-film-muted">value {readout && image ? valueAt(image, ...readout) : '—'}</p>
                <p className="col-span-2 bg-film px-4 py-2.5 text-film-muted">
                  {readLabel && readLabel !== 'background'
                    ? `${labelName(readLabel)}, ${info?.supervised.includes(readLabel) ? 'supervised' : 'silent and suppressed'}`
                    : 'Hover the scan to read a voxel'}
                </p>
              </div>
            </div>

            <div className="mt-8 grid grid-cols-12 gap-x-6 gap-y-8">
              <div className="col-span-12 md:col-span-7">
                <p className="mb-3 text-small text-muted">Channels present in this case</p>
                <ul className="border-t border-rule">
                  {info?.present.map((l) => {
                    const sup = info.supervised.includes(l);
                    return (
                      <li key={l} className="grid grid-cols-[1rem_1fr_auto] items-center gap-x-3 border-b border-rule py-2.5 text-small">
                        <Swatch kind={sup ? 'supervised' : 'silent'} />
                        <span>
                          <span className="font-medium">{labelName(l)}</span>
                          <span className="text-muted"> {sup ? 'supervised by this corpus' : 'no corpus supervises it here; zeroed by the server'}</span>
                        </span>
                        <span className="font-mono text-muted num">{int(info.voxels[l] ?? 0)} vx</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
              <div className="col-span-12 md:col-span-5">
                <button
                  type="button"
                  onClick={run}
                  disabled={running || !info}
                  className="flex w-full items-center justify-between gap-3 bg-fg px-4 py-3.5 text-left text-surface hover:bg-cobalt disabled:opacity-60"
                >
                  <span className="font-medium" aria-live="polite">{running ? stage ?? 'Running network' : 'Run network on this case'}</span>
                  <LuPlay className="size-4" />
                </button>
                <p className="mt-3 text-small text-muted">
                  {api.online
                    ? api.health.inference_ready
                      ? `Uses checkpoint ${api.health.checkpoint}.`
                      : 'The API is running but has no trained checkpoint yet.'
                    : 'The trained network runs in your browser, on your device. The first run downloads it once and can take a minute.'}
                </p>
                {message && <p className="mt-3 border-l-2 border-vermilion pl-3 text-small" role="status">{message}</p>}
                {pred && info && (() => {
                  // Voxels the network gave to an organ this scan's corpus never labels: a real error, shown hatched.
                  const counts = new Map<number, number>();
                  for (const v of pred.volume.data) if (v) counts.set(v, (counts.get(v) ?? 0) + 1);
                  const stray = [...counts].filter(([id]) => !info.supervised.includes(LABEL_ORDER[id]));
                  return stray.length ? (
                    <p className="mt-3 border-l-2 border-vermilion pl-3 text-small">
                      The network also marked {stray.map(([id, n]) => `${int(n)} voxels as ${labelName(LABEL_ORDER[id])}`).join(' and ')}, which this scan does not contain (hatched in red): a mistake of the model, left visible on purpose.
                    </p>
                  ) : null;
                })()}
                {pred && info?.networkDice && (
                  <p className="mt-3 border-l-2 border-cobalt pl-3 text-small">
                    Against the expert on this held-out patient:{' '}
                    {Object.entries(info.networkDice).map(([k, v]) => `${labelName(k as Label)} Dice ${v == null ? 'n/a' : v.toFixed(2)}`).join(', ')}.
                  </p>
                )}
                {pred && (
                  <dl className="mt-5 grid grid-cols-3 border-t border-rule pt-4" aria-label="Prognosis output">
                    {Object.entries(pred.response.survival_probability).map(([k, v]) => (
                      <div key={k}>
                        <dt className="text-small text-muted">Survival {k}</dt>
                        <dd className="display num mt-1 text-h3">{Math.round(v * 100)}%</dd>
                      </div>
                    ))}
                    <p className="col-span-3 mt-4 text-small text-muted">
                      Run {pred.response.run_id ?? 'unknown'} in {pred.response.seconds}s. Suppressed: {pred.response.suppressed.map(labelName).join(', ') || 'none'}.
                      Survival estimates come from a network tested on only 30 patients (C-index 0.71): research output, not a forecast for anyone.
                    </p>
                  </dl>
                )}
              </div>
            </div>
          </div>
        </div>
      </Container>
    </section>
  );
}


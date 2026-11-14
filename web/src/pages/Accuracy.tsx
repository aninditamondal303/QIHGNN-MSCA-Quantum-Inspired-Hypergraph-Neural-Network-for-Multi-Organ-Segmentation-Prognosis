import { Container, PageHead } from '../components/chrome';
import { Results } from '../components/Results';
import { Link, useTitle } from '../lib/router';
import { CLASS_LABEL, CLASSES } from '../lib/screen';
import { ci, pct, useScreen, type ScreenMetrics } from '../lib/screenMetrics';
import type { MetricsFile, TrainingHistory, Verification } from '../lib/types';

function Card({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="bg-surface p-5">
      <p className="text-small text-muted">{k}</p>
      <p className="display num mt-3 text-[clamp(2rem,1.4rem+1.6vw,3rem)] leading-none">{v}</p>
      {sub && <p className="mt-2 font-mono text-[0.72rem] text-muted num">{sub}</p>}
    </div>
  );
}

function Block({ id, title, lead, children }: { id: string; title: string; lead: string; children: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-t`} className="border-t border-rule-strong py-14">
      <div className="grid grid-cols-12 gap-x-6 gap-y-6">
        <div className="col-span-12 lg:col-span-4">
          <h2 id={`${id}-t`} className="display text-h3">{title}</h2>
          <p className="mt-3 text-small text-muted">{lead}</p>
        </div>
        <div className="col-span-12 lg:col-span-8">{children}</div>
      </div>
    </section>
  );
}

function Confusion({ m }: { m: NonNullable<ScreenMetrics['classifier']>['test'] }) {
  const max = Math.max(...m.confusion.flat());
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-small">
        <caption className="mb-2 text-left text-muted">Rows: true class. Columns: predicted class. Held-out test images.</caption>
        <thead>
          <tr>
            <th className="p-2" />
            {CLASSES.map((c) => <th key={c} scope="col" className="p-2 text-left font-normal text-muted">{CLASS_LABEL[c]}</th>)}
          </tr>
        </thead>
        <tbody>
          {m.confusion.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="p-2 text-left font-normal">{CLASS_LABEL[CLASSES[i]]}</th>
              {row.map((v, j) => (
                <td key={j} className="p-0">
                  <span className={`block p-2 text-right font-mono num ${i === j ? 'text-surface' : ''}`} style={{ background: i === j ? `color-mix(in srgb, var(--cobalt) ${20 + 80 * (v / max)}%, transparent)` : v ? `color-mix(in srgb, var(--vermilion) ${15 + 60 * (v / max)}%, transparent)` : 'transparent' }}>
                    {v}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Scatter({ pairs }: { pairs: [number, number][] }) {
  const W = 360;
  const max = Math.max(1, ...pairs.flat());
  const s = (v: number) => 30 + (v / max) * (W - 40);
  return (
    <svg viewBox={`0 0 ${W} ${W}`} className="w-full max-w-[360px]" role="img" aria-label="Estimated against true tumour volume on held-out scans">
      <line x1={s(0)} y1={W - s(0)} x2={s(max)} y2={W - s(max)} stroke="var(--rule-strong)" strokeDasharray="3 3" />
      {pairs.map(([t, e], i) => <circle key={i} cx={s(t)} cy={W - s(e)} r={3.5} fill="var(--cobalt)" opacity={0.75} />)}
      <text x={W / 2} y={W - 4} textAnchor="middle" className="fill-[var(--muted)] font-mono text-[10px]">expert volume, cm³ (0–{max.toFixed(0)})</text>
      <text x={10} y={W / 2} transform={`rotate(-90 10 ${W / 2})`} textAnchor="middle" className="fill-[var(--muted)] font-mono text-[10px]">estimated, cm³</text>
    </svg>
  );
}

function KM({ km }: { km: NonNullable<ScreenMetrics['survival']>['risk_groups']['km'] }) {
  const all = Object.values(km);
  const maxT = Math.max(...all.flatMap((g) => g.t));
  const W = 480;
  const H = 220;
  const x = (t: number) => 34 + (t / maxT) * (W - 44);
  const y = (s: number) => 8 + (1 - s) * (H - 34);
  const step = (t: number[], s: number[]) => t.map((ti, i) => (i ? `H${x(ti).toFixed(1)}V${y(s[i]).toFixed(1)}` : `M${x(ti)},${y(s[i])}`)).join('');
  const colours = { low: 'var(--cobalt)', intermediate: 'var(--muted)', high: 'var(--vermilion)' } as const;
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Kaplan-Meier survival by predicted risk third">
        {[0, 0.5, 1].map((v) => <text key={v} x={28} y={y(v) + 4} textAnchor="end" className="fill-[var(--muted)] font-mono text-[10px]">{v * 100}%</text>)}
        {[0, 12, 24, 36, 48, 60].filter((mo) => mo * 30.44 <= maxT).map((mo) => <text key={mo} x={x(mo * 30.44)} y={H - 6} textAnchor="middle" className="fill-[var(--muted)] font-mono text-[10px]">{mo}m</text>)}
        {(Object.keys(km) as (keyof typeof km)[]).map((g) => <path key={g} d={step(km[g].t, km[g].s)} fill="none" stroke={colours[g]} strokeWidth={2} />)}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-4 text-small text-muted">
        {(Object.keys(km) as (keyof typeof km)[]).map((g) => (
          <span key={g} className="inline-flex items-center gap-2"><span className="inline-block h-0.5 w-5" style={{ background: colours[g] }} />{g} risk (n = {km[g].n})</span>
        ))}
      </figcaption>
    </figure>
  );
}

export function Accuracy({ metrics3d, history, verification, loading3d }: { metrics3d: MetricsFile | null; history: TrainingHistory | null; verification: Verification | null; loading3d: boolean }) {
  useTitle('Accuracy');
  const { metrics, ready } = useScreen();
  const t = metrics?.classifier?.test;
  const ext = metrics?.classifier?.external_test;
  const seg = metrics?.segmenter;
  const s = metrics?.survival;
  return (
    <>
      <PageHead eyebrow="Accuracy and validation" title="Measured, not claimed.">
        Every number below comes from images and patients that were held out of training, or from an independent dataset. Ranges are 95%
        confidence intervals.
      </PageHead>
      <Container className="pb-16">
        {!ready ? (
          <p className="border-l-2 border-vermilion pl-3 text-small">The screening models are being trained and evaluated; their results will appear here when that run finishes.</p>
        ) : (
          <>
            <div className="grid gap-px bg-rule sm:grid-cols-2 lg:grid-cols-4">
              <Card k="Tumours detected (sensitivity)" v={pct(t!.tumour_vs_no_tumour.sensitivity)} sub={ci(t!.tumour_vs_no_tumour.sensitivity_ci)} />
              <Card k="Tumour-free scans cleared (specificity)" v={pct(t!.tumour_vs_no_tumour.specificity)} sub={ci(t!.tumour_vs_no_tumour.specificity_ci)} />
              <Card k="External dataset accuracy" v={pct(ext?.accuracy)} sub={ext ? `${ext.n} images, ${ci(ext.accuracy_ci)}` : 'not available'} />
              <Card k="Survival ranking (C-index)" v={s ? s.cv_c_index.full.mean.toFixed(2) : '—'} sub={s ? `age alone ${s.cv_c_index.age_only.mean.toFixed(2)}` : ''} />
            </div>

            <Block id="detect" title="Tumour check" lead={`${t!.n} held-out test images: ${metrics!.classifier!.architecture}. Calibrated with temperature ${metrics!.classifier!.temperature.toFixed(2)}.`}>
              <div className="grid gap-px bg-rule sm:grid-cols-3">
                <Card k="5-class accuracy" v={pct(t!.accuracy)} sub={ci(t!.accuracy_ci)} />
                <Card k="Macro F1" v={t!.macro_f1.toFixed(3)} />
                <Card k="Tumour vs none, AUC" v={t!.tumour_vs_no_tumour.auc?.toFixed(3) ?? '—'} />
              </div>
              <div className="mt-8"><Confusion m={t!} /></div>
              <ul className="mt-8 grid gap-x-8 gap-y-2 text-small sm:grid-cols-2">
                {CLASSES.map((c) => {
                  const r = t!.per_class[c];
                  return r ? <li key={c} className="flex justify-between border-b border-rule py-1.5"><span>{CLASS_LABEL[c]} recall</span><span className="font-mono num">{pct(r.recall)} <span className="text-muted">({r.n})</span></span></li> : null;
                })}
              </ul>
              <p className="mt-6 text-small">
                <span className="font-medium">Calibration.</span> Expected calibration error {pct(t!.ece)}{t!.ece_before_calibration != null && ` (${pct(t!.ece_before_calibration)} before temperature scaling)`}.
                {' '}<span className="font-medium">Abstention.</span> Below {pct(t!.abstention.threshold, 0)} confidence the tool declines to answer: it answers {pct(t!.abstention.coverage)} of test images and is right on {pct(t!.abstention.accuracy_when_answered)} of those.
              </p>
              <p className="mt-3 text-small text-muted">
                By source: {Object.entries(t!.by_source).map(([k, v]) => `${k} ${pct(v.accuracy)} (${v.n})`).join(' · ')}.
              </p>
            </Block>

            {ext && (
              <Block id="external" title="Independent dataset" lead="Brain MRI images from a different public collection (yes/no tumour), never used for training. Images that matched any of ours were removed first.">
                <div className="grid gap-px bg-rule sm:grid-cols-3">
                  <Card k="Sensitivity" v={pct(ext.sensitivity)} sub={ci(ext.sensitivity_ci)} />
                  <Card k="Specificity" v={pct(ext.specificity)} sub={ci(ext.specificity_ci)} />
                  <Card k="AUC" v={ext.auc?.toFixed(3) ?? '—'} sub={`${ext.n} images`} />
                </div>
                <p className="mt-4 text-small text-muted">
                  {ext.removed_as_copies_of_training_data} images were removed as copies of our training data; {ext.rejected_as_not_brain_mri} were flagged as not a brain MRI.
                  A drop from the in-distribution numbers is expected and is the more realistic estimate.
                </p>
              </Block>
            )}

            {seg && (
              <Block id="outline" title="Tumour outline" lead={`${seg.architecture}. Patient-level split of BraTS 2020: no test patient contributed any training slice.`}>
                <table className="w-full text-small">
                  <thead><tr className="border-b border-rule-strong text-left text-muted"><th className="py-2 font-normal">MRI sequence</th><th className="py-2 font-normal">Dice, tumour slices</th><th className="py-2 font-normal">False alarms, tumour-free slices</th></tr></thead>
                  <tbody>
                    {Object.entries(seg.brats_slices).map(([k, v]) => (
                      <tr key={k} className="border-b border-rule"><td className="py-2">{k}</td><td className="py-2 font-mono num">{v.dice_on_tumour_slices.mean?.toFixed(3)} <span className="text-muted">(n {v.dice_on_tumour_slices.n})</span></td><td className="py-2 font-mono num">{pct(v.false_alarm_rate_on_tumour_free_slices)}</td></tr>
                    ))}
                  </tbody>
                </table>
                {seg.figshare_test && <p className="mt-4 text-small">Skull-on T1 contrast images: Dice {seg.figshare_test.dice.mean?.toFixed(3)} (n {seg.figshare_test.dice.n}). <span className="text-muted">{seg.figshare_test.note}</span></p>}
                {Object.entries(seg.brats_3d_test).map(([k, v]) => (
                  <div key={k} className="mt-8 grid gap-6 sm:grid-cols-[1fr_auto]">
                    <div className="text-small">
                      <p className="font-medium">Whole 3D scans, {k}, {v.patients} test patients</p>
                      <p className="mt-2">3D Dice {v.dice3d.mean?.toFixed(3)}; volume error median {v.volume_abs_error_cc.median?.toFixed(1)} cm³ ({pct(v.volume_rel_error.median)}); correlation r = {v.volume_pearson_r?.toFixed(3)}.</p>
                    </div>
                    {k === 'FLAIR' && <Scatter pairs={v.pairs} />}
                  </div>
                ))}
              </Block>
            )}

            {s && (
              <Block id="prognosis" title="Prognosis" lead={`Cox model on ${s.n} BraTS 2020 glioma patients (${s.events} deaths observed). Repeated 5-fold cross-validation; risk groups from out-of-fold predictions.`}>
                <table className="w-full text-small">
                  <thead><tr className="border-b border-rule-strong text-left text-muted"><th className="py-2 font-normal">Model</th><th className="py-2 font-normal">C-index (mean ± sd)</th></tr></thead>
                  <tbody>
                    {Object.entries(s.cv_c_index).map(([k, v]) => (
                      <tr key={k} className="border-b border-rule"><td className="py-2">{k.replace(/_/g, ' ')}</td><td className="py-2 font-mono num">{v.mean.toFixed(3)} ± {v.sd.toFixed(3)}</td></tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-8"><KM km={s.risk_groups.km} /></div>
                <p className="mt-3 text-small">Log-rank test across the three groups: p = {s.risk_groups.logrank_p < 0.001 ? s.risk_groups.logrank_p.toExponential(1) : s.risk_groups.logrank_p.toFixed(3)}.
                  {' '}BraTS survival classes (under 10, 10 to 15, over 15 months): {pct(s.survival_class_accuracy.accuracy)} correct, against {pct(s.survival_class_accuracy.majority_class_baseline)} for always guessing the commonest class.
                </p>
                <p className="mt-3 text-small text-muted">Survival is hard to predict from imaging and age alone; published BraTS challenge results sit in a similar range. The tool reports the uncertainty band for that reason.</p>
              </Block>
            )}
          </>
        )}
        <section className="border-t border-rule-strong pt-6 text-small text-muted">
          <p>How the data was split and de-duplicated: <Link to="/data">Data & privacy</Link>. What the numbers do not cover: <Link to="/about#limitations">limitations</Link>.</p>
        </section>
      </Container>
      <Results metrics={metrics3d} history={history} verification={verification} loading={loading3d} />
    </>
  );
}

import { dateTime, fixed } from '../lib/format';
import type { MetricsFile, TrainingHistory, Verification } from '../lib/types';
import { Container, SectionHead } from './chrome';
import { labelName } from './Legend';

const STEPS = [
  { cmd: 'python scripts/download_data.py', what: 'Download the three Kaggle datasets into data/raw/' },
  { cmd: 'python -m qihgnn.data.inspection', what: 'Inspect formats and decide which datasets supervise which organ' },
  { cmd: 'python -m qihgnn.data.cache', what: 'Preprocess once into 128 × 128 × 64 volumes with patient-level splits' },
  { cmd: 'python -m qihgnn.train --config configs/full.json', what: 'Train the full model (GPU; Kaggle or Colab works)' },
  { cmd: 'python -m qihgnn.train --config configs/baseline.json', what: 'Train the matched baseline, then each ablation' },
  { cmd: 'python -m qihgnn.evaluate --checkpoint runs/<full>/best.pt runs/<baseline>/best.pt --label full baseline', what: 'Write results/metrics.json; this page reads it' },
];

function HistoryChart({ history }: { history: TrainingHistory }) {
  const pts = history.history.filter((h) => h.train_marginal !== null);
  if (pts.length < 2) return null;
  const vals = pts.flatMap((p) => [p.train_marginal ?? 0, p.val_marginal ?? 0]);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const W = 560;
  const H = 180;
  const x = (i: number) => 36 + (i / (pts.length - 1)) * (W - 46);
  const y = (v: number) => 12 + (1 - (v - lo) / (hi - lo || 1)) * (H - 36);
  const line = (k: 'train_marginal' | 'val_marginal') => pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[k] ?? lo).toFixed(1)}`).join('');
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Marginal loss by epoch, training and validation">
        <line x1={36} x2={W - 10} y1={H - 24} y2={H - 24} stroke="var(--rule-strong)" />
        <path d={line('train_marginal')} fill="none" stroke="var(--fg)" strokeWidth="1.5" />
        <path d={line('val_marginal')} fill="none" stroke="var(--cobalt)" strokeWidth="1.5" />
        <text x={36} y={H - 8} className="fill-muted font-mono text-[10px]">epoch 0</text>
        <text x={W - 10} y={H - 8} textAnchor="end" className="fill-muted font-mono text-[10px]">epoch {pts.length - 1}</text>
        <text x={30} y={y(hi) + 4} textAnchor="end" className="fill-muted font-mono text-[10px]">{hi.toFixed(2)}</text>
        <text x={30} y={y(lo) + 4} textAnchor="end" className="fill-muted font-mono text-[10px]">{lo.toFixed(2)}</text>
      </svg>
      <figcaption className="mt-2 text-small text-muted">
        Marginal loss, training (ink) and validation (cobalt). Run {history.run_id} on {history.device}, {history.train_cases} training cases
        {history.train_config.dataset === 'phantom' ? ', synthetic phantoms: this shows the pipeline runs, not how well the model segments.' : '.'}
      </figcaption>
    </figure>
  );
}

const RUN_NAMES: Record<string, string> = { full: 'QIHGNN-MSCA', baseline: 'Matched baseline' };

export function Results({ metrics, history, verification, loading }: { metrics: MetricsFile | null; history: TrainingHistory | null; verification: Verification | null; loading: boolean }) {
  const runs = metrics?.runs ?? [];
  const organs = Array.from(new Set(runs.flatMap((r) => Object.keys(r.segmentation))));
  const placeholderOrgans = ['brain', 'liver'];
  return (
    <section id="results" aria-labelledby="results-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="06"
          id="results"
          title="Results, measured and not"
          standfirst={runs.length
            ? 'The 3D research network against a matched baseline trained the same way, on held-out test patients (53 brain, 19 liver, 30 with survival data). Dice: overlap with the expert outline (1 is perfect). HD95: boundary distance in voxels (lower is better).'
            : 'Scores appear here only when the evaluation script has written them. Until a converged run and a matched baseline exist, segmentation and prognosis are marked not yet measured rather than estimated.'}
        />
        <div className="mt-14 grid grid-cols-12 gap-x-6 gap-y-12">
          <div className="col-span-12 lg:col-span-7">
            <h3 className="font-semibold">Segmentation and prognosis</h3>
            {loading ? (
              <p className="mt-4 text-small text-muted">Reading results</p>
            ) : runs.length ? (
              <>
                <p className="mt-1 text-small text-muted">Evaluated {dateTime(metrics?.generated_at)} on {runs[0].evaluated_on}.</p>
                <div className="mt-5 overflow-x-auto">
                  <table className="w-full min-w-[520px] text-left text-small">
                    <thead>
                      <tr className="border-b border-rule-strong text-muted">
                        <th className="py-2 pr-4 font-normal">Organ</th>
                        {runs.map((r) => <th key={r.label} className="py-2 pr-4 font-normal">{RUN_NAMES[r.label] ?? r.label}<br /><span className="font-mono text-[0.68rem]">Dice / HD95</span></th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {organs.map((o) => (
                        <tr key={o} className="border-b border-rule">
                          <td className="py-2.5 pr-4">{labelName(o)}</td>
                          {runs.map((r) => {
                            const s = r.segmentation[o];
                            return <td key={r.label} className="py-2.5 pr-4 font-mono num">{s ? `${fixed(s.dice_mean)} ± ${fixed(s.dice_sd, 2)} / ${fixed(s.hd95_mean, 1)}` : '—'}</td>;
                          })}
                        </tr>
                      ))}
                      <tr className="border-b border-rule">
                        <td className="py-2.5 pr-4">C-index</td>
                        {runs.map((r) => <td key={r.label} className="py-2.5 pr-4 font-mono num">{r.prognosis ? fixed(r.prognosis.c_index) : 'no survival cases'}</td>)}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <>
                <table className="mt-5 w-full text-left text-small">
                  <thead>
                    <tr className="border-b border-rule-strong text-muted">
                      {['Measure', 'Full model', 'Matched baseline'].map((h) => <th key={h} className="py-2 pr-4 font-normal">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {[...placeholderOrgans.map((o) => `Dice, ${labelName(o)}`), 'HD95', 'C-index', 'Time-dependent AUC, 1/3/5 y'].map((m) => (
                      <tr key={m} className="border-b border-rule">
                        <td className="py-3 pr-4">{m}</td>
                        <td className="py-3 pr-4"><span className="hatch-fine inline-block px-1.5 py-0.5"><span className="bg-surface px-1 font-mono text-[0.7rem]">not yet measured</span></span></td>
                        <td className="py-3 pr-4"><span className="hatch-fine inline-block px-1.5 py-0.5"><span className="bg-surface px-1 font-mono text-[0.7rem]">not yet measured</span></span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-6 max-w-[62ch] text-small text-muted">
                  The report’s preliminary run reached a Dice of 0.173 after far fewer than the configured 200 epochs. That run is not reproduced in
                  this repository and is not a performance result; it only showed that the pipeline executes end to end.
                </p>
              </>
            )}
            {history && <div className="mt-10"><HistoryChart history={history} /></div>}
          </div>

          <div className="col-span-12 lg:col-span-5">
            {runs.length ? (
              <>
                <h3 className="font-semibold">What this shows</h3>
                <ul className="mt-5 space-y-4 border-t border-rule pt-5 text-small">
                  <li>One 3D network learns brain lesions and the liver from datasets that each label only one organ. The marginal loss stops it being punished for organs a dataset left unlabelled.</li>
                  <li>The baseline is the same network with the project’s three additions removed (MPO layers, the hypergraph and multi-scale cross-attention), trained on the same data for the same 30 epochs. The gap between the columns is what those additions contribute.</li>
                  <li>The same network also ranks patients by survival risk (C-index: 0.5 is chance, 1 is perfect).</li>
                  <li>Test patients were never used in training. Kidney, cervix and spleen have no 3D labels in the data, so those outputs are switched off.</li>
                </ul>
              </>
            ) : (
              <>
                <h3 className="font-semibold">How the empty cells get filled</h3>
                <ol className="mt-5 border-t border-rule">
                  {STEPS.map((s, i) => (
                    <li key={s.cmd} className="grid grid-cols-[1.75rem_1fr] gap-x-3 border-b border-rule py-3">
                      <span className="font-mono text-small text-muted num">{i + 1}</span>
                      <span>
                        <code className="block break-words font-mono text-[0.75rem]">{s.cmd}</code>
                        <span className="mt-1 block text-small text-muted">{s.what}</span>
                      </span>
                    </li>
                  ))}
                </ol>
                <p className="mt-6 text-small text-muted">
                  Already measured: the mathematics and the parameter budget, {verification?.all_passed ? 'all checks passing' : 'see section 04'}.
                  Step 1 runs from the repository root, the rest from the ml/ folder.
                </p>
              </>
            )}
          </div>
        </div>
      </Container>
    </section>
  );
}

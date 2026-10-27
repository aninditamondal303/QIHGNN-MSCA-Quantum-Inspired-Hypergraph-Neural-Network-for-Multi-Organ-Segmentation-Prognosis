import { useState } from 'react';
import { STAGES } from '../content/stages';
import { int, pct } from '../lib/format';
import type { Verification } from '../lib/types';
import { Container, SectionHead } from './chrome';
import { Equation } from './Equation';

const FALLBACK: Record<string, number> = {
  encoder: 3559808, mpo: 17152, hypergraph: 232468, msca: 2108160, shared: 132608, segmentation_head: 2130214, prognosis_head: 166661,
};

export function Architecture({ verification }: { verification: Verification | null }) {
  const budget = verification?.parameters.stages ?? FALLBACK;
  const total = Object.values(budget).reduce((a, b) => a + b, 0);
  const [active, setActive] = useState(4); // open on the hypergraph stage
  const stage = STAGES.find((s) => s.n === active) ?? STAGES[0];
  const params = (keys: string[] | null) => (keys ? keys.reduce((a, k) => a + (budget[k] ?? 0), 0) : 0);
  const stageParams = params(stage.budgetKey);

  return (
    <section id="architecture" aria-labelledby="architecture-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="02"
          id="architecture"
          title="Eight stages, one encoding"
          standfirst="Stages three to five are the contribution. The encoder and decoder are conventional and hold most of the weights. Select a stage to see its shape, its share of the budget and the equation it computes."
        />

        {/* Parameter budget as one bar: width = share of 8.35 M weights. */}
        <div className="mt-14">
          <div className="mb-2 flex items-baseline justify-between text-small text-muted">
            <span>Parameter budget by stage</span>
            <span className="font-mono num">{int(total)} weights, {(total * 4 / 2 ** 20).toFixed(2)} MB at fp32</span>
          </div>
          <div className="flex h-10 w-full border border-rule-strong" role="group" aria-label="Parameter budget, select a stage">
            {STAGES.filter((s) => s.budgetKey).map((s) => {
              const p = params(s.budgetKey);
              const on = s.n === active;
              return (
                <button
                  key={s.key}
                  type="button"
                  onClick={() => setActive(s.n)}
                  aria-pressed={on}
                  aria-label={`${s.name}: ${int(p)} parameters`}
                  style={{ width: `${Math.max((p / total) * 100, 1.4)}%` }}
                  className={`relative h-full border-r border-surface last:border-r-0 transition-colors ${
                    on ? 'bg-fg' : s.proposed ? 'bg-cobalt hover:bg-cobalt/80' : 'bg-fg/15 hover:bg-fg/25'
                  }`}
                >
                  {p / total > 0.08 && (
                    <span className={`absolute left-2 top-1/2 -translate-y-1/2 font-mono text-[0.68rem] ${on ? 'text-surface' : 'text-fg'}`}>
                      {s.short} {pct(p / total, 0)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-small text-muted">
            <span className="inline-flex items-center gap-2"><span className="size-3 bg-cobalt" />Proposed stages: MPO, hypergraph, multi-scale attention ({pct(params(['mpo', 'hypergraph', 'msca']) / total)})</span>
            <span className="inline-flex items-center gap-2"><span className="size-3 bg-fg/15" />Conventional stages</span>
          </div>
        </div>

        <div className="mt-12 grid grid-cols-12 gap-x-6 gap-y-10">
          <ol className="col-span-12 lg:col-span-5" aria-label="Pipeline stages">
            {STAGES.map((s) => {
              const on = s.n === active;
              return (
                <li key={s.key}>
                  <button
                    type="button"
                    onClick={() => setActive(s.n)}
                    aria-current={on ? 'step' : undefined}
                    className={`group grid w-full grid-cols-[2.25rem_1fr_auto] items-baseline gap-x-3 border-t border-rule py-3.5 text-left transition-colors ${on ? 'bg-fg text-surface' : 'hover:bg-raised'}`}
                  >
                    <span className={`pl-2 font-mono text-small num ${on ? 'text-surface/70' : 'text-muted'}`}>{String(s.n).padStart(2, '0')}</span>
                    <span className="min-w-0">
                      <span className="block font-medium">
                        {s.name}
                        {s.proposed && <span className={`ml-2 inline-block size-2 align-middle ${on ? 'bg-surface' : 'bg-cobalt'}`} aria-label="proposed stage" />}
                      </span>
                      <span className={`mt-0.5 block font-mono text-[0.7rem] ${on ? 'text-surface/70' : 'text-muted'}`}>{s.shapeOut}</span>
                    </span>
                    <span className={`pr-2 font-mono text-small num ${on ? 'text-surface' : 'text-muted'}`}>{s.budgetKey ? int(params(s.budgetKey)) : '—'}</span>
                  </button>
                </li>
              );
            })}
          </ol>

          <article className="col-span-12 lg:col-span-7 lg:border-l lg:border-rule lg:pl-10" aria-live="polite">
            <p className="font-mono text-small text-muted">Stage {String(stage.n).padStart(2, '0')} of 08{stage.proposed ? ', proposed' : ''}</p>
            <h3 className="display mt-3 text-[clamp(2rem,1.4rem+2.2vw,3.4rem)] leading-[0.95]">{stage.name}</h3>
            <p className="mt-6 max-w-[60ch] text-body">{stage.body}</p>
            {stage.tex && (
              <div className="mt-8 border-y border-rule py-5">
                <Equation tex={stage.tex} label={`Equation for ${stage.name}`} />
              </div>
            )}
            <dl className="mt-8 grid grid-cols-2 gap-x-6 gap-y-6 sm:grid-cols-4">
              <div>
                <dt className="text-small text-muted">Parameters</dt>
                <dd className="display num mt-1 text-h3 tracking-[-0.03em]">{stage.budgetKey ? int(stageParams) : '0'}</dd>
              </div>
              <div>
                <dt className="text-small text-muted">Share</dt>
                <dd className="display num mt-1 text-h3 tracking-[-0.03em]">{stage.budgetKey ? pct(stageParams / total, stageParams / total < 0.01 ? 2 : 1) : '—'}</dd>
              </div>
              <div className="col-span-2">
                <dt className="text-small text-muted">Tensor in → out</dt>
                <dd className="mt-2 font-mono text-small leading-relaxed">
                  {stage.shapeIn}
                  <br />→ {stage.shapeOut}
                </dd>
              </div>
            </dl>
            {stage.budgetKey && verification && (
              <p className="mt-8 text-small text-muted">
                {verification.parameters.matches_report
                  ? 'Counted by an independent pure-Python counter and equal to the report’s Table 5.7.'
                  : 'The counted parameters differ from the report’s Table 5.7; see results/verification.json.'}
              </p>
            )}
          </article>
        </div>
      </Container>
    </section>
  );
}

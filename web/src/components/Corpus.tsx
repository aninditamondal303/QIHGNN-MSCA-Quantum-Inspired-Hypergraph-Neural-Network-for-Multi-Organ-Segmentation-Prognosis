import type { CorpusEntry, CorpusFile, Label } from '../lib/types';
import { Container, SectionHead } from './chrome';
import { labelName, Swatch } from './Legend';

const ORGANS: Label[] = ['brain', 'liver', 'kidney', 'cervix', 'spleen'];

type CellState = 'supervised' | 'pending' | 'refused' | 'silent';

function cellState(c: CorpusEntry, organ: Label): CellState {
  if (!c.claimed_channels.includes(organ)) return 'silent';
  if (c.status === 'admitted' && c.supervises.includes(organ)) return 'supervised';
  if (c.status === 'pending inspection') return 'pending';
  return 'refused';
}

function Cell({ state, label }: { state: CellState; label: string }) {
  const text: Record<CellState, string> = {
    supervised: 'supervised',
    pending: 'claimed, awaiting inspection',
    refused: 'claimed, not admitted',
    silent: 'silent',
  };
  return (
    <td className="border-b border-l border-rule p-0">
      <div
        className={`relative h-12 ${state === 'supervised' ? 'bg-cobalt' : state === 'silent' ? 'hatch-fine opacity-60' : ''}`}
        title={`${label}: ${text[state]}`}
      >
        {state === 'pending' && <span className="absolute inset-1.5 border border-dashed border-cobalt" />}
        {state === 'refused' && (
          <svg className="absolute inset-0 size-full" preserveAspectRatio="none" viewBox="0 0 10 10" aria-hidden="true">
            <path d="M0 0 L10 10 M10 0 L0 10" stroke="var(--fg)" strokeWidth="0.15" vectorEffect="non-scaling-stroke" />
          </svg>
        )}
        <span className="sr-only">{text[state]}</span>
      </div>
    </td>
  );
}

const STATUS_STYLE: Record<CorpusEntry['status'], string> = {
  admitted: 'bg-cobalt text-white',
  'pending inspection': 'border border-cobalt text-cobalt',
  rejected: 'border border-vermilion text-vermilion-text',
  'not registered': 'border border-rule-strong text-muted',
};

export function Corpus({ corpus }: { corpus: CorpusFile | null }) {
  const corpora = corpus?.corpora ?? [];
  const sup = new Set(corpus?.supervisedChannels ?? ['background']);
  return (
    <section id="corpus" aria-labelledby="corpus-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="05"
          id="corpus"
          title="Who supervises what"
          standfirst="A dataset may supervise an organ only if it has pixel-level masks for it in volumetric CT or MRI. Everything else is silent, and silence is never read as background."
        />
        <div className="mt-14 grid grid-cols-12 gap-x-6 gap-y-10">
          <div className="col-span-12 overflow-x-auto xl:col-span-8">
            <table className="w-full min-w-[640px] border-collapse text-left">
              <caption className="sr-only">Supervision matrix: datasets by organ channel</caption>
              <thead>
                <tr className="border-b border-rule-strong">
                  <th scope="col" className="w-[34%] pb-2 text-small font-normal text-muted">Dataset</th>
                  {ORGANS.map((o) => (
                    <th key={o} scope="col" className="border-l border-rule px-2 pb-2 text-small font-normal text-muted">{labelName(o)[0].toUpperCase() + labelName(o).slice(1)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corpora.map((c) => (
                  <tr key={c.key}>
                    <th scope="row" className="border-b border-rule py-2 pr-3 align-middle font-normal">
                      <span className="block font-medium">{c.name}</span>
                      <span className="block font-mono text-[0.7rem] text-muted">{c.kaggle ?? 'not in the current dataset list'}</span>
                    </th>
                    {ORGANS.map((o) => <Cell key={o} state={cellState(c, o)} label={`${c.name}, ${labelName(o)}`} />)}
                  </tr>
                ))}
                <tr>
                  <th scope="row" className="pt-3 pr-3 align-middle text-small font-semibold">What the network may report</th>
                  {ORGANS.map((o) => (
                    <td key={o} className="border-l border-rule px-2 pt-3 align-middle">
                      <span className="inline-flex items-center gap-1.5 text-small">
                        <Swatch kind={sup.has(o) ? 'supervised' : 'silent'} />
                        {sup.has(o) ? 'reported' : 'suppressed'}
                      </span>
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
            <div className="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-small text-muted">
              <span className="inline-flex items-center gap-2"><span className="size-3 bg-cobalt" />Supervised</span>
              <span className="inline-flex items-center gap-2"><span className="size-3 border border-dashed border-cobalt" />Claimed, awaiting inspection</span>
              <span className="inline-flex items-center gap-2">
                <svg className="size-3 border border-rule" viewBox="0 0 10 10" aria-hidden="true"><path d="M0 0 L10 10 M10 0 L0 10" stroke="var(--fg)" strokeWidth="0.8" /></svg>
                Claimed, not admitted
              </span>
              <span className="inline-flex items-center gap-2"><span className="hatch-fine size-3 opacity-60" />Silent in this dataset</span>
            </div>
            {!corpus?.inspected && (
              <p className="mt-6 max-w-[70ch] border-l-2 border-cobalt pl-4 text-small">
                No dataset has been inspected on this machine yet, so every organ is currently suppressed. Download the data and run
                <code className="mx-1 font-mono">python -m qihgnn.data.inspection</code>; this table then shows the admitted datasets.
              </p>
            )}
          </div>

          <div className="col-span-12 xl:col-span-4">
            <ol className="border-t border-rule">
              {corpora.map((c) => (
                <li key={c.key} className="border-b border-rule py-4">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="font-medium">{c.name}</span>
                    <span className={`shrink-0 px-1.5 py-0.5 font-mono text-[0.68rem] ${STATUS_STYLE[c.status]}`}>{c.status}</span>
                  </div>
                  <p className="mt-1.5 text-small text-muted">{c.reason || c.notes}</p>
                  <p className="mt-1.5 font-mono text-[0.7rem] text-muted">{[c.modality, c.expected_format].filter(Boolean).join('. ')}.</p>
                </li>
              ))}
            </ol>
          </div>

          <div className="col-span-12 border-t border-rule pt-8 lg:col-span-8 lg:col-start-5">
            <p className="text-small font-semibold">Corpus recorded in the project report: {corpus?.reportCorpus.cases ?? 709} usable cases</p>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-small">
                <thead className="text-muted">
                  <tr className="border-b border-rule-strong">
                    {['Corpus', 'Modality', 'Format', 'Cases', 'Supervises'].map((h) => <th key={h} className="py-2 pr-4 font-normal">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {(corpus?.reportCorpus.rows ?? []).map((r) => (
                    <tr key={r.name} className="border-b border-rule">
                      <td className="py-2 pr-4">{r.name}</td>
                      <td className="py-2 pr-4 text-muted">{r.modality}</td>
                      <td className="py-2 pr-4 text-muted">{r.format}</td>
                      <td className="py-2 pr-4 font-mono num">{r.cases ?? '—'}</td>
                      <td className={`py-2 pr-4 ${r.supervises === 'rejected' ? 'text-vermilion-text' : ''}`}>{r.supervises === 'rejected' ? 'rejected: classification labels only' : labelName(r.supervises)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </Container>
    </section>
  );
}

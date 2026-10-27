import { useMemo, useState } from 'react';
import { LuSearch } from 'react-icons/lu';
import { PAPERS, THEMES, type Theme } from '../content/literature';
import { Container, SectionHead } from './chrome';

type Filter = 'all' | Theme;

export function Literature() {
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState<number | null>(null);
  const [all, setAll] = useState(false);
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return PAPERS.filter((p) => (filter === 'all' || p.theme === filter) && (!q || `${p.title} ${p.authors} ${p.approach} ${p.gap}`.toLowerCase().includes(q)));
  }, [filter, query]);
  const filters: { v: Filter; label: string; n: number }[] = [
    { v: 'all', label: 'All', n: PAPERS.length },
    ...(Object.keys(THEMES) as Theme[]).map((t) => ({ v: t, label: THEMES[t], n: PAPERS.filter((p) => p.theme === t).length })),
  ];

  return (
    <section id="literature" aria-labelledby="literature-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="07"
          id="literature"
          title="Thirty papers, thirty gaps"
          standfirst="The literature review behind the design. Each entry records what the work contributes and the gap it leaves, which is the part this project responds to."
        />
        <div className="mt-14 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div role="radiogroup" aria-label="Theme" className="flex flex-wrap border-l border-t border-rule-strong">
            {filters.map((f) => (
              <button
                key={f.v}
                type="button"
                role="radio"
                aria-checked={filter === f.v}
                onClick={() => setFilter(f.v)}
                className={`border-b border-r border-rule-strong px-3 py-2 text-small ${filter === f.v ? 'bg-fg text-surface' : 'hover:bg-raised'}`}
              >
                {f.label} <span className={`font-mono text-[0.7rem] ${filter === f.v ? 'text-surface/70' : 'text-muted'}`}>{f.n}</span>
              </button>
            ))}
          </div>
          <label className="flex w-full items-center gap-2 border-b border-rule-strong py-2 md:w-80">
            <LuSearch className="size-4 text-muted" aria-hidden="true" />
            <span className="sr-only">Search papers</span>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search titles, authors, gaps" className="w-full bg-transparent text-small outline-none placeholder:text-muted" type="search" />
          </label>
        </div>

        <div className="mt-8 border-t border-rule-strong">
          <div className="hidden grid-cols-[3rem_4rem_minmax(0,1.4fr)_minmax(0,1fr)] gap-x-6 border-b border-rule py-2 text-small text-muted md:grid">
            <span>No.</span><span>Year</span><span>Paper</span><span>Gap it leaves</span>
          </div>
          <ul>
            {(all || query || filter !== 'all' ? rows : rows.slice(0, 8)).map((p) => {
              const isOpen = open === p.n;
              return (
                <li key={p.n} className="border-b border-rule">
                  <button
                    type="button"
                    aria-expanded={isOpen}
                    onClick={() => setOpen(isOpen ? null : p.n)}
                    className="grid w-full grid-cols-[3rem_1fr] gap-x-4 py-4 text-left transition-colors hover:bg-raised md:grid-cols-[3rem_4rem_minmax(0,1.4fr)_minmax(0,1fr)] md:gap-x-6"
                  >
                    <span className="font-mono text-small text-muted num">3.{p.n}</span>
                    <span className="hidden font-mono text-small num md:block">{p.year}</span>
                    <span>
                      <span className="block font-medium leading-snug">{p.title}</span>
                      <span className="mt-1 block text-small text-muted">{p.authors}<span className="md:hidden">, {p.year}</span></span>
                    </span>
                    <span className="col-start-2 mt-2 text-small md:col-start-auto md:mt-0">{p.gap}</span>
                  </button>
                  {isOpen && (
                    <div className="grid grid-cols-[3rem_1fr] gap-x-4 pb-5 md:grid-cols-[3rem_4rem_minmax(0,1.4fr)_minmax(0,1fr)] md:gap-x-6">
                      <span className="md:col-span-2" />
                      <p className="text-small"><span className="text-muted">Approach. </span>{p.approach}</p>
                      <p className="col-start-2 mt-2 text-small text-muted md:col-start-auto md:mt-0">Theme: {THEMES[p.theme]}</p>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
          {!all && !query && filter === 'all' && rows.length > 8 && (
            <button type="button" onClick={() => setAll(true)} className="mt-6 border border-rule-strong px-4 py-2.5 text-small hover:bg-fg hover:text-surface">
              Show all {rows.length} papers
            </button>
          )}
          {rows.length === 0 && <p className="py-8 text-small text-muted">No paper matches “{query}”. Clear the search or pick another theme.</p>}
        </div>
      </Container>
    </section>
  );
}

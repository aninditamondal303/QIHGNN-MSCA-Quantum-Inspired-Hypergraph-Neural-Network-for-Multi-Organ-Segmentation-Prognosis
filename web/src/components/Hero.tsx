import { useEffect, useMemo, useState } from 'react';
import { LuChevronLeft, LuChevronRight } from 'react-icons/lu';
import type { CaseInfo, Verification } from '../lib/types';
import { defaultChannel, LABEL_ORDER, overlaysFor, useCase } from '../lib/useCase';
import { labelExtent } from '../lib/volume';
import { Container } from './chrome';
import { ScanHud } from './fx';
import { FilmCanvas } from './FilmCanvas';
import { Legend } from './Legend';

export function Hero({ cases, children }: { cases: CaseInfo[]; verification?: Verification | null; children: React.ReactNode }) {
  const [which, setWhich] = useState(0);
  const info = cases[which] ?? null;
  const channel = info ? defaultChannel(info) : 0;
  const { image, labels } = useCase(info, channel);
  const [z, setZ] = useState<number | null>(null);
  const [developed, setDeveloped] = useState(false);

  // Slices that contain labelled anatomy, used for the contact sheet.
  const sheet = useMemo(() => {
    if (!labels || !info) return [] as number[];
    const ext = info.present.map((p) => labelExtent(labels, LABEL_ORDER.indexOf(p))).filter(Boolean) as { first: number; last: number }[];
    const first = Math.min(...ext.map((e) => e.first), labels.d - 1);
    const last = Math.max(...ext.map((e) => e.last), 0);
    const n = 6;
    return Array.from({ length: n }, (_, i) => Math.round(first + ((last - first) * (i + 0.5)) / n));
  }, [labels, info]);

  useEffect(() => {
    setZ(null);
  }, [which]);
  useEffect(() => {
    if (z === null && sheet.length) setZ(sheet[Math.floor(sheet.length / 2)]);
  }, [sheet, z]);
  useEffect(() => {
    if (image) {
      const t = window.setTimeout(() => setDeveloped(true), 1800);
      return () => window.clearTimeout(t);
    }
  }, [image]);

  const overlays = info ? overlaysFor(info) : [];
  const silent = info ? info.present.filter((p) => !info.supervised.includes(p)) : [];
  const supervised = info ? info.present.filter((p) => info.supervised.includes(p)) : [];

  const step = (d: number) => cases.length && setWhich((w) => (w + d + cases.length) % cases.length);

  return (
    <section id="top" aria-labelledby="hero-title" className="pb-16 pt-10 md:pt-16 lg:pb-24">
      <Container>
        <div className="grid grid-cols-12 gap-x-6 gap-y-10">
          <div className="@container col-span-12 flex flex-col lg:col-span-7">{children}</div>

          <figure className="col-span-12 lg:col-span-5" aria-label="Research model viewer">
            <div className="film">
              <div className="flex items-center justify-between border-b border-film-rule px-4 py-2.5">
                <div className="flex min-w-0 items-baseline gap-3">
                  <span className="font-mono text-small">{info?.id ?? 'Loading'}</span>
                  {info && <span className="meta truncate text-film-muted">{info.modality}, {info.channels[channel]?.name}</span>}
                </div>
                <div className="flex items-center">
                  <span className="meta mr-2 text-film-muted num">{cases.length ? `${which + 1}/${cases.length}` : ''}</span>
                  <button type="button" onClick={() => step(-1)} className="grid size-8 place-items-center hover:bg-white/10" aria-label="Previous case">
                    <LuChevronLeft className="size-4" />
                  </button>
                  <button type="button" onClick={() => step(1)} className="grid size-8 place-items-center hover:bg-white/10" aria-label="Next case">
                    <LuChevronRight className="size-4" />
                  </button>
                </div>
              </div>
              <ScanHud readout={[['CASE', info?.id ?? '—'], ['SEQ', info?.channels[channel]?.name ?? '—'], ['SLICE', `${z ?? 0}`], ['MODE', info?.synthetic ? 'PHANTOM' : 'HELD-OUT']]}>
              <div className="relative">
                <FilmCanvas
                  image={image}
                  labels={labels}
                  plane="axial"
                  index={z ?? 0}
                  overlays={overlays}
                  opacity={0.12}
                  spacing={info?.spacing}
                  develop={!developed}
                  ariaLabel={info ? `${info.title}, axial slice ${z ?? 0}` : 'Loading scan'}
                />
                {info?.synthetic && (
                  <span className="meta absolute left-3 top-3 bg-film/80 px-1.5 py-0.5 text-film-muted">Synthetic phantom</span>
                )}
                <span className="meta num absolute bottom-3 right-3 bg-film/80 px-1.5 py-0.5 text-film-muted">z {String(z ?? 0).padStart(2, '0')}</span>
              </div>
              </ScanHud>
              <div className="grid grid-cols-6 gap-px border-t border-film-rule bg-film-rule" role="group" aria-label="Contact sheet">
                {sheet.map((k) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setZ(k)}
                    aria-pressed={k === z}
                    aria-label={`Show slice ${k}`}
                    className={`relative bg-film ${k === z ? 'outline outline-2 -outline-offset-2 outline-[#5474ff]' : 'opacity-70 hover:opacity-100'}`}
                  >
                    <FilmCanvas image={image} labels={labels} plane="axial" index={k} overlays={overlays} opacity={0.12} spacing={info?.spacing} ariaLabel={`Slice ${k}`} />
                    <span className="meta num absolute bottom-0.5 left-1 text-[0.6rem] text-film-muted">{k}</span>
                  </button>
                ))}
              </div>
            </div>
            <figcaption className="mt-3 flex flex-col gap-2">
              <Legend supervised={supervised} silent={silent} />
              {info && <p className="text-small text-muted">{info.note}</p>}
            </figcaption>
          </figure>
        </div>
      </Container>
    </section>
  );
}

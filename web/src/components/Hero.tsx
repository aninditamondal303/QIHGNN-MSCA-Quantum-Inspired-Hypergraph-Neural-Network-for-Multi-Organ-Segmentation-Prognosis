import { useEffect, useMemo, useState } from 'react';
import { LuChevronLeft, LuChevronRight } from 'react-icons/lu';
import { int } from '../lib/format';
import type { CaseInfo, Verification } from '../lib/types';
import { defaultChannel, LABEL_ORDER, overlaysFor, useCase } from '../lib/useCase';
import { labelExtent } from '../lib/volume';
import { Container } from './chrome';
import { FilmCanvas } from './FilmCanvas';
import { Legend } from './Legend';

function Figure({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-rule pt-3">
      <p className="display num text-[clamp(1.9rem,1.2rem+1.6vw,2.75rem)] leading-none tracking-[-0.04em]">{value}</p>
      <p className="mt-2 max-w-[30ch] text-small text-muted">{children}</p>
    </div>
  );
}

export function Hero({ cases, verification }: { cases: CaseInfo[]; verification: Verification | null }) {
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
  const unl = verification?.unlabelled_prediction;
  const checks = verification ? Object.values(verification.checks) : [];

  const step = (d: number) => cases.length && setWhich((w) => (w + d + cases.length) % cases.length);

  return (
    <section id="top" aria-labelledby="hero-title" className="pb-16 pt-8 md:pt-12 lg:pb-24">
      <Container>
        <div className="grid grid-cols-12 gap-x-6 gap-y-10">
          <div className="@container col-span-12 flex flex-col lg:col-span-7">
            {/* Sized in container units: "BACKGROUND." is 8.31 em wide in Syne Bold, so 100cqi / 8.45 fills the column. */}
            <h1 id="hero-title" className="display leading-[0.86]" style={{ fontSize: 'calc(100cqi / 8.45)' }}>
              <span className="block">Silence</span>
              <span className="block pl-[39%]">is not</span>
              <span className="block">background.</span>
            </h1>
            <p className="mt-8 max-w-[52ch] text-lead font-light">
              One network that outlines organs in CT and MRI and estimates survival risk from the same encoding. It learns from public
              datasets that each label a single organ, without being taught that every unlabelled organ is empty tissue.
            </p>
            <p className="mt-6 max-w-[60ch] text-small text-muted">
              B.Tech project, Department of Computational Intelligence, SRM Institute of Science and Technology. Satakshi Srivastava, Anindita
              Mondal and Aayushman Ghatak, guided by Dr G Tamilmani.
            </p>
            <div className="mt-auto grid grid-cols-2 gap-x-6 gap-y-7 pt-12">
              <Figure value={verification ? int(verification.parameters.total) : '8,347,071'}>
                Parameters across eight stages, equal to the report’s count stage by stage.
              </Figure>
              <Figure value="66,577">Attention entries per sample, where full voxel attention needs 1.1 × 10¹².</Figure>
              <Figure value={unl ? `+${unl.max_marginal_change.toFixed(2)}` : '+0.00'}>
                Loss charged for recognising an organ the dataset never labelled, against +{unl ? unl.min_naive_increase.toFixed(2) : '0.68'} under a naive loss.
              </Figure>
              <Figure value={verification ? `${checks.filter(Boolean).length}/${checks.length}` : '—'}>
                Numerical checks passing in the last recorded verification run.
              </Figure>
            </div>
          </div>

          <figure className="col-span-12 lg:col-span-5" aria-label="Case viewer">
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

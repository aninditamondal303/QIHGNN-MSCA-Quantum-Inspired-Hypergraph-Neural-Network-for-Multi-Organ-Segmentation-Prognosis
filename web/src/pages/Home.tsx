import { LuArrowRight, LuBrain, LuChartLine, LuScanSearch, LuShieldCheck } from 'react-icons/lu';
import { Container } from '../components/chrome';
import { CountUp, Reveal, Ticker } from '../components/fx';
import { Hero } from '../components/Hero';
import { Link, useTitle } from '../lib/router';
import { pct, useScreen } from '../lib/screenMetrics';
import type { CaseInfo } from '../lib/types';

function Stat({ value, label, n, format }: { value: string; label: string; n?: number; format?: (v: number) => string }) {
  return (
    <div className="border-t border-rule pt-3">
      <p className="display num text-[clamp(1.9rem,1.2rem+1.6vw,2.75rem)] leading-none tracking-[-0.04em]">{n !== undefined && format ? <CountUp to={n} format={format} /> : value}</p>
      <p className="mt-2 max-w-[30ch] text-small text-muted">{label}</p>
    </div>
  );
}

export function Home({ cases }: { cases: CaseInfo[] }) {
  useTitle('');
  const { metrics, ready } = useScreen();
  const t = metrics?.classifier?.test;
  const ext = metrics?.classifier?.external_test;
  const seg = metrics?.segmenter?.brats_3d_test?.FLAIR;
  const surv = metrics?.survival?.cv_c_index?.full;
  const m = (v: string) => (ready ? v : '—');

  return (
    <>
      <Hero cases={cases}>
        <p className="meta mb-6 text-muted">Brain tumour detection · outlining · prognosis</p>
        {/* Sized in container units so the longest line fills the column at any width. */}
        <h1 id="hero-title" className="display leading-[0.86]" style={{ fontSize: 'calc(100cqi / 9.3)' }}>
          <span className="block">See it.</span>
          <span className="block pl-[12%]">Measure it.</span>
          <span className="block">Explain it.</span>
        </h1>
        <p className="mt-8 max-w-[54ch] text-lead font-light">
          Upload a brain MRI. QIHGNN-MSCA checks it for a tumour, outlines it, measures it and estimates survival, privately, inside your
          browser. Every number on this site is measured on scans the models never saw.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link to="/analyse" className="inline-flex h-12 items-center gap-2 bg-fg px-5 font-medium text-surface no-underline hover:bg-cobalt">
            Analyse a scan <LuArrowRight className="size-4" aria-hidden="true" />
          </Link>
          <Link to="/accuracy" className="inline-flex h-12 items-center gap-2 border border-rule-strong px-5 no-underline hover:bg-raised">
            See the accuracy
          </Link>
        </div>
        <div className="mt-auto grid grid-cols-2 gap-x-6 gap-y-7 pt-12">
          <Stat value={m(pct(t?.tumour_vs_no_tumour.sensitivity))} n={ready ? t?.tumour_vs_no_tumour.sensitivity : undefined} format={(v) => pct(v)} label="of tumours detected on held-out test scans (sensitivity)." />
          <Stat value={m(pct(t?.tumour_vs_no_tumour.specificity))} n={ready ? t?.tumour_vs_no_tumour.specificity : undefined} format={(v) => pct(v)} label="of tumour-free scans correctly cleared (specificity)." />
          <Stat value={m(pct(ext?.accuracy))} n={ready ? ext?.accuracy : undefined} format={(v) => pct(v)} label="accuracy on an independent dataset from a different source." />
          <Stat value={ready && seg?.dice3d.mean != null ? seg.dice3d.mean.toFixed(2) : '—'} n={ready && seg?.dice3d.mean != null ? seg.dice3d.mean : undefined} format={(v) => v.toFixed(2)} label="Dice overlap of 3D tumour outlines with expert segmentations." />
        </div>
      </Hero>

      <Ticker
        items={ready && t ? [
          `${pct(t.tumour_vs_no_tumour.sensitivity)} sensitivity`,
          `${pct(t.tumour_vs_no_tumour.specificity)} specificity`,
          ext ? `${pct(ext.accuracy)} on outside data` : 'tested on outside data',
          seg?.dice3d.mean != null ? `${seg.dice3d.mean.toFixed(2)} Dice in 3D` : '3D outlines',
          surv ? `C-index ${surv.mean.toFixed(2)}` : 'survival curves',
          '0 bytes uploaded',
        ] : ['Detect', 'Outline', 'Measure', 'Explain', 'Calibrated', 'Private', '0 bytes uploaded']}
      />

      <section className="border-t border-rule-strong py-16 md:py-24" aria-labelledby="what-title">
        <Container>
          <div className="grid grid-cols-12 gap-x-6 gap-y-10">
            <h2 id="what-title" className="display col-span-12 text-h2 md:col-span-5"><Reveal>Three answers from one scan.</Reveal></h2>
            <div className="col-span-12 grid gap-px bg-rule md:col-span-7 md:grid-cols-3">
              {[
                { icon: LuScanSearch, t: 'Detect', d: 'No tumour, glioma, meningioma or pituitary tumour. Unclear images and images that are not brain MRIs are flagged, not guessed.', s: ready && t ? `${pct(t.accuracy)} accuracy over 5 classes` : 'Accuracy published after training' },
                { icon: LuBrain, t: 'Outline', d: 'The tumour drawn on the scan. On a 3D scan, every slice is outlined and the volume is measured in cm³.', s: ready && seg ? `median volume error ${seg.volume_abs_error_cc.median?.toFixed(1)} cm³` : 'Error published after training' },
                { icon: LuChartLine, t: 'Prognosis', d: 'An estimated survival curve for glioma, from age, tumour volume and extent of surgery, with its uncertainty band.', s: ready && surv ? `concordance ${surv.mean.toFixed(2)}, cross-validated` : 'Concordance published after training' },
              ].map(({ icon: Icon, t: title, d, s }, i) => (
                <Reveal key={title} delay={i * 120} className="flex bg-surface">
                <article className="flex flex-col p-6">
                  <Icon className="size-6 text-cobalt" aria-hidden="true" />
                  <h3 className="display mt-6 text-h3">{title}</h3>
                  <p className="mt-3 text-small">{d}</p>
                  <p className="meta mt-auto pt-6 text-muted">{s}</p>
                </article>
                </Reveal>
              ))}
            </div>
          </div>
        </Container>
      </section>

      <section className="film py-16 md:py-24" aria-labelledby="honest-title">
        <Container>
          <div className="grid grid-cols-12 gap-x-6 gap-y-8">
            <h2 id="honest-title" className="display col-span-12 text-h2 text-film-fg md:col-span-6"><Reveal>Numbers you can trust.</Reveal></h2>
            <ul className="col-span-12 space-y-6 text-film-fg md:col-span-6">
              <li>
                <p className="font-medium">No duplicate leakage.</p>
                <p className="mt-1 text-small text-film-muted">
                  The most popular brain-MRI dataset has copies of its test images in its training set, which inflates the 98–99% accuracies
                  often quoted.{metrics?.dedup_audit ? ` We found ${metrics.dedup_audit.official_test_images_with_a_copy_in_training} such test images and` : ' We'} grouped
                  near-duplicates so a scan and its copies always fall on the same side of the split.
                </p>
              </li>
              <li>
                <p className="font-medium">Tested on outside data.</p>
                <p className="mt-1 text-small text-film-muted">Accuracy is also reported on an independent dataset the models never trained on, after removing any image it shares with ours.</p>
              </li>
              <li>
                <p className="font-medium">Knows when it doesn’t know.</p>
                <p className="mt-1 text-small text-film-muted">Probabilities are calibrated, and the tool declines to answer below the confidence at which it was validated.</p>
              </li>
            </ul>
          </div>
          <Link to="/accuracy" className="mt-10 inline-flex items-center gap-2 text-film-fg">Read the full evaluation <LuArrowRight className="size-4" aria-hidden="true" /></Link>
        </Container>
      </section>

      <section className="py-16 md:py-24" aria-labelledby="private-title">
        <Container>
          <div className="grid grid-cols-12 gap-x-6 gap-y-8">
            <div className="col-span-12 md:col-span-5">
              <LuShieldCheck className="size-8 text-cobalt" aria-hidden="true" />
              <h2 id="private-title" className="display mt-5 text-h2"><Reveal>Your scan stays with you.</Reveal></h2>
            </div>
            <div className="col-span-12 space-y-4 md:col-span-6 md:col-start-7">
              <p className="text-lead font-light">
                The models are downloaded once and run inside your browser with WebAssembly. Images are read from your disk into memory and are
                never sent to a server, stored or logged.
              </p>
              <Link to="/data" className="inline-flex items-center gap-2">How data is handled <LuArrowRight className="size-4" aria-hidden="true" /></Link>
            </div>
          </div>
        </Container>
      </section>

      <section className="border-t border-rule-strong py-16 md:py-24" aria-labelledby="research-title">
        <Container>
          <div className="grid grid-cols-12 gap-x-6 gap-y-8">
            <p className="meta col-span-12 text-muted md:col-span-3">Under the hood</p>
            <div className="col-span-12 md:col-span-9">
              <h2 id="research-title" className="display max-w-[18ch] text-h2"><Reveal>A research model that learns from incomplete labels.</Reveal></h2>
              <p className="mt-6 max-w-[62ch] text-lead font-light">
                Public datasets each label one organ. QIHGNN-MSCA is an 8.3-million-parameter 3D network that trains on all of them together
                without being taught that every unlabelled organ is empty tissue, and predicts survival from the same encoding. The viewer at
                the top of this page shows it on a synthetic phantom.
              </p>
              <Link to="/how-it-works" className="mt-6 inline-flex items-center gap-2">How it works <LuArrowRight className="size-4" aria-hidden="true" /></Link>
            </div>
          </div>
        </Container>
      </section>
    </>
  );
}

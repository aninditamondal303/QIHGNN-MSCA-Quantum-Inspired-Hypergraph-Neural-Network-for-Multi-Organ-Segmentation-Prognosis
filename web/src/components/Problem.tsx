import { Container, SectionHead } from './chrome';
import { Swatch } from './Legend';

// One voxel inside a liver, read by a model trained on a brain-only batch.
const VOXEL = [
  { label: 'Background', p: 0.18 },
  { label: 'Brain', p: 0.04 },
  { label: 'Liver', p: 0.62 },
  { label: 'Kidney', p: 0.06 },
  { label: 'Cervix', p: 0.03 },
  { label: 'Spleen', p: 0.07 },
];
const SUPERVISED = new Set(['Background', 'Brain']);

function Bar({ merged }: { merged: boolean }) {
  const unsup = VOXEL.filter((v) => !SUPERVISED.has(v.label)).reduce((a, v) => a + v.p, 0);
  const parts = merged
    ? [
        { label: 'Background', p: VOXEL[0].p, kind: 'bg' as const },
        { label: 'Unlabelled channels, folded into background', p: unsup, kind: 'folded' as const },
        { label: 'Brain', p: VOXEL[1].p, kind: 'sup' as const },
      ]
    : VOXEL.map((v) => ({ label: v.label, p: v.p, kind: SUPERVISED.has(v.label) ? (v.label === 'Background' ? ('bg' as const) : ('sup' as const)) : ('wrong' as const) }));
  return (
    <div className="flex h-9 w-full border border-rule-strong" role="img" aria-label={parts.map((p) => `${p.label} ${Math.round(p.p * 100)}%`).join(', ')}>
      {parts.map((part) => (
        <div
          key={part.label}
          style={{ width: `${part.p * 100}%` }}
          className={`relative h-full border-r border-surface last:border-r-0 ${
            part.kind === 'bg' ? 'bg-fg/15' : part.kind === 'sup' ? 'bg-cobalt' : part.kind === 'folded' ? 'bg-fg/15' : 'hatch'
          }`}
          title={`${part.label}: ${Math.round(part.p * 100)}%`}
        >
          {part.p > 0.12 && (
            <span className="absolute left-1.5 top-1/2 hidden -translate-y-1/2 whitespace-nowrap bg-surface px-1 font-mono text-[0.68rem] text-fg md:block">{part.label === 'Unlabelled channels, folded into background' ? 'merged into background' : part.label.toLowerCase()} {Math.round(part.p * 100)}%</span>
          )}
        </div>
      ))}
    </div>
  );
}

export function Problem() {
  return (
    <section id="problem" aria-labelledby="problem-title" className="py-20 lg:py-28">
      <Container>
        <SectionHead
          n="01"
          id="problem"
          title="Why one model is hard"
          standfirst="Public datasets each label one organ, attention over a whole volume does not fit in memory, and segmentation and prognosis are usually built as two separate systems."
        />
        <div className="mt-14 grid grid-cols-12 gap-x-6 gap-y-14">
          <div className="col-span-12 lg:col-span-5">
            <p className="display num text-figure tracking-[-0.045em]">1.1 × 10¹²</p>
            <p className="mt-4 max-w-[38ch] text-body text-muted">
              Pairwise entries in one self-attention map over a 128 × 128 × 64 volume. Stored as 32-bit floats that is 4.4 terabytes, for
              one head of one layer of one scan.
            </p>
          </div>
          <div className="col-span-12 lg:col-span-7">
            <div className="gap-10 text-body md:columns-2 [&>div]:mb-8 [&>div]:break-inside-avoid">
              <div>
                <h3 className="mb-2 font-semibold">Each dataset is silent about most organs</h3>
                <p>
                  BraTS outlines brain tumours, LiTS the liver, KiTS the kidneys. Train on all three with an ordinary loss and every brain scan
                  teaches the network that a liver is background. The model is punished for being right about organs nobody annotated.
                </p>
              </div>
              <div>
                <h3 className="mb-2 font-semibold">Global context does not fit</h3>
                <p>
                  Attention relates distant anatomy, but its cost grows with the square of the voxel count. Most volumetric models keep it
                  affordable by attending inside local windows, which gives up the global view that made attention worth using.
                </p>
              </div>
              <div>
                <h3 className="mb-2 font-semibold">Two pipelines learn the same tissue twice</h3>
                <p>
                  A mask is an intermediate product. Clinicians act on risk. Extracting hand-picked features from a mask and feeding them to a
                  separate survival model discards everything the features did not anticipate.
                </p>
              </div>
              <div>
                <h3 className="mb-2 font-semibold">What this project does about it</h3>
                <p>
                  A marginal loss folds unlabelled channels into background before scoring. Attention runs over four fixed token budgets.
                  Both heads read one 512-number encoding, and organs no dataset supervises are switched off at inference.
                </p>
              </div>
            </div>
          </div>

          <figure className="col-span-12 border-t border-rule pt-8 lg:col-span-10 lg:col-start-3">
            <div className="grid grid-cols-12 gap-x-6 gap-y-5">
              <p className="col-span-12 text-small font-semibold md:col-span-3">One voxel inside a liver, from a brain-only training batch</p>
              <div className="col-span-12 space-y-5 md:col-span-9">
                <div>
                  <p className="mb-2 flex items-center gap-2 text-small">
                    <Swatch kind="silent" /> Naive loss: the 62% on liver, and every other unlabelled channel, counts as error
                  </p>
                  <Bar merged={false} />
                </div>
                <div>
                  <p className="mb-2 flex items-center gap-2 text-small">
                    <Swatch kind="supervised" /> Marginal loss: unlabelled mass is merged into background, so the liver prediction costs nothing
                  </p>
                  <Bar merged />
                </div>
              </div>
            </div>
          </figure>
        </div>
      </Container>
    </section>
  );
}

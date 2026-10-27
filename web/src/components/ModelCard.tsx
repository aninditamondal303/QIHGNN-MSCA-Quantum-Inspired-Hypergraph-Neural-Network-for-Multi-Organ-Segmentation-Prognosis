import type { CasesFile, CorpusFile, Label } from '../lib/types';
import { Drawer } from './chrome';
import { labelName, Swatch } from './Legend';

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-rule py-5">
      <h3 className="mb-2 font-semibold">{title}</h3>
      <div className="space-y-2 text-small">{children}</div>
    </section>
  );
}

export function ModelCard({ open, onClose, cases, corpus }: { open: boolean; onClose: () => void; cases: CasesFile | null; corpus: CorpusFile | null }) {
  const labels = (cases?.labels ?? []) as Label[];
  const sup = new Set(corpus?.supervisedChannels ?? ['background']);
  return (
    <Drawer open={open} onClose={onClose} title="Model card">
      <p className="text-small">
        QIHGNN-MSCA, version 0.4.0. A 3D network with 8,347,071 parameters that segments five organ channels and estimates survival risk
        from one shared encoding.
      </p>
      <p className="mt-3 border-l-2 border-vermilion pl-3 text-small">Research prototype. Not a medical device and not for clinical use.</p>
      <Block title="Intended use">
        <p>Research on training one segmentation model across datasets that each label different organs, and on joint segmentation and prognosis.</p>
      </Block>
      <Block title="Not intended for">
        <p>Diagnosis, treatment planning or any decision about a patient. Reporting on any organ that no dataset supervised.</p>
      </Block>
      <Block title="Output channels">
        <ul className="border-t border-rule">
          {labels.filter((l) => l !== 'background').map((l) => (
            <li key={l} className="grid grid-cols-[1rem_1fr] gap-x-3 border-b border-rule py-2">
              <Swatch kind={sup.has(l) ? 'supervised' : 'silent'} />
              <span>
                <span className="font-medium">{labelName(l)}</span>
                <span className="text-muted"> {cases?.labelDescriptions[l]} {sup.has(l) ? '' : 'Currently suppressed.'}</span>
              </span>
            </li>
          ))}
        </ul>
      </Block>
      <Block title="Training data">
        {(corpus?.corpora ?? []).map((c) => (
          <p key={c.key}><span className="font-medium">{c.name}</span> <span className="text-muted">({c.status}). {c.reason || c.notes}</span></p>
        ))}
      </Block>
      <Block title="Evaluation">
        <p>Per-organ Dice and HD95, Harrell’s C-index and time-dependent AUC at 1, 3 and 5 years, against a matched baseline and three ablations. Scores are reported only after the evaluation script has run.</p>
      </Block>
      <Block title="Known limitations">
        <ul className="list-disc space-y-1.5 pl-4">
          <li>No converged run or baseline comparison has been reported yet.</li>
          <li>Each dataset supervises one organ, so adjacent organs in a single scan have not been evaluated together.</li>
          <li>The Cox term learns an ordering of risk; calibration depends on the horizon term and real survival data.</li>
          <li>With a softmax head the exclusion loss is identically zero.</li>
          <li>Slices exported as display images lose Hounsfield units and are replicated across input channels.</li>
        </ul>
      </Block>
      <Block title="Data and privacy">
        <p>Public research datasets only, used under their own licences. The scans on this site are synthetic phantoms generated in code.</p>
      </Block>
    </Drawer>
  );
}

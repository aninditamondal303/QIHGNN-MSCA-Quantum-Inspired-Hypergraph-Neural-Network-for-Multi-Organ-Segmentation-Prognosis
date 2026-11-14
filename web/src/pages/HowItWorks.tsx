import { Architecture } from '../components/Architecture';
import { Container, PageHead } from '../components/chrome';
import { Problem } from '../components/Problem';
import { Workbench } from '../components/Workbench';
import type { ApiState } from '../lib/data';
import { useTitle } from '../lib/router';
import { useScreen } from '../lib/screenMetrics';
import type { CasesFile, Verification } from '../lib/types';

const STEPS = [
  { t: 'Read', d: 'JPG/PNG slices are converted to grey levels exactly as during training; NIfTI volumes are read with their voxel size, slice by slice.' },
  { t: 'Check', d: 'An EfficientNet-B0 classifier (4.0 M parameters, ImageNet-initialised) gives calibrated probabilities for no tumour, glioma, meningioma, pituitary tumour and not a brain MRI. Below the validated confidence it abstains.' },
  { t: 'Explain', d: 'A class activation map shows which regions of the image carried the evidence for the answer.' },
  { t: 'Outline', d: 'A 2D U-Net (1.9 M parameters), trained on all four MRI sequences and on skull-on and skull-stripped images, marks the tumour pixel by pixel.' },
  { t: 'Measure', d: 'On a 3D scan every slice is outlined and the voxels are summed with the scan’s own voxel size to give a volume in cm³.' },
  { t: 'Estimate', d: 'A Cox proportional-hazards model on age, tumour volume and extent of surgery, refitted 200 times on bootstrap samples, gives the survival curve and its band.' },
];

export function HowItWorks({ cases, api, verification }: { cases: CasesFile | null; api: ApiState; verification: Verification | null }) {
  useTitle('How it works');
  const { metrics } = useScreen();
  const exp = metrics?.export;
  return (
    <>
      <PageHead eyebrow="How it works" title="From pixels to a prognosis.">
        Six steps run in your browser when you analyse a scan. Below them is the research model the project began with: one 3D network that
        learns from datasets that each label a single organ.
      </PageHead>
      <section aria-labelledby="pipeline-title" className="pb-20">
        <Container>
          <h2 id="pipeline-title" className="sr-only">Analysis pipeline</h2>
          <ol className="grid gap-px bg-rule md:grid-cols-3">
            {STEPS.map((s, i) => (
              <li key={s.t} className="bg-surface p-6">
                <span className="font-mono text-small text-muted">0{i + 1}</span>
                <h3 className="display mt-4 text-h3">{s.t}</h3>
                <p className="mt-3 text-small">{s.d}</p>
              </li>
            ))}
          </ol>
          <p className="mt-6 max-w-[70ch] text-small text-muted">
            Both networks are exported to ONNX and run with ONNX Runtime Web (WebAssembly).
            {exp && ` Each export is checked against PyTorch on the same input: largest difference ${Math.max(exp.classifier.max_abs_diff_vs_torch ?? 0, exp.segmenter.max_abs_diff_vs_torch ?? 0).toExponential(1)}.`}
            {' '}Image preprocessing in the browser mirrors the training code line for line.
          </p>
        </Container>
      </section>
      <div className="space-y-24 pb-24 md:space-y-36">
        <Problem />
        <Architecture verification={verification} />
        <Workbench cases={cases} api={api} />
      </div>
    </>
  );
}

import { LuArrowUpRight } from 'react-icons/lu';
import { Container, PageHead } from '../components/chrome';
import { BUILD } from '../lib/build';
import { Link, useTitle } from '../lib/router';

const DIFFERENT = [
  ['Three cancers, one place', 'Brain tumours on MRI, liver tumours on CT and cervical cells on Pap smears. Most public projects stop at one classification label; here an upload gives the type, the region, its size and, for brain tumours, a survival estimate.'],
  ['Honest evaluation', 'Duplicate-aware splits, an independent test set, confidence intervals and a published list of what the numbers do not cover.'],
  ['Calibrated, and allowed to say “I don’t know”', 'Confidence is calibrated on held-out data; the tool abstains below the validated level and rejects images that are not brain MRIs.'],
  ['Private by design', 'Inference runs in the browser. There is no server that could receive a scan.'],
  ['A research model underneath', 'A 3D network trained across datasets that each label a single organ, with a loss that does not punish it for recognising organs a dataset left unlabelled.'],
];

const FAQ = [
  ['Can I use this to find out whether I have a brain tumour?', 'No. It is a research project and has not been clinically validated. Only a doctor reading the full scan with your history can tell you that. If you are worried, see a doctor.'],
  ['What kind of image works best?', 'Brain: an axial brain MRI slice as JPG or PNG, or a whole 3D MRI as NIfTI. Liver: an axial abdominal CT slice, or a whole CT as NIfTI in Hounsfield units. Pap smear: a microscope image of a Papanicolaou-stained smear. Photos of a screen and X-rays are outside what the models were trained on.'],
  ['Why does it sometimes refuse to answer?', 'When its calibrated confidence is below the level validated on held-out data, or when the image does not look like a brain MRI, it says so rather than guessing.'],
  ['How accurate is it?', 'See the Accuracy page: every figure there is measured on images and patients that were not used for training, including an independent dataset.'],
  ['Where does my scan go?', 'Nowhere. It is processed in your browser and discarded when you close the tab.'],
];

export function About() {
  useTitle('About');
  return (
    <>
      <PageHead eyebrow="About" title="Built by Anindita Mondal.">
        QIHGNN-MSCA is an independent project in medical imaging AI: from a 3D research network for partially labelled multi-organ data to a
        private, browser-based tool for brain, liver and cervical cancer imaging, with its accuracy measured in the open.
      </PageHead>
      <Container className="pb-24">
        <section aria-labelledby="diff-t" className="border-t border-rule-strong py-12">
          <h2 id="diff-t" className="display text-h3">What makes it different</h2>
          <ol className="mt-6 grid gap-px bg-rule md:grid-cols-2">
            {DIFFERENT.map(([t, d], i) => (
              <li key={t} className="bg-surface p-6">
                <span className="font-mono text-small text-muted">0{i + 1}</span>
                <h3 className="mt-3 font-semibold">{t}</h3>
                <p className="mt-2 text-small text-muted">{d}</p>
              </li>
            ))}
          </ol>
        </section>

        <section id="limitations" aria-labelledby="lim-t" className="grid grid-cols-12 gap-x-6 gap-y-6 border-t border-rule-strong py-12">
          <h2 id="lim-t" className="display col-span-12 text-h3 lg:col-span-4">Intended use and limitations</h2>
          <div className="col-span-12 space-y-4 text-body lg:col-span-8">
            <p className="border-l-2 border-vermilion pl-3">Research and education only. Not a medical device, not cleared by any regulator, and not for diagnosis or treatment decisions.</p>
            <ul className="list-disc space-y-2 pl-5 text-small">
              <li>Trained on public datasets that do not represent every scanner, protocol, population or tumour type. Rarer tumours (for example metastases or lymphoma) are not among the classes.</li>
              <li>A single 2D slice can miss a tumour that is elsewhere in the brain. A whole 3D scan is more reliable.</li>
              <li>The outline was trained on whole-tumour regions (tumour plus surrounding swelling) and does not separate sub-regions.</li>
              <li>Liver: trained on 2D CT slices from 131 LiTS patients that were already windowed for display; CT scans uploaded in Hounsfield units are shown with a standard abdominal window. Almost all LiTS patients have tumours, so the ability to clear tumour-free scans rests on few examples.</li>
              <li>Pap smears: the training images outline cells and nuclei but carry no diagnosis. The tool measures nucleus-to-cytoplasm ratios and flags outliers; it does not grade a smear (for example as LSIL or HSIL). Staining and magnification differ between laboratories.</li>
              <li>The prognosis model was fitted on 236 pre-operative glioma patients. It ignores treatment, genetics (such as IDH and MGMT status) and performance status, which matter a great deal; it is a population estimate.</li>
              <li>Accuracy on the independent dataset is lower than on the held-out split of the training data; real-world accuracy can be lower still.</li>
            </ul>
          </div>
        </section>

        <section aria-labelledby="faq-t" className="border-t border-rule-strong py-12">
          <h2 id="faq-t" className="display text-h3">Questions</h2>
          <div className="mt-6 divide-y divide-rule border-y border-rule">
            {FAQ.map(([q, a]) => (
              <details key={q} className="group py-4">
                <summary className="cursor-pointer list-none font-medium marker:hidden">
                  <span className="mr-3 inline-block font-mono text-muted transition-transform group-open:rotate-45">+</span>{q}
                </summary>
                <p className="mt-3 max-w-[70ch] pl-7 text-small text-muted">{a}</p>
              </details>
            ))}
          </div>
        </section>

        <section aria-labelledby="stack-t" className="grid grid-cols-12 gap-x-6 gap-y-6 border-t border-rule-strong py-12">
          <h2 id="stack-t" className="display col-span-12 text-h3 lg:col-span-4">How it is built</h2>
          <div className="col-span-12 text-small lg:col-span-8">
            <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2">
              {[
                ['Models', 'PyTorch: EfficientNet-B0, 2D U-Net, 8-stage 3D research network'],
                ['Statistics', 'NumPy Cox model with bootstrap bands, Wilson intervals'],
                ['Training', 'Kaggle GPU notebooks driven from a script'],
                ['In the browser', 'ONNX Runtime Web (WebAssembly), React, TypeScript'],
                ['Design', 'Tailwind CSS, Syne and Plus Jakarta Sans, hand-drawn charts in SVG'],
                ['Testing', 'pytest, numerical verification, unit tests and Playwright browser tests'],
              ].map(([k, v]) => (
                <div key={k} className="border-t border-rule pt-2"><dt className="text-muted">{k}</dt><dd className="mt-1">{v}</dd></div>
              ))}
            </dl>
            <p className="mt-8 flex flex-wrap gap-6">
              <Link to="/how-it-works">How it works</Link>
              <Link to="/accuracy">Accuracy</Link>
              {BUILD.repo && <a href={BUILD.repo} target="_blank" rel="noreferrer" className="link-underline inline-flex items-center gap-1">Source code<LuArrowUpRight className="size-3.5" aria-hidden="true" /></a>}
            </p>
          </div>
        </section>
      </Container>
    </>
  );
}

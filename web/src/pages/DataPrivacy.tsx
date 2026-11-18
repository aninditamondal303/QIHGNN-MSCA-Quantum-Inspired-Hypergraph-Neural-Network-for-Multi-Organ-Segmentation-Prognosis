import { LuArrowUpRight, LuShieldCheck } from 'react-icons/lu';
import { Container, PageHead } from '../components/chrome';
import { Corpus } from '../components/Corpus';
import { useTitle } from '../lib/router';
import { useScreen } from '../lib/screenMetrics';
import type { CorpusFile } from '../lib/types';

const DATASETS: { name: string; role: string; what: string; href?: string }[] = [
  { name: 'Brain Tumor MRI Dataset', role: 'Tumour check: training, validation and test', what: 'About 7,000 MRI slices in four groups: no tumour, glioma, meningioma, pituitary.' },
  { name: 'Brain MRI Images for Brain Tumor Detection', role: 'Independent test only', what: 'About 250 MRI images labelled tumour / no tumour, from a different collection.' },
  { name: 'BraTS 2020', role: 'Outline, volume, prognosis; glioma slices for the tumour check', what: '369 pre-operative glioma MRI scans in four sequences with expert tumour segmentations; survival data for 236 patients.', href: 'https://www.med.upenn.edu/cbica/brats2020/' },
  { name: 'Brain tumour segmentation (2D)', role: 'Outline on skull-on images', what: 'About 3,000 contrast-enhanced T1 slices with tumour masks.' },
  { name: 'LiTS (2D slices)', role: 'Liver and liver-tumour outline; “not a brain MRI” examples; liver channel of the research model', what: '131 abdominal CT scans as slices, with liver and tumour labelled separately.', href: 'https://competitions.codalab.org/competitions/17094' },
  { name: 'LiTS (original CT volumes)', role: 'Liver and liver-tumour outline, whole-scan verdict', what: '131 CT scans in Hounsfield units with liver and tumour labelled; the model the website runs is trained on these.', href: 'https://competitions.codalab.org/competitions/17094' },
  { name: 'CHAOS', role: 'Tumour-free livers: training and the whole-scan specificity test', what: '40 abdominal CT scans of healthy livers from a second hospital; 20 with the liver outlined.', href: 'https://chaos.grand-challenge.org/' },
  { name: 'SIPaKMeD', role: 'Cell grading', what: 'About 4,000 single cervical cells cut from stained slides, each graded by cytologists into one of five types.', href: 'https://doi.org/10.1109/ICIP.2018.8451588' },
  { name: 'Cervical cytology', role: 'Cell and nucleus outline, N:C ratio; “not a brain MRI” examples', what: '686 stained microscope images with about 34,000 cells and their nuclei outlined. No diagnosis labels.' },
];

export function DataPrivacy({ corpus }: { corpus: CorpusFile | null }) {
  useTitle('Privacy');
  const { metrics } = useScreen();
  const a = metrics?.dedup_audit;
  const cd = metrics?.classifier_data;
  return (
    <>
      <PageHead eyebrow="Data & privacy" title="Public data in. Nothing of yours out.">
        The models learned only from public research datasets. The scans you analyse are processed on your own device.
      </PageHead>
      <Container className="pb-16">
        <section aria-labelledby="privacy-t" className="grid grid-cols-12 gap-x-6 gap-y-6 border-t border-rule-strong py-12">
          <div className="col-span-12 lg:col-span-4">
            <LuShieldCheck className="size-7 text-cobalt" aria-hidden="true" />
            <h2 id="privacy-t" className="display mt-4 text-h3">Your scans</h2>
          </div>
          <ul className="col-span-12 space-y-4 text-body lg:col-span-8">
            <li><span className="font-medium">Processed locally.</span> The models run in your browser with WebAssembly. Files are read into memory and never sent anywhere.</li>
            <li><span className="font-medium">Not stored.</span> Closing or reloading the tab discards the image and the result. There are no accounts, cookies or analytics.</li>
            <li><span className="font-medium">What is downloaded.</span> The website, the two model files and the WebAssembly runtime, all served from the same site.</li>
            <li><span className="font-medium">Remove identifiers yourself.</span> DICOM headers and burned-in text can identify a patient; export an anonymised image before sharing it with anyone.</li>
          </ul>
        </section>

        <section aria-labelledby="ds-t" className="border-t border-rule-strong py-12">
          <h2 id="ds-t" className="display text-h3">Training and test data</h2>
          <div className="mt-6 overflow-x-auto">
            <table className="w-full min-w-[640px] text-small">
              <thead><tr className="border-b border-rule-strong text-left text-muted"><th className="py-2 pr-4 font-normal">Dataset</th><th className="py-2 pr-4 font-normal">Used for</th><th className="py-2 font-normal">Contents</th></tr></thead>
              <tbody>
                {DATASETS.map((d) => (
                  <tr key={d.name} className="border-b border-rule align-top">
                    <td className="py-3 pr-4">{d.href ? <a href={d.href} target="_blank" rel="noreferrer" className="link-underline inline-flex items-center gap-1 font-medium">{d.name}<LuArrowUpRight className="size-3.5" aria-hidden="true" /></a> : <span className="font-medium">{d.name}</span>}</td>
                    <td className="py-3 pr-4">{d.role}</td>
                    <td className="py-3 text-muted">{d.what}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-small text-muted">Each dataset is used under the terms published on its page. None of it is redistributed here except a few held-out test images shown as samples.</p>
        </section>

        <section aria-labelledby="split-t" className="grid grid-cols-12 gap-x-6 gap-y-6 border-t border-rule-strong py-12">
          <div className="col-span-12 lg:col-span-4">
            <h2 id="split-t" className="display text-h3">How the data was split</h2>
          </div>
          <div className="col-span-12 space-y-4 text-body lg:col-span-8">
            <p>
              Splits are made per group, never per image: per patient for BraTS and LiTS, per slide for the cytology images, and per cluster of
              near-identical images for the 2D brain datasets. Near-identical images are found with a perceptual hash that survives resizing and re-encoding.
            </p>
            {a && (
              <p>
                In the brain-MRI dataset we found {a.duplicate_clusters} groups of near-duplicates covering {a.images_in_duplicate_clusters} of {a.images} images.
                {' '}{a.official_test_images_with_a_copy_in_training} of the dataset’s own {a.official_test_images} test images have a copy in its training folder, and
                {' '}{a.clusters_with_conflicting_labels} groups carry conflicting labels. With the official split those copies would be scored as unseen.
              </p>
            )}
            {cd && (
              <p className="text-small text-muted">
                Tumour-check images: {cd.train.n} training, {cd.val.n} validation, {cd.test.n} test.
              </p>
            )}
          </div>
        </section>
      </Container>
      <Corpus corpus={corpus} />
    </>
  );
}

import { useState } from 'react';
import { VERSION } from './lib/build';
import { useApiHealth, useJson } from './lib/data';
import type { CasesFile, CorpusFile, MetricsFile, TrainingHistory, Verification } from './lib/types';
import { Architecture } from './components/Architecture';
import { Header } from './components/chrome';
import { Corpus } from './components/Corpus';
import { Footer } from './components/Footer';
import { Hero } from './components/Hero';
import { Lab } from './components/Lab';
import { Literature } from './components/Literature';
import { ModelCard } from './components/ModelCard';
import { Problem } from './components/Problem';
import { Results } from './components/Results';
import { Workbench } from './components/Workbench';


export function App() {
  const api = useApiHealth();
  const cases = useJson<CasesFile>('data/cases.json');
  const verification = useJson<Verification>('data/verification.json');
  const corpus = useJson<CorpusFile>('data/corpus.json');
  // Optional results exist only after training/evaluation; the manifest lists what was exported.
  const manifest = useJson<{ files: string[] }>('data/manifest.json');
  const has = (f: string) => (manifest.data?.files.includes(f) ? `data/${f}` : null);
  const metrics = useJson<MetricsFile>(manifest.loading ? null : has('metrics.json'));
  const history = useJson<TrainingHistory>(manifest.loading ? null : has('training_history.json'));
  const [card, setCard] = useState(false);

  return (
    <>
      <a href="#problem" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:bg-fg focus:px-3 focus:py-2 focus:text-surface">
        Skip to content
      </a>
      <Header api={api} version={VERSION} onOpenCard={() => setCard(true)} />
      <main>
        <Hero cases={cases.data?.cases ?? []} verification={verification.data} />
        <Problem />
        <Architecture verification={verification.data} />
        <Workbench cases={cases.data} api={api} />
        <Lab verification={verification.data} api={api} />
        <Corpus corpus={corpus.data} />
        <Results metrics={metrics.data} history={history.data} verification={verification.data} loading={manifest.loading || metrics.loading} />
        <Literature />
      </main>
      <Footer verification={verification.data} />
      <ModelCard open={card} onClose={() => setCard(false)} cases={cases.data} corpus={corpus.data} />
    </>
  );
}

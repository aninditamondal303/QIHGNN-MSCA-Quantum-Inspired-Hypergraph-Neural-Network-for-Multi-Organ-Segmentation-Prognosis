import { useEffect, useState } from 'react';
import { Header } from './components/chrome';
import { Footer } from './components/Footer';
import { useApiHealth, useJson } from './lib/data';
import { useRoute } from './lib/router';
import type { CasesFile, CorpusFile, MetricsFile, TrainingHistory, Verification } from './lib/types';
import { About } from './pages/About';
import { Accuracy } from './pages/Accuracy';
import { Analyse } from './pages/Analyse';
import { DataPrivacy } from './pages/DataPrivacy';
import { Home } from './pages/Home';
import { HowItWorks } from './pages/HowItWorks';
import { Intro } from './pages/Intro';
import { NotFound } from './pages/NotFound';

export function App() {
  const path = useRoute();
  const api = useApiHealth();
  const cases = useJson<CasesFile>('data/cases.json');
  const verification = useJson<Verification>('data/verification.json');
  const corpus = useJson<CorpusFile>('data/corpus.json');
  // Research-model results exist only after training/evaluation; the manifest lists what was exported.
  const manifest = useJson<{ files: string[] }>('data/manifest.json');
  const has = (f: string) => (manifest.data?.files.includes(f) ? `data/${f}` : null);
  const metrics = useJson<MetricsFile>(manifest.loading ? null : has('metrics.json'));
  const history = useJson<TrainingHistory>(manifest.loading ? null : has('training_history.json'));

  // The intro plays at the site root once per visit (or with ?intro); deep links go straight to their page.
  const [entered, setEntered] = useState(() => {
    if (new URLSearchParams(window.location.search).has('intro')) return false;
    try {
      return sessionStorage.getItem('qihgnn-entered') === '1';
    } catch {
      return false;
    }
  });
  const enter = () => {
    try {
      sessionStorage.setItem('qihgnn-entered', '1');
    } catch {
      /* storage unavailable: the intro simply shows again next time */
    }
    if (window.location.search) window.history.replaceState({}, '', window.location.pathname);
    setEntered(true);
    window.scrollTo({ top: 0 });
  };

  useEffect(() => {
    if (entered) document.getElementById('main')?.focus({ preventScroll: true });
  }, [path, entered]);

  if (path === '/' && !entered) return <Intro onEnter={enter} />;

  let page;
  switch (path) {
    case '/':
      page = <Home cases={cases.data?.cases ?? []} />;
      break;
    case '/analyse':
      page = <Analyse />;
      break;
    case '/how-it-works':
      page = <HowItWorks cases={cases.data} api={api} verification={verification.data} />;
      break;
    case '/accuracy':
      page = <Accuracy metrics3d={metrics.data} history={history.data} verification={verification.data} loading3d={manifest.loading || metrics.loading} />;
      break;
    case '/data':
      page = <DataPrivacy corpus={corpus.data} />;
      break;
    case '/about':
      page = <About />;
      break;
    default:
      page = <NotFound />;
  }

  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:bg-fg focus:px-3 focus:py-2 focus:text-surface">
        Skip to content
      </a>
      <Header api={api} path={path} />
      <main id="main" tabIndex={-1} className="outline-none" key={path}>
        <div className="page-enter">{page}</div>
      </main>
      <Footer verification={verification.data} />
    </>
  );
}

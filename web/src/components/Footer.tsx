import { BUILD, MODEL, VERSION } from '../lib/build';
import { Link, openIntro, ROUTES } from '../lib/router';
import type { Verification } from '../lib/types';
import { Container, Mark } from './chrome';

function Col({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="meta mb-3 text-muted">{title}</p>
      <div className="text-small">{children}</div>
    </div>
  );
}

export function Footer({ verification }: { verification: Verification | null }) {
  return (
    <footer className="no-print border-t border-rule-strong pb-10 pt-12">
      <Container>
        <div className="grid grid-cols-12 gap-x-6 gap-y-10">
          <div className="col-span-12 lg:col-span-5">
            <Link to="/" className="flex items-center gap-3 no-underline">
              <Mark className="size-9 shrink-0" />
              <span className="display text-[1.35rem]">TriScan AI</span>
            </Link>
            <p className="mt-4 max-w-[46ch] text-small text-muted">
              Private AI for medical scans: brain MRI, liver CT and cervical cell analysis that runs entirely in your browser, powered by the {MODEL} network.
            </p>
            <p className="mt-6 text-small">Designed and built by <span className="font-medium">Anindita Mondal</span>. © {new Date().getFullYear()}</p>
          </div>
          <div className="col-span-12 grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-3 lg:col-span-7">
            <Col title="Product">
              <ul className="space-y-2">
                {ROUTES.map((r) => <li key={r.path}><Link to={r.path} className="no-underline hover:underline">{r.name}</Link></li>)}
              </ul>
            </Col>
            <Col title="Resources">
              <ul className="space-y-2">
                <li><Link to="/about#limitations" className="no-underline hover:underline">Intended use and limitations</Link></li>
                <li><Link to="/data" className="no-underline hover:underline">Privacy</Link></li>
                <li><button type="button" onClick={openIntro} className="text-left hover:underline">Replay the intro</button></li>
              </ul>
            </Col>
            <Col title="Release">
              <p className="font-mono text-[0.75rem] leading-relaxed text-muted">
                TriScan AI v{VERSION}<br />
                model {MODEL}<br />
                {BUILD.commit ? `commit ${BUILD.commit}` : 'local build'}<br />
                {verification ? `maths verified ${verification.generated_at.slice(0, 10)}` : ''}
              </p>
            </Col>
          </div>
        </div>
        <p className="mt-12 border-t border-rule-strong pt-4 text-small">
          Research and education only. Not a medical device and not for diagnosis or treatment. Scans you analyse never leave your device.
        </p>
      </Container>
    </footer>
  );
}

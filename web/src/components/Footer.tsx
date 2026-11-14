import { LuArrowUpRight } from 'react-icons/lu';
import { BUILD, VERSION } from '../lib/build';
import { Link, ROUTES } from '../lib/router';
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
              <span className="display text-[1.35rem]">QIHGNN-MSCA</span>
            </Link>
            <p className="mt-4 max-w-[46ch] text-small text-muted">
              Brain tumour detection, outlining and prognosis in the browser, with its accuracy measured in the open.
            </p>
            <p className="mt-6 text-small">A project by <span className="font-medium">Anindita Mondal</span>.</p>
          </div>
          <div className="col-span-12 grid grid-cols-2 gap-x-6 gap-y-8 sm:grid-cols-3 lg:col-span-7">
            <Col title="Site">
              <ul className="space-y-2">
                {ROUTES.map((r) => <li key={r.path}><Link to={r.path} className="no-underline hover:underline">{r.name}</Link></li>)}
              </ul>
            </Col>
            <Col title="Project">
              <ul className="space-y-2">
                <li><Link to="/about#limitations" className="no-underline hover:underline">Intended use and limitations</Link></li>
                <li><Link to="/data" className="no-underline hover:underline">Privacy</Link></li>
                {BUILD.repo && (
                  <li><a href={BUILD.repo} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 no-underline hover:underline">Source code<LuArrowUpRight className="size-3.5" aria-hidden="true" /></a></li>
                )}
              </ul>
            </Col>
            <Col title="Build">
              <p className="font-mono text-[0.75rem] leading-relaxed text-muted">
                v{VERSION}<br />
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

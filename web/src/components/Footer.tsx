import { useEffect, useState } from 'react';
import { LuArrowUpRight } from 'react-icons/lu';
import { BUILD, VERSION } from '../lib/build';
import type { Verification } from '../lib/types';
import { Container, Mark } from './chrome';

const DATASETS = [
  { name: 'LiTS on Kaggle', href: 'https://www.kaggle.com/datasets/harshwardhanbhangale/lits-dataset' },
  { name: 'BraTS 2020 on Kaggle', href: 'https://www.kaggle.com/datasets/awsaf49/brats2020-training-data' },
  { name: 'Cervical segmentation on Kaggle', href: 'https://www.kaggle.com/datasets/balajikartheek/cervical-cancer-segmentation' },
];

function useClock(): string {
  const fmt = () => new Date().toLocaleTimeString('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const [t, setT] = useState(fmt);
  useEffect(() => {
    const id = window.setInterval(() => setT(fmt()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return t;
}

function Col({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-rule py-4">
      <p className="mb-2 text-small text-muted">{title}</p>
      <div className="text-small">{children}</div>
    </div>
  );
}

export function Footer({ verification }: { verification: Verification | null }) {
  const clock = useClock();
  return (
    <footer className="mt-10 border-t border-rule-strong pb-10 pt-10">
      <Container>
        <div className="grid grid-cols-12 gap-x-6">
          <div className="col-span-12 mb-8 flex items-start gap-4 lg:col-span-4 lg:mb-0">
            <Mark className="size-10 shrink-0" />
            <p className="display text-[1.35rem] leading-[1.05]">
              Quantum-inspired hypergraph network with multi-scale cross-attention
            </p>
          </div>
          <div className="col-span-12 grid grid-cols-2 gap-x-6 md:grid-cols-4 lg:col-span-8">
            <Col title="Team">
              Satakshi Srivastava<br />Anindita Mondal<br />Aayushman Ghatak
            </Col>
            <Col title="Guide">
              Dr G Tamilmani<br />
              <span className="text-muted">Computational Intelligence, SRM IST, Kattankulathur</span>
            </Col>
            <Col title="Build">
              <span className="font-mono text-[0.75rem]">
                v{VERSION}<br />
                {BUILD.commit ? `commit ${BUILD.commit}` : 'local build'}<br />
                {verification ? `verified ${verification.generated_at.slice(0, 10)}` : 'not verified'}
              </span>
            </Col>
            <Col title="Kattankulathur, IST">
              <span className="font-mono text-[0.75rem] num" aria-live="off">{clock}</span>
            </Col>
            <div className="col-span-2 md:col-span-4">
              <Col title="Data and code">
                <ul className="flex flex-col gap-1.5 md:flex-row md:flex-wrap md:gap-x-6">
                  {DATASETS.map((d) => (
                    <li key={d.href}>
                      <a href={d.href} target="_blank" rel="noreferrer" className="link-underline inline-flex items-center gap-1">
                        {d.name}<LuArrowUpRight className="size-3.5" aria-hidden="true" />
                      </a>
                    </li>
                  ))}
                  {BUILD.repo && (
                    <li>
                      <a href={BUILD.repo} target="_blank" rel="noreferrer" className="link-underline inline-flex items-center gap-1">
                        Source repository<LuArrowUpRight className="size-3.5" aria-hidden="true" />
                      </a>
                    </li>
                  )}
                </ul>
              </Col>
            </div>
          </div>
        </div>
        <p className="mt-10 border-t border-rule-strong pt-4 text-small">
          Research prototype. Not a medical device and not for clinical use. All scans on this page are synthetic phantoms unless marked
          otherwise; no patient data is used.
        </p>
      </Container>
    </footer>
  );
}

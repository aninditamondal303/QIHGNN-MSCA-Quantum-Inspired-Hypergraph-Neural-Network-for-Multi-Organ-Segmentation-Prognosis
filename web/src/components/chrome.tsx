import { useEffect, useState, type ReactNode } from 'react';
import { LuArrowRight, LuMenu, LuMoon, LuScanLine, LuSun, LuX } from 'react-icons/lu';
import { Link, openIntro, ROUTES } from '../lib/router';
import type { ApiState } from '../lib/data';

export function Container({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`mx-auto w-full max-w-[1560px] px-4 sm:px-6 lg:px-10 ${className}`}>{children}</div>;
}

export function Mark({ className = '' }: { className?: string }) {
  // Three region nodes, one hyperedge frame (cobalt) and the diagonal of silence (vermilion).
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <rect x="3.5" y="3.5" width="25" height="25" fill="none" stroke="var(--cobalt)" strokeWidth="2" />
      <path d="M4 4 L28 28" stroke="var(--vermilion)" strokeWidth="2" />
      <circle cx="10" cy="11" r="2.6" fill="currentColor" />
      <circle cx="22" cy="9.5" r="2.6" fill="currentColor" />
      <circle cx="16.5" cy="22.5" r="2.6" fill="currentColor" />
    </svg>
  );
}

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'light');
  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('qihgnn-theme', next);
    } catch {
      /* private mode: the choice lasts for this visit only */
    }
    setTheme(next);
  };
  return [theme, toggle];
}

export function StatusDot({ api }: { api: ApiState }) {
  if (!api.checked) return <span className="meta text-muted">Checking API</span>;
  if (api.online) {
    return (
      <span className="inline-flex items-center gap-2" title={api.health.inference_ready ? 'API online, checkpoint loaded' : 'API online, no checkpoint yet'}>
        <span className="relative flex size-2">
          <span className="absolute inline-flex size-full animate-ping bg-cobalt opacity-60" />
          <span className="relative inline-flex size-2 bg-cobalt" />
        </span>
        <span className="meta text-muted">API {api.health.inference_ready ? 'live' : 'online'}</span>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2" title="The Python API is not running. Everything except running the network works.">
      <span className="inline-flex size-2 border border-muted" />
      <span className="meta text-muted">Static preview</span>
    </span>
  );
}

const NAV = ROUTES.filter((r) => r.path !== '/' && r.path !== '/analyse');

export function Header({ api, path }: { api: ApiState; path: string }) {
  const [theme, toggle] = useTheme();
  const [menu, setMenu] = useState(false);
  useEffect(() => setMenu(false), [path]);
  return (
    <header className="sticky top-0 z-40 border-b border-rule bg-surface/95 backdrop-blur-[2px]">
      <Container className="flex h-16 items-center gap-4">
        <Link
          to="/"
          className="flex items-center gap-2.5 no-underline"
          aria-label="QIHGNN-MSCA: back to the interactive intro"
          title="Back to the interactive intro"
          onClick={(e) => {
            if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
            e.preventDefault();
            openIntro();
          }}
        >
          <Mark className="size-7" />
          <span className="display text-[1.1rem] tracking-[-0.02em]">QIHGNN-MSCA</span>
        </Link>
        <span className="hidden lg:inline-flex">
          <StatusDot api={api} />
        </span>
        <nav aria-label="Main" className="ml-auto hidden lg:block">
          <ul className="flex items-center gap-1">
            {NAV.map((r) => (
              <li key={r.path}>
                <Link
                  to={r.path}
                  aria-current={path === r.path ? 'page' : undefined}
                  className={`px-3 py-2 text-small no-underline transition-colors ${path === r.path ? 'text-fg underline decoration-cobalt decoration-2 underline-offset-[7px]' : 'text-muted hover:text-fg'}`}
                >
                  {r.name}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="ml-auto flex items-center gap-1 lg:ml-2">
          <button
            type="button"
            onClick={openIntro}
            className="inline-flex h-10 items-center gap-2 px-2.5 text-small text-fg hover:bg-raised"
            aria-label="Back to the interactive intro"
            title="Back to the interactive intro"
          >
            <LuScanLine className="size-[18px]" aria-hidden="true" />
            <span className="hidden xl:inline">Intro</span>
          </button>
          <button
            type="button"
            onClick={toggle}
            className="grid size-10 place-items-center text-fg hover:bg-raised"
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {theme === 'dark' ? <LuSun className="size-[18px]" /> : <LuMoon className="size-[18px]" />}
          </button>
          <Link
            to="/analyse"
            aria-current={path === '/analyse' ? 'page' : undefined}
            className="hidden h-10 items-center gap-2 bg-fg px-4 text-small font-medium text-surface no-underline hover:bg-cobalt sm:inline-flex"
          >
            Analyse a scan <LuArrowRight className="size-4" aria-hidden="true" />
          </Link>
          <button
            type="button"
            className="grid size-10 place-items-center lg:hidden"
            aria-expanded={menu}
            aria-controls="mobile-nav"
            aria-label={menu ? 'Close menu' : 'Open menu'}
            onClick={() => setMenu((m) => !m)}
          >
            {menu ? <LuX className="size-5" /> : <LuMenu className="size-5" />}
          </button>
        </div>
      </Container>
      {menu && (
        <nav id="mobile-nav" aria-label="Main" className="border-t border-rule bg-surface lg:hidden">
          <Container className="grid grid-cols-1 py-2 sm:grid-cols-2">
            {ROUTES.map((r) => (
              <Link key={r.path} to={r.path} aria-current={path === r.path ? 'page' : undefined} className="flex items-center justify-between border-b border-rule py-3.5 no-underline">
                <span className={`display text-lead ${path === r.path ? 'text-cobalt' : ''}`}>{r.name}</span>
                <LuArrowRight className="size-4 text-muted" aria-hidden="true" />
              </Link>
            ))}
            <button type="button" onClick={openIntro} className="flex items-center justify-between border-b border-rule py-3.5 text-left">
              <span className="display text-lead">Interactive intro</span>
              <LuScanLine className="size-4 text-muted" aria-hidden="true" />
            </button>
            <div className="py-3 sm:col-span-2"><StatusDot api={api} /></div>
          </Container>
        </nav>
      )}
    </header>
  );
}

/** Page opener used by every inner page: eyebrow, large title, standfirst. */
export function PageHead({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="pb-10 pt-12 md:pb-14 md:pt-20">
      <Container>
        <p className="meta mb-5 text-muted">{eyebrow}</p>
        <h1 className="display max-w-[16ch] text-h2">{title}</h1>
        {children && <div className="mt-6 max-w-[62ch] text-lead font-light">{children}</div>}
      </Container>
    </div>
  );
}

export function SectionHead({ n, id, title, standfirst, children }: { n: string; id: string; title: string; standfirst: string; children?: ReactNode }) {
  return (
    <div className="border-t border-rule-strong pt-5">
      <div className="grid grid-cols-12 gap-x-6 gap-y-4">
        <p className="col-span-12 font-mono text-small text-muted md:col-span-1" aria-hidden="true">
          {n}
        </p>
        <h2 id={`${id}-title`} className="display col-span-12 text-h2 md:col-span-7">
          {title}
        </h2>
        <p className="col-span-12 self-end text-body text-muted md:col-span-4 measure">{standfirst}</p>
      </div>
      {children}
    </div>
  );
}

export function Drawer({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: ReactNode }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby="drawer-title">
      <button type="button" className="absolute inset-0 bg-ink/50" aria-label="Close" onClick={onClose} />
      <aside className="absolute right-0 top-0 flex h-full w-full max-w-[640px] flex-col border-l border-rule-strong bg-surface">
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-rule px-5">
          <h2 id="drawer-title" className="display text-lead">{title}</h2>
          <button type="button" onClick={onClose} className="grid size-9 place-items-center hover:bg-raised" aria-label="Close model card" autoFocus>
            <LuX className="size-5" />
          </button>
        </div>
        <div className="overflow-y-auto px-5 py-6">{children}</div>
      </aside>
    </div>
  );
}

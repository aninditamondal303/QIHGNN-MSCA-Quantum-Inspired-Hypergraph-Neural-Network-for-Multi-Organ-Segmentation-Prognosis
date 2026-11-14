import { useEffect, useState, type AnchorHTMLAttributes, type MouseEvent } from 'react';

/** Path-based routing that works under any base path (GitHub Pages serves the site from /<repo>/).
 *  The <base href> in index.html is set at build time; every route is relative to it. */
export const BASE = new URL(document.baseURI).pathname.replace(/[^/]*$/, '');

export const ROUTES = [
  { path: '/', name: 'Home' },
  { path: '/analyse', name: 'Analyse a scan' },
  { path: '/how-it-works', name: 'How it works' },
  { path: '/accuracy', name: 'Accuracy' },
  { path: '/data', name: 'Data & privacy' },
  { path: '/about', name: 'About' },
] as const;

export function currentPath(): string {
  let p = window.location.pathname;
  if (p.startsWith(BASE)) p = p.slice(BASE.length);
  p = '/' + p.replace(/^\/+|\/+$/g, '');
  return p === '/index.html' ? '/' : p;
}

export function href(to: string): string {
  return BASE + to.replace(/^\//, '');
}

export function navigate(to: string): void {
  const [path, hash] = to.split('#');
  window.history.pushState({}, '', href(path) + (hash ? `#${hash}` : ''));
  window.dispatchEvent(new Event('routechange'));
  if (hash) requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView());
  else window.scrollTo({ top: 0 });
}

/** Back to the interactive intro (the scanner) from anywhere on the site. */
export function openIntro(): void {
  if (currentPath() !== '/') window.history.pushState({}, '', href('/'));
  window.dispatchEvent(new Event('openintro'));
  window.dispatchEvent(new Event('routechange'));
}

export function useRoute(): string {
  const [path, setPath] = useState(currentPath);
  useEffect(() => {
    const on = () => setPath(currentPath());
    window.addEventListener('popstate', on);
    window.addEventListener('routechange', on);
    return () => {
      window.removeEventListener('popstate', on);
      window.removeEventListener('routechange', on);
    };
  }, []);
  return path;
}

export function Link({ to, onClick, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { to: string }) {
  const [path, hash] = to.split('#');
  return (
    <a
      {...rest}
      href={href(path) + (hash ? `#${hash}` : '')}
      onClick={(e: MouseEvent<HTMLAnchorElement>) => {
        onClick?.(e);
        if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        e.preventDefault();
        navigate(to);
      }}
    />
  );
}

export function useTitle(title: string): void {
  useEffect(() => {
    document.title = title ? `${title} · QIHGNN-MSCA` : 'QIHGNN-MSCA · Brain tumour detection, outlining and prognosis';
  }, [title]);
}

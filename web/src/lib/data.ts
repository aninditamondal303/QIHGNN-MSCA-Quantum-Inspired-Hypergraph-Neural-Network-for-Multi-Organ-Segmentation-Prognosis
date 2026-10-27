import { useEffect, useState } from 'react';
import type { Health } from './types';

/** Static JSON written by `python -m qihgnn.export_web`. Paths are relative so the site works under any base path. */
export async function fetchJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path, { cache: 'no-cache' });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

/** Pass `null` to skip the request (e.g. an optional file the manifest says does not exist). */
export function useJson<T>(path: string | null): { data: T | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; loading: boolean }>({ data: null, loading: path !== null });
  useEffect(() => {
    let alive = true;
    if (path === null) {
      setState({ data: null, loading: false });
      return;
    }
    fetchJson<T>(path).then((data) => alive && setState({ data, loading: false }));
    return () => {
      alive = false;
    };
  }, [path]);
  return state;
}

export type ApiState = { online: false; checked: boolean } | { online: true; checked: true; health: Health };

/** Checks /api/health on load and whenever the tab regains focus. The site works without the API
 * (static preview); the header says which mode it is in. */
export function useApiHealth(): ApiState {
  const [state, setState] = useState<ApiState>({ online: false, checked: false });
  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const res = await fetch('api/health', { cache: 'no-store' });
        const body = res.ok ? ((await res.json()) as Health) : null;
        if (!alive) return;
        setState(body && body.status === 'ok' ? { online: true, checked: true, health: body } : { online: false, checked: true });
      } catch {
        if (alive) setState({ online: false, checked: true });
      }
    };
    check();
    const onFocus = () => document.visibilityState === 'visible' && check();
    document.addEventListener('visibilitychange', onFocus);
    return () => {
      alive = false;
      document.removeEventListener('visibilitychange', onFocus);
    };
  }, []);
  return state;
}

export async function apiError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string; how_to_fix?: string };
    return [body.detail, body.how_to_fix].filter(Boolean).join(' ');
  } catch {
    return `The server answered ${res.status}.`;
  }
}

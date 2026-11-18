import { useEffect, useState } from 'react';

/** Offline support and install prompt. The service worker is registered on the deployed site only,
 *  so local development and the browser tests always see fresh files. */
const LOCAL = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname);

export function registerServiceWorker(): void {
  if (LOCAL || !('serviceWorker' in navigator)) return;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch(() => {
      /* offline support is a bonus; the app works without it */
    });
  });
}

interface InstallEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

let deferred: InstallEvent | null = null;
const listeners = new Set<() => void>();
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferred = e as InstallEvent;
  listeners.forEach((f) => f());
});
window.addEventListener('appinstalled', () => {
  deferred = null;
  listeners.forEach((f) => f());
});

/** Returns an install function while the browser offers installation, otherwise null. */
export function useInstall(): (() => Promise<void>) | null {
  const [, tick] = useState(0);
  useEffect(() => {
    const f = () => tick((n) => n + 1);
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  }, []);
  if (!deferred) return null;
  return async () => {
    const d = deferred;
    if (!d) return;
    await d.prompt();
    await d.userChoice;
    deferred = null;
    listeners.forEach((f) => f());
  };
}

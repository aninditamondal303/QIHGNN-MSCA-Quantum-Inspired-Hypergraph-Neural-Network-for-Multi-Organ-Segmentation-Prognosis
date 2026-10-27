declare const __BUILD__: { time: string; commit: string | null; repo: string | null };

/** Injected by scripts/build.mjs (GIT_COMMIT and REPO_URL environment variables are optional). */
export const BUILD = typeof __BUILD__ !== 'undefined' ? __BUILD__ : { time: new Date().toISOString(), commit: null, repo: null };
export const VERSION = '0.4.0';

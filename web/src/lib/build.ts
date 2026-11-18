declare const __BUILD__: { time: string; commit: string | null; repo: string | null };

/** Injected by scripts/build.mjs (GIT_COMMIT and REPO_URL environment variables are optional). */
export const BUILD = typeof __BUILD__ !== 'undefined' ? __BUILD__ : { time: new Date().toISOString(), commit: null, repo: null };
export const VERSION = '1.0.0';

/** Product name; QIHGNN-MSCA is the research network inside it. */
export const PRODUCT = 'TriScan AI';
export const MODEL = 'QIHGNN-MSCA';
export const TAGLINE = 'Medical scan analysis in your browser';

// Production build: esbuild bundles the React/TypeScript app, Tailwind v4 compiles the CSS,
// and public/ is copied alongside. Output goes to dist/, which the API serves at /.
import { build as esbuild, context } from 'esbuild';
import { compile } from 'tailwindcss';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(ROOT, 'src');
const DIST = join(ROOT, 'dist');

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(p)));
    else out.push(p);
  }
  return out;
}

// Tailwind v4: compile the stylesheet, then generate utilities for every class-like token in src/.
export async function buildCss() {
  const entry = join(SRC, 'styles.css');
  const css = await readFile(entry, 'utf8');
  const compiler = await compile(css, {
    base: SRC,
    from: entry,
    loadStylesheet: async (id, base) => {
      const path = id === 'tailwindcss' ? require.resolve('tailwindcss/index.css') : resolve(base, id);
      return { path, base: dirname(path), content: await readFile(path, 'utf8') };
    },
  });
  const candidates = new Set();
  for (const file of await walk(SRC)) {
    if (!/\.(tsx?|html)$/.test(file)) continue;
    const text = await readFile(file, 'utf8');
    // Class names can contain brackets, commas and parentheses (e.g. text-[clamp(1rem,2vw,3rem)]), so split on
    // whitespace and quotes first; a finer split adds fragments from template literals. Unknown tokens are ignored.
    for (const token of text.split(/[\s"'`]+/)) {
      if (!token) continue;
      candidates.add(token);
      for (const part of token.split(/[{}()<>;=,]+/)) if (part) candidates.add(part);
    }
  }
  return compiler.build([...candidates]);
}

function hash(text) {
  return createHash('sha256').update(text).digest('hex').slice(0, 10);
}

export const esbuildOptions = (dev) => ({
  entryPoints: { app: join(SRC, 'main.tsx'), infer: join(SRC, 'workers', 'infer.ts') },
  bundle: true,
  outdir: join(DIST, 'assets'),
  format: 'esm',
  splitting: true,
  chunkNames: 'chunks/[name]-[hash]',
  sourcemap: true,
  minify: !dev,
  target: ['es2022', 'chrome111', 'safari16', 'firefox115'],
  jsx: 'automatic',
  loader: { '.woff2': 'file', '.woff': 'file', '.ttf': 'file' },
  assetNames: 'fonts/[name]-[hash]',
  define: {
    'process.env.NODE_ENV': JSON.stringify(dev ? 'development' : 'production'),
    __BUILD__: JSON.stringify({ time: new Date().toISOString(), commit: process.env.GIT_COMMIT || null, repo: process.env.REPO_URL || null }),
  },
  nodePaths: process.env.NODE_PATH ? process.env.NODE_PATH.split(':') : [],
  logLevel: 'warning',
});

export async function writeHtml(cssHash = 'dev', jsHash = 'dev', dev = false) {
  const template = await readFile(join(ROOT, 'index.html'), 'utf8');
  const html = template
    .replace('%CSS%', `assets/styles.css?v=${cssHash}`)
    .replace('%KATEX%', `assets/app.css?v=${jsHash}`)
    .replace('%JS%', `assets/app.js?v=${jsHash}`)
    .replace('%BASE%', process.env.BASE_PATH || '/')
    .replace('%RELOAD%', dev ? `<script>new EventSource('/esbuild').addEventListener('change',()=>location.reload())</script>` : '');
  await writeFile(join(DIST, 'index.html'), html);
  // Every route gets its own index.html so deep links return 200 on static hosts; 404.html covers the rest.
  for (const route of ['analyse', 'how-it-works', 'accuracy', 'data', 'about']) {
    await mkdir(join(DIST, route), { recursive: true });
    await writeFile(join(DIST, route, 'index.html'), html);
  }
  await writeFile(join(DIST, '404.html'), html);
}

/** Offline support: the app shell, models and runtime are cached on first use, in a cache named after this build,
 *  so a new deploy replaces old models instead of mixing them. Pages are network-first, everything else cache-first. */
export async function writeServiceWorker(version) {
  const sw = `const CACHE = 'triscan-${version}';
self.addEventListener('install', (e) => { self.skipWaiting(); e.waitUntil(caches.open(CACHE).then((c) => c.addAll(['./', './analyse/', './manifest.webmanifest', './icons/icon-192.png']))); });
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith('triscan-') && k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin || req.url.includes('/api/') || req.url.includes('/esbuild')) return;
  if (req.mode === 'navigate') {
    e.respondWith(fetch(req).then((res) => { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); return res; })
      .catch(() => caches.match(req).then((r) => r || caches.match('./'))));
    return;
  }
  e.respondWith(caches.match(req).then((hit) => hit || fetch(req).then((res) => {
    if (res.ok && res.status === 200) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
    return res;
  })));
});
`;
  await writeFile(join(DIST, 'sw.js'), sw);
}

/** Self-host ONNX Runtime Web's WebAssembly backend (glue + binary); the loader is bundled by esbuild. */
export async function copyOrt() {
  // The package's exports map hides its dist/ folder from require.resolve, so locate the package directly.
  const base = (require.resolve.paths('onnxruntime-web') || []).map((p) => join(p, 'onnxruntime-web', 'dist')).find((p) => existsSync(p));
  if (!base) throw new Error('onnxruntime-web is not installed: run npm install in web/');
  const dist = base;
  await mkdir(join(DIST, 'ort'), { recursive: true });
  for (const f of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) await cp(join(dist, f), join(DIST, 'ort', f));
}

async function main() {
  await rm(DIST, { recursive: true, force: true });
  await mkdir(join(DIST, 'assets'), { recursive: true });
  await cp(join(ROOT, 'public'), DIST, { recursive: true });
  await copyOrt();
  const [css, result] = await Promise.all([buildCss(), esbuild({ ...esbuildOptions(false), metafile: true })]);
  await writeFile(join(DIST, 'assets', 'styles.css'), css);
  const js = await readFile(join(DIST, 'assets', 'app.js'), 'utf8');
  await writeHtml(hash(css), hash(js));
  await writeServiceWorker(hash(css + js));
  const bytes = Object.entries(result.metafile.outputs).filter(([k]) => k.endsWith('.js'))[0][1].bytes;
  console.log(`built dist/ — app.js ${(bytes / 1024).toFixed(0)} kB, styles.css ${(css.length / 1024).toFixed(0)} kB`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

export { context };

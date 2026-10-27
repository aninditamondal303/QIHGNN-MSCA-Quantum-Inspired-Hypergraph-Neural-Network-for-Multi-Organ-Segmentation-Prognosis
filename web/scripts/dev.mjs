// Development server on http://localhost:5173
// - rebuilds JS (esbuild) and CSS (Tailwind) on every change in src/ and reloads the page
// - proxies /api/* to the Python API on http://127.0.0.1:8765 (start it with `uvicorn api.app:app --port 8765`)
import { context } from 'esbuild';
import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { watch } from 'node:fs';
import http from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCss, esbuildOptions, writeHtml } from './build.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(ROOT, 'dist');
const PORT = Number(process.env.PORT || 5173);
const API = process.env.API_URL || 'http://127.0.0.1:8765';
const noWatch = process.argv.includes('--no-watch');

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.woff': 'font/woff', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.svg': 'image/svg+xml',
  '.map': 'application/json', '.txt': 'text/plain; charset=utf-8',
};
const clients = new Set();

async function rebuildCss() {
  try {
    await writeFile(join(DIST, 'assets', 'styles.css'), await buildCss());
  } catch (err) {
    console.error('[css]', err.message);
  }
}

async function start() {
  await mkdir(join(DIST, 'assets'), { recursive: true });
  await cp(join(ROOT, 'public'), DIST, { recursive: true });
  await writeHtml('dev', String(Date.now()), !noWatch);
  await rebuildCss();
  const ctx = await context({
    ...esbuildOptions(true),
    plugins: [{
      name: 'reload',
      setup(build) {
        build.onEnd((r) => {
          if (r.errors.length) return;
          for (const res of clients) res.write('event: change\ndata: js\n\n');
        });
      },
    }],
  });
  await ctx.rebuild();
  if (!noWatch) {
    await ctx.watch();
    let timer;
    watch(join(ROOT, 'src'), { recursive: true }, () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        await rebuildCss();
        for (const res of clients) res.write('event: change\ndata: css\n\n');
      }, 80);
    });
  }

  http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === '/esbuild') {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      clients.add(res);
      req.on('close', () => clients.delete(res));
      return;
    }
    if (url.pathname.startsWith('/api/')) {
      const target = new URL(url.pathname + url.search, API);
      const proxy = http.request(target, { method: req.method, headers: { ...req.headers, host: target.host } }, (up) => {
        res.writeHead(up.statusCode || 502, up.headers);
        up.pipe(res);
      });
      proxy.on('error', () => {
        res.writeHead(502, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ detail: 'The API is not running.', how_to_fix: 'From the repository root run: uvicorn api.app:app --port 8765' }));
      });
      req.pipe(proxy);
      return;
    }
    let path = normalize(decodeURIComponent(url.pathname)).replace(/^([/\\])+/, '');
    if (!path || path.endsWith('/')) path += 'index.html';
    const file = join(DIST, path);
    if (!file.startsWith(DIST)) {
      res.writeHead(403).end();
      return;
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' });
      res.end(body);
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
    }
  }).listen(PORT, () => {
    console.log(`\n  QIHGNN-MSCA web  http://localhost:${PORT}\n  API proxy        /api -> ${API}\n`);
  });
}

start().catch((err) => {
  console.error(err);
  process.exit(1);
});

// End-to-end smoke test in a real Chromium: `npm run build && npm run test:e2e`.
// Starts the preview server on a free port, walks through every interactive section,
// and fails on any uncaught page error or horizontal overflow.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PORT = 5199;
const URL = `http://localhost:${PORT}/`;

const server = spawn(process.execPath, ['scripts/dev.mjs', '--no-watch'], {
  cwd: ROOT,
  env: { ...process.env, PORT: String(PORT), API_URL: 'http://127.0.0.1:9' },
  stdio: ['ignore', 'pipe', 'inherit'],
});
await Promise.race([
  once(server.stdout, 'data'),
  new Promise((_, reject) => setTimeout(() => reject(new Error('preview server did not start')), 20000)),
]);

const results = [];
async function step(name, fn) {
  try {
    await fn();
    results.push(['ok', name]);
  } catch (err) {
    results.push(['FAIL', name, err.message.split('\n')[0]]);
  }
}

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(URL, { waitUntil: 'networkidle' });

  await step('hero renders headline and figures', async () => {
    assert.match(await page.locator('h1').innerText(), /SILENCE\s+IS NOT\s+BACKGROUND\./i);
    assert.ok(await page.getByText('8,347,071').first().isVisible());
  });

  await step('header reports static preview when the API is down', async () => {
    await page.getByText('Static preview').waitFor({ timeout: 5000 });
  });

  await step('theme toggle switches and persists', async () => {
    const before = await page.evaluate(() => document.documentElement.dataset.theme);
    await page.getByRole('button', { name: /Switch to (dark|light) theme/ }).click();
    const after = await page.evaluate(() => document.documentElement.dataset.theme);
    assert.notEqual(before, after);
    assert.equal(await page.evaluate(() => localStorage.getItem('qihgnn-theme')), after);
  });

  await step('hero case switcher cycles cases', async () => {
    const id = () => page.locator('figure[aria-label="Case viewer"] .font-mono').first().innerText();
    const a = await id();
    await page.getByRole('button', { name: 'Next case' }).click();
    assert.notEqual(await id(), a);
  });

  await step('architecture stage selection updates the detail panel', async () => {
    await page.locator('#architecture').getByRole('button', { name: /Multi-scale cross-attention/ }).first().click();
    await page.locator('#architecture h3', { hasText: /Multi-scale cross-attention/i }).waitFor();
    assert.ok(await page.locator('#architecture .katex').first().isVisible());
  });

  await step('workbench shows silent organs for the abdomen case', async () => {
    const wb = page.locator('#workbench');
    await wb.getByRole('radio', { name: /PH-ABDOMEN-01/ }).click();
    await wb.getByText('no corpus supervises it here; zeroed by the server').first().waitFor();
    await wb.getByRole('radio', { name: 'Sagittal' }).click();
    assert.match(await wb.getByText(/sagittal \d+ \/ \d+/i).first().innerText(), /sagittal/i);
  });

  await step('run network explains how to start the API', async () => {
    await page.getByRole('button', { name: 'Run network on this case' }).click();
    await page.locator('#workbench [role=status]').waitFor({ timeout: 5000 });
  });

  await step('lab: naive loss rises while marginal loss stays fixed', async () => {
    const lab = page.locator('#verification');
    const slider = lab.getByRole('slider', { name: 'Moved' });
    const read = async () => Number((await lab.locator('p.display').nth(1).innerText()).trim());
    const readM = async () => Number((await lab.locator('p.display').nth(0).innerText()).trim());
    await slider.fill('10');
    const n1 = await read();
    const m1 = await readM();
    await slider.fill('80');
    assert.ok((await read()) > n1, 'naive loss should increase');
    assert.equal(await readM(), m1);
  });

  await step('literature filters and search', async () => {
    const lit = page.locator('#literature');
    await lit.getByRole('radio', { name: /Partial supervision/ }).click();
    assert.equal(await lit.locator('ul > li').count(), 10);
    await lit.getByRole('radio', { name: /^All/ }).click();
    await lit.getByPlaceholder('Search titles, authors, gaps').fill('hypergraph');
    assert.ok((await lit.locator('ul > li').count()) >= 3);
  });

  await step('model card opens and closes with Escape', async () => {
    await page.getByRole('button', { name: 'Model card' }).first().click();
    await page.getByRole('dialog').waitFor();
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('dialog').count(), 0);
  });

  await step('no uncaught page errors', async () => {
    assert.deepEqual(errors, []);
  });

  const mobile = await browser.newPage({ viewport: { width: 360, height: 780 }, isMobile: true, hasTouch: true });
  await mobile.goto(URL, { waitUntil: 'networkidle' });
  await step('no horizontal scroll at 360 px', async () => {
    assert.ok((await mobile.evaluate(() => document.documentElement.scrollWidth)) <= 360);
  });
  await step('mobile contents menu opens', async () => {
    await mobile.getByRole('button', { name: 'Open contents' }).click();
    assert.ok(await mobile.locator('#mobile-nav').isVisible());
  });
} finally {
  await browser.close();
  server.kill();
}

for (const [status, name, msg] of results) console.log(`${status === 'ok' ? '  ok  ' : '  FAIL'} ${name}${msg ? `: ${msg}` : ''}`);
const failed = results.filter((r) => r[0] !== 'ok').length;
console.log(`${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

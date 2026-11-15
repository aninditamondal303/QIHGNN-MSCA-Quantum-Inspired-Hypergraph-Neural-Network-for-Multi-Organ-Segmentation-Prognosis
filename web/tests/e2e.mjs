// End-to-end smoke test in a real Chromium: `npm run build && npm run test:e2e`.
// Starts the preview server on a free port, walks through every page and interactive element,
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

  await step('intro: opens the site, organ buttons switch the subject, skip enters', async () => {
    const intro = page.getByRole('dialog', { name: 'Introduction' });
    await intro.waitFor();
    await page.getByRole('button', { name: /02 Liver CT/ }).click();
    await page.getByText('the liver.').waitFor();
    await page.getByRole('button', { name: /Skip intro/ }).click();
    await page.locator('h1', { hasText: /See it/i }).waitFor();
  });

  await step('intro: plays whenever the site is opened, not when moving between pages; deep links skip it', async () => {
    await page.getByRole('link', { name: 'About', exact: true }).first().click();
    await page.getByRole('link', { name: 'Home', exact: true }).first().click();
    await page.locator('h1', { hasText: /See it/i }).waitFor();
    assert.equal(await page.getByRole('dialog', { name: 'Introduction' }).count(), 0);
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.getByRole('dialog', { name: 'Introduction' }).waitFor();
    await page.keyboard.press('Escape');
    await page.locator('h1', { hasText: /See it/i }).waitFor();
    const fresh = await browser.newPage();
    await fresh.goto(URL + 'about', { waitUntil: 'networkidle' });
    assert.equal(await fresh.getByRole('dialog', { name: 'Introduction' }).count(), 0);
    await fresh.close();
  });

  await step('intro: header button and logo bring the scanner back from any page', async () => {
    await page.goto(URL + 'about', { waitUntil: 'networkidle' });
    await page.getByRole('button', { name: 'Back to the interactive intro' }).click();
    await page.getByRole('dialog', { name: 'Introduction' }).waitFor();
    assert.ok(!page.url().includes('/about'), page.url());
    await page.keyboard.press('Escape');
    await page.locator('h1', { hasText: /See it/i }).waitFor();
    await page.goto(URL + 'accuracy', { waitUntil: 'networkidle' });
    await page.getByRole('link', { name: /back to the interactive intro/i }).click();
    await page.getByRole('dialog', { name: 'Introduction' }).waitFor();
    await page.getByRole('button', { name: /Skip intro/ }).click();
    await page.locator('h1', { hasText: /See it/i }).waitFor();
  });

  await step('intro: press and hold scans, then enters the site', async () => {
    const p2 = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    await p2.goto(URL, { waitUntil: 'networkidle' });
    await p2.mouse.move(900, 400);
    await p2.mouse.down();
    await p2.waitForTimeout(2600);
    await p2.mouse.up();
    await p2.locator('h1', { hasText: /See it/i }).waitFor({ timeout: 5000 });
    await p2.close();
  });

  await step('home renders headline, calls to action and ticker', async () => {
    assert.match(await page.locator('h1').innerText(), /SEE IT\.\s+MEASURE IT\.\s+EXPLAIN IT\./i);
    assert.ok(await page.getByRole('link', { name: /Analyse a scan/ }).first().isVisible());
    assert.ok(await page.locator('.ticker').isVisible());
  });

  await step('header reports static preview when the API is down', async () => {
    await page.getByText('Static preview').first().waitFor({ timeout: 5000 });
  });

  await step('theme toggle switches and persists', async () => {
    const before = await page.evaluate(() => document.documentElement.dataset.theme);
    await page.getByRole('button', { name: /Switch to (dark|light) theme/ }).click();
    const after = await page.evaluate(() => document.documentElement.dataset.theme);
    assert.notEqual(before, after);
    assert.equal(await page.evaluate(() => localStorage.getItem('qihgnn-theme')), after);
  });

  await step('research viewer cycles cases', async () => {
    const id = () => page.locator('figure[aria-label="Research model viewer"] .font-mono').first().innerText();
    const a = await id();
    await page.getByRole('button', { name: 'Next case' }).click();
    assert.notEqual(await id(), a);
  });

  await step('header links navigate between pages without reloading', async () => {
    await page.evaluate(() => { window.__sameDoc = true; });
    for (const [name, path, heading] of [['How it works', '/how-it-works', /pixels to a prognosis/i], ['Accuracy', '/accuracy', /measured, not claimed/i], ['Data & privacy', '/data', /nothing of yours out/i], ['About', '/about', /Anindita Mondal/i]]) {
      await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name }).click();
      await page.waitForURL(`**${path}`);
      assert.match(await page.locator('h1').innerText(), heading);
    }
    assert.equal(await page.evaluate(() => window.__sameDoc), true, 'navigation reloaded the page');
  });

  await step('deep links and the 404 page work', async () => {
    await page.goto(URL + 'accuracy', { waitUntil: 'networkidle' });
    assert.match(await page.locator('h1').innerText(), /measured, not claimed/i);
    await page.goto(URL + 'no-such-page', { waitUntil: 'networkidle' });
    assert.match(await page.locator('h1').innerText(), /nothing on this slice/i);
    await page.goBack();
    assert.match(await page.locator('h1').innerText(), /measured, not claimed/i);
  });

  await step('how it works: architecture stages and workbench', async () => {
    await page.goto(URL + 'how-it-works', { waitUntil: 'networkidle' });
    await page.locator('#architecture').getByRole('button', { name: /Multi-scale cross-attention/ }).first().click();
    await page.locator('#architecture h3', { hasText: /Multi-scale cross-attention/i }).waitFor();
    const wb = page.locator('#workbench');
    await wb.getByRole('radio', { name: /PH-ABDOMEN-01/ }).click();
    await wb.getByText('no corpus supervises it here; zeroed by the server').first().waitFor();
    await page.getByRole('button', { name: 'Run network on this case' }).click();
    await page.locator('#workbench [role=status]').waitFor({ timeout: 5000 });
  });

  await step('about: questions expand', async () => {
    await page.goto(URL + 'about', { waitUntil: 'networkidle' });
    const q = page.locator('details').first();
    await q.locator('summary').click();
    assert.equal(await q.evaluate((d) => d.open), true);
  });

  await step('analyse: upload area, and a held-out sample is analysed in the browser when models are published', async () => {
    await page.goto(URL + 'analyse', { waitUntil: 'networkidle' });
    await page.getByText('Drop a scan here').waitFor();
    const published = await page.evaluate(async () => (await fetch('screen/classifier.onnx', { method: 'HEAD' })).ok);
    if (!published) return;
    await page.getByRole('button', { name: /Glioma|No tumour|Meningioma|Pituitary/ }).first().click();
    await page.getByText(/Tumour detected|No tumour found|Not confident enough|does not look like a brain MRI/i).first().waitFor({ timeout: 60000 });
    assert.ok(await page.getByText('Calibrated probabilities').isVisible());
  });

  await step('cervical cells: a held-out slide is measured in the browser when the model is published', async () => {
    await page.goto(URL + 'analyse?type=cervix', { waitUntil: 'networkidle' });
    const published = await page.evaluate(async () => (await fetch('screen/cervix.onnx', { method: 'HEAD' })).ok);
    if (!published) return;
    await page.locator('section', { hasText: 'held-out test image' }).getByRole('button').first().click();
    const h = page.getByText(/\d+ cells measured/).first();
    await h.waitFor({ timeout: 90000 });
    const n = Number((await h.textContent()).match(/\d+/)[0]);
    assert.ok(n >= 20 && n <= 200, `cells measured ${n}`);
  });

  await step('liver CT: a held-out CT scan is measured in the browser, close to the expert', async () => {
    await page.goto(URL + 'analyse?type=liver', { waitUntil: 'networkidle' });
    const published = await page.evaluate(async () => (await fetch('screen/liver_tumour.onnx', { method: 'HEAD' })).ok);
    if (!published) return;
    const idx = await page.evaluate(async () => (await fetch('screen/samples/organs_index.json')).json());
    const vol = idx.find((s) => s.kind === 'liver-volume');
    await page.locator('section', { hasText: 'held-out test image' }).getByRole('button', { name: /3D CT/ }).click();
    await page.getByText(/possible lesions? outlined|No liver lesion outlined/).first().waitFor({ timeout: 240000 });
    const txt = await page.locator('.print-report').innerText();
    const liverCc = Number(txt.match(/Liver volume\s+([\d.]+)/i)[1]);
    assert.ok(Math.abs(liverCc - vol.true_liver_cc) / vol.true_liver_cc < 0.1, `liver ${liverCc} vs expert ${vol.true_liver_cc}`);
  });

  await step('no uncaught page errors', async () => {
    assert.deepEqual(errors, []);
  });

  const mobile = await browser.newPage({ viewport: { width: 360, height: 780 }, isMobile: true, hasTouch: true });
  for (const path of ['', '?intro', 'analyse', 'how-it-works', 'accuracy', 'data', 'about']) {
    await mobile.goto(URL + path, { waitUntil: 'networkidle' });
    await step(`no horizontal scroll at 360 px: /${path}`, async () => {
      assert.ok((await mobile.evaluate(() => document.documentElement.scrollWidth)) <= 360);
    });
  }
  await step('mobile menu offers the interactive intro', async () => {
    await mobile.getByRole('button', { name: 'Open menu' }).click();
    assert.ok(await mobile.locator('#mobile-nav').getByRole('button', { name: 'Interactive intro' }).isVisible());
    await mobile.getByRole('button', { name: 'Close menu' }).click();
  });

  await step('mobile menu opens and navigates', async () => {
    await mobile.getByRole('button', { name: 'Open menu' }).click();
    await mobile.locator('#mobile-nav').getByRole('link', { name: 'Accuracy' }).click();
    await mobile.waitForURL('**/accuracy');
    assert.equal(await mobile.locator('#mobile-nav').count(), 0);
  });
} finally {
  await browser.close();
  server.kill();
}

for (const [status, name, msg] of results) console.log(`${status === 'ok' ? '  ok  ' : '  FAIL'} ${name}${msg ? `: ${msg}` : ''}`);
const failed = results.filter((r) => r[0] !== 'ok').length;
console.log(`${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

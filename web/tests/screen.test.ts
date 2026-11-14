// Parity between the browser preprocessing/statistics (src/lib/screen.ts) and the Python training code.
// The fixtures were produced by ml/qihgnn/screen (see the generator in docs/DECISIONS.md, "screening parity").
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { letterboxParams, predictSurvival, segNormalise, survivalDesign, toUint8, tumourProbability, verdict, type SurvivalModel } from '../src/lib/screen';

const F = JSON.parse(readFileSync(new URL('./screen.fixtures.json', import.meta.url), 'utf8'));

test('segmenter normalisation matches numpy (percentiles of non-zero pixels)', () => {
  const got = segNormalise(Uint8Array.from(F.u8));
  F.seg.forEach((v: number, i: number) => assert.ok(Math.abs(got[i] - v) < 1e-5, `pixel ${i}: ${got[i]} vs ${v}`));
});

test('MRI slice to 0..255 matches numpy, including negative and zero voxels', () => {
  assert.deepEqual(Array.from(toUint8(Float32Array.from(F.f))), F.to8);
});

test('letterbox geometry matches Pillow-side letterbox', () => {
  assert.deepEqual(letterboxParams(200, 300, 192), { scale: 0.64, top: 0, left: 32, h: 192, w: 128 });
  assert.deepEqual(letterboxParams(240, 240, 192), { scale: 0.8, top: 0, left: 0, h: 192, w: 192 });
});

test('survival curve equals the Python Cox model', () => {
  const c = F.cox;
  const model: SurvivalModel = {
    features: [], n: 80, events: 60,
    model: { mu: c.mu, sd: c.sd, beta: c.beta, grid_days: c.grid, H0: c.H0, hazard_ratio_per_sd: [], boot_beta: [c.beta], boot_H0: [c.H0] },
  };
  const x = survivalDesign(67, 85, 'STR');
  x.forEach((v, i) => assert.ok(Math.abs(v - c.x[i]) < 1e-9));
  const p = predictSurvival(model, x);
  p.survival.forEach((s, i) => assert.ok(Math.abs(s - c.S[i]) < 1e-9, `t=${c.grid[i]}`));
  assert.equal(p.medianDays, c.median);
});

test('verdict abstains below threshold and rejects non-brain images', () => {
  assert.equal(verdict([0.1, 0.1, 0.1, 0.1, 0.6], 0.6).kind, 'not-brain');
  assert.equal(verdict([0.3, 0.25, 0.2, 0.15, 0.1], 0.6).kind, 'uncertain');
  assert.equal(verdict([0.9, 0.04, 0.03, 0.02, 0.01], 0.6).kind, 'no-tumour');
  const v = verdict([0.02, 0.9, 0.05, 0.02, 0.01], 0.6);
  assert.equal(v.kind, 'tumour');
  assert.ok(Math.abs(tumourProbability([0.02, 0.9, 0.05, 0.02, 0.01]) - 0.97 / 0.99) < 1e-12);
});

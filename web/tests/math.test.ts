// Cross-language check: the TypeScript ports must reproduce the Python reference (fixtures written by
// `python -m qihgnn.verify`) to 1e-9.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { attentionCost, marginalLoss, mergeUnsupervised, mpoRow, naiveLoss, reallocate, shiftToChannel } from '../src/lib/math.ts';
import { sci } from '../src/lib/format.ts';

const fx = JSON.parse(readFileSync(new URL('../src/lib/fixtures.json', import.meta.url), 'utf8'));

test('marginal and naive losses match Python', () => {
  const { probs, labels, supervised, marginal_loss, naive_loss } = fx.marginal;
  assert.ok(Math.abs(marginalLoss(probs, labels, supervised) - marginal_loss) < 1e-9);
  assert.ok(Math.abs(naiveLoss(probs, labels) - naive_loss) < 1e-9);
});

test('marginal loss is invariant under reallocation of unsupervised mass; naive loss is not', () => {
  const { probs, labels, supervised } = fx.marginal;
  const base = marginalLoss(probs, labels, supervised);
  const naive = naiveLoss(probs, labels);
  for (const target of [1, 3, 4, 5]) {
    for (const t of [0.25, 0.5, 1]) {
      const q = reallocate(probs, supervised, target, t);
      assert.ok(Math.abs(marginalLoss(q, labels, supervised) - base) < 1e-12, `target ${target} t ${t}`);
      for (let i = 0; i < labels.length; i++) {
        const col = q.reduce((a: number, row: number[]) => a + row[i], 0);
        assert.ok(Math.abs(col - 1) < 1e-12);
      }
    }
    assert.ok(Math.abs(naiveLoss(reallocate(probs, supervised, target, 1), labels) - naive) > 1e-3);
  }
});

test('merge preserves total mass and supervised channels', () => {
  const { probs, supervised } = fx.marginal;
  const m = mergeUnsupervised(probs, supervised);
  for (let i = 0; i < probs[0].length; i++) {
    const a = probs.reduce((s: number, r: number[]) => s + r[i], 0);
    const b = m.reduce((s: number, r: number[]) => s + r[i], 0);
    assert.ok(Math.abs(a - b) < 1e-12);
    assert.equal(m[2][i], probs[2][i]);
  }
});

test('attention cost matches Python at every fixture resolution', () => {
  for (const row of fx.attention) {
    const c = attentionCost(row.shape);
    assert.equal(c.multi_scale, row.multi_scale);
    assert.equal(c.voxels, row.voxels);
    assert.ok(Math.abs(c.naive - row.naive) / row.naive < 1e-12);
  }
  assert.equal(attentionCost([128, 128, 64]).multi_scale, 66577);
});

test('MPO compression table matches Python', () => {
  for (const row of fx.mpo) {
    const r = mpoRow(row.bond);
    assert.equal(r.parameters, row.parameters);
    assert.ok(Math.abs(r.compression - row.compression) < 1e-12);
  }
});

test('scientific formatting', () => {
  assert.equal(sci(1099511627776), '1.100 × 10¹²');
  assert.equal(sci(66577), '66,577');
});

test('predicting an unlabelled organ is free under the marginal loss and penalised by the naive loss', () => {
  const { probs, labels, supervised } = fx.marginal;
  const m0 = marginalLoss(probs, labels, supervised);
  let prev = naiveLoss(probs, labels);
  for (const t of [0.2, 0.4, 0.6, 0.8]) {
    const q = shiftToChannel(probs, labels, 3, t);
    assert.ok(Math.abs(marginalLoss(q, labels, supervised) - m0) < 1e-12);
    const n = naiveLoss(q, labels);
    assert.ok(n > prev);
    prev = n;
  }
});

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { combine, fullBox, LIVER, liverBox, stage2Input, TUM, uncrop } from '../src/lib/organs';
import type { Letterbox } from '../src/lib/screen';

// Produced by ml/qihgnn/screen/liver_cascade.py (liver_box, stage2_input, uncrop, combine) on the same synthetic inputs.
const fixtures = JSON.parse(readFileSync(new URL('./liver.fixtures.json', import.meta.url), 'utf8')) as {
  cy: number; cx: number; ry: number; rx: number; k: number; H: number; W: number; lb: { scale: number; top: number; left: number } | null;
  box: [number, number, number]; fbox: [number, number, number];
  in_sum: number; in_at: number[]; un_sum: number; un_at: number[]; counts: number[]; tum_first: number; tum_last: number;
}[];

const IDX = [0, 777, 5000, 12345, 20000, 36863];

test('liver stage 2: crop box, resize, uncrop and clean-up match Python', () => {
  for (const f of fixtures) {
    const N = LIVER;
    const s1 = new Uint8Array(N * N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        if (((y - f.cy) / f.ry) ** 2 + ((x - f.cx) / f.rx) ** 2 <= 1) s1[i] = 1;
        if ((y - f.cy) ** 2 + (x - f.cx) ** 2 < 36) s1[i] = 2;
      }
    }
    const full = { data: new Uint8Array(f.H * f.W), w: f.W, h: f.H };
    for (let y = 0; y < f.H; y++) for (let x = 0; x < f.W; x++) full.data[y * f.W + x] = (x * 7 + y * 13 + f.k * 5) % 256;
    const lb = f.lb ? ({ ...f.lb, h: 0, w: 0 } as Letterbox) : undefined;
    const box = liverBox(s1)!;
    assert.deepEqual([box.y0, box.x0, box.s], f.box);
    const fb = fullBox(box, lb);
    assert.deepEqual([fb.y0, fb.x0, fb.s], f.fbox);
    const x = stage2Input(full, box, lb);
    assert.ok(Math.abs(x.reduce((a, b) => a + b, 0) - f.in_sum) < 0.05, 'input sum');
    IDX.forEach((i, j) => assert.ok(Math.abs(x[i] - f.in_at[j]) < 1e-5, `input at ${i}`));
    const prob = new Float32Array(TUM * TUM);
    for (let y = 0; y < TUM; y++) for (let xx = 0; xx < TUM; xx++) prob[y * TUM + xx] = 0.5 + 0.5 * Math.sin(y / 9 + f.k) * Math.cos(xx / 7);
    const u = uncrop(prob, box);
    assert.ok(Math.abs(u.reduce((a, b) => a + b, 0) - f.un_sum) < 0.05, 'uncrop sum');
    [0, 12345, 32896, 40000, 65535, f.cy * N + f.cx].forEach((i, j) => assert.ok(Math.abs(u[i] - f.un_at[j]) < 1e-5, `uncrop at ${i}`));
    const c = combine(s1, u, 0.6);
    assert.deepEqual([0, 1, 2].map((v) => c.reduce((a, b) => a + (b === v ? 1 : 0), 0)), f.counts);
    assert.equal(c.indexOf(2), f.tum_first);
    assert.equal(c.lastIndexOf(2), f.tum_last);
  }
});

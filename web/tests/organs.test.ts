// The browser's cell separation must give exactly the cells the Python evaluation measured.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { cellsFromClasses, lesions3d } from '../src/lib/organs';

const F = JSON.parse(readFileSync(new URL('./organs.fixtures.json', import.meta.url), 'utf8'));

test('cell separation matches qihgnn.screen.organs.cells_from_classes', () => {
  const { owner, cells } = cellsFromClasses(Uint8Array.from(F.cls), F.w, F.h);
  assert.deepEqual(Array.from(owner), F.owner);
  assert.equal(cells.length, F.cells.length);
  cells.forEach((c, i) => {
    const p = F.cells[i];
    assert.equal(c.id, p.id);
    assert.equal(c.nucleusPx, p.nucleus_px);
    assert.equal(c.cytoplasmPx, p.cytoplasm_px);
    assert.ok(Math.abs(c.nc - p.nc) < 1e-12 && Math.abs(c.cx - p.cx) < 1e-9 && Math.abs(c.cy - p.cy) < 1e-9);
  });
});

test('3D lesion count joins slices and drops specks', () => {
  const w = 8;
  const h = 8;
  const s = () => new Uint8Array(w * h);
  const a = s();
  const b = s();
  const c = s();
  for (let y = 1; y < 5; y++) for (let x = 1; x < 5; x++) a[y * w + x] = b[y * w + x] = 2; // 32 voxels over two slices
  c[7 * w + 7] = 2; // a 1-voxel speck
  assert.deepEqual(lesions3d([a, b, c], w, h), [32]);
  assert.deepEqual(lesions3d([a, b, c], w, h, 1), [32, 1]);
});

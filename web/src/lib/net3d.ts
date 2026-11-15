/* The trained 3D QIHGNN-MSCA network, run in the browser (ONNX Runtime in a Web Worker), on a published case.
   Input: the case's four channels from their 8-bit PNG mosaics, mapped back to training values with the stored
   ranges, laid out (1, 4, H, W, D). Output: label map with unsupervised channels already suppressed, plus prognosis. */
import { session, yieldToBrowser } from './screen';
import type { CaseInfo, PredictResponse } from './types';
import { volumeFor } from './useCase';
import type { Volume } from './volume';

export async function runNet3d(info: CaseInfo, onStage?: (s: string) => void): Promise<{ volume: Volume; response: PredictResponse }> {
  const t0 = performance.now();
  if (!info.channelRanges) throw new Error('This case has no stored intensity ranges, so the network cannot run on it here.');
  onStage?.('Loading the four image channels');
  const chans = await Promise.all(info.channels.map((c) => volumeFor(c.file, info)));
  const [H, W, D] = info.shape;
  const C = chans.length;
  const x = new Float32Array(C * H * W * D);
  for (let c = 0; c < C; c++) {
    await yieldToBrowser();
    const v = chans[c];
    const [lo, hi] = info.channelRanges![c];
    const k255 = (hi - lo) / 255;
    // Volume data is (k * h + y) * w + x; the network wants ((c * H + y) * W + x) * D + k.
    for (let k = 0; k < D; k++)
      for (let y = 0; y < H; y++) {
        const src = (k * H + y) * W;
        const dst = (c * H + y) * W;
        for (let xx = 0; xx < W; xx++) x[(dst + xx) * D + k] = lo + v.data[src + xx] * k255;
      }
  }
  const cov = new Float32Array(info.covariates ?? [0, 0, 0, 0, 0]);
  onStage?.('Running the network (first run downloads it)');
  const s = await session('net3d');
  const out = await s.run({ image: { data: x, dims: [1, C, H, W, D] }, covariates: { data: cov, dims: [1, cov.length] } });
  await yieldToBrowser();
  const lab = out.label.data as unknown as Uint8Array;
  const vol = new Uint8Array(H * W * D);
  for (let k = 0; k < D; k++)
    for (let y = 0; y < H; y++)
      for (let xx = 0; xx < W; xx++) vol[(k * H + y) * W + xx] = lab[(y * W + xx) * D + k];
  const hor = out.horizon.data;
  return {
    volume: { data: vol, h: H, w: W, d: D },
    response: {
      case_id: info.id,
      run_id: info.model?.run_id ?? null,
      shape: [H, W, D],
      mosaic_columns: info.mosaicColumns,
      label_png_base64: '',
      suppressed: info.model?.suppressed ?? [],
      log_risk: out.log_risk.data[0],
      survival_probability: { '1y': hor[0], '3y': hor[1], '5y': hor[2] },
      aleatoric_sd: Math.exp(0.5 * out.log_variance.data[0]),
      seconds: Math.round((performance.now() - t0) / 100) / 10,
    },
  };
}

/// <reference lib="webworker" />
/* Runs the ONNX models off the page's main thread, so long 3D jobs never freeze the page. The page sends
   { id, base, model, feeds } and gets { id, ok, out } back; tensors travel as transferable Float32Arrays. */
import type { InferenceSession } from 'onnxruntime-web';

type Feed = { data: Float32Array; dims: number[] };
type Msg = { id: number; base: string; model: string; feeds: Record<string, Feed> };

const ortPromise = import('onnxruntime-web/wasm');
const sessions = new Map<string, Promise<InferenceSession>>();

self.onmessage = async (e: MessageEvent<Msg>) => {
  const { id, base, model, feeds } = e.data;
  try {
    const ort = await ortPromise;
    ort.env.wasm.wasmPaths = base + 'ort/';
    ort.env.wasm.numThreads = 1;
    let s = sessions.get(model);
    if (!s) {
      s = ort.InferenceSession.create(base + 'screen/' + model + '.onnx', { executionProviders: ['wasm'] });
      sessions.set(model, s);
    }
    const session = await s;
    const inputs: Record<string, InstanceType<typeof ort.Tensor>> = {};
    for (const [k, f] of Object.entries(feeds)) inputs[k] = new ort.Tensor('float32', f.data, f.dims);
    const out = await session.run(inputs);
    const res: Record<string, Feed> = {};
    const transfer: Transferable[] = [];
    for (const [k, t] of Object.entries(out)) {
      const data = t.data as Float32Array;
      res[k] = { data, dims: [...t.dims] };
      transfer.push(data.buffer);
    }
    (self as unknown as Worker).postMessage({ id, ok: true, out: res }, transfer);
  } catch (err) {
    sessions.delete(model);
    (self as unknown as Worker).postMessage({ id, ok: false, error: String(err) });
  }
};

import { useEffect, useState } from 'react';
import type { CaseInfo, Label } from './types';
import { loadVolume, type Volume } from './volume';

const cache = new Map<string, Promise<Volume>>();

export function volumeFor(file: string, info: CaseInfo): Promise<Volume> {
  const key = `data/cases/${file}`;
  let p = cache.get(key);
  if (!p) {
    p = loadVolume(key, info.shape, info.mosaicColumns);
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return p;
}

export interface LoadedCase {
  image: Volume | null;
  labels: Volume | null;
  error: string | null;
}

/** Loads one image channel and the label volume of a case; switching channel reuses the label volume. */
export function useCase(info: CaseInfo | null, channel: number): LoadedCase {
  const [state, setState] = useState<LoadedCase>({ image: null, labels: null, error: null });
  useEffect(() => {
    if (!info) return;
    let alive = true;
    const ch = info.channels[Math.min(channel, info.channels.length - 1)];
    Promise.all([volumeFor(ch.file, info), volumeFor(info.label, info)])
      .then(([image, labels]) => alive && setState({ image, labels, error: null }))
      .catch((e: unknown) => alive && setState({ image: null, labels: null, error: e instanceof Error ? e.message : String(e) }));
    return () => {
      alive = false;
    };
  }, [info, channel]);
  return state;
}

export type Overlay = 'supervised' | 'suppressed' | 'hidden';

export const LABEL_ORDER: Label[] = ['background', 'brain', 'liver', 'kidney', 'cervix', 'spleen'];

/** Supervised channels draw in cobalt; channels present in the anatomy but unsupervised draw as hatched silence. */
export function overlaysFor(info: CaseInfo, supervisedOverride?: Label[]): Overlay[] {
  const sup = new Set(supervisedOverride ?? info.supervised);
  return LABEL_ORDER.map((l, i) => (i === 0 ? 'hidden' : sup.has(l) ? 'supervised' : 'suppressed'));
}

export function defaultChannel(info: CaseInfo): number {
  if (info.modality === 'MRI') return Math.max(0, info.channels.findIndex((c) => c.name === 'FLAIR'));
  return 0;
}

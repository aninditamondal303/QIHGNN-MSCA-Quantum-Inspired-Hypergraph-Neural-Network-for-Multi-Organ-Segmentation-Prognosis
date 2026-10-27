const NAMES: Record<string, string> = { brain: 'brain lesion', liver: 'liver', kidney: 'kidney', cervix: 'cervix', spleen: 'spleen', background: 'background' };

export function labelName(l: string): string {
  return NAMES[l] ?? l;
}

export function list(items: string[]): string {
  const n = items.map(labelName);
  if (n.length <= 1) return n.join('');
  return `${n.slice(0, -1).join(', ')} and ${n[n.length - 1]}`;
}

export function Swatch({ kind }: { kind: 'supervised' | 'silent' }) {
  return kind === 'supervised' ? (
    <span className="inline-block size-3 shrink-0 bg-cobalt" aria-hidden="true" />
  ) : (
    <span className="hatch-fine inline-block size-3 shrink-0 outline outline-1 -outline-offset-1 outline-vermilion" aria-hidden="true" />
  );
}

export function Legend({ supervised, silent }: { supervised: string[]; silent: string[] }) {
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-1.5 text-small">
      <span className="inline-flex items-center gap-2">
        <Swatch kind="supervised" />
        <span>
          Supervised{supervised.length ? `: ${list(supervised)}` : ': none in this case'}
        </span>
      </span>
      {silent.length > 0 && (
        <span className="inline-flex items-center gap-2">
          <Swatch kind="silent" />
          <span>Silent, suppressed at inference: {list(silent)}</span>
        </span>
      )}
    </div>
  );
}

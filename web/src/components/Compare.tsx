import { useEffect, useRef, useState, type DependencyList } from 'react';

/** Before/after view: the AI result on the left of the handle, the untouched scan on the right. Drag anywhere on the
    image (or use the arrow keys on the handle) to move the split. */
export function CompareCanvas({ paint, deps, label, className }: {
  paint: (c: HTMLCanvasElement, overlay: boolean) => void;
  deps: DependencyList;
  label: string;
  className: string;
}) {
  const base = useRef<HTMLCanvasElement>(null);
  const top = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [split, setSplit] = useState(60);
  useEffect(() => {
    if (base.current) paint(base.current, false);
    if (top.current) paint(top.current, true);
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps

  const moveTo = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    setSplit(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };
  return (
    <div
      ref={box}
      className="develop-sweep relative cursor-ew-resize select-none"
      style={{ touchAction: 'pan-y' }}
      onPointerDown={(e) => {
        dragging.current = true;
        (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
        moveTo(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && moveTo(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
    >
      <canvas ref={base} className={className} aria-label={`${label}: original scan`} />
      <canvas ref={top} className={`${className} absolute inset-0 !h-full`} style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} aria-label={label} />
      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${split}%` }} aria-hidden="true">
        <div className="h-full w-px -translate-x-1/2 bg-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.35)]" />
        <span className="absolute left-0 top-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white font-mono text-[0.8rem] text-black shadow-lg">⇆</span>
      </div>
      <span className="pointer-events-none absolute left-2 top-2 bg-black/60 px-1.5 py-0.5 font-mono text-[0.65rem] uppercase tracking-[0.14em] text-white" aria-hidden="true">AI</span>
      <span className="pointer-events-none absolute right-2 top-2 bg-black/60 px-1.5 py-0.5 font-mono text-[0.65rem] uppercase tracking-[0.14em] text-white" aria-hidden="true">Original</span>
      <input
        type="range"
        min={0}
        max={100}
        value={Math.round(split)}
        onChange={(e) => setSplit(Number(e.target.value))}
        className="sr-only"
        aria-label="Compare the AI result with the original scan"
      />
    </div>
  );
}

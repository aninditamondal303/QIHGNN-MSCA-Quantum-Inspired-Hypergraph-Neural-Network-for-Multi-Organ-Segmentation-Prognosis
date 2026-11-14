import { useEffect, useRef, useState, type ReactNode } from 'react';

const reduced = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Fades and lifts its children in when they scroll into view (once). */
export function Reveal({ children, delay = 0, className = '' }: { children: ReactNode; delay?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (reduced()) return setShown(true);
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        setShown(true);
        io.disconnect();
      }
    }, { rootMargin: '0px 0px -12% 0px' });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={ref} className={`reveal ${shown ? 'reveal-in' : ''} ${className}`} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

/** Counts from 0 to `to` once visible. Formats with `format`. */
export function CountUp({ to, format, duration = 1400 }: { to: number; format: (v: number) => string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [v, setV] = useState(reduced() ? to : 0);
  useEffect(() => {
    if (reduced()) return setV(to);
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      const t0 = performance.now();
      const tick = (t: number) => {
        const k = Math.min(1, (t - t0) / duration);
        setV(to * (1 - Math.pow(1 - k, 3)));
        if (k < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    });
    io.observe(el);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [to, duration]);
  return <span ref={ref}>{format(v)}</span>;
}

/** A huge, slowly scrolling band of text. Content is duplicated so the loop is seamless. */
export function Ticker({ items }: { items: string[] }) {
  const row = items.map((t, i) => (
    <span key={i} className="flex items-center gap-[0.35em] pr-[0.6em]">
      <span>{t}</span>
      <span className="text-vermilion" aria-hidden="true">/</span>
    </span>
  ));
  return (
    <div className="ticker overflow-hidden border-y border-rule-strong py-5" aria-label={items.join('. ')}>
      <div className="ticker-track display flex w-max whitespace-nowrap text-[clamp(2.4rem,1.4rem+4vw,5.5rem)] leading-none" aria-hidden="true">
        {row}
        {row}
      </div>
    </div>
  );
}

/** Laser sweep, HUD corners and a cursor reticle laid over a scan panel. Purely decorative. */
export function ScanHud({ children, readout }: { children: ReactNode; readout: [string, string][] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  return (
    <div
      ref={ref}
      className="scan-hud relative overflow-hidden"
      onPointerMove={(e) => {
        const r = ref.current!.getBoundingClientRect();
        setPos({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
      }}
      onPointerLeave={() => setPos(null)}
    >
      {children}
      <div className="scan-beam pointer-events-none absolute inset-x-0 top-0 h-[22%]" aria-hidden="true" />
      <div className="pointer-events-none absolute right-3 top-3 space-y-0.5 text-right font-mono text-[0.68rem] leading-tight text-[#9fb3ff]" aria-hidden="true">
        {readout.map(([k, v]) => (
          <p key={k} className="hud-line"><span className="text-film-muted">{k}</span> {v}</p>
        ))}
      </div>
      {pos && (
        <div className="pointer-events-none absolute inset-0" aria-hidden="true">
          <div className="absolute inset-y-0 w-px bg-[#5474ff]/60" style={{ left: `${pos.x * 100}%` }} />
          <div className="absolute inset-x-0 h-px bg-[#5474ff]/60" style={{ top: `${pos.y * 100}%` }} />
          <div className="absolute size-8 -translate-x-1/2 -translate-y-1/2 border border-[#9fb3ff]" style={{ left: `${pos.x * 100}%`, top: `${pos.y * 100}%` }} />
          <span className="absolute translate-x-3 translate-y-3 bg-film/80 px-1 font-mono text-[0.65rem] text-[#9fb3ff]" style={{ left: `${pos.x * 100}%`, top: `${pos.y * 100}%` }}>
            {(pos.x * 128).toFixed(0)}, {(pos.y * 128).toFixed(0)}
          </span>
        </div>
      )}
    </div>
  );
}

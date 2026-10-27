import { useEffect, useState } from 'react';

// KaTeX is ~260 kB, so it loads as a separate chunk the first time an equation is shown.
let katexModule: Promise<typeof import('katex')> | null = null;
const loadKatex = () => (katexModule ??= import('katex'));

export function Equation({ tex, display = true, label }: { tex: string; display?: boolean; label?: string }) {
  const [html, setHtml] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    loadKatex()
      .then((k) => {
        if (alive) setHtml(k.default.renderToString(tex, { displayMode: display, throwOnError: false, output: 'htmlAndMathml' }));
      })
      .catch(() => alive && setHtml(null));
    return () => {
      alive = false;
    };
  }, [tex, display]);
  return (
    <div className="max-w-full overflow-x-auto overflow-y-hidden py-1 [&_.katex-display]:my-0 [&_.katex-display]:text-left [&_.katex]:text-[1.08em]" aria-label={label}>
      {html ? <div dangerouslySetInnerHTML={{ __html: html }} /> : <code className="font-mono text-small text-muted">{tex}</code>}
    </div>
  );
}

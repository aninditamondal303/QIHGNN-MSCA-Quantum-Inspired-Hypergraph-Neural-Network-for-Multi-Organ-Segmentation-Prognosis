const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };

export function int(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/** 1099511627776 -> "1.100 × 10¹²" */
export function sci(n: number, digits = 3): string {
  if (n === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(n)));
  if (exp >= -2 && exp < 6) return n.toLocaleString('en-US', { maximumFractionDigits: digits });
  const mant = n / 10 ** exp;
  const sup = String(exp).split('').map((ch) => SUP[ch] ?? ch).join('');
  return `${mant.toFixed(digits)} × 10${sup}`;
}

export function fixed(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return n.toFixed(digits);
}

export function pct(n: number, digits = 1): string {
  return `${(n * 100).toFixed(digits)}%`;
}

export function dateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC';
}

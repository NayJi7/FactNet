/** Handle display: drop ".bsky.social", keep custom domains (reuters.com is the info). */

const DEFAULT_SUFFIX = ".bsky.social";

export function shortHandle(handle: string): string {
  if (!handle) return "";
  return handle.endsWith(DEFAULT_SUFFIX)
    ? handle.slice(0, -DEFAULT_SUFFIX.length)
    : handle;
}

/** bluesky glyph as an svg string (also used in the canvas tooltip). currentColor. */
export const personGlyph = (size = 11, colour = "currentColor") =>
  `<svg width="${size}" height="${size}" viewBox="0 0 16 16" fill="none" ` +
  `stroke="${colour}" stroke-width="1.4" stroke-linecap="round" ` +
  `style="vertical-align:-1px;display:inline-block">` +
  `<circle cx="8" cy="5.2" r="2.8"/><path d="M2.8 14c0-2.9 2.3-4.6 5.2-4.6s5.2 1.7 5.2 4.6"/>` +
  `</svg>`;

export function Account({ handle, className = "" }: { handle: string; className?: string }) {
  const short = shortHandle(handle);
  return (
    <span className={`inline-flex items-baseline gap-1.5 ${className}`} title={handle}>
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none"
           stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"
           className="shrink-0 translate-y-[1px] text-ink-faint" aria-hidden>
        <circle cx="8" cy="5.2" r="2.8" />
        <path d="M2.8 14c0-2.9 2.3-4.6 5.2-4.6s5.2 1.7 5.2 4.6" />
      </svg>
      <span className="font-mono">{short}</span>
    </span>
  );
}

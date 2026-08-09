import { useEffect, useState } from "react";

/**
 * The two institutions behind the work.
 *
 * The marks are optional files: if one is missing the name is set in type
 * instead, so the interface never shows a broken image during a demonstration.
 * Drop `um.png` and `cytech.png` (or .svg) into web/public/logos/ to use them.
 */
const MARKS = [
  { key: "um", name: "Universiti Malaya", detail: "FSKTM" },
  { key: "ct", name: "CY Tech", detail: "CY Cergy Paris Université" },
];

const EXTENSIONS = ["svg", "png", "webp", "jpg"];

function Mark({ file, name, detail }: { file: string; name: string; detail: string }) {
  const [src, setSrc] = useState<string | null>(null);

  // The image is probed by decoding it, not by asking the server: a dev server
  // answers 200 to any path, so a HEAD request would report every mark present
  // and the footer would fill with broken images during a demonstration.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (const ext of EXTENSIONS) {
        const url = `/logos/${file}.${ext}`;
        const ok = await new Promise<boolean>((resolve) => {
          const probe = new Image();
          probe.onload = () => resolve(probe.naturalWidth > 1);
          probe.onerror = () => resolve(false);
          probe.src = url;
        });
        if (cancelled) return;
        if (ok) { setSrc(url); return; }
      }
    })();
    return () => { cancelled = true; };
  }, [file]);

  return (
    <div className="flex items-center gap-3">
      {src ? (
        <img src={src} alt={name} className="h-9 w-auto max-w-[132px] object-contain opacity-85" />
      ) : (
        <span className="grid h-9 w-9 shrink-0 place-items-center border border-rule text-[12px] font-semibold text-ink-faint">
          {name.split(" ").map((w) => w[0]).join("").slice(0, 2)}
        </span>
      )}
      <span className="leading-tight">
        <span className="block text-[12.5px] font-medium">{name}</span>
        <span className="block text-[11.5px] text-ink-faint">{detail}</span>
      </span>
    </div>
  );
}

export default function Institutions() {
  return (
    <div className="flex flex-wrap items-center gap-x-9 gap-y-4">
      {MARKS.map((m) => (
        <Mark key={m.key} file={m.key} name={m.name} detail={m.detail} />
      ))}
    </div>
  );
}

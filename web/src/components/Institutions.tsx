import { useEffect, useState } from "react";

/** UM + CY Tech logos in the footer (web/public/logos/), falls back to the name if missing. */
const MARKS = [
  { key: "um", name: "Universiti Malaya", detail: "FSKTM",
    url: "https://fsktm.um.edu.my/" },
  { key: "ct", name: "CY Tech", detail: "CY Cergy Paris Université",
    url: "https://cytech.cyu.fr/" },
];

const EXTENSIONS = ["svg", "png", "webp", "jpg"];

function Mark({ file, name, detail, url }:
              { file: string; name: string; detail: string; url: string }) {
  const [src, setSrc] = useState<string | null>(null);

  // check by actually loading the image, vite dev returns 200 for any path
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
    <a href={url} target="_blank" rel="noreferrer"
       title={`${name}, ${detail}`}
       className="flex items-center gap-3 transition-opacity hover:opacity-70">
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
    </a>
  );
}

export default function Institutions() {
  return (
    <div className="flex flex-wrap items-center gap-x-9 gap-y-4">
      {MARKS.map((m) => (
        <Mark key={m.key} file={m.key} name={m.name} detail={m.detail} url={m.url} />
      ))}
    </div>
  );
}

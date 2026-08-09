import type { ModelCard, Sample, SampleDetail, Trace } from "./types";

async function call<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    // the engine sends a sentence a person can act on; keep it rather than
    // replacing it with a status code
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `request failed (${response.status})`);
  }
  return response.json();
}

export const getModels = () =>
  call<{ content: ModelCard[]; graph: ModelCard[] }>("/models");

export const getSamples = () => call<{ samples: Sample[] }>("/samples");

export const getSampleDetail = (id: number) =>
  call<SampleDetail>(`/samples/${id}`);

export const getVerdict = (payload: {
  text?: string;
  cascade?: unknown;
  sample_id?: number;
  model?: string;
  origin?: string;
}) => call<Trace>("/verdict", payload);

export const fetchCascade = (url: string) =>
  call<{ cascade: Record<string, any> }>("/fetch", { url });

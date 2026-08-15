import type { ModelCard, Sample, SampleDetail, Step, Trace } from "./types";

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

export interface JobInfo {
  id: string;
  tab: string;
  label: string;
  state: "running" | "done" | "failed";
  stages: number;
  elapsed: number;
}

/** What the engine is reading right now, if anything. */
export const getCurrentJob = () => call<{ job: JobInfo | null }>("/jobs/current");

/**
 * Consume one reading, stage by stage, off an open response.
 *
 * Frames are `event: name` then `data: json`, separated by a blank line, and a
 * chunk can split one in half: whatever follows the last blank line is held
 * back until the rest of it arrives.
 */
async function consume(
  response: Response,
  onStep: (step: Step) => void,
  onJob?: (job: JobInfo) => void,
): Promise<Trace> {
  if (!response.ok || !response.body) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const steps: Step[] = [];
  let buffer = "";
  let summary: Omit<Trace, "steps"> | null = null;
  let failure: string | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const name = /^event: (.+)$/m.exec(frame)?.[1];
      const body = /^data: (.+)$/m.exec(frame)?.[1];
      if (!name || !body) continue;
      const parsed = JSON.parse(body);
      if (name === "step") { steps.push(parsed); onStep(parsed); }
      else if (name === "job") onJob?.(parsed);
      else if (name === "done") summary = parsed;
      else if (name === "failed") failure = parsed.detail;
    }
  }
  if (failure) throw new Error(failure);
  if (!summary) throw new Error("the reading ended before it produced a verdict");
  return { ...summary, steps } as Trace;
}

/**
 * Ask for a reading and follow it.
 *
 * EventSource cannot POST, so the stream is read off the fetch body directly.
 * The reading is a job on the server: this connection is a viewer of it, and
 * dropping the connection does not stop the work.
 */
export async function streamVerdict(
  payload: Record<string, unknown>,
  onStep: (step: Step) => void,
  onJob?: (job: JobInfo) => void,
): Promise<Trace> {
  return consume(
    await fetch("/api/verdict/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
    onStep, onJob,
  );
}

/** Rejoin a reading already under way, replaying the stages it has produced. */
export async function followJob(
  id: string,
  onStep: (step: Step) => void,
  onJob?: (job: JobInfo) => void,
): Promise<Trace> {
  return consume(await fetch(`/api/jobs/${id}/stream`), onStep, onJob);
}

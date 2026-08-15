export type FigureKind =
  | "tokens" | "bars" | "line" | "graph" | "compare" | "table";

export interface Figure {
  kind: FigureKind;
  title: string;
  data: Record<string, any>;
  caption: string;
}

/** Which half of the project a stage or a table comes from. */
export type Module = "content" | "propagation" | "both";

export interface Step {
  key: string;
  title: string;
  summary: string;
  detail: Record<string, any>;
  figures: Figure[];
  status: "ok" | "skipped" | "warning";
  note: string;
  module: Module | "";
}

export interface Trace {
  input_kind: string;
  steps: Step[];
  verdict: number | null;
  label: string;
  confidence: "low" | "moderate" | "high";
  provenance: Record<string, any>;
  warnings: string[];
}

export interface ModelCard {
  key: string;
  name: string;
  kind: "content" | "graph";
  macro_f1: number | null;
  trained_on: string;
  note: string;
  primary: boolean;
  available: boolean;
}

export interface Sample {
  id: number;
  handle: string;
  text: string;
  accounts: number;
  source_domain: string;
  label: number | null;
  why: string;
}

export interface Person {
  handle: string;
  kind: string;
  hops: number | null;
  followers: number;
  posts: number;
}

export interface SampleDetail {
  id: number;
  uri: string;
  url: string;
  source_handle: string;
  source_name: string;
  text: string;
  created_at: string;
  reposts: number;
  replies: number;
  likes: number;
  source_domain: string;
  source_label: string;
  label: number | null;
  why: string;
  shape: { accounts: number; depth: number; breadth: number; direct: number };
  graph: { nodes: any[]; links: any[]; levels?: any[] };
  graph_caption: string;
  people: Person[];
  people_shown: number;
}

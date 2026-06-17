const BASE = import.meta.env.VITE_API_URL ?? "/api";

export interface Blank {
  index: number;
  answer: string;
  hint: string;
  masked_text: string;
}

export interface Story {
  template: string;
  blanks: Blank[];
  story: string;
  opening: string;
}

export interface Candidate {
  word: string;
  score: number;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Erreur ${response.status}`);
  }
  return response.json();
}

export async function fetchOpenings(): Promise<string[]> {
  const response = await fetch(`${BASE}/openings`);
  const data = await response.json();
  return data.openings;
}

export function generate(
  opening: string | null,
  n_blanks: number,
  seed?: number
): Promise<Story> {
  return post<Story>("/generate", { opening, n_blanks, seed });
}

export async function fill(
  masked_text: string,
  top_k = 5
): Promise<Candidate[]> {
  const data = await post<{ candidates: Candidate[] }>("/fill", {
    masked_text,
    top_k,
  });
  return data.candidates;
}

export interface ModelInfo {
  id: string;
  label: string;
  available: boolean;
}

export interface ModelsState {
  models: ModelInfo[];
  active: string;
  active_label: string;
  active_checkpoint: string | null;
  active_loaded: boolean;
}

export async function fetchModels(): Promise<ModelsState> {
  const response = await fetch(`${BASE}/models`);
  return response.json();
}

export function setModel(
  lineage: string
): Promise<{ active: string; active_label: string; active_checkpoint: string }> {
  return post("/model", { lineage });
}

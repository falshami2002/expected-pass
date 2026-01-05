export type Team = "attack" | "defense";

export type PlayerPlacement = {
  x: number; // 0..1
  y: number; // 0..1
  team: Team;
};

export type PredictRequest = {
  players: PlayerPlacement[]; // length 22
  ball: { x: number; y: number };
  passer_index: number;
  receiver_index: number;
};

export type PredictResponse = { xpass: number };

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function predictXpass(payload: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Predict failed (${res.status}): ${text || res.statusText}`);
  }

  return res.json();
}

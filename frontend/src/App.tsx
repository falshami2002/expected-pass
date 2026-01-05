import { useMemo, useRef, useState } from "react";
import { predictXpass, type PlayerPlacement, type Team } from "./lib/api";

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function makeDefaultPlayers(): PlayerPlacement[] {
  const A = "attack" as const;
  const D = "defense" as const;

  return [
    { x: 0.06, y: 0.5, team: A },

    { x: 0.14, y: 0.15, team: A },
    { x: 0.14, y: 0.38, team: A },
    { x: 0.14, y: 0.62, team: A },
    { x: 0.14, y: 0.85, team: A },

    { x: 0.30, y: 0.30, team: A },
    { x: 0.30, y: 0.50, team: A },
    { x: 0.30, y: 0.70, team: A },

    { x: 0.45, y: 0.20, team: A },
    { x: 0.46, y: 0.50, team: A },
    { x: 0.45, y: 0.80, team: A },

    { x: 0.94, y: 0.5, team: D },

    { x: 0.86, y: 0.15, team: D },
    { x: 0.86, y: 0.38, team: D },
    { x: 0.86, y: 0.62, team: D },
    { x: 0.86, y: 0.85, team: D },

    { x: 0.70, y: 0.30, team: D },
    { x: 0.70, y: 0.50, team: D },
    { x: 0.70, y: 0.70, team: D },

    { x: 0.55, y: 0.20, team: D },
    { x: 0.54, y: 0.50, team: D },
    { x: 0.55, y: 0.80, team: D },
  ];
}

export default function App() {
  const [players, setPlayers] = useState<PlayerPlacement[]>(makeDefaultPlayers);
  const [passerIndex, setPasserIndex] = useState(0);
  const [receiverIndex, setReceiverIndex] = useState(1);

  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);
  const [selectingRole, setSelectingRole] = useState<"passer" | "receiver" | null>(null);

  const [xpass, setXpass] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const pitchRef = useRef<HTMLDivElement | null>(null);

  const passerTeam: Team = players[passerIndex]?.team ?? "attack";

  const canPredict = useMemo(() => {
    if (players.length !== 22) return false;
    if (passerIndex === receiverIndex) return false;
    if (!players[passerIndex] || !players[receiverIndex]) return false;
    return true;
  }, [players, passerIndex, receiverIndex]);

  function eventToPitchXY(e: React.PointerEvent) {
    const el = pitchRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const x = clamp01((e.clientX - rect.left) / rect.width);
    const y = clamp01((e.clientY - rect.top) / rect.height);
    return { x, y };
  }

  function setPlayerPos(i: number, x: number, y: number) {
    setPlayers((prev) => {
      const next = [...prev];
      next[i] = { ...next[i], x, y };
      return next;
    });
  }

  async function onPredict() {
    setErr("");
    setLoading(true);
    setXpass(null);

    try {
      const passer = players[passerIndex];
      const res = await predictXpass({
        players,
        ball: { x: passer.x, y: passer.y },
        passer_index: passerIndex,
        receiver_index: receiverIndex,
      });
      setXpass(res.xpass);
    } catch (e: any) {
      setErr(e?.message ?? "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setPlayers(makeDefaultPlayers());
    setPasserIndex(0);
    setReceiverIndex(1);
    setDraggingIndex(null);
    setSelectingRole(null);
    setXpass(null);
    setErr("");
  }

  return (
    <div className="w-full overflow-x-hidden">
      <div className="min-h-dvh bg-slate-950 text-slate-100">
        <div className="mx-auto max-w-6xl p-6">
          <header className="mb-6 flex items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">xPass Dashboard</h1>
              <p className="text-sm text-slate-400">
                Select passer/receiver, then drag players on the pitch.
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={reset}
                className="rounded-xl bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700"
              >
                Reset
              </button>
              <button
                disabled={!canPredict || loading}
                onClick={onPredict}
                className="rounded-xl bg-emerald-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-emerald-400 disabled:opacity-50"
              >
                {loading ? "Predicting..." : "Predict xPass"}
              </button>
            </div>
          </header>

          <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
            <div
              ref={pitchRef}
              onPointerMove={(e) => {
                if (draggingIndex == null) return;
                const xy = eventToPitchXY(e);
                if (!xy) return;
                setPlayerPos(draggingIndex, xy.x, xy.y);
              }}
              onPointerUp={() => setDraggingIndex(null)}
              onPointerCancel={() => setDraggingIndex(null)}
              className="relative aspect-[3/2] w-full overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 touch-none"
            >
              <div className="absolute inset-0 opacity-60">
                <div className="absolute inset-4 rounded-xl border border-slate-700" />
                <div className="absolute left-1/2 top-4 bottom-4 w-px -translate-x-1/2 bg-slate-700" />
                <div className="absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full border border-slate-700" />
              </div>

              {players.map((p, i) => {
                const isPasser = i === passerIndex;
                const isReceiver = i === receiverIndex;
                const isDragging = i === draggingIndex;

                const base = p.team === "attack" ? "bg-sky-400" : "bg-rose-400";
                const ring = isPasser
                  ? "ring-4 ring-emerald-300"
                  : isReceiver
                  ? "ring-4 ring-amber-300"
                  : isDragging
                  ? "ring-4 ring-slate-200"
                  : "";

                return (
                  <button
                    key={i}
                    onPointerDown={(ev) => {
                      ev.stopPropagation();
                      ev.preventDefault();

                      if (selectingRole === "passer") {
                        setPasserIndex(i);
                        setSelectingRole(null);
                        return;
                      }
                      if (selectingRole === "receiver") {
                        setReceiverIndex(i);
                        setSelectingRole(null);
                        return;
                      }

                      setDraggingIndex(i);
                      ev.currentTarget.setPointerCapture(ev.pointerId);

                      const xy = eventToPitchXY(ev);
                      if (xy) setPlayerPos(i, xy.x, xy.y);
                    }}
                    className={`absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full ${base} ${ring} cursor-grab active:cursor-grabbing`}
                    style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
                    title={`Player ${i}`}
                  />
                );
              })}
            </div>

            <aside className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
              <div className="mb-4 flex items-center justify-between">
                <div className="text-sm font-medium">Controls</div>
                <div className="text-xs text-slate-400">Payload: 22 players</div>
              </div>

              <div className="space-y-3 text-sm">
                <div className="grid grid-cols-2 gap-2">
                  <label className="space-y-1">
                    <div className="text-xs text-slate-400">Passer index</div>
                    <input
                      type="number"
                      min={0}
                      max={21}
                      value={passerIndex}
                      onChange={(e) => setPasserIndex(Number(e.target.value))}
                      className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2"
                    />
                  </label>

                  <label className="space-y-1">
                    <div className="text-xs text-slate-400">Receiver index</div>
                    <input
                      type="number"
                      min={0}
                      max={21}
                      value={receiverIndex}
                      onChange={(e) => setReceiverIndex(Number(e.target.value))}
                      className="w-full rounded-xl border border-slate-800 bg-slate-950 px-3 py-2"
                    />
                  </label>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectingRole("passer")}
                    className={`flex-1 rounded-xl px-3 py-2 ${
                      selectingRole === "passer"
                        ? "bg-emerald-400 text-slate-950"
                        : "bg-slate-800 hover:bg-slate-700"
                    }`}
                  >
                    Select passer
                  </button>
                  <button
                    onClick={() => setSelectingRole("receiver")}
                    className={`flex-1 rounded-xl px-3 py-2 ${
                      selectingRole === "receiver"
                        ? "bg-amber-400 text-slate-950"
                        : "bg-slate-800 hover:bg-slate-700"
                    }`}
                  >
                    Select receiver
                  </button>
                </div>

                <button
                  onClick={() => {
                    setSelectingRole(null);
                    setDraggingIndex(null);
                  }}
                  className="w-full rounded-xl bg-slate-800 px-3 py-2 hover:bg-slate-700"
                >
                  Stop editing
                </button>

                <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                  <div className="text-xs text-slate-400">Result</div>
                  <div className="mt-1 text-lg font-semibold">
                    {xpass == null ? "—" : `${(xpass * 100).toFixed(1)}%`}
                  </div>
                  {err && <div className="mt-2 text-xs text-rose-300">{err}</div>}
                </div>

                <div className="text-xs text-slate-500">
                  {selectingRole
                    ? `Click a player to set ${selectingRole}.`
                    : "Drag players to reposition them."}
                </div>

                <div className="text-xs text-slate-500">
                  Passer team: <span className="text-slate-200">{passerTeam}</span>
                </div>
              </div>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useRef } from "react";

export type LogKind = "info" | "agent" | "flag" | "done" | "error";

export interface LogEntry {
  ts: string;       // "HH:MM:SS"
  text: string;
  kind: LogKind;
}

interface Props {
  entries: LogEntry[];
}

export function AnalysisLog({ entries }: Props) {
  const streamRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the latest entry whenever the list grows.
  useEffect(() => {
    const el = streamRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries.length]);

  return (
    <section className="analysis-log">
      <p className="section-label">Analysis log</p>
      <div className="analysis-log-stream" ref={streamRef}>
        {entries.length === 0 ? (
          <p className="analysis-log-empty">Waiting for first event…</p>
        ) : (
          entries.map((entry, i) => (
            <div
              key={`${entry.ts}-${i}`}
              className={`analysis-log-entry kind-${entry.kind}`}
            >
              <span className="analysis-log-time">{entry.ts}</span>
              <span className="analysis-log-text">{entry.text}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export function nowStamp(): string {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

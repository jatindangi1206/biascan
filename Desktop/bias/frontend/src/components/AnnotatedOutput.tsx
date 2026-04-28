import { useMemo, useState } from "react";
import type { Annotation, BiasType } from "../types";
import { BIAS_COLORS, BIAS_LABELS } from "../types";

interface Props {
  text: string;
  annotations: Annotation[];
}

interface Segment {
  start: number;
  end: number;
  text: string;
  anns: Annotation[];
}

function segment(text: string, annotations: Annotation[]): Segment[] {
  if (annotations.length === 0) return [{ start: 0, end: text.length, text, anns: [] }];
  const points = new Set<number>([0, text.length]);
  for (const a of annotations) {
    points.add(Math.max(0, Math.min(text.length, a.span_start)));
    points.add(Math.max(0, Math.min(text.length, a.span_end)));
  }
  const sorted = Array.from(points).sort((a, b) => a - b);
  const out: Segment[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    const s = sorted[i];
    const e = sorted[i + 1];
    if (e <= s) continue;
    const anns = annotations.filter((a) => a.span_start < e && a.span_end > s);
    out.push({ start: s, end: e, text: text.slice(s, e), anns });
  }
  return out;
}

export function AnnotatedOutput({ text, annotations }: Props) {
  const [active, setActive] = useState<number | null>(null);
  const [filter, setFilter] = useState<BiasType | "all">("all");

  const visible = useMemo(
    () => annotations.filter((a) => filter === "all" || a.bias_type === filter),
    [annotations, filter]
  );
  const segments = useMemo(() => segment(text, visible), [text, visible]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const a of annotations) c[a.bias_type] = (c[a.bias_type] ?? 0) + 1;
    return c;
  }, [annotations]);

  return (
    <div className="panel">
      <div className="panel-header">
        <p className="overline">02 — Annotated output</p>
        <h2 className="panel-title">Highlighted synthesis text</h2>
      </div>

      <div className="filter-row">
        <button
          type="button"
          className={`filter-chip ${filter === "all" ? "active" : ""}`}
          onClick={() => setFilter("all")}
        >
          All ({annotations.length})
        </button>
        {(Object.keys(BIAS_LABELS) as BiasType[]).map((bt) => {
          const c = counts[bt] ?? 0;
          if (c === 0) return null;
          const palette = BIAS_COLORS[bt];
          return (
            <button
              key={bt}
              type="button"
              className={`filter-chip ${filter === bt ? "active" : ""}`}
              onClick={() => setFilter(bt)}
              style={{
                background: palette.bg,
                color: palette.fg,
                borderColor: filter === bt ? palette.fg : palette.border,
              }}
            >
              {BIAS_LABELS[bt]} ({c})
            </button>
          );
        })}
      </div>

      <div className="annotated-text">
        {segments.map((seg, i) => {
          if (seg.anns.length === 0) {
            return <span key={i}>{seg.text}</span>;
          }
          const top = seg.anns.slice().sort((a, b) => b.confidence - a.confidence)[0];
          const palette = BIAS_COLORS[top.bias_type];
          const conflict = seg.anns.some((a) => a.conflict) || seg.anns.length > 1;
          return (
            <span
              key={i}
              className={`hl ${conflict ? "hl-conflict" : ""}`}
              style={{
                background: palette.bg,
                borderBottom: `2px solid ${palette.fg}`,
              }}
              onMouseEnter={() => setActive(i)}
              onMouseLeave={() => setActive(null)}
              title={seg.anns
                .map((a) => `${BIAS_LABELS[a.bias_type]} · ${a.severity} · ${(a.confidence * 100).toFixed(0)}%`)
                .join("\n")}
            >
              {seg.text}
              {active === i && (
                <span className="hl-pop">
                  {seg.anns.map((a, j) => {
                    const p = BIAS_COLORS[a.bias_type];
                    return (
                      <span key={j} className="hl-pop-row">
                        <span
                          className="hl-pop-tag"
                          style={{ background: p.bg, color: p.fg }}
                        >
                          {a.agent_name} · {BIAS_LABELS[a.bias_type]}
                        </span>
                        <span className="hl-pop-meta">
                          {a.severity} · {(a.confidence * 100).toFixed(0)}%
                        </span>
                        {a.clean_alternative && (
                          <span className="hl-pop-clean">→ {a.clean_alternative}</span>
                        )}
                      </span>
                    );
                  })}
                </span>
              )}
            </span>
          );
        })}
      </div>

      <p className="hint" style={{ marginTop: 12 }}>
        Hover any highlighted span to see which agents flagged it.
        Multi-agent overlaps are marked as conflicts (preserved, not dropped).
      </p>
    </div>
  );
}

import type { CSSProperties } from "react";
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
  if (annotations.length === 0) {
    return [{ start: 0, end: text.length, text, anns: [] }];
  }

  const points = new Set<number>([0, text.length]);
  for (const annotation of annotations) {
    points.add(Math.max(0, Math.min(text.length, annotation.span_start)));
    points.add(Math.max(0, Math.min(text.length, annotation.span_end)));
  }

  const sorted = Array.from(points).sort((a, b) => a - b);
  const output: Segment[] = [];

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const start = sorted[i];
    const end = sorted[i + 1];
    if (end <= start) continue;
    const anns = annotations.filter(
      (annotation) => annotation.span_start < end && annotation.span_end > start
    );
    output.push({ start, end, text: text.slice(start, end), anns });
  }

  return output;
}

export function AnnotatedOutput({ text, annotations }: Props) {
  const [active, setActive] = useState<number | null>(null);
  const [filter, setFilter] = useState<BiasType | "all">("all");

  const visible = useMemo(
    () =>
      annotations.filter(
        (annotation) => filter === "all" || annotation.bias_type === filter
      ),
    [annotations, filter]
  );

  const counts = useMemo(() => {
    const next: Record<string, number> = {};
    for (const annotation of annotations) {
      next[annotation.bias_type] = (next[annotation.bias_type] ?? 0) + 1;
    }
    return next;
  }, [annotations]);

  const segments = useMemo(() => segment(text, visible), [text, visible]);

  return (
    <section className="annotated-section">
      <div className="section-head">
        <p className="section-label">Annotated</p>
        <div className="annotation-filters">
          <button
            type="button"
            className={`annotation-filter ${filter === "all" ? "active" : ""}`}
            onClick={() => setFilter("all")}
          >
            All
          </button>
          {(Object.keys(BIAS_LABELS) as BiasType[]).map((biasType) => {
            const count = counts[biasType] ?? 0;
            if (count === 0) return null;
            return (
              <button
                key={biasType}
                type="button"
                className={`annotation-filter ${
                  filter === biasType ? "active" : ""
                }`}
                onClick={() => setFilter(biasType)}
              >
                {BIAS_LABELS[biasType]} ({count})
              </button>
            );
          })}
        </div>
      </div>

      <div className="annotated-copy">
        {segments.map((seg, index) => {
          if (seg.anns.length === 0) {
            return <span key={index}>{seg.text}</span>;
          }

          const lead = seg.anns.slice().sort((a, b) => b.confidence - a.confidence)[0];
          const palette = BIAS_COLORS[lead.bias_type];
          const hasConflict =
            seg.anns.some((annotation) => annotation.conflict) || seg.anns.length > 1;

          return (
            <span
              key={index}
              className={`annotation-mark ${hasConflict ? "conflict" : ""}`}
              style={{ "--mark-color": palette.border } as CSSProperties}
              onMouseEnter={() => setActive(index)}
              onMouseLeave={() => setActive(null)}
            >
              {seg.text}
              {active === index && (
                <span className="annotation-popover">
                  {seg.anns.map((annotation, rowIndex) => (
                    <span key={rowIndex} className="annotation-popover-row">
                      <span className="annotation-popover-title">
                        {annotation.agent_name} · {BIAS_LABELS[annotation.bias_type]}
                      </span>
                      <span className="annotation-popover-meta">
                        {annotation.severity} ·{" "}
                        {(annotation.confidence * 100).toFixed(0)}%
                      </span>
                      {annotation.clean_alternative && (
                        <span className="annotation-popover-clean">
                          {annotation.clean_alternative}
                        </span>
                      )}
                    </span>
                  ))}
                </span>
              )}
            </span>
          );
        })}
      </div>

      <p className="annotation-note">
        Hover an annotation to see the agent&apos;s evidence.
      </p>
    </section>
  );
}

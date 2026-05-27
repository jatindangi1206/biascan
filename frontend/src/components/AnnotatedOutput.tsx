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

// Known "why this is biased" reasoning keys, ordered by priority. These live
// either directly on `extras` or one level down inside `extras.chain_of_thought`.
const REASON_FIELDS = [
  "reasoning",
  "evidence_symmetry",              // ARGUS
  "syntactic_context_of_boosters",  // LIBRA
  "evidenced_scope",                // LENS
  "evidence_quality",
  "evidence_design",                // VIGIL
  "alternative_frame",              // QUILL
  "linguistic_evidence",
] as const;

const TAG_FIELDS = ["mechanism", "frame_type"] as const;

// Keys to ignore when falling back to "longest string in chain_of_thought" —
// these echo the source claim, not the reasoning for flagging it.
const CLAIM_KEY_RE = /claim|extract|present|acknowledged/i;

function pickString(source: Record<string, unknown>, keys: readonly string[]): string | null {
  for (const k of keys) {
    const v = source[k];
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return null;
}

export function extractReasoning(extras: Record<string, unknown>): string | null {
  const cot =
    extras.chain_of_thought && typeof extras.chain_of_thought === "object"
      ? (extras.chain_of_thought as Record<string, unknown>)
      : null;

  // Priority: known reasoning keys, top-level first, then inside chain_of_thought.
  const direct = pickString(extras, REASON_FIELDS);
  if (direct) return direct;
  if (cot) {
    const nested = pickString(cot, REASON_FIELDS);
    if (nested) return nested;

    // Fallback: longest non-claim string anywhere in chain_of_thought.
    const candidates: string[] = [];
    for (const [k, v] of Object.entries(cot)) {
      if (typeof v !== "string" || !v.trim()) continue;
      if (CLAIM_KEY_RE.test(k)) continue;
      candidates.push(v.trim());
    }
    if (candidates.length > 0) {
      candidates.sort((a, b) => b.length - a.length);
      return candidates[0];
    }
  }
  return null;
}

function succinctReason(annotation: Annotation): { tag: string | null; why: string | null } {
  const tag = pickString(annotation.extras, TAG_FIELDS);
  let why = extractReasoning(annotation.extras);
  if (why) {
    const firstStop = why.search(/(?<=[.!?])\s/);
    if (firstStop > 0 && firstStop < 160) why = why.slice(0, firstStop + 1);
    if (why.length > 180) why = why.slice(0, 177).trimEnd() + "…";
  }
  return { tag, why };
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
                  {seg.anns.map((annotation, rowIndex) => {
                    const { tag, why } = succinctReason(annotation);
                    return (
                      <span key={rowIndex} className="annotation-popover-row">
                        <span className="annotation-popover-title">
                          {annotation.agent_name} · {BIAS_LABELS[annotation.bias_type]}
                        </span>
                        <span className="annotation-popover-meta">
                          {annotation.severity} ·{" "}
                          {(annotation.confidence * 100).toFixed(0)}%
                          {tag && <span className="annotation-popover-tag">{tag.replace(/_/g, " ")}</span>}
                        </span>
                        {why && (
                          <span className="annotation-popover-why">
                            <em>Why:</em> {why}
                          </span>
                        )}
                        {annotation.clean_alternative && (
                          <span className="annotation-popover-clean">
                            {annotation.clean_alternative}
                          </span>
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

      <p className="annotation-note">
        Hover an annotation to see the agent&apos;s evidence.
      </p>
    </section>
  );
}

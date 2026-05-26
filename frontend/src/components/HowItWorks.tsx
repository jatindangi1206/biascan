interface Props {
  onBack: () => void;
}

export function HowItWorks({ onBack }: Props) {
  return (
    <section className="howitworks-stage">
      <button type="button" className="nav-link howitworks-back" onClick={onBack}>
        ← Back
      </button>

      <h1 className="howitworks-title">How it works</h1>
      <p className="howitworks-copy">
        Five agents read your synthesis in parallel — each tuned to one cognitive
        bias. AEGIS resolves conflicts, an evidence index cross-checks each flag,
        and a density-aware score summarises what was found.
      </p>

      <div className="video-frame">
        {/* TODO: replace VIDEO_ID with the actual YouTube video ID */}
        <iframe
          src="https://www.youtube-nocookie.com/embed/VIDEO_ID"
          title="BiasScan walkthrough"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>
    </section>
  );
}

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

      <div className="video-frame">
        <iframe
          src="https://www.youtube-nocookie.com/embed/vlxvTlvt9hk"
          title="BiasScan walkthrough"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>

      <section className="howitworks-panel" aria-labelledby="privacy-data-handling">
        <h2 id="privacy-data-handling" className="howitworks-section-title">
          Privacy &amp; Data Handling
        </h2>
        <div className="howitworks-copy-stack">
          <p className="howitworks-copy">
            BiasScan does not permanently store uploaded documents, prompts, or
            analysis outputs on our servers.
          </p>
          <p className="howitworks-copy">
            When users connect their own LLM API keys, submitted content is
            processed directly through the selected AI provider. Data handling,
            retention, and processing are governed by the provider&apos;s own
            infrastructure and privacy policies.
          </p>
          <p className="howitworks-copy">
            BiasScan itself does not access, retain, or reuse submitted content
            beyond the active analysis session.
          </p>
        </div>
      </section>

      <section className="howitworks-panel" aria-labelledby="transparency-limitations">
        <h2 id="transparency-limitations" className="howitworks-section-title">
          Transparency &amp; Limitations
        </h2>
        <div className="howitworks-copy-stack">
          <p className="howitworks-copy">
            Bias detection is inherently subjective and context-dependent.
          </p>
          <p className="howitworks-copy">
            The platform attempts to surface potential framing patterns,
            omission signals, evidence imbalance, and narrative inconsistencies,
            but results should be interpreted as probabilistic analytical
            signals rather than objective truth.
          </p>
          <p className="howitworks-copy">The system may:</p>
          <ul className="howitworks-list">
            <li>Miss subtle contextual nuance</li>
            <li>Produce false positives or conflicting interpretations</li>
            <li>Reflect limitations of underlying language models</li>
            <li>Vary across providers and model versions</li>
          </ul>
          <p className="howitworks-copy">
            BiasScan is built around transparency, inspectable reasoning, and
            research-oriented evaluation of AI-assisted analysis systems.
          </p>
        </div>
      </section>
    </section>
  );
}

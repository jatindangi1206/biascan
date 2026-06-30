import data from "../leaderboard.json";

interface Props {
  onBack: () => void;
}

export function Leaderboard({ onBack }: Props) {
  return (
    <section className="howitworks-stage">
      <button type="button" className="nav-link howitworks-back" onClick={onBack}>
        ← Back
      </button>

      <h1 className="howitworks-title">Model Leaderboard</h1>
      <p className="howitworks-copy">
        To support model selection, we publish how supported models perform on
        our systematic-review benchmark. Each score is the mean bias score
        detected across the benchmark documents, averaged over {data.n_runs} runs
        per document. Higher indicates a more sensitive detector.
      </p>

      <section className="howitworks-panel">
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th className="lb-rank">#</th>
              <th>Model</th>
              <th className="lb-num">Overall</th>
              <th className="lb-num">MS-Gut Review</th>
              <th className="lb-num">Nutraceuticals Review</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) => (
              <tr key={r.model}>
                <td className="lb-rank">{r.rank}</td>
                <td className="lb-model">{r.model}</td>
                <td className="lb-num lb-overall">{r.overall.toFixed(3)}</td>
                <td className="lb-num">{r.ms_gut.toFixed(3)}</td>
                <td className="lb-num">{r.nutra.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <p className="howitworks-copy lb-note">
        This leaderboard is preliminary. We are still expanding the benchmark and
        actively evaluating additional models — rankings and scores will be
        updated as testing continues. Last updated {data.updated}.
      </p>
    </section>
  );
}

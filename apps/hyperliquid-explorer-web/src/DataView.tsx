import { type Detail, runUrl } from "./api";
export function DataView({ detail }: { detail: Detail }) {
  return (
    <>
      <section className="panel">
        <h2>Data & evidence</h2>
        <p className="muted">
          Saved report artifacts are the source of truth. This explorer does not
          recalculate strategies or modify report files.
        </p>
        <div className="analysis-grid">
          {[
            ["Provenance", detail.provenance],
            ["Configuration", detail.config],
            ["Reconciliation", detail.reconciliation],
          ].map(([title, value]) => (
            <section key={String(title)}>
              <h3>{String(title)}</h3>
              <pre>{JSON.stringify(value, null, 2)}</pre>
            </section>
          ))}
        </div>
      </section>
      <section className="panel">
        <h2>Original artifacts</h2>
        <p className="muted">
          Downloads retain original values—including raw ratios suppressed in
          the synthetic demo UI.
        </p>
        <div className="artifact-grid">
          {detail.artifacts.map((name) => (
            <a
              key={name}
              href={runUrl(detail.id, `/artifacts/${encodeURIComponent(name)}`)}
              download
            >
              <span>↓</span>
              {name}
            </a>
          ))}
        </div>
      </section>
    </>
  );
}

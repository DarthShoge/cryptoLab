import type { usePreflight } from "./usePreflight";

export function RunPreflight({
  state,
}: {
  state: ReturnType<typeof usePreflight>;
}) {
  return (
    <section className="notice" aria-label="Run readiness" aria-live="polite">
      {state.missingDataset && (
        <p>Select an available local dataset to check this draft.</p>
      )}
      {state.checking && <p>Checking draft…</p>}
      {state.error && (
        <>
          <p>{state.error}</p>
          <button onClick={state.retry}>Retry validation</button>
        </>
      )}
      {state.data && (
        <>
          <strong>
            {state.ready ? "Ready to submit" : "Cannot run with these settings"}
          </strong>
          {state.data.issues.map((issue, index) => (
            <p key={`${issue.code}-${index}`}>{issue.message}</p>
          ))}
          <p>
            Required history: {state.data.required_start} →{" "}
            {state.data.required_end} (UTC).
          </p>
          <small>
            Full checksum and market-data validation still runs on submission
            and execution. Settings are not changed automatically.
          </small>
        </>
      )}
    </section>
  );
}

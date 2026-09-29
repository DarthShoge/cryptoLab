export function Status({
  loading,
  error,
  retry,
}: {
  loading?: boolean;
  error?: string;
  retry?: () => void;
}) {
  if (error)
    return (
      <div className="empty error" role="alert">
        <strong>Unable to show this data</strong>
        <p>{error}</p>
        {retry && <button onClick={retry}>Try again</button>}
      </div>
    );
  if (loading)
    return (
      <div className="skeleton" role="status" aria-label="Loading report data">
        <span />
        <span />
        <span />
      </div>
    );
  return null;
}

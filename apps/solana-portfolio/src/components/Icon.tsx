type Name = "orbit" | "overview" | "chart" | "positions" | "activity" | "settings" | "arrow" | "refresh" | "close" | "external" | "wallet";
const paths: Record<Name, React.ReactNode> = {
  orbit: <><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></>,
  overview: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
  chart: <><path d="M3 3v18h18M5 16l5-6 4 3 6-8"/></>,
  positions: <><path d="M3 8h18M5 8v12m7-12v12m7-12v12M3 21h18M12 3 3 7h18z"/></>,
  activity: <><path d="M3 8h15l-4-4m4 12H3l4 4M18 8v0M3 16v0"/></>,
  settings: <><circle cx="12" cy="12" r="3"/><path d="m9 3-1 3-3 1-2 3 2 2-1 3 3 2 2 3h4l1-3 3-1 2-3-2-2 1-3-3-2-2-3z"/></>,
  arrow: <path d="M5 17 19 3M5 3h14v14"/>,
  refresh: <><path d="M20 9a8 8 0 1 0 0 7M20 3v6h-6"/></>,
  close: <path d="m5 5 14 14M19 5 5 19"/>,
  external: <><path d="M14 3h7v7M21 3 10 14M10 3H3v18h18v-7"/></>,
  wallet: <><rect x="3" y="5" width="18" height="15" rx="3"/><path d="M3 8V5l13-3v3M16 12h5v5h-5z"/></>,
};
export function Icon({ name, size = 18 }: { name: Name; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

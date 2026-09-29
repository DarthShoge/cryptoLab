import LegacyExplorer from "./LegacyExplorer";
import { Lab } from "./Lab";
export default function App() {
  return window.location.pathname === "/reports" ? <LegacyExplorer /> : <Lab />;
}

export const money = (value: number | null | undefined, digits = 2) => value == null ? "—" : value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: digits, minimumFractionDigits: digits });
export const number = (value: number | null | undefined, digits = 2) => value == null ? "—" : value.toLocaleString("en-US", { maximumFractionDigits: digits });
export const percent = (value: number | null | undefined, signed = false) => value == null ? "—" : `${signed && value > 0 ? "+" : ""}${value.toFixed(2)}%`;
export const short = (address: string, length = 4) => address.length > 18 ? `${address.slice(0, length)}…${address.slice(-length)}` : address;
export const date = (time: number) => new Date(time * 1000).toLocaleDateString("en-GB", { day: "2-digit", month: "short", timeZone: "UTC" });
export const datetime = (time: number) => `${new Date(time * 1000).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" })} · ${new Date(time * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" })} UTC`;

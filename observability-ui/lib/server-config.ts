// Server-only — never imported from client components
export const API_BASE = process.env.API_BASE ?? "http://localhost:8000";
export const API_KEY = process.env.API_KEY ?? "change-me";
export const API_HEADERS = { "X-API-Key": API_KEY, "Content-Type": "application/json" };

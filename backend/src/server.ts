/** Process entry point: `npm run dev` / `npm start`. */
import { createApp } from "./app.js";
import { loadDotEnv, loadSettings } from "./config.js";

loadDotEnv();
const settings = loadSettings();
const app = await createApp(settings);
const shutdown = () => void app.close().then(() => process.exit(0));
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
await app.listen({ port: settings.port, host: settings.host });
console.log(`Advance RAG listening on http://localhost:${settings.port}`);

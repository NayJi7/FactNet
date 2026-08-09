import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // the engine runs separately; proxying keeps the browser on one origin
    proxy: { "/api": `http://127.0.0.1:${process.env.API_PORT ?? 8000}` },
  },
});

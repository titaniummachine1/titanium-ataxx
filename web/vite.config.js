import { defineConfig } from "vite";

export default defineConfig({
  // relative base so the built site works from any static host / subpath
  base: "./",
  server: {
    port: 5173,
  },
  build: {
    target: "es2022",
  },
});

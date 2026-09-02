import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // the booster is fetched from /model.json, so the entry chunk stays small
    chunkSizeWarningLimit: 700,
  },
})

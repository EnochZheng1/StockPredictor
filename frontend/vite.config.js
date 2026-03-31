import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, path.resolve(__dirname, '..'), '')
  return {
    plugins: [react()],
    server: {
      port: parseInt(env.VITE_PORT || '4290'),
    },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './src/test/setup.js',
      coverage: {
        provider: 'v8',
        reporter: ['text', 'html'],
        reportsDirectory: '../reports/frontend-coverage',
        include: ['src/**/*.{js,jsx}'],
        exclude: ['src/test/**', 'src/main.jsx'],
      },
    },
  }
})

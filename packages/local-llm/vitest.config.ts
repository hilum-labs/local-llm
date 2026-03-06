import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    testTimeout: 30000,
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/types.ts',
        'src/openai-types.ts',
        'src/index.ts',
        'src/native.ts',
      ],
      thresholds: {
        statements: 70,
        branches: 85,
        functions: 65,
        lines: 70,
      },
    },
  },
})

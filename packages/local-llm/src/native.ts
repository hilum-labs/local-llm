import { createRequire } from 'node:module';
import {
  peekNativeAddon,
  setNativeAddon,
  setNativeAddonLoader,
  type NativeAddon,
  type NativeContext,
  type NativeModel,
  type NativeMtmdContext,
} from 'local-llm-js-core/native';

let cached: NativeAddon | null = null;

export { setNativeAddon };
export type { NativeAddon, NativeContext, NativeModel, NativeMtmdContext };

function resolveNodeNativeAddon(): NativeAddon {
  const require = createRequire(import.meta.url);
  const platformPkg = `@local-llm/${process.platform}-${process.arch}`;

  try {
    return require(platformPkg) as NativeAddon;
  } catch {}

  try {
    return require('@local-llm/native') as NativeAddon;
  } catch {}

  throw new Error(
    `No native binary found for ${process.platform}-${process.arch}.\n` +
    `Install the platform package: npm install ${platformPkg}\n` +
    `Or build from source: cd packages/native && npm run build`,
  );
}

export function loadNativeAddon(): NativeAddon {
  if (cached) return cached;

  const existing = peekNativeAddon();
  if (existing) {
    cached = existing;
    return cached;
  }

  cached = resolveNodeNativeAddon();
  setNativeAddon(cached);
  return cached;
}

setNativeAddonLoader(loadNativeAddon);

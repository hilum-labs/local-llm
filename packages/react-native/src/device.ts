import NativeLocalLLM from './NativeLocalLLM';

export interface DeviceCapabilities {
  totalRAM: number;        // bytes
  availableRAM: number;    // bytes (respects iOS jetsam limits)
  gpuName: string;         // e.g. "Apple A16 GPU"
  metalFamily: number;     // Apple GPU family (5 = A12+, 7 = A14+, 9 = A17+)
  metalVersion: number;    // 1, 2, or 3
  iosVersion: string;      // e.g. "17.2.1"
  isLowPowerMode: boolean;
}

export function getDeviceCapabilities(): DeviceCapabilities {
  return NativeLocalLLM.getDeviceCapabilities() as DeviceCapabilities;
}

export function canRunModel(modelSizeBytes: number): {
  canRun: boolean;
  reason?: string;
  suggestion?: string;
} {
  const caps = getDeviceCapabilities();
  const availableMB = caps.availableRAM / (1024 * 1024);
  const modelSizeMB = modelSizeBytes / (1024 * 1024);

  // Model needs ~1.2x its file size in RAM (weights + KV cache + overhead)
  const estimatedUsageMB = modelSizeMB * 1.2;

  if (estimatedUsageMB > availableMB * 0.8) {
    return {
      canRun: false,
      reason: `Model needs ~${Math.round(estimatedUsageMB)} MB but only ${Math.round(availableMB)} MB available`,
      suggestion: modelSizeMB > 2000
        ? 'Try a Q4_K_M quantized variant or a smaller model'
        : 'Close other apps to free memory',
    };
  }

  return { canRun: true };
}

export function recommendQuantization(): string {
  const caps = getDeviceCapabilities();
  const totalGB = caps.totalRAM / (1024 * 1024 * 1024);

  if (totalGB >= 8) return 'Q8_0';    // iPhone 16 Pro (8 GB)
  if (totalGB >= 6) return 'Q6_K';    // iPhone 14/15 Pro (6 GB)
  if (totalGB >= 4) return 'Q4_K_M';  // iPhone 11-13, 14/15 base (4 GB)
  return 'Q3_K_S';                     // iPhone 8/X (3 GB) -- minimal
}

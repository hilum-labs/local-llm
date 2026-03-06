const path = require("path");
const fs = require("fs");

// Find the compiled .node addon
const addonPath = path.join(
  __dirname,
  "..",
  "packages",
  "native",
  "build",
  "Release",
  "hilum_native.node"
);

if (!fs.existsSync(addonPath)) {
  console.error("FAIL: Addon not found at", addonPath);
  process.exit(1);
}

const addon = require(addonPath);

// Check exports
const exportedKeys = Object.keys(addon).sort();
console.log("Exported functions:", exportedKeys);

let passed = true;

if (!exportedKeys.includes("backendInfo")) {
  console.error("FAIL: missing backendInfo export");
  passed = false;
}

if (!exportedKeys.includes("backendVersion")) {
  console.error("FAIL: missing backendVersion export");
  passed = false;
}

// Call backendVersion
const version = addon.backendVersion();
console.log("Backend version:", version);
if (!version.includes("local-llm-native")) {
  console.error("FAIL: unexpected version string");
  passed = false;
}

// Call backendInfo
const info = addon.backendInfo();
console.log("System info:", info);

// On macOS, check for Metal support
if (process.platform === "darwin") {
  if (!info.includes("Metal")) {
    console.warn("WARN: Metal not detected in system info (may be expected on some Macs)");
  }
}

if (passed) {
  console.log("\nPASS: All Phase 1 checks passed.");
} else {
  console.error("\nFAIL: Some checks failed.");
  process.exit(1);
}

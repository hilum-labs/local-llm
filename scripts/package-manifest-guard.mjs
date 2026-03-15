#!/usr/bin/env node

import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const [, , command, manifestArg] = process.argv;

if (!command || !manifestArg) {
  console.error('Usage: node scripts/package-manifest-guard.mjs <prepare|restore|verify|verify-tarball> <path>');
  process.exit(1);
}

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const manifestPath = path.resolve(repoRoot, manifestArg);
const backupPath = `${manifestPath}.bak`;

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function replaceWorkspaceProtocols(pkg) {
  const version = pkg.version;
  for (const field of ['dependencies', 'optionalDependencies', 'peerDependencies']) {
    const deps = pkg[field];
    if (!deps) continue;
    for (const [name, spec] of Object.entries(deps)) {
      if (typeof spec === 'string' && spec.startsWith('workspace:')) {
        deps[name] = version;
      }
    }
  }
}

if (command === 'prepare') {
  const pkg = readJson(manifestPath);
  writeFileSync(backupPath, readFileSync(manifestPath));
  replaceWorkspaceProtocols(pkg);
  writeJson(manifestPath, pkg);
  process.exit(0);
}

if (command === 'restore') {
  if (!existsSync(backupPath)) {
    process.exit(0);
  }
  writeFileSync(manifestPath, readFileSync(backupPath));
  rmSync(backupPath, { force: true });
  process.exit(0);
}

if (command === 'verify') {
  const raw = readFileSync(manifestPath, 'utf8');
  if (raw.includes('workspace:')) {
    console.error(`workspace protocol leak detected in ${manifestArg}`);
    process.exit(1);
  }
  process.exit(0);
}

if (command === 'verify-tarball') {
  const tarballPath = manifestPath;
  const result = spawnSync('tar', ['-xOf', tarballPath, 'package/package.json'], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  if (result.status !== 0) {
    process.stderr.write(result.stderr || '');
    console.error(`failed to inspect packed manifest in ${manifestArg}`);
    process.exit(1);
  }

  if ((result.stdout || '').includes('workspace:')) {
    console.error(`workspace protocol leak detected in packed manifest ${manifestArg}`);
    process.exit(1);
  }

  process.exit(0);
}

console.error(`Unknown command: ${command}`);
process.exit(1);

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const packageDirs = process.argv.slice(2);
if (packageDirs.length === 0) {
  throw new Error('usage: node scripts/publish-packages.mjs <package-dir> [...]');
}

const sleep = (milliseconds) => {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
};

function isPublished(name, version) {
  const result = spawnSync('npm', ['view', `${name}@${version}`, 'version'], {
    encoding: 'utf8',
  });
  if (result.status === 0) return true;

  const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  if (/E404|404 Not Found|No match found/i.test(output)) return false;

  process.stderr.write(output);
  throw new Error(`could not query npm for ${name}@${version}`);
}

function publish(packageDir, name, version) {
  const spec = `${name}@${version}`;
  if (isPublished(name, version)) {
    console.log(`Skipping ${spec}: already published`);
    return;
  }

  const delays = [15_000, 30_000, 45_000, 60_000];
  for (let attempt = 0; attempt <= delays.length; attempt += 1) {
    console.log(`Publishing ${spec} (attempt ${attempt + 1}/${delays.length + 1})...`);
    const result = spawnSync('npm', ['publish', '--access', 'public', packageDir], {
      stdio: 'inherit',
    });
    if (result.status === 0 || isPublished(name, version)) return;
    if (attempt === delays.length) break;

    console.log(`npm has not accepted ${spec}; retrying in ${delays[attempt] / 1000}s`);
    sleep(delays[attempt]);
  }

  throw new Error(`failed to publish ${spec}`);
}

for (const [index, packageDir] of packageDirs.entries()) {
  const absolutePackageDir = resolve(packageDir);
  const manifest = JSON.parse(
    readFileSync(resolve(absolutePackageDir, 'package.json'), 'utf8'),
  );
  publish(absolutePackageDir, manifest.name, manifest.version);

  // npm can briefly reject a write while processing the preceding packument.
  if (index < packageDirs.length - 1) sleep(15_000);
}

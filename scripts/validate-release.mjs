import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2).filter((arg) => arg !== '--');
const tag = args.find((arg) => arg !== '--check-registry') ?? process.env.GITHUB_REF_NAME;
const checkRegistry = args.includes('--check-registry');
if (!tag?.startsWith('v') || !/^\d+\.\d+\.\d+$/.test(tag.slice(1))) {
  throw new Error(`expected a v-prefixed semantic version tag; received ${JSON.stringify(tag)}`);
}

const version = tag.slice(1);
const manifests = [
  'package.json',
  'packages/local-llm/package.json',
  'packages/native/package.json',
  'packages/platforms/darwin-arm64/package.json',
  'packages/platforms/darwin-x64/package.json',
  'packages/platforms/linux-x64/package.json',
  'packages/platforms/win32-x64/package.json',
].map((path) => ({ path, value: JSON.parse(readFileSync(path, 'utf8')) }));

for (const { path, value } of manifests) {
  if (value.version !== version) {
    throw new Error(`${path} has version ${value.version}; expected ${version} from tag ${tag}`);
  }
}

const mainPackage = manifests.find(({ path }) => path === 'packages/local-llm/package.json').value;
const platformPackages = manifests.filter(({ path }) => path.startsWith('packages/platforms/'));
for (const { value } of platformPackages) {
  if (mainPackage.optionalDependencies?.[value.name] !== version) {
    throw new Error(`${mainPackage.name} must pin ${value.name} to ${version}`);
  }
}

if (checkRegistry) {
  for (const { value } of [
    manifests.find(({ path }) => path === 'packages/local-llm/package.json'),
    ...platformPackages,
  ]) {
    const spec = `${value.name}@${version}`;
    const result = spawnSync('npm', ['view', spec, 'version'], { encoding: 'utf8' });
    if (result.status === 0) {
      throw new Error(`${spec} is already published`);
    }

    const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
    if (!/E404|404 Not Found/i.test(output)) {
      process.stderr.write(output);
      throw new Error(`could not verify npm availability for ${spec}`);
    }
  }
}

console.log(`Validated ${tag} metadata for ${platformPackages.length + 1} public packages`);

// UI + API test for the backend-environment card: installed / newest version per backend, an install-or-update
// button that names the version it will install, progress while it runs, and the failure path.
// The server runs a fake scripts/setup_backend.py (MQDE_SETUP_BACKEND_BIN) driven by a state file, so nothing is installed.
//
//   node web/e2e/backend_envs.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../..');
const require = createRequire(path.resolve(repo, process.env.PLAYWRIGHT_DIR ?? 'demo') + '/');
let chromium;
try { ({ chromium } = require('playwright')); } catch {
  console.error(`SKIP  Playwright is not installed under ${process.env.PLAYWRIGHT_DIR ?? 'demo'}/ (cd demo && npm install), nothing was run`);
  process.exit(3);
}
const bin = path.resolve(repo, process.env.MQDE_BIN ?? ['build/dev/bin/mqde', 'build/mac/bin/mqde'].find((p) => fs.existsSync(path.join(repo, p))) ?? 'build/dev/bin/mqde');
if (!fs.existsSync(bin)) { console.error(`mqde binary not found: ${bin}`); process.exit(2); }
if (!fs.existsSync(path.join(repo, 'web/dist/index.html'))) { console.error('web/dist is not built (cd web && npm run build)'); process.exit(2); }

const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-envs-'));
const stateFile = path.join(tmp, 'state.json');
const callLog = path.join(tmp, 'calls.log');
const row = (id, installed, version) => ({ id, installed, version, path: installed ? `/envs/${id}/bin/x` : null, source: installed ? 'managed' : null, managed: installed, envs_dir: '/envs' });
const writeState = (s) => fs.writeFileSync(stateFile, JSON.stringify(s));
writeState({
  rows: [row('vllm', true, '0.30.0'), row('sglang', true, '0.5.22.dev586+g86624f14e'), row('llamacpp', false, null)],
  latest: { vllm: '0.31.0', sglang: '0.5.21', llamacpp: 'b11565' },
  newer: { sglang: false }, // the installed 0.5.22.dev is ahead of the newest release 0.5.21 (the real script compares versions itself)
});
// Behaves like setup_backend.py's `status --json [--check]` and `install <backend> [--version v]`.
const fake = path.join(tmp, 'fake_setup.py');
fs.writeFileSync(fake, `#!/usr/bin/env python3
import json, os, sys, time
st = os.environ["FAKE_STATE"]; d = json.load(open(st)); a = sys.argv[1:]
open(os.environ["FAKE_CALLS"], "a").write(" ".join(a) + "\\n")
if a[0] == "status":
    for r in d["rows"]:
        if "--check" in a:
            r["latest"] = d["latest"][r["id"]]
            r["update_available"] = (not r["installed"]) or (r["latest"] != r["version"] and d.get("newer", {}).get(r["id"], True))
    print(json.dumps(d["rows"]))
elif a[0] == "install":
    b = a[1]; v = a[a.index("--version") + 1] if "--version" in a else d["latest"][b]
    print("resolving", b, v, flush=True); time.sleep(4); print("installing", b, v, flush=True)
    if v == "9.9.9":
        print("error: no matching distribution", file=sys.stderr); sys.exit(1)
    for r in d["rows"]:
        if r["id"] == b: r.update(installed=True, version=v, source="managed", managed=True)
    json.dump(d, open(st, "w")); print(b, v, "ready")
`);
fs.chmodSync(fake, 0o755);

const port = await freePort();
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)],
{ stdio: ['ignore', 'ignore', 'ignore'], env: { ...process.env, MQDE_SETUP_BACKEND_BIN: fake, FAKE_STATE: stateFile, FAKE_CALLS: callLog } });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }
const calls = () => (fs.existsSync(callLog) ? fs.readFileSync(callLog, 'utf8').split('\n').filter(Boolean) : []);
const post = (p, body) => fetch(`${base}/api/v1${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  let dialogs = 0;
  let accept = true;
  page.on('dialog', (d) => { dialogs++; accept ? d.accept() : d.dismiss(); });

  // ---- 1. installed and newest versions, and the button names the version ----
  await page.goto(`${base}/hardware`, { waitUntil: 'load' });
  const txt = (id) => page.getByTestId(id).innerText();
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-latest-vllm"]')?.textContent?.includes('0.31.0'), null, { timeout: 15000 });
  check('vLLM shows its installed version', (await txt('backend-env-version-vllm')) === 'v0.30.0');
  check('and the newest one', (await txt('backend-env-latest-vllm')) === 'v0.31.0');
  check('the update button says which version it will install', (await txt('backend-env-action-vllm')) === '更新到 v0.31.0');
  check('a source build ahead of the newest release is already up to date', (await txt('backend-env-action-sglang')) === '已是最新 v0.5.22.dev586+g86624f14e' && await page.getByTestId('backend-env-action-sglang').isDisabled());
  check('a missing backend offers to install the newest', (await txt('backend-env-action-llamacpp')) === '安装 b11565', await txt('backend-env-action-llamacpp'));
  check('llama.cpp build tags are not given a "v"', (await txt('backend-env-latest-llamacpp')) === 'b11565');
  check('the check goes through the server with --check', calls().some((c) => c === 'status --json --check') && calls().some((c) => c === 'status --json'), JSON.stringify(calls()));

  // ---- 2. declining the confirmation starts nothing ----
  accept = false;
  await page.getByTestId('backend-env-action-vllm').click();
  await new Promise((r) => setTimeout(r, 400));
  check('declining the confirmation runs no install', dialogs === 1 && !calls().some((c) => c.startsWith('install')), JSON.stringify(calls()));

  // ---- 3. an update: progress, a second install refused, then the new version ----
  accept = true;
  await page.getByTestId('backend-env-action-vllm').click();
  await page.getByTestId('backend-install').waitFor({ timeout: 10000 });
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-install-log"]')?.textContent?.includes('resolving vllm 0.31.0'), null, { timeout: 10000 });
  check('the install runs the script with the pinned version', calls().includes('install vllm --version 0.31.0'), JSON.stringify(calls()));
  check('progress is shown while it runs', (await txt('backend-install-state')) === '安装中');
  check('other buttons are disabled meanwhile', await page.getByTestId('backend-env-action-llamacpp').isDisabled() && await page.getByTestId('backend-env-check').isDisabled());
  const second = await post('/backend-envs/sglang/install', {});
  check('a second install is refused with 409 while one runs', second.status === 409, `${second.status} ${await second.text()}`);
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-install-state"]')?.textContent === '完成', null, { timeout: 20000 });
  check('the log shows what the installer printed', (await txt('backend-install-log')).includes('vllm 0.31.0 ready'));
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-version-vllm"]')?.textContent === 'v0.31.0', null, { timeout: 15000 });
  check('afterwards the installed version is the new one', true);
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-action-vllm"]')?.textContent?.startsWith('已是最新'), null, { timeout: 15000 });
  check('and the button reads "already newest v0.31.0"', (await txt('backend-env-action-vllm')) === '已是最新 v0.31.0' && await page.getByTestId('backend-env-action-vllm').isDisabled());

  // ---- 4. a failing install says so and keeps the working version ----
  const s = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  s.latest.vllm = '9.9.9';
  writeState(s);
  await page.getByTestId('backend-env-check').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-action-vllm"]')?.textContent === '更新到 v9.9.9', null, { timeout: 15000 });
  check('"check for updates" picks up a newer release', true);
  await page.getByTestId('backend-env-action-vllm').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-install-state"]')?.textContent?.startsWith('失败'), null, { timeout: 20000 });
  check('a failed install shows the exit code and the error', (await txt('backend-install-state')).includes('退出码 1') && (await txt('backend-install-log')).includes('no matching distribution'));
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-action-vllm"]')?.textContent === '更新到 v9.9.9', null, { timeout: 15000 });
  check('the installed version is unchanged after the failure', (await txt('backend-env-version-vllm')) === 'v0.31.0');

  // ---- 4b. a backend pinned by an environment variable cannot be updated from here (the variable would win) ----
  const pinned = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  pinned.rows.find((r) => r.id === 'sglang').source = 'env';
  writeState(pinned);
  await page.getByTestId('backend-env-check').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-action-sglang"]')?.textContent === '由环境变量指定', null, { timeout: 15000 });
  check('an env-var-pinned backend says so and cannot be "updated"', await page.getByTestId('backend-env-action-sglang').isDisabled());
  check('with an explanation on hover', ((await page.getByTestId('backend-env-action-sglang').getAttribute('title')) ?? '').includes('优先级高于托管环境'));

  // ---- 5. the API refuses anything but a known backend and a plain version ----
  for (const [p, b, why] of [['/backend-envs/pip/install', {}, 'unknown backend'], ['/backend-envs/vllm/install', { version: '1.0; rm -rf /' }, 'shell metacharacters'],
    ['/backend-envs/vllm/install', { version: '$(id)' }, 'command substitution'], ['/backend-envs/vllm/install', { version: 5 }, 'non-string version']]) {
    const r = await post(p, b);
    check(`the API rejects ${why} with 400`, r.status === 400, `${r.status}`);
  }
  check('rejected requests ran nothing', calls().filter((c) => c.startsWith('install')).length === 2, JSON.stringify(calls().filter((c) => c.startsWith('install'))));
  check('an unknown install id is 404', (await fetch(`${base}/api/v1/backend-envs/install/deadbeef`)).status === 404);

  // ---- 6. a llama.cpp built from 3rdpart/llama.cpp (the official CUDA build does not run on the dev host) ----
  const srcState = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
  Object.assign(srcState.rows.find((r) => r.id === 'llamacpp'), { installed: true, version: 'b11574', source: 'managed', managed: true, source_build: true });
  writeState(srcState);
  await page.getByTestId('backend-env-check').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="backend-env-action-llamacpp"]')?.textContent?.startsWith('重新构建'), null, { timeout: 15000 });
  check('a source build offers a rebuild, not "更新到 bNNNN" (which would install the CPU release over it)', (await txt('backend-env-action-llamacpp')) === '重新构建（源码 b11574）', await txt('backend-env-action-llamacpp'));
  check('its "latest" is the submodule, not the newest release', (await txt('backend-env-latest-llamacpp')) === '随子模块');
  check('the source column says so', (await page.getByTestId('backend-env-llamacpp').innerText()).includes('源码构建'));
  const before = calls().length;
  await page.getByTestId('backend-env-action-llamacpp').click();
  await page.waitForFunction((n) => true, before, { timeout: 1000 });
  await page.waitForTimeout(800);
  const last = calls().slice(before).find((c) => c.startsWith('install'));
  check('clicking it runs install without pinning a release version (the script then rebuilds from source)', last === 'install llamacpp', JSON.stringify(calls().slice(before)));

  check('no page errors the whole run', errors.length === 0, errors.slice(0, 2).join(' | '));
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
}
const bad = results.filter((x) => !x).length;
console.log(`\n${results.length - bad}/${results.length} checks passed${failed ? ' (aborted by an error)' : ''}`);
process.exit(failed || bad ? 1 : 0);

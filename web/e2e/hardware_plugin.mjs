// UI test for T-1101: a HAL plugin's devices show up on the hardware page without any change above the loader.
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/hardware_plugin.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/hardware_plugin.mjs
//
// Exit code 0 = every check passed. Each check prints PASS / FAIL with what it saw.
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
const pluginDir = path.join(path.dirname(bin), 'plugins');
if (!fs.existsSync(path.join(pluginDir, 'template.so'))) { console.error(`template plugin not built: ${pluginDir}/template.so`); process.exit(2); }

async function withServer(env, fn) {
  const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
  const port = await freePort();
  const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
    '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'], env: { ...process.env, MQDE_SETUP_BACKEND_BIN: path.join(here, 'fake_setup_backend.py'), ...env } });
  const base = `http://127.0.0.1:${port}`;
  try {
    for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }
    await fn(base);
  } finally { server.kill(); }
}

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  await withServer({ MQDE_TEMPLATE_DEVICES: '2' }, async (base) => {
    const api = await (await fetch(`${base}/api/v1/hardware`)).json();
    const tmpl = api.devices.filter((d) => d.vendor === 'template');
    check('the API lists both simulated devices with kind npu and the vendor', tmpl.length === 2 && tmpl.every((d) => d.kind === 'npu'), JSON.stringify(tmpl.map((d) => [d.kind, d.index])));
    check('the built-in CPU is still first', api.devices[0].kind === 'cpu');
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`${base}/hardware`, { waitUntil: 'load' });
    await page.getByText('MQDE Template Accelerator 0').waitFor({ timeout: 20000 });
    const r0 = (await page.getByTestId('probe-device-npu-0').innerText()).replace(/\s+/g, ' ');
    const r1 = (await page.getByTestId('probe-device-npu-1').innerText()).replace(/\s+/g, ' ');
    check('probe card shows device 0: NPU, tmpl1, 16.00 GiB free, vendor', /NPU/.test(r0) && r0.includes('tmpl1') && r0.includes('16.00 GiB') && r0.includes('template'), r0);
    check('probe card shows device 1 separately (12.00 GiB free)', r1.includes('12.00 GiB') && r0 !== r1, r1);
    check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));

    // ---- T-1307: hardware_db profile rows expand into a detail view with a vendor source link ----
    await page.getByTestId('profile-row-nvidia/rtx4090').waitFor({ timeout: 10000 });
    check('detail view is closed before any row is clicked', (await page.getByTestId('profile-detail-nvidia/rtx4090').count()) === 0);
    await page.getByTestId('profile-row-nvidia/rtx4090').click();
    await page.getByTestId('profile-detail-nvidia/rtx4090').waitFor({ timeout: 5000 });
    const detailText = (await page.getByTestId('profile-detail-nvidia/rtx4090').innerText()).replace(/\s+/g, ' ');
    check('detail view shows more than the table row (native formats spelled out)', detailText.includes('原生格式'));
    const sourceHref = await page.getByTestId('profile-source-nvidia/rtx4090').getAttribute('href');
    check('detail view links to the real vendor datasheet from data/hardware_db', sourceHref != null && sourceHref.startsWith('https://'), sourceHref);
    await page.getByTestId('profile-row-nvidia/rtx4090').click();
    check('clicking again closes the detail view', (await page.getByTestId('profile-detail-nvidia/rtx4090').count()) === 0);
    await page.close();
  });
  await withServer({}, async (base) => {
    const api = await (await fetch(`${base}/api/v1/hardware`)).json();
    check('without MQDE_TEMPLATE_DEVICES the plugin reports nothing, so no fake device is listed', api.devices.every((d) => d.vendor !== 'template'));
  });
  await withServer({ MQDE_TEMPLATE_DEVICES: '2', MQDE_HAL_DISABLE: 'template' }, async (base) => {
    const api = await (await fetch(`${base}/api/v1/hardware`)).json();
    check('MQDE_HAL_DISABLE=template hides it even when it has devices', api.devices.every((d) => d.vendor !== 'template'));
  });
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
}
const bad = results.filter((x) => !x).length;
console.log(`\n${results.length - bad}/${results.length} checks passed${failed ? ' (aborted by an error)' : ''}`);
process.exit(failed || bad ? 1 : 0);

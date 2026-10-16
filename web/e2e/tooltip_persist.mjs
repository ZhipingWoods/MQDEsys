// T-D: a hover tooltip showing a numeric value (download progress %) used to flicker instead of staying put
// while the cursor rested on it. Root cause: Meter used the native `title=` attribute, and the download
// job list polls for new progress every second (Fit.tsx HubPanel) -- rewriting `title` while the browser's
// native tooltip is already showing resets its hover timer, so the percentage flashes and vanishes instead
// of updating in place. Fixed by switching to a CSS `data-tip` tooltip (`:hover::after`), which tracks the
// pointer being over the element, not the attribute's identity, so a value update mid-hover cannot dismiss it.
//
// This test drives a real download job through Fit.tsx's "HF 搜索与下载" panel with the network endpoints
// intercepted (same technique newjob_wizard.mjs uses to stay offline), so the 1-second polling loop is the
// real one in the running app, not simulated.
//
//   node web/e2e/tooltip_persist.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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
// a dedicated allow-root, not the shared os.tmpdir() -- other e2e runs leave fixture model directories
// there, and scanning hundreds of them for the local-library table just slows this test down for nothing
const allow = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-tooltip-allow-'));
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-tooltip-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', allow, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));

  // the download: starts at 10% and advances 15 points on every status poll, so each 1-second tick the
  // UI's own interval makes actually changes the number -- the exact condition that used to reset the
  // native tooltip's hover timer and make it flicker
  let pct = 10;
  await page.route('**/api/v1/hf/search*', (route) => route.fulfill({ json: { models: [{ name: 'org/tiny-model', parameters_raw: 5e8, hf_downloads: 100, pipeline_tag: 'text-generation' }] } }));
  // one real endpoint, POST /api/v1/download, serves both "list this repo's weight groups" (body.list=true)
  // and "start downloading one group" -- same as the live server, so branch on the request body like it does
  await page.route('**/api/v1/download', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    const body = route.request().postDataJSON();
    if (body?.list) return route.fulfill({ json: { groups: [{ group: 'safetensors', bytes: 1e9, files: 2, quality: 1 }], chosen: 'safetensors' } });
    route.fulfill({ json: { id: 'dl-1', repo: 'org/tiny-model', group: 'safetensors', dir: '/tmp/x', state: 'running', bytes_done: 1e8, bytes_total: 1e9, files_done: 0, files_total: 2 } });
  });
  await page.route('**/api/v1/download/dl-1/status', (route) => {
    pct = Math.min(99, pct + 15);
    route.fulfill({ json: { id: 'dl-1', repo: 'org/tiny-model', group: 'safetensors', dir: '/tmp/x', state: 'running', bytes_done: Math.round(1e9 * pct / 100), bytes_total: 1e9, files_done: 0, files_total: 2 } });
  });

  await page.goto(`${base}/fit`, { waitUntil: 'load' });
  await page.getByPlaceholder('如 Qwen3 8B GGUF').scrollIntoViewIfNeeded();
  await page.getByPlaceholder('如 Qwen3 8B GGUF').fill('tiny');
  await page.getByRole('button', { name: '查询', exact: true }).click();
  await page.getByText('org/tiny-model').first().waitFor({ timeout: 10000 });
  await page.getByText('org/tiny-model').first().click();
  await page.getByRole('button', { name: '下载', exact: true }).first().waitFor({ timeout: 10000 });
  await page.getByRole('button', { name: '下载', exact: true }).first().click();

  const meter = page.locator('.meter').first();
  await meter.waitFor({ timeout: 10000 });
  check('the meter uses the CSS tooltip attribute, not the native title=', (await meter.getAttribute('title')) === null && (await meter.getAttribute('data-tip')) !== null);

  await meter.hover();
  const tipVisible = () => meter.evaluate((el) => getComputedStyle(el, '::after').content !== 'none' && getComputedStyle(el, '::after').content !== '""');
  await page.waitForFunction((sel) => {
    const el = document.querySelector(sel);
    const c = el && getComputedStyle(el, '::after').content;
    return c && c !== 'none' && c !== '""';
  }, '.meter', { timeout: 5000 });
  check('hovering shows the tooltip (the ::after pseudo-element renders content)', await tipVisible());
  const tip1 = await meter.getAttribute('data-tip');

  // stay hovered (cursor not moved) across two real polling ticks; the value must update AND the tooltip
  // must still be showing both times -- this is the actual bug: it used to disappear right here
  await page.waitForTimeout(1100);
  check('the value changed after one real polling tick while still hovered', (await meter.getAttribute('data-tip')) !== tip1, await meter.getAttribute('data-tip'));
  check('the tooltip is still rendered after that tick (this is what used to flicker away)', await tipVisible());
  const tip2 = await meter.getAttribute('data-tip');

  await page.waitForTimeout(1100);
  check('the value changed again on the next tick', (await meter.getAttribute('data-tip')) !== tip2, await meter.getAttribute('data-tip'));
  check('the tooltip is still rendered after a second tick, continuously hovered the whole time', await tipVisible());

  // moving away and back still works normally (this isn't a tooltip that's stuck permanently visible)
  await page.mouse.move(5, 5);
  await page.waitForTimeout(150);
  check('moving the mouse away hides the tooltip (pure CSS :hover, not stuck on)', !(await tipVisible()));

  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  failed = results.some((r) => !r);
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${String(e.message).split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
  fs.rmSync(runs, { recursive: true, force: true });
  fs.rmSync(allow, { recursive: true, force: true });
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed || results.some((r) => !r) ? 1 : 0);

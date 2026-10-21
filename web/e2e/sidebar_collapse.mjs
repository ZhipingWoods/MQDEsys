// The left navigation can be collapsed to an icon-only rail (button or "[" key), the choice survives a reload,
// a collapsed item still names itself on hover, and the page content gets the freed width.
//
//   node web/e2e/sidebar_collapse.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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
const allow = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-sidebar-allow-'));
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-sidebar-runs-'));
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
  await page.goto(`${base}/jobs`, { waitUntil: 'load' });
  const side = page.getByTestId('sidebar');
  const width = async () => (await side.boundingBox()).width;
  const mainWidth = async () => (await page.locator('main.main').boundingBox()).width;
  const w0 = await width(), m0 = await mainWidth();
  check('expanded by default, labels visible', w0 > 200 && (await page.locator('.nav-label').first().isVisible()), `${w0}px`);

  await page.getByTestId('sidebar-toggle').click();
  await page.waitForTimeout(300); // the width transition
  const w1 = await width(), m1 = await mainWidth();
  check('the button collapses it to an icon rail', w1 < 80 && (await side.getAttribute('data-collapsed')) === '1', `${w1}px`);
  check('labels are hidden, every icon is still there and clickable', !(await page.locator('.nav-label').first().isVisible()) && (await page.locator('.nav a .ico').count()) === 11);
  check('the content area gets the freed width', m1 > m0 + 100, `${m0} -> ${m1}`);
  const link = page.locator('.nav a[href="/compare"]');
  await link.hover();
  const tip = await link.evaluate((el) => getComputedStyle(el, '::after').content);
  check('a collapsed item names itself on hover', tip.includes('配方对比'), tip);
  await link.click();
  await page.waitForURL(/\/compare$/);
  check('navigation still works from the rail', (await side.getAttribute('data-collapsed')) === '1');

  await page.reload({ waitUntil: 'load' });
  check('the choice survives a reload', (await side.getAttribute('data-collapsed')) === '1' && (await width()) < 80);

  await page.locator('body').click({ position: { x: 700, y: 10 } });
  await page.keyboard.press('[');
  await page.waitForTimeout(300);
  check('"[" expands it again', (await side.getAttribute('data-collapsed')) === '0' && (await width()) > 200);
  await page.goto(`${base}/new`, { waitUntil: 'load' });
  const input = page.locator('input').first();
  await input.click();
  await page.keyboard.press('[');
  check('"[" typed into a text field does not toggle the sidebar', (await side.getAttribute('data-collapsed')) === '0');
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  failed = results.some((r) => !r);
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${String(e.message).split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
  fs.rmSync(allow, { recursive: true, force: true });
  fs.rmSync(runs, { recursive: true, force: true });
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed ? 1 : 0);

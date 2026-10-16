// Fixes from the 2026-10-13 frontend audit (every route crawled with real jobs): a stale ?tab= no longer leaves a job
// page blank; the job list and job header name the measured device instead of "GPU –" once the pool slot is gone;
// the catalog is grouped category -> family instead of one flat 200-row table.
//
//   node web/e2e/audit_fixes.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const VLM = '20261013-000000-auditv1', LLM = '20261013-000000-auditl1';
const mk = (id, report, yaml) => {
  const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(report));
  fs.writeFileSync(path.join(d, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), yaml.join('\n') + '\n');
};
const vrep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/vlm_dq.json'), 'utf8'));
vrep.env = { torch: '2.x', device: 'cuda:0', device_name: 'NVIDIA GeForce RTX 4090' };
mk(VLM, vrep, ['metadata: {name: audit-vlm}', `model: {type: vlm, path: ${os.tmpdir()}/m}`, 'quant: {recipe: [w4a16]}', 'deploy: {device: gpu}', `output_dir: ${runs}`]);
const lrep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
lrep.env = { ...(lrep.env ?? {}), quant_device: 'cpu', eval_device: 'cuda:0', quant_device_name: 'cpu', eval_device_name: 'NVIDIA A100-SXM4-80GB' };
mk(LLM, lrep, ['metadata: {name: audit-llm}', `model: {type: llm, path: ${os.tmpdir()}/m}`, 'quant: {recipe: [w4a16, w4a8]}', 'deploy: {device: gpu}', `output_dir: ${runs}`]);

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }

const post1 = (p, b) => fetch(`${base}/api/v1${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) });
const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));

  // ---- ?dry_run=1: validate a job without queueing it ----
  const okJob = { domain: 'llm', model: { path: path.join(os.tmpdir(), 'm'), type: 'llm' }, quant: { recipe: ['w4a16'], algorithm: 'rtn' }, deploy: { device: 'gpu' } };
  const before = (await (await fetch(`${base}/api/v1/jobs`)).json()).length;
  const dv = await post1('/jobs?dry_run=1', okJob);
  check('dry_run on a valid job says valid and 200', dv.status === 200 && (await dv.json()).valid === true);
  const di = await post1('/jobs?dry_run=1', { ...okJob, quant: { recipe: ['nope'] } });
  const dj = await di.json();
  check('dry_run on an invalid job is the same 400 with the problems a real submit gives', di.status === 400 && dj.problems?.[0]?.field === '/quant/recipe', JSON.stringify(dj).slice(0, 160));
  check('and neither queued anything', (await (await fetch(`${base}/api/v1/jobs`)).json()).length === before, String(before));
  const real = await post1('/jobs', { ...okJob, quant: { recipe: ['nope'] } });
  check('without dry_run the invalid job is refused identically', real.status === 400 && (await real.json()).problems?.[0]?.field === '/quant/recipe');

  // ---- device names ----
  const list = await (await fetch(`${base}/api/v1/jobs`)).json();
  check('the job list API carries the measured device name from the report', list.find((j) => j.job_id === VLM)?.device_name === 'NVIDIA GeForce RTX 4090' && list.find((j) => j.job_id === LLM)?.device_name === 'NVIDIA A100-SXM4-80GB', JSON.stringify(list.map((j) => [j.job_id, j.device_name])));
  await page.goto(`${base}/jobs`, { waitUntil: 'load' });
  await page.getByText(VLM).waitFor({ timeout: 15000 });
  const rowV = (await page.locator('tr', { hasText: VLM }).innerText()).replace(/\s+/g, ' ');
  const rowL = (await page.locator('tr', { hasText: LLM }).innerText()).replace(/\s+/g, ' ');
  check('the job list shows the device (shortened), not "–", for a GPU job whose pool slot is forgotten', rowV.includes('RTX 4090') && rowL.includes('A100-SXM4-80GB'), rowV + ' // ' + rowL);
  await page.goto(`${base}/jobs/${VLM}`, { waitUntil: 'load' });
  await page.locator('p.subtitle').waitFor({ timeout: 15000 });
  await page.waitForFunction(() => /RTX 4090|GPU/.test(document.querySelector('p.subtitle')?.textContent ?? ''), null, { timeout: 15000 });
  const sub = await page.locator('p.subtitle').innerText();
  check('the job header names the device instead of "GPU –"', sub.includes('NVIDIA GeForce RTX 4090') && !sub.includes('GPU –'), sub.slice(0, 160));

  // ---- stale ?tab= ----
  await page.goto(`${base}/jobs/${VLM}?tab=W8A16`, { waitUntil: 'load' });  // this job only has W4A16
  await page.getByTestId('vlm-summary-table').waitFor({ timeout: 15000 });
  check('a VLM job with a stale ?tab= falls back to the overview instead of a blank page', true);
  await page.goto(`${base}/jobs/${LLM}?tab=NOPE`, { waitUntil: 'load' });
  await page.getByText('配方汇总').waitFor({ timeout: 15000 });
  check('an LLM job too', true);
  await page.goto(`${base}/jobs/${LLM}?tab=W4A8`, { waitUntil: 'load' });
  await page.getByText('逐层量化误差热力图').first().waitFor({ timeout: 15000 });
  check('a valid ?tab= still opens that recipe', true);

  // ---- catalog grouping ----
  await page.goto(`${base}/catalog`, { waitUntil: 'load' });
  await page.getByTestId('catalog-count').waitFor({ timeout: 15000 });
  await page.waitForFunction(() => /\d+ 条 · \d+ 个类别/.test(document.querySelector('[data-testid="catalog-count"]')?.textContent ?? ''), null, { timeout: 15000 });
  const total = Number((await page.getByTestId('catalog-count').innerText()).match(/(\d+) 条/)[1]);
  const heads = await page.locator('tr.group-head').count();
  const bodyRows = await page.locator('tbody tr').count();
  check(`the catalog opens as ${heads} collapsed category rows, not ${total} model rows`, heads > 1 && bodyRows === heads && total > heads, `${heads} heads, ${bodyRows} rows, ${total} models`);
  const h0 = await page.evaluate(() => document.documentElement.scrollHeight);
  check('collapsed, the page is a fraction of the old 7000-pixel flat table', h0 < 1500, `${h0}px`);
  const first = page.locator('tr.group-head').first();
  const firstN = Number((await first.innerText()).match(/(\d+) 个模型/)[1]);
  await first.click();
  await page.waitForFunction((n) => document.querySelectorAll('tbody tr').length > n, heads, { timeout: 5000 });
  const afterOpen = await page.locator('tbody tr').count();
  const subs = await page.locator('tr.group-sub').count();
  check('opening a category shows its families and models', afterOpen === heads + firstN + subs && subs >= 1, `${afterOpen} rows = ${heads} heads + ${firstN} models + ${subs} family rows`);
  await page.getByTestId('catalog-toggle-all').click();
  await page.waitForFunction((t) => document.querySelectorAll('tbody tr').length >= t, total, { timeout: 5000 });
  check('全部展开 shows every model', (await page.locator('tbody tr').count()) >= total + heads);
  await page.getByPlaceholder('模糊搜索，如 qwen25、llama3（按字符顺序匹配，不用连着打全）').fill('qwen2.5');
  await page.waitForFunction(() => { const c = document.querySelector('[data-testid="catalog-count"]')?.textContent ?? ''; return /^\d+ 条/.test(c) && !c.startsWith(String(document.querySelectorAll('tbody tr').length)); }, null, { timeout: 10000 }).catch(() => {});
  await page.waitForTimeout(600);
  const open = await page.locator('tr.group-head .muted').first().innerText();
  check('a search flattens the groups (every matching model visible, no collapsed category)', open.trim() === '▾' && (await page.getByTestId('catalog-toggle-all').count()) === 0, open);

  // T-B: "qwen25" is not a literal substring of "Qwen2.5-..." (there's a "." in the way) -- only a fuzzy,
  // in-order-characters match finds it. This is the real end-to-end path (server's hub::search, not the
  // unit-level fuzzy_score tests), against the actual catalog fixture.
  const searchBox = page.getByPlaceholder('模糊搜索，如 qwen25、llama3（按字符顺序匹配，不用连着打全）');
  await searchBox.fill('qwen25');
  await page.waitForTimeout(600);
  const fuzzyRows = await page.locator('tbody tr td', { hasText: 'Qwen2.5' }).count();
  check('a fuzzy query with no literal substring in the name ("qwen25") still finds Qwen2.5 models', fuzzyRows > 0, `${fuzzyRows} rows mentioning Qwen2.5`);
  await searchBox.fill('zzzzznosuchmodel');
  await page.waitForTimeout(600);
  const noHitCount = Number((await page.getByTestId('catalog-count').innerText()).match(/(\d+) 条/)?.[1] ?? '-1');
  check('a query matching nothing shows zero results, not everything', noHitCount === 0, String(noHitCount));
  await searchBox.fill('');
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
}
const nbad = results.filter((x) => !x).length;
console.log(`\n${results.length - nbad}/${results.length} checks passed`);
process.exit(failed || nbad ? 1 : 0);

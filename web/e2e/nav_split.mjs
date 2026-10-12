// UI test for T-1306: "误差分析" and "优化建议" promoted from tabs buried in the job detail page into
// independent top-level nav items / standalone routes (/analysis/:jobId, /suggest/:jobId), reachable
// either cold from the sidebar or with context via a link on the job detail page.
//
//   node web/e2e/nav_split.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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

const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
const JOB = '20261008-000002-cccccc';
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const src = path.join(runs, JOB);
fs.mkdirSync(src);
fs.writeFileSync(path.join(src, 'report.json'), JSON.stringify(fixture));
fs.writeFileSync(path.join(src, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
fs.writeFileSync(path.join(src, 'job.yaml'), [
  'metadata: {name: nav-split-fixture}', `model: {type: llm, path: ${os.tmpdir()}/model}`,
  'quant:', '  recipe: [w4a16, w4a8]', '  algorithm: gptq',
  'deploy: {device: cpu}', 'analysis: {sensitivity: block}', `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  // ---- 1. sidebar has the two new top-level items ----
  await page.goto(`${base}/jobs/${JOB}?tab=W4A16`, { waitUntil: 'load' });
  check('sidebar has 误差分析', await page.getByRole('link', { name: '误差分析' }).count() > 0);
  check('sidebar has 优化建议', await page.getByRole('link', { name: '优化建议' }).count() > 0);

  // ---- 2. the job detail page links to both, with the current recipe as context ----
  await page.getByTestId('open-analysis-link').waitFor({ timeout: 10000 });
  check('analysis link carries the job id and recipe', (await page.getByTestId('open-analysis-link').getAttribute('href')) === `/analysis/${JOB}?recipe=W4A16`);
  check('suggest link carries the job id and recipe', (await page.getByTestId('open-suggest-link').getAttribute('href')) === `/suggest/${JOB}?recipe=W4A16`);

  // ---- 3. the standalone /analysis page renders the same error-analysis content, out of the job tabs ----
  await page.getByTestId('open-analysis-link').click();
  await page.waitForURL(`**/analysis/${JOB}?recipe=W4A16`);
  await page.getByTestId('worst-row').first().waitFor({ timeout: 10000 });
  check('standalone /analysis shows worst-layer rows', (await page.getByTestId('worst-row').count()) > 0);
  check('breadcrumb links back to the job', (await page.getByRole('link', { name: JOB }).count()) > 0);

  // ---- 4. the standalone /suggest page renders the sensitivity/mixed-precision suggestion ----
  await page.goto(`${base}/suggest/${JOB}?recipe=W4A16`, { waitUntil: 'load' });
  await page.getByText('敏感度分析').first().waitFor({ timeout: 10000 });
  check('standalone /suggest shows the sensitivity chart and suggestion text', (await page.getByText('预计值是各组单独恢复 FP').count()) > 0);
  check('and the clone-as-new-job button', (await page.getByRole('button', { name: /按建议生成新作业/ }).count()) > 0);

  // ---- 5. cold entry (no job picked) explains itself instead of crashing ----
  await page.goto(`${base}/analysis`, { waitUntil: 'load' });
  check('cold /analysis points at the jobs list', (await page.getByRole('link', { name: /作业与执行/ }).count()) > 0);
  await page.goto(`${base}/suggest`, { waitUntil: 'load' });
  check('cold /suggest points at the jobs list too', (await page.getByRole('link', { name: /作业与执行/ }).count()) > 0);

  const unexpected = errors;
  check('no page errors the whole run', unexpected.length === 0, unexpected.slice(0, 2).join(' | '));
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

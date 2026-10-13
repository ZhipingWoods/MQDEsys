// UI test for the VLM dynamic-quantization suggestion (T-R10): the card lists the layers the job's own per-layer error
// flagged (real MiniCPM-V-4.6 GPTQ report), the clone button submits a job with exactly those skip rules, and the
// standalone 优化建议 page shows the same card for a VLM job (it used to call it a VLA job).
//
//   node web/e2e/vlm_suggest.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const rep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/vlm_dq.json'), 'utf8'));
const sug = rep.recipes.W4A16.summary.suggestion;
const JOB = '20261013-000000-vlmdq1', OLD = '20261013-000000-vlmold';
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-vlm-'));
const mk = (id, report, name) => {
  const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(report));
  fs.writeFileSync(path.join(d, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), [`metadata: {name: ${name}}`, `model: {type: vlm, path: ${MODEL}}`, 'quant: {recipe: [w4a16], algorithm: gptq}', 'calibration: {dataset: synthetic, n_samples: 8}', 'evaluation: {dataset: synthetic, n_samples: 6, max_new_tokens: 12}', `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));
};
mk(JOB, rep, 'vlm-dq');
const old = JSON.parse(JSON.stringify(rep)); delete old.recipes.W4A16.summary.suggestion; // a report written before this feature
mk(OLD, old, 'vlm-old');

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));

  await page.goto(`${base}/jobs/${JOB}`, { waitUntil: 'load' });
  await page.getByText('W4A16 误差分析').first().click();
  await page.getByTestId('vlm-suggest-table').waitFor({ timeout: 20000 });
  const rows = (await page.getByTestId('vlm-suggest-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check(`the card lists the ${sug.layers.length} flagged layers worst first, with their SQNR and the component median`,
    rows.length === sug.layers.length && sug.layers.every((l, i) => rows[i].includes(l.name) && rows[i].includes(l.out_sqnr_db.toFixed(1)) && rows[i].includes(l.median_db.toFixed(1))), rows.join(' // '));
  const card = (await page.getByTestId('vlm-suggest').innerText()).replace(/\s+/g, ' ');
  check('it states the extra weight cost', card.includes(`${(sug.extra_weight_bytes / 2 ** 20).toFixed(2)} MiB`), card.slice(0, 260));
  const btn = page.getByTestId('vlm-suggest-clone');
  check('the clone button names the recipe and the layer count', (await btn.innerText()).includes('W4A16') && (await btn.innerText()).includes(`跳过 ${sug.layers.length} 层`));

  await btn.click();
  await page.waitForURL((u) => u.pathname.startsWith('/jobs/') && !u.pathname.endsWith(JOB), { timeout: 15000 });
  const newId = page.url().split('/jobs/')[1];
  const y = fs.readFileSync(path.join(runs, newId, 'job.yaml'), 'utf8');
  check('a new job was created, named -dq, with the same model and data', /name: vlm-dq-dq/.test(y) && /type: vlm/.test(y) && /dataset: synthetic/.test(y) && /algorithm: gptq/.test(y), y.split('\n').slice(0, 8).join(' | '));
  check('its quant.mixed_precision is exactly the suggested skip rules (anchored regexes)',
    sug.rules.every((r) => y.includes(r.match) || y.includes(r.match.replace(/\\/g, '\\\\'))) && (y.match(/skip: true/g) ?? []).length === sug.rules.length, y.slice(y.indexOf('mixed_precision'), y.indexOf('mixed_precision') + 260).replace(/\n/g, ' | '));
  check('the original job is unchanged', !fs.readFileSync(path.join(runs, JOB, 'job.yaml'), 'utf8').includes('mixed_precision'));

  await page.goto(`${base}/suggest/${JOB}?recipe=W4A16`, { waitUntil: 'load' });
  await page.getByTestId('vlm-suggest').waitFor({ timeout: 15000 });
  const sp = await page.locator('body').innerText();
  check('the standalone 优化建议 page shows the card and does not call it a VLA job', !sp.includes('是 VLA 作业') && (await page.getByTestId('vlm-suggest-table').count()) === 1);

  await page.goto(`${base}/jobs/${OLD}?tab=W4A16`, { waitUntil: 'load' });
  await page.getByTestId('vlm-suggest-none').waitFor({ timeout: 15000 });
  check('a report from before this feature says so instead of showing an empty card', (await page.getByTestId('vlm-suggest').count()) === 0);
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
}
const bad = results.filter((x) => !x).length;
console.log(`\n${results.length - bad}/${results.length} checks passed`);
process.exit(failed || bad ? 1 : 0);

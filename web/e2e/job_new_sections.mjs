// UI test for T-0807: the LLM job page shows the logits-probe table and the lm-eval accuracy table from a report.
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/job_new_sections.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/job_new_sections.mjs
//
// Exit code 0 = every check passed. Each check prints PASS / FAIL with what it saw.
import { spawn, spawnSync } from 'node:child_process';
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

const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_vllm.json'), 'utf8'));
const JOB = '20261009-000000-dddddd';
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));

// Extend the real vLLM report with what newer jobs also write: logits-probe results and lm-eval results, unquantized baseline included.
const rep = JSON.parse(JSON.stringify(fixture));
const rec = rep.recipes.W4A16;
rec.logits = { kl: 0.1327, top1_agreement: 0.7894, ref_top1_prob_drop: 0.0612, positions: 4096, windows: 8, exact: false, k: 256 };
rep.logits_probe = { windows: 8, seq_len: 512, k: 256, exact: false, reference_mib: 12.4 };
const lm = (v1, v2) => ({ status: 'succeeded', version: '0.4.13', limit: 50, num_fewshot: 5,
  tasks: { gsm8k: { alias: 'gsm8k', samples: { original: 1319, effective: 50 }, metrics: {
    'exact_match,strict-match': { metric: 'exact_match', filter: 'strict-match', value: v1, stderr: 0.0571 },
    'exact_match,flexible-extract': { metric: 'exact_match', filter: 'flexible-extract', value: v2, stderr: null } } } } });
rep.baseline_lm_eval = lm(0.2, 0.28);
const fmts = Object.keys(rec.export);
rec.export[fmts[0]].vllm.lm_eval = lm(0.02, 0.06);
if (fmts[1]) rec.export[fmts[1]].vllm.lm_eval = { status: 'failed', error: 'RuntimeError: lm-eval exited with 1\nmore' };
const src = path.join(runs, JOB);
fs.mkdirSync(src);
fs.writeFileSync(path.join(src, 'report.json'), JSON.stringify(rep));
// a finished job also has its HTML pages, written by the real renderer
for (const lang of ['en', 'zh']) {
  const r = spawnSync(process.env.PYTHON ?? 'python3', ['-m', 'mqde.report', 'html', path.join(src, 'report.json'), '--lang', lang], { env: { ...process.env, PYTHONPATH: path.join(repo, 'python') }, encoding: 'utf8' });
  if (r.status !== 0) { console.error('could not render the HTML report:', r.stderr); process.exit(2); }
}
fs.writeFileSync(path.join(src, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
fs.writeFileSync(path.join(src, 'job.yaml'), ['metadata: {name: new-sections}', `model: {type: llm, path: ${os.tmpdir()}/model}`, 'quant: {recipe: [w4a16]}', `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));
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
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`${base}/jobs/${JOB}`, { waitUntil: 'load' });
  await page.getByTestId('logits-card').waitFor({ timeout: 20000 });
  const lc = (await page.getByTestId('logits-W4A16').innerText()).replace(/\s+/g, ' ');
  check('logits table shows KL 0.1327, Top-1 agreement 78.9%, drop 0.0612 and 4,096 positions', lc.includes('0.1327') && lc.includes('78.9%') && lc.includes('0.0612') && lc.includes('4,096'), lc);
  const note = (await page.getByTestId('logits-card').innerText());
  check('the top-k lower-bound caveat is shown, with k and the reference size', note.includes('下界') && note.includes('top-256') && note.includes('12 MiB'), '');

  const card = page.getByTestId('lmeval-card');
  await card.waitFor({ timeout: 5000 });
  const rows = (await card.locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  const nBase = rows.filter((x) => x.startsWith('未量化（基线）')).length;
  check('the unquantized baseline rows come first (2 metrics), before any quantized row', nBase === 2 && rows.slice(0, 2).every((x) => x.startsWith('未量化（基线）')) && rows.some((x) => x.startsWith('未量化（基线）') && x.includes('0.2000 ± 0.0571')), rows.slice(0, 3).join(' // ').slice(0, 160));
  check('a quantized row shows value and standard error, and a missing stderr shows the value alone', rows.some((r) => r.includes('0.0200 ± 0.0571')) && rows.some((r) => /0\.0600( |$)/.test(r) && !r.includes('0.0600 ±')), rows.join(' // ').slice(0, 200));
  check('sample counts and settings are shown (50 samples, lm-eval 0.4.13, limit 50, 5-shot)', rows[0].includes(' 50 ') && rows[0].includes('lm-eval 0.4.13') && rows[0].includes('limit 50') && rows[0].includes('5-shot'), rows[0]);
  check('a failed lm-eval run is a row with its first error line, not a crash', fmts[1] ? rows.some((r) => r.includes('失败') && r.includes('RuntimeError')) : true, fmts[1] ?? 'only one export format');
  check('the caveat about few samples is there', (await card.innerText()).includes('标准误'));
  const html = await (await fetch(`${base}/api/v1/jobs/${JOB}/report.en.html`)).text();
  check('the HTML report of the same job shows the same baseline number (0.2000)', html.includes('0.2000') && html.includes('Unquantized (baseline)'));
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
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

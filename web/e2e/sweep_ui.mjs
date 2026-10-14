// T-R12 UI: the new-job wizard can ask for a concurrency sweep of the served model, and the deploy-job / LLM-job report
// pages draw it (throughput bars + TTFT lines, a per-level table with p50/p99, the saturation level highlighted and
// explained).
//
//   node web/e2e/sweep_ui.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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

// The deploy step is three backend cards (llama.cpp / vLLM / SGLang). `pickBackend` turns exactly that one on, like the dropdown
// this replaced did; 'none' turns them all off. For a reused artifact the cards are radio buttons, so choosing is enough.
async function pickBackend(page, id) {
  const ids = ['vllm', 'sglang', 'llamacpp'];
  if (id !== 'none') {
    const t = page.getByTestId(`backend-toggle-${id}`);
    if ((await t.getAttribute('type')) !== 'radio') for (const b of ids) if (b !== id) { const o = page.getByTestId(`backend-toggle-${b}`); if (await o.isChecked()) await o.uncheck(); }
    await t.check();
  } else for (const b of ids) { const o = page.getByTestId(`backend-toggle-${b}`); if (await o.isChecked()) await o.uncheck(); }
}
const enabledBackends = (page) => page.locator('[data-testid^="backend-card-"]:not(.off) .backend-name').allInnerTexts();
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-model-'));
{ const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000, max_position_embeddings: 512, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const header = { 'model.embed_tokens.weight': { dtype: 'BF16', shape: [1000, 64], data_offsets: [0, 128000] } };
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb])); }
const pt = (c, tps, t50, t99) => ({ concurrency: c, ok: Math.max(16, 2 * c), failed: 0, output_tokens_per_s: tps, ttft_ms: { p50: t50, p99: t99 }, tpot_ms: { p50: 5, p99: 9 }, itl_ms: { p50: 5.5, p99: 12 } });
const sweep = { points: [pt(1, 180, 30, 40), pt(2, 340, 33, 50), pt(4, 610, 40, 80), pt(8, 650, 120, 900), pt(16, 655, 400, 2500)], max_ttft_p99_ms: 1000,
  peak: pt(16, 655, 400, 2500), saturation: { concurrency: 4, reason: 'throughput', next_gain_pct: 100 * (650 / 610 - 1) } };
const DEP = '20261013-000000-sweepd', LLM = '20261013-000000-sweepl', DEP2 = '20261013-000000-sweeper', DEP3 = '20261013-000000-sweepnc';
const mkjob = (id, report, yaml) => { const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(report));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), yaml.join('\n') + '\n'); };
const dep = (sw) => ({ job_id: DEP, kind: 'llm_deploy', verdict: 'succeeded', spec: { name: 'sweep-deploy', model_path: MODEL, artifact: { job: LLM, recipe: 'W4A16', format: 'awq' }, eval_seq_len: 256, eval_windows: 4, device: 'cuda:0' },
  stages: ['validate', 'eval_prepare', 'deploy_vllm', 'report'].map((stage) => ({ stage, status: 'succeeded', seconds: 1 })), artifact: { job: LLM, recipe: 'W4A16', format: 'awq', path: '/x/w4a16-awq', parent_real_ppl: 20.4 },
  data: { eval: { dataset: 'wikitext-2 test', windows: 4, seq_len: 256 } }, deploy: { vllm: { status: 'succeeded', startup_s: 40, served_ppl: 20.45, served_vs_real_ppl_pct: 0.2, sweep: sw } } });
const ymlDeploy = (n) => [`metadata: {name: ${n}}`, `model: {type: llm, path: ${MODEL}}`, `artifact: {job: ${LLM}, recipe: W4A16, format: awq}`, 'deploy: {device: gpu, vllm: {}}', `output_dir: ${runs}`];
mkjob(DEP, dep(sweep), ymlDeploy('sweep-deploy'));
mkjob(DEP2, { ...dep({ error: 'RuntimeError: server died at concurrency 8' }), job_id: DEP2 }, ymlDeploy('sweep-error'));
{ const nc = dep(sweep); nc.job_id = DEP3; nc.artifact = { ...nc.artifact, ppl_comparable: false, parent_eval: { windows: 40, seq_len: 512 } };
  nc.deploy.vllm.served_vs_real_ppl_pct = null; mkjob(DEP3, nc, ymlDeploy('sweep-nc')); }
{ const ok = dep(sweep); ok.job_id = DEP; ok.artifact = { ...ok.artifact, ppl_comparable: true, parent_eval: { windows: 4, seq_len: 256 } }; mkjob(DEP + '-ok', ok, ymlDeploy('sweep-ok')); }
const lrep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
lrep.data = { ...(lrep.data ?? {}), eval: { dataset: 'wikitext-2 test', windows: 40, seq_len: 512 } };
lrep.recipes.W4A16.export.awq.vllm = { status: 'succeeded', startup_s: 30, served_ppl: 20.5, served_vs_real_ppl_pct: 0.1, sweep, load: { ok: 16, failed: 0, spec: { concurrency: 4, requests: 16, prompt_len: 256, n_predict: 64 }, output_tokens_per_s: 600 } };
mkjob(LLM, lrep, ['metadata: {name: sweep-llm}', `model: {type: llm, path: ${MODEL}}`, 'quant: {recipe: [w4a16]}', `output_dir: ${runs}`]);

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
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const until = (fn) => page.waitForFunction(fn, null, { timeout: 10000 });

  // ---- wizard ----
  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  await page.getByTestId('model-path').fill(MODEL);
  check('no sweep option while nothing is deployed', (await page.getByTestId('sweep-options').count()) === 0);
  await pickBackend(page, 'vllm');
  await page.getByTestId('opt-sweep').waitFor({ timeout: 10000 });
  check('picking vLLM offers the concurrency sweep, off by default', !(await page.getByTestId('opt-sweep').isChecked()) && (await preview()).deploy.vllm?.sweep === undefined);
  await page.getByTestId('opt-sweep').check();
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm?.sweep; } catch { return false; } });
  check('checking it puts the default ladder in deploy.vllm.sweep', JSON.stringify((await preview()).deploy.vllm.sweep.levels) === '[1,2,4,8,16]' && (await preview()).deploy.vllm.sweep.requests_per_level === 16 && (await preview()).deploy.vllm.sweep.max_ttft_p99_ms === undefined);
  await page.getByTestId('opt-sweep-levels').fill('2, 8,32');
  await page.getByTestId('opt-sweep-budget').fill('800');
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm?.sweep?.max_ttft_p99_ms === 800; } catch { return false; } });
  check('custom levels and a TTFT p99 budget reach the job', JSON.stringify((await preview()).deploy.vllm.sweep.levels) === '[2,8,32]');
  for (const bad of ['1,1', '0,2', 'a,b', '1,,2', '']) {
    await page.getByTestId('opt-sweep-levels').fill(bad);
    await page.waitForTimeout(150);
    check(`levels "${bad}" are refused: no sweep in the job and submit is disabled`, (await preview()).deploy.vllm.sweep === undefined && (await page.getByRole('button', { name: '提交作业' }).isDisabled()), JSON.stringify((await preview()).deploy.vllm));
  }
  await page.getByTestId('opt-sweep-levels').fill('1,4');
  await pickBackend(page, 'sglang');
  await until(() => { try { const j = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return j?.deploy?.sglang && !j?.deploy?.vllm; } catch { return false; } });
  check('the same sweep goes with SGLang when that is the backend', JSON.stringify((await preview()).deploy.sglang.sweep.levels) === '[1,4]');
  await pickBackend(page, 'llamacpp');
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.llamacpp !== undefined; } catch { return false; } });
  check('llama.cpp has no sweep (llama-server is measured by its own bench): the option disappears', (await page.getByTestId('sweep-options').count()) === 0 && !JSON.stringify((await preview()).deploy).includes('sweep'));

  // ---- reuse mode + derive keeps it ----
  await page.goto(`${base}/new?mode=all&from=${DEP}&step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.artifact; } catch { return false; } });
  check('an artifact (HF) job offers the sweep too', (await page.getByTestId('opt-sweep').count()) === 1);
  await page.getByTestId('opt-sweep').check();
  await page.getByTestId('opt-sweep-levels').fill('1,2');
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm?.sweep; } catch { return false; } });
  check('and writes it into the deploy job', JSON.stringify((await preview()).deploy.vllm.sweep.levels) === '[1,2]');
  await pickBackend(page, 'sglang');
  await until(() => { try { const j = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return j?.deploy?.sglang && !j?.deploy?.vllm; } catch { return false; } });
  check('SGLang on a reused artifact carries the sweep as well', JSON.stringify((await preview()).deploy.sglang.sweep?.levels) === '[1,2]', JSON.stringify((await preview()).deploy));
  await pickBackend(page, 'vllm');
  await until(() => { try { const j = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return j?.deploy?.vllm?.sweep; } catch { return false; } });
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL((u) => u.pathname.startsWith('/jobs/') && !u.pathname.endsWith(DEP), { timeout: 15000 });
  const child = page.url().split('/jobs/')[1];
  const cj = (await (await fetch(`${base}/api/v1/jobs/${child}`)).json()).job;
  check('the server accepted it with the sweep intact', JSON.stringify(cj.deploy.vllm.sweep.levels) === '[1,2]', JSON.stringify(cj.deploy));
  const parent = JSON.parse(fs.readFileSync(path.join(runs, DEP, 'report.json'), 'utf8'));
  fs.writeFileSync(path.join(runs, DEP, 'job.yaml'), ymlDeploy('sweep-deploy').map((l) => l.startsWith('deploy:') ? 'deploy: {device: gpu, vllm: {sweep: {levels: [1, 2, 4], max_ttft_p99_ms: 750}}}' : l).join('\n') + '\n');
  void parent;
  await page.goto(`${base}/new?mode=all&from=${DEP}&step=eval`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm?.sweep; } catch { return false; } });
  const dv = (await preview()).deploy.vllm.sweep;
  check('deriving from a job that swept keeps its levels and budget', JSON.stringify(dv.levels) === '[1,2,4]' && dv.max_ttft_p99_ms === 750, JSON.stringify(dv));

  // ---- the report pages ----
  await page.goto(`${base}/jobs/${DEP}`, { waitUntil: 'load' });
  await page.getByTestId('sweep-card').waitFor({ timeout: 15000 });
  const rows = (await page.getByTestId('sweep-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('one table row per level, ascending', rows.length === 5 && rows.map((r) => r.split(' ')[0]).join(',') === '1,2,4,8,16', rows.join(' // '));
  check('each row has throughput and p50 / p99 for TTFT, TPOT and ITL', rows[2].includes('610') && rows[2].includes('40 ms / 80 ms') && rows[2].includes('5.0 ms / 9.0 ms') && rows[2].includes('5.5 ms / 12 ms'), rows[2]);
  check('the saturation level (4) is the highlighted row', ((await page.getByTestId('sweep-row-4').getAttribute('class')) ?? '').includes('worst') && !((await page.getByTestId('sweep-row-8').getAttribute('class')) ?? '').includes('worst'));
  const verdict = await page.getByTestId('sweep-verdict').innerText();
  check('the verdict says which level, why, the next level\'s gain, the budget and the peak', verdict.includes('建议并发 4') && verdict.includes('不到 10%') && verdict.includes('+6.6%') && verdict.includes('1000 ms') && verdict.includes('655 tok/s @ 并发 16'), verdict);
  check('the chart is drawn (one svg)', (await page.getByTestId('sweep-card').locator('svg').count()) >= 1);
  await page.goto(`${base}/jobs/${DEP2}`, { waitUntil: 'load' });
  await page.getByTestId('sweep-error').waitFor({ timeout: 15000 });
  check('a sweep that failed says so and the rest of the result is still shown', (await page.getByTestId('sweep-error').innerText()).includes('server died at concurrency 8') && (await page.getByTestId('deploy-table').innerText()).includes('20.450'));
  await page.goto(`${base}/jobs/${LLM}`, { waitUntil: 'load' });
  await page.getByTestId('sweep-card').waitFor({ timeout: 15000 });
  check('an LLM job that swept its own export shows the card too, named by recipe / format / backend', (await page.getByTestId('sweep-card').locator('h2').innerText()).includes('W4A16 · awq · vLLM'));
  // ---- like-for-like PPL ----
  await page.goto(`${base}/jobs/${DEP3}`, { waitUntil: 'load' });
  await page.getByTestId('ppl-not-comparable').waitFor({ timeout: 15000 });
  const nct = await page.getByTestId('ppl-not-comparable').innerText();
    check('it names this job\'s and the parent\'s windows x seq_len and does not call the numbers wrong', nct.includes('40 × 512') && nct.includes('不能直接比') && nct.includes('仍是准的'), nct);
  check('and the "vs parent" column shows a dash, not a made-up percentage', (await page.getByTestId('deploy-table').locator('tbody tr').first().innerText()).includes('–'));
  await page.goto(`${base}/jobs/${DEP}-ok`, { waitUntil: 'load' });
  await page.getByTestId('deploy-table').waitFor({ timeout: 15000 });
  check('a comparable one has no warning and shows the percentage', (await page.getByTestId('ppl-not-comparable').count()) === 0 && (await page.getByTestId('deploy-table').innerText()).includes('+0.20%'));
  await page.goto(`${base}/new?mode=all&from=${LLM}&step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('artifact-eval').waitFor({ timeout: 15000 });
  check('in the wizard, reusing an artifact with the parent\'s protocol shows no warning', (await page.getByTestId('artifact-eval-differs').count()) === 0);
  await page.getByTestId('artifact-windows').fill('4');
  await page.getByTestId('artifact-eval-differs').waitFor({ timeout: 5000 });
  const warn = await page.getByTestId('artifact-eval-differs').innerText();
  check('changing the windows warns right there, naming 4 x 512 against the parent\'s 40 x 512', warn.includes('4 窗口 × 512') && warn.includes('40 × 512') && warn.includes('不可比'), warn);
  await page.getByTestId('artifact-windows').fill('40');
  await page.waitForFunction(() => !document.querySelector('[data-testid="artifact-eval-differs"]'), null, { timeout: 5000 });
  await page.getByTestId('artifact-eval').locator('input[type=number]').first().fill('256');  // 序列长度
  await page.getByTestId('artifact-eval-differs').waitFor({ timeout: 5000 });
  check('a different sequence length alone warns as well', (await page.getByTestId('artifact-eval-differs').innerText()).includes('40 窗口 × 256'));
  await page.getByTestId('artifact-eval').locator('input[type=number]').first().fill('512');
  await page.waitForFunction(() => !document.querySelector('[data-testid="artifact-eval-differs"]'), null, { timeout: 5000 });
  check('and the warning goes away when it matches again', true);
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

// T-R13 UI: the Compare page puts the concurrency sweeps of every finished job side by side -- one throughput line per
// served model, a table sorted by peak throughput with the saturation level and TTFT p99, and a warning that two engines'
// numbers are not comparable. Sweeps that failed are not drawn.
//
//   node web/e2e/perf_compare.mjs         # needs: built web/dist, built mqde, Playwright (see demo/)
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
const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-model-'));
{ const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000, max_position_embeddings: 512, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const header = { 'model.embed_tokens.weight': { dtype: 'BF16', shape: [1000, 64], data_offsets: [0, 128000] } };
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb])); }
const pt = (c, tps, t50, t99) => ({ concurrency: c, ok: Math.max(16, 2 * c), failed: 0, output_tokens_per_s: tps, ttft_ms: { p50: t50, p99: t99 }, tpot_ms: { p50: 5, p99: 9 }, itl_ms: { p50: 5.5, p99: 12 } });
const sweep = { points: [pt(1, 180, 30, 40), pt(2, 340, 33, 50), pt(4, 610, 40, 80), pt(8, 650, 120, 900)], max_ttft_p99_ms: 1000,
  peak: pt(8, 650, 120, 900), saturation: { concurrency: 4, reason: 'throughput', next_gain_pct: 6.6 } };
const fast = { points: [pt(1, 300, 20, 30), pt(4, 1100, 25, 60), pt(16, 2900, 90, 400)], peak: pt(16, 2900, 90, 400), saturation: { concurrency: 16, reason: 'not-reached', next_gain_pct: null } };  // the real shape of a sweep that never saturated
const LLM = '20261013-000000-perfllm', DEPV = '20261013-000000-perfdepv', DEPS = '20261013-000000-perfdeps', DEPX = '20261013-000000-perfdepx', DEPE = '20261013-000000-perfdepe';
const mkjob = (id, report) => { const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(report));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), `metadata: {name: ${id}}\nmodel: {type: llm, path: ${MODEL}}\noutput_dir: ${runs}\n`); };
const dep = (id, backend, sw) => ({ job_id: id, kind: 'llm_deploy', verdict: 'succeeded', spec: { name: id, model_path: MODEL, artifact: { job: LLM, recipe: 'W4A16', format: 'awq' }, eval_seq_len: 256, eval_windows: 4, device: 'cuda:0' },
  stages: [], artifact: { job: LLM, recipe: 'W4A16', format: 'awq', path: '/x' }, deploy: { [backend]: { status: 'succeeded', sweep: sw } } });
const lrep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
lrep.recipes.W4A16.export.awq.vllm = { status: 'succeeded', sweep };
mkjob(LLM, lrep);
mkjob(DEPV, dep(DEPV, 'vllm', fast));
mkjob(DEPS, dep(DEPS, 'sglang', { ...fast, peak: { ...fast.peak, output_tokens_per_s: 3300 } }));
mkjob(DEPX, dep(DEPX, 'vllm', { error: 'RuntimeError: server died' }));
mkjob(DEPE, dep(DEPE, 'vllm', { points: [], peak: null, saturation: null }));  // a sweep that measured nothing
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

  await page.goto(`${base}/compare`, { waitUntil: 'load' });
  await page.getByTestId('perf-compare').waitFor({ timeout: 15000 });
  const rows = (await page.getByTestId('perf-compare-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('one row per finished sweep (LLM job + 2 deploy jobs); a failed sweep and an empty one are not drawn', rows.length === 3 && !rows.join().includes(DEPX.slice(-6)) && !rows.join().includes(DEPE.slice(-6)), rows.join(' // '));
  check('sorted by peak throughput, highest first', rows.map((r) => r.split(' ').at(-6)).join(',') === '3300,2900,650' || /3300.*2900.*650/.test(rows.join(' // ')), rows.join(' // '));
  check('each row names engine, peak, the level it peaked at, saturation and TTFT p99',
    rows[0].includes('sglang') && rows[0].includes('3300') && rows[0].includes('16') && rows[0].includes('未饱和') && rows[0].includes('400 ms') &&
    rows[2].includes('vllm') && rows[2].includes('650') && rows[2].includes('900 ms') && / 4 /.test(rows[2]), rows.join(' // '));
  check('a job link goes to the job page', ((await page.getByTestId(`perf-row-${DEPV}-vllm`).locator('a').getAttribute('href')) ?? '').endsWith(`/jobs/${DEPV}`));
  check('two engines in the table: the not-comparable warning is shown', /不可直接比较/.test(await page.getByTestId('perf-engine-note').innerText()));
  check('the chart is drawn', (await page.getByTestId('perf-compare').locator('canvas, svg').count()) > 0);
  const other = await page.getByTestId('perf-compare').screenshot().then((b) => b.length > 1000);
  check('card renders non-blank', other);
  check('no page errors', errors.length === 0, errors.join(' | '));
  failed = results.some((r) => !r);
} finally {
  await browser.close();
  stop();
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed || results.some((r) => !r) ? 1 : 0);

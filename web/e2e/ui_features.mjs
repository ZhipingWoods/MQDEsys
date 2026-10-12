// UI test for T-1101/T-0304: the web UI triggers the new backend features (plugin benchmarks, backend profile in the estimate, job options).
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/ui_features.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/ui_features.mjs
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
if (!fs.existsSync(path.join(pluginDir, 'cpu.so'))) { console.error(`cpu plugin not built in ${pluginDir}`); process.exit(2); }

// A Qwen2-style model directory with header-only safetensors (the estimator never reads the payload).
const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-model-'));
{
  const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 896, intermediate_size: 4864, num_hidden_layers: 24, num_attention_heads: 14,
    num_key_value_heads: 2, vocab_size: 151936, max_position_embeddings: 32768, tie_word_embeddings: true, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const h = cfg.hidden_size, kv = (h / cfg.num_attention_heads) * cfg.num_key_value_heads, i = cfg.intermediate_size;
  const t = { 'model.embed_tokens.weight': [cfg.vocab_size, h], 'model.norm.weight': [h] };
  for (let l = 0; l < cfg.num_hidden_layers; l++) {
    const p = `model.layers.${l}.`;
    Object.assign(t, { [p + 'self_attn.q_proj.weight']: [h, h], [p + 'self_attn.k_proj.weight']: [kv, h], [p + 'self_attn.v_proj.weight']: [kv, h], [p + 'self_attn.o_proj.weight']: [h, h],
      [p + 'mlp.gate_proj.weight']: [i, h], [p + 'mlp.up_proj.weight']: [i, h], [p + 'mlp.down_proj.weight']: [h, i] });
  }
  let off = 0; const header = {};
  for (const [k, shape] of Object.entries(t)) { const n = shape.reduce((a, b) => a * b, 1) * 2; header[k] = { dtype: 'BF16', shape, data_offsets: [off, off + n] }; off += n; }
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb]));
}

const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'], env: { ...process.env, MQDE_TEMPLATE_DEVICES: '1', MQDE_PLUGIN_DIR: pluginDir, MQDE_SETUP_BACKEND_BIN: path.join(here, 'fake_setup_backend.py') } });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }
const post = async (u, b) => (await fetch(`${base}/api/v1${u}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) }));

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

  // ---- 1. hardware page: plugins card and micro-benchmark buttons ----
  await page.goto(`${base}/hardware`, { waitUntil: 'load' });
  await page.getByTestId('plugins-card').waitFor({ timeout: 20000 });
  const api = await (await fetch(`${base}/api/v1/hardware/plugins`)).json();
  check('the plugins card lists every plugin the server reports', (await page.locator('[data-testid^="plugin-status-"]').count()) === api.length, `${api.length} plugins`);
  const statusOf = async (n) => (await page.getByTestId(`plugin-status-${n}`).innerText());
  check('cpu shows as loaded, template (1 simulated device) as loaded', (await statusOf('cpu')) === '已加载' && (await statusOf('template')) === '已加载');
  check('cuda without NVIDIA software shows "no device", not an error', ['未发现设备', '已加载'].includes(await statusOf('cuda')), await statusOf('cuda'));

  await page.getByTestId('bench-bw-cpu').click();
  await page.getByTestId('bench-results-cpu').waitFor({ timeout: 60000 });
  const bw = (await page.getByTestId('bench-results-cpu').innerText()).replace(/\s+/g, ' ');
  const bwNum = parseFloat(/([\d.]+) GB\/s/.exec(bw)?.[1] ?? 'NaN');
  check('the bandwidth button runs the real benchmark and shows a GB/s number', bwNum > 1 && bwNum < 5000, bw);
  await page.getByTestId('bench-gemm-cpu').click();
  await page.waitForFunction(() => /TFLOPS/.test(document.querySelector('[data-testid="bench-results-cpu"]')?.textContent ?? ''), null, { timeout: 120000 });
  const gm = (await page.getByTestId('bench-results-cpu').innerText()).replace(/\s+/g, ' ');
  check('the GEMM button adds a TFLOPS line above the first result (newest first)', /gemm fp32.*TFLOPS/.test(gm) && gm.indexOf('TFLOPS') < gm.indexOf('GB/s'), gm);
  await page.getByTestId('bench-bw-template').click();
  await page.getByTestId('bench-results-template').waitFor({ timeout: 20000 });
  check('the simulated template plugin answers too (500 GB/s by design)', (await page.getByTestId('bench-results-template').innerText()).includes('500'));
  // an error from the server is shown on the row, not swallowed
  await page.locator('[data-testid="plugin-cpu"] input[type=number]').fill('7');
  await page.getByTestId('bench-bw-cpu').click();
  await page.getByTestId('bench-err-cpu').waitFor({ timeout: 20000 });
  check('a benchmark on a device that does not exist shows the server error on that row', (await page.getByTestId('bench-err-cpu').innerText()).includes('no such device'));
  // the API itself rejects bad input
  check('the API rejects an unknown kind (400)', (await post('/hardware/profile', { plugin: 'cpu', device: 0, kind: 'teleport' })).status === 400);
  check('the API says 404 for an unknown plugin', (await post('/hardware/profile', { plugin: 'nope', device: 0, kind: 'bandwidth' })).status === 404);

  // ---- 2. estimate page: backend profile changes the estimate ----
  await page.goto(`${base}/estimate`, { waitUntil: 'load' });
  await page.getByTestId('est-backend').waitFor({ timeout: 20000 });
  const opts = await page.getByTestId('est-backend').locator('option').allInnerTexts();
  check('the backend selector offers hf_transformers and vllm with their activation model', opts.some((o) => o.includes('vllm') && o.includes('按批')) && opts.some((o) => o.includes('hf_transformers') && o.includes('按序列')), opts.join(' | '));
  const input = page.locator('input').first();
  await input.fill(MODEL);
  const estCalls = [];
  page.on('request', (r) => { if (r.url().endsWith('/api/v1/estimate')) estCalls.push(r.postData()); });
  await page.getByTestId('est-backend').selectOption('vllm');
  await page.waitForFunction(() => true, null, {});
  await new Promise((r) => setTimeout(r, 1200));
  check('choosing vllm sends backend=vllm to the estimate endpoint', estCalls.some((b) => b && JSON.parse(b).backend === 'vllm'), `${estCalls.length} request(s)`);
  const ask = async (backend) => (await (await post('/estimate', { model: MODEL, hardware: 'nvidia/rtx4090', recipes: ['w4a16'], backend, runtime: { concurrency: 1, images_per_step: 2, text_tokens: 42, image_resolution: 448, gpu_memory_utilization: 0.9, extra_modalities: [], context_tokens: 4096 } })).json()).estimates[0].max_concurrency;
  const [cHf, cVl] = [await ask('hf_transformers'), await ask('vllm')];
  check('the two backends give different max concurrency (batched is far higher)', cVl > 2 * cHf, `hf ${cHf}, vllm ${cVl}`);

  // ---- 3. new job: the options reach the submitted job ----
  await page.goto(`${base}/new`, { waitUntil: 'load' });
  await page.getByTestId('domain-tab-llm').click();
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  check('options are off by default: no analysis, no vllm in the job', (await preview()).analysis === undefined && (await preview()).deploy.vllm === undefined);
  await page.getByTestId('opt-probe').check();
  check('logits probe on adds analysis.logits_probe', JSON.stringify((await preview()).analysis?.logits_probe) === '{}');
  await page.getByTestId('opt-probe-exact').check();
  check('exact KL adds exact: true', (await preview()).analysis.logits_probe.exact === true);
  await page.getByTestId('opt-vllm').check();
  await page.getByTestId('opt-vllm-util').fill('0.4');
  let j = await preview();
  check('vLLM on adds deploy.vllm with the chosen memory utilization', j.deploy.vllm?.gpu_memory_utilization === 0.4 && j.deploy.vllm.lm_eval === undefined, JSON.stringify(j.deploy.vllm));
  await page.getByTestId('opt-lmeval').check();
  await page.getByTestId('opt-lmeval-tasks').fill('gsm8k, arc_easy');
  j = await preview();
  check('lm-eval adds its tasks (split and trimmed), limit and few-shot under deploy.vllm', JSON.stringify(j.deploy.vllm.lm_eval.tasks) === '["gsm8k","arc_easy"]' && j.deploy.vllm.lm_eval.limit === 50 && j.deploy.vllm.lm_eval.num_fewshot === 5, JSON.stringify(j.deploy.vllm.lm_eval));
  await page.getByTestId('opt-vllm').uncheck();
  check('switching vLLM off removes lm-eval with it', (await preview()).deploy.vllm === undefined);
  // submit through the real server: the job is accepted and its stored spec carries the options
  await page.getByTestId('opt-vllm').check();
  await page.getByTestId('opt-lmeval').check();
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL(/\/jobs\/\d{8}-\d{6}-[0-9a-f]{6}$/, { timeout: 30000 });
  const id = page.url().split('/').pop();
  const stored = (await (await fetch(`${base}/api/v1/jobs/${id}`)).json()).job;
  check('the server stored the submitted job with logits_probe, vllm and lm_eval', stored?.analysis?.logits_probe?.exact === true && stored?.deploy?.vllm?.lm_eval?.tasks?.[0] === 'gsm8k', JSON.stringify([stored?.analysis, stored?.deploy?.vllm]).slice(0, 160));
  check('no page errors during the whole run', errors.filter((e) => !/403|404|400|422/.test(e)).length === 0, errors.filter((e) => !/403|404|400|422/.test(e)).slice(0, 2).join(' | '));
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

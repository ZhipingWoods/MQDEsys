// UI test for the decoupled pipeline modes: full, quantize-only (with a reusable export) and
// evaluate-only against an already-quantized checkpoint.
//
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/pipeline_modes.mjs
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
  console.error('SKIP  Playwright is not installed under demo/');
  process.exit(3);
}

const bin = path.resolve(repo, process.env.MQDE_BIN ?? 'build/dev/bin/mqde');
if (!fs.existsSync(bin)) { console.error(`mqde binary not found: ${bin}`); process.exit(2); }

const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-model-'));
{
  const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 896, intermediate_size: 4864, num_hidden_layers: 2,
    num_attention_heads: 14, num_key_value_heads: 2, vocab_size: 151936, max_position_embeddings: 32768, tie_word_embeddings: true, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const h = cfg.hidden_size, kv = (h / cfg.num_attention_heads) * cfg.num_key_value_heads, i = cfg.intermediate_size;
  const t = { 'model.embed_tokens.weight': [cfg.vocab_size, h], 'model.norm.weight': [h] };
  for (let l = 0; l < cfg.num_hidden_layers; l++) {
    const p = `model.layers.${l}.`;
    Object.assign(t, { [p + 'self_attn.q_proj.weight']: [h, h], [p + 'self_attn.k_proj.weight']: [kv, h], [p + 'self_attn.v_proj.weight']: [kv, h],
      [p + 'self_attn.o_proj.weight']: [h, h], [p + 'mlp.gate_proj.weight']: [i, h], [p + 'mlp.up_proj.weight']: [i, h], [p + 'mlp.down_proj.weight']: [h, i] });
  }
  let off = 0; const header = {};
  for (const [k, shape] of Object.entries(t)) { const n = shape.reduce((a, b) => a * b, 1) * 2; header[k] = { dtype: 'BF16', shape, data_offsets: [off, off + n] }; off += n; }
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb]));
}
// A plausible already-quantized checkpoint directory: only the layout the detector reads is needed here.
const QUANT = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-q4-'));
fs.writeFileSync(path.join(QUANT, 'config.json'), JSON.stringify({ model_type: 'qwen2', quantization_config: { quant_method: 'gptq' } }));

const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)],
  { stdio: ['ignore', 'ignore', 'ignore'], env: { ...process.env, MQDE_SETUP_BACKEND_BIN: path.join(here, 'fake_setup_backend.py') } });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const untilPreview = (src) => page.waitForFunction((body) => {
    try { return new Function('j', `return (${body});`)(JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')); } catch { return false; }
  }, src, { timeout: 10000 });

  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  await page.getByTestId('job-mode').waitFor({ timeout: 15000 });
  await page.getByTestId('model-path').fill(MODEL);
  await page.waitForTimeout(600); // let the first debounced estimate (against the default path) settle
  errors.length = 0;
  check('the LLM wizard offers three explicit modes', (await page.getByTestId('job-mode-full').count()) === 1 && (await page.getByTestId('job-mode-quantize').count()) === 1 && (await page.getByTestId('job-mode-eval').count()) === 1);

  // ---- evaluate-only: no quant section, explicit stages, quantized_path, server accepts it ----
  await page.getByTestId('job-mode-eval').click();
  await page.getByTestId('quantized-path').waitFor({ timeout: 5000 });
  await page.getByTestId('quantized-path').fill(QUANT);
  await untilPreview('j.stages?.[0] === "evaluate" && !!j.model?.quantized_path');
  const je = await preview();
  check('evaluate-only sends stages [evaluate] and model.quantized_path', JSON.stringify(je.stages) === '["evaluate"]' && je.model.quantized_path === QUANT, JSON.stringify(je.stages));
  check('and no quant section at all', je.quant === undefined);
  check('offline benchmarks are offered in evaluate-only mode', await page.getByTestId('offline-benchmarks').isVisible());
  check('and the JobSpec carries them', (je.evaluation?.benchmarks ?? []).includes('wikitext2_ppl'), JSON.stringify(je.evaluation?.benchmarks));
  check('the focused evaluate-only page omits the quantize steps entirely', (await page.getByTestId('quant-steps').count()) === 0 && (await page.getByTestId('step-tab-2').count()) === 0);
  const de = await fetch(`${base}/api/v1/jobs?dry_run=1`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(je) });
  check('the preview is accepted by the real planner', de.ok, `${de.status} ${await de.text()}`);
  // The requirement is "only name the dataset and the quantized checkpoint": the base model path is optional.
  await page.getByTestId('eval-advanced').locator('summary').click();
  await page.getByTestId('model-path').fill('');
  await untilPreview('j.stages?.[0] === "evaluate" && j.model?.path === "" && !!j.model?.quantized_path');
  const de2 = await fetch(`${base}/api/v1/jobs?dry_run=1`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(await preview()) });
  check('and it still works with only the quantized checkpoint + eval dataset', de2.ok && await page.getByRole('button', { name: '提交作业' }).isEnabled(), `${de2.status}`);
  await page.getByTestId('model-path').fill(MODEL);

  // ---- quantize-only: a real checkpoint gets written (export), no evaluation section ----
  await page.getByTestId('job-mode-quantize').click();
  await untilPreview('j.stages?.includes("quantize") && j.evaluation === undefined');
  const jq = await preview();
  check('quantize-only sends [quantize, export] and keeps the export format', JSON.stringify(jq.stages) === '["quantize","export"]' && (jq.export?.formats ?? []).length > 0, JSON.stringify(jq.stages));
  check('and has no evaluation section', jq.evaluation === undefined);
  check('the export card explains where the weights land', (await page.getByTestId('quantize-export').innerText()).includes('runs/'));
  check('the serving section is hidden', (await page.getByTestId('backend-cards').count()) === 0);
  const dq = await fetch(`${base}/api/v1/jobs?dry_run=1`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(jq) });
  check('quantize-only is accepted by the planner', dq.ok, `${dq.status} ${await dq.text()}`);

  // ---- back to the full pipeline ----
  await page.getByTestId('job-mode-full').click();
  await untilPreview('j.stages?.includes("quantize") && j.stages?.includes("evaluate")');
  const jf = await preview();
  check('full restores quantize + evaluate + export', jf.stages.includes('quantize') && jf.stages.includes('evaluate') && jf.stages.includes('export'), JSON.stringify(jf.stages));

  check('no page errors', errors.length === 0, errors.join(' | '));
  if (results.includes(false)) failed = true;
} finally {
  await browser.close();
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed ? 1 : 0);

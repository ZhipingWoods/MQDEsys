// T-C: the "量化误差预览" (quantization error preview) silently failed for almost every ordinary-sized
// weight matrix, because the size cap meant for cheap histogram sampling (4,000,000 elements, element-level
// striding) was reused as the exactness gate for the preview itself. Any 2-D weight bigger than that -- most
// real linear layers -- got sampled, failed the equality check, and the preview was skipped entirely. The
// frontend then showed "not supported", which misdiagnosed a size problem as an unsupported-tensor problem.
//
// This fixture's mlp.up_proj is 2048x2048 = 4,194,304 elements, just over the old 4,000,000 threshold: before
// the fix this tensor got no preview at all; after the fix it gets an exact one (it is well under the new
// 64,000,000-element cap used for an explicitly requested preview).
//
//   node web/e2e/quant_preview.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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

// BF16: the top 16 bits of the IEEE-754 float32 bit pattern (truncated, not rounded -- plenty exact enough
// for a UI smoke test; the C++ unit tests pin the real rounding behavior).
const f32buf = Buffer.alloc(4);
function bf16(v) {
  f32buf.writeFloatLE(v);
  return f32buf.readUInt16LE(2);
}
function writeMatrixBf16(rows, cols, fn) {
  const out = Buffer.alloc(rows * cols * 2);
  let i = 0;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out.writeUInt16LE(bf16(fn(r, c)), (i++) * 2);
  return out;
}
// small, varied, deterministic weight-like values (no true zero rows/cols, which would be a degenerate
// quantization case and isn't what this test is about)
const val = (r, c) => Math.sin((r * 97 + c * 13) * 0.01) * 0.05;

const ROOT = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-qp-'));
const MODEL = path.join(ROOT, 'tiny-big-layer');
fs.mkdirSync(MODEL);
const H = 2048, I = 2048, KV = 512, VOCAB = 64; // mlp.up_proj = H x I = 4,194,304 elements > the old 4M bug threshold
{
  const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: H, intermediate_size: I, num_hidden_layers: 1,
    num_attention_heads: 16, num_key_value_heads: 4, vocab_size: VOCAB, max_position_embeddings: 4096, tie_word_embeddings: true, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const tensors = {
    'model.embed_tokens.weight': [VOCAB, H],
    'model.norm.weight': [H],
    'model.layers.0.self_attn.q_proj.weight': [H, H],
    'model.layers.0.self_attn.k_proj.weight': [KV, H],
    'model.layers.0.self_attn.v_proj.weight': [KV, H],
    'model.layers.0.self_attn.o_proj.weight': [H, H],
    'model.layers.0.mlp.gate_proj.weight': [I, H],
    'model.layers.0.mlp.up_proj.weight': [I, H],   // the one over the old 4M threshold
    'model.layers.0.mlp.down_proj.weight': [H, I],
    'model.layers.0.input_layernorm.weight': [H],
    'model.layers.0.post_attention_layernorm.weight': [H],
  };
  const buffers = {};
  for (const [name, shape] of Object.entries(tensors)) {
    const [rows, cols] = shape.length === 2 ? shape : [1, shape[0]];
    buffers[name] = writeMatrixBf16(rows, cols, val);
  }
  let off = 0; const header = {};
  const parts = [];
  for (const [name, shape] of Object.entries(tensors)) {
    const buf = buffers[name];
    header[name] = { dtype: 'BF16', shape, data_offsets: [off, off + buf.length] };
    off += buf.length;
    parts.push(buf);
  }
  const hb = Buffer.from(JSON.stringify(header));
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb, ...parts]));
}

const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-qp-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', ROOT, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

// ---------------------------------------------------------------- API-level: the real bug, directly
{
  const r = await fetch(`${base}/api/v1/models:tensor`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: MODEL, tensor: 'model.layers.0.mlp.up_proj.weight', recipe: 'w4a16' }) });
  const j = await r.json();
  check('a weight matrix just over the old 4,000,000-element threshold gets a preview, not "unsupported"',
    r.status === 200 && !!j.quant_preview, JSON.stringify(j).slice(0, 200));
  check('it is exact (well under the new cap), not an approximation', j.quant_preview?.exact === true, JSON.stringify(j.quant_preview));
  check('rows_sampled equals rows_total when exact', j.quant_preview?.rows_sampled === I && j.quant_preview?.rows_total === I, JSON.stringify(j.quant_preview));
  check('the metrics look like real numbers, not placeholders', isFinite(j.quant_preview?.metrics?.cosine) && j.quant_preview.metrics.cosine > 0.9, JSON.stringify(j.quant_preview?.metrics));
}
{
  // a non-quantizable-shape case (1-D) must still correctly report no preview, same as before this fix
  const r = await fetch(`${base}/api/v1/models:tensor`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: MODEL, tensor: 'model.norm.weight', recipe: 'w4a16' }) });
  const j = await r.json();
  check('a 1-D tensor still has no quant_preview (row-sampling only applies to 2-D weights)', r.status === 200 && !j.quant_preview, JSON.stringify(j).slice(0, 150));
}

// ---------------------------------------------------------------- through the real UI
const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));

  await page.goto(`${base}/graph`, { waitUntil: 'load' });
  await page.locator('select').first().waitFor({ timeout: 15000 });
  const modelOpt = await page.locator('select').first().locator('option', { hasText: 'tiny-big-layer' }).first().getAttribute('value');
  await page.locator('select').first().selectOption(modelOpt);
  // the default-focus effect auto-selects the single block once the model's graph loads, so the mlp nodes
  // render without any tree interaction -- just wait for them
  await page.locator('.gnode', { hasText: 'up_proj' }).first().waitFor({ timeout: 15000 });
  await page.locator('.gnode', { hasText: 'mlp.up_proj' }).first().click();
  // clicking the node auto-selects its "weight" tensor (Props' own effect), no need to click the tensor table row
  await page.getByText('量化误差预览').waitFor({ timeout: 10000 });
  await page.waitForFunction(() => /SQNR|不支持预览/.test(document.body.innerText), null, { timeout: 15000 });
  const bodyText = await page.locator('body').innerText();
  check('the real UI shows the preview metrics, not "不支持预览"', /SQNR/.test(bodyText) && !/不支持预览/.test(bodyText), bodyText.includes('SQNR') ? 'has SQNR' : 'missing SQNR');
  check('no "approximate" note for a tensor that was read exactly', (await page.getByTestId('quant-preview-approx').count()) === 0);
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  failed = results.some((r) => !r);
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${String(e.message).split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
  fs.rmSync(ROOT, { recursive: true, force: true });
  fs.rmSync(runs, { recursive: true, force: true });
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed || results.some((r) => !r) ? 1 : 0);

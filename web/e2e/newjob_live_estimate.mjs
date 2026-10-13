// UI test for T-0304: the new-job page's live estimate sidebar (same C++ estimator as the estimate page, debounced).
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/newjob_live_estimate.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/newjob_live_estimate.mjs
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


// A Qwen2-style model directory with a config and a header-only safetensors file (the estimator never reads the payload).
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
      [p + 'mlp.gate_proj.weight']: [i, h], [p + 'mlp.up_proj.weight']: [i, h], [p + 'mlp.down_proj.weight']: [h, i], [p + 'input_layernorm.weight']: [h], [p + 'post_attention_layernorm.weight']: [h] });
  }
  let off = 0; const header = {};
  for (const [k, shape] of Object.entries(t)) { const n = shape.reduce((a, b) => a * b, 1) * 2; header[k] = { dtype: 'BF16', shape, data_offsets: [off, off + n] }; off += n; }
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb]));
}
const HW = 'nvidia/rtx4090';
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

const RT = { concurrency: 1, images_per_step: 2, text_tokens: 42, image_resolution: 448, gpu_memory_utilization: 0.9, extra_modalities: [] };
const api = async (recipes, hw = HW, rt = RT) => (await (await fetch(`${base}/api/v1/estimate`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: MODEL, hardware: hw, recipes, runtime: rt }) })).json());
// The page prints fmtBytes(x): reproduce its rounding from the raw number so the comparison is on what a user reads.
const fmtBytes = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0; let v = b; while (Math.abs(v) >= 1024 && i < u.length - 1) { v /= 1024; i++; } return `${v.toFixed(i === 0 ? 0 : 2)} ${u[i]}`; };

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const calls = [];
  page.on('request', (r) => r.url().endsWith('/api/v1/estimate') && calls.push(Date.now()));
  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  const side = page.getByTestId('live-estimate');
  await side.waitFor({ timeout: 20000 });
  // The wizard opens on LLM; this test compares the four VLA default recipes, so pick that tab explicitly.
  await page.getByTestId('domain-tab-vla').click();
  const input = page.getByTestId('model-path');
  await input.fill(MODEL);
  await page.waitForSelector('[data-testid="live-w4a16"]', { timeout: 20000 });

  const want = await api(['w8a16', 'fp8', 'w4a16', 'w4a8']);
  check('the sidebar has a row for each picked recipe', (await side.locator('tbody tr').count()) === 4, String(await side.locator('tbody tr').count()));
  let ok = true;
  const seen = [];
  for (const e of want.estimates) {
    const row = (await side.getByTestId(`live-${e.recipe_id}`).innerText()).replace(/\s+/g, ' ');
    seen.push(row);
    ok = ok && row.includes(fmtBytes(e.breakdown.weights)) && row.includes(fmtBytes(e.peak_bytes)) && row.includes(e.feasible ? '✓' : '✗');
  }
  check('weights, peak and feasibility equal the estimate endpoint\'s numbers', ok, seen[0]);
  const w = want.estimates.map((e) => e.breakdown.weights);
  check('the estimates differ between recipes (the sidebar is not static text)', new Set(w).size > 1, w.join(','));

  // picking another recipe changes the rows; unpicking removes one
  const before = calls.length;
  await page.getByText('w2a16', { exact: false }).first().click();
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="live-estimate"] tbody tr').length === 5, null, { timeout: 20000 });
  const want5 = await api(['w8a16', 'fp8', 'w4a16', 'w4a8', 'w2a16']);
  const r5 = (await side.getByTestId(`live-${want5.estimates[4].recipe_id}`).innerText()).replace(/\s+/g, ' ');
  check('adding a recipe adds its row with the right weights', r5.includes(fmtBytes(want5.estimates[4].breakdown.weights)), r5);
  check('that took a request (and not one per keystroke)', calls.length > before, `${calls.length - before} request(s)`);

  // debounce: typing many characters quickly causes one request, not one per character
  const MODEL2 = `${MODEL}-b`;
  fs.cpSync(MODEL, MODEL2, { recursive: true });
  await input.fill('');
  await new Promise((r) => setTimeout(r, 500));
  const n0 = calls.length;
  await input.pressSequentially(MODEL2, { delay: 5 });
  await page.waitForSelector('[data-testid="live-w4a16"]', { timeout: 20000 });
  await new Promise((r) => setTimeout(r, 600));
  check('typing a path character by character sends few requests (debounced)', calls.length - n0 <= 3 && calls.length - n0 >= 1, `${calls.length - n0} for ${MODEL2.length} characters`);

  // hardware changes capacity: a small GPU makes the large recipe infeasible or the numbers differ
  const hwSelect = page.getByTestId('hw-select');
  const options = await hwSelect.locator('option').evaluateAll((os) => os.map((o) => o.value));
  const other = options.find((o) => o !== HW);
  if (other) {
    await hwSelect.selectOption(other);
    await page.waitForFunction((h) => document.querySelector('[data-testid="live-estimate"] .note')?.textContent?.includes(h), other, { timeout: 20000 });
    const wantO = await api(['w8a16', 'fp8', 'w4a16', 'w4a8', 'w2a16'], other);
    const cap = fmtBytes(wantO.estimates[0].capacity_bytes);
    await page.waitForFunction((c) => document.querySelector('[data-testid="live-estimate"]')?.innerText.includes(c), cap, { timeout: 20000 });
    check(`switching hardware to ${other} shows that hardware's capacity (${cap})`, true);
  } else check('a second hardware profile exists to switch to', false);

  // an unreadable model shows the server's error in the sidebar, and clearing recipes empties it
  await input.fill('/definitely/not/a/model');
  await page.waitForSelector('[data-testid="live-estimate-error"]', { timeout: 20000 });
  check('a bad model path shows an error in the sidebar', (await page.getByTestId('live-estimate-error').innerText()).length > 0, (await page.getByTestId('live-estimate-error').innerText()).slice(0, 80));
  check('the job form itself is still usable after that error', await page.getByRole('button', { name: '提交作业' }).isEnabled());
  const unexpected = errors.filter((e) => !/403 \(Forbidden\)/.test(e)); // from the empty / nonexistent paths this test types on purpose
  check('no console or page errors besides that deliberate 403', unexpected.length === 0, unexpected.slice(0, 2).join(' | '));
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

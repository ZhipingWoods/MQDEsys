// UI test for the new-job wizard's 部署建议 card: POST /recommend ranks recipe x backend for the target device by the
// chosen objective, shows the accuracy risk -- measured by a finished job on the same model (a real Qwen2.5-0.5B
// report, python/tests/fixtures/reports/llm_full.json: W4A16 +12.7% PPL, W4A8 +21.2%) or a bit-width prior --
// and 采用 fills the wizard with that option.
//
//   node web/e2e/advice.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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

// A real (if tiny) safetensors header, same shape the other new-job e2e tests use -- the estimator reads
// tensor shapes/dtypes from this header and errors on anything else, so a placeholder zero-filled file
// (tried first, then fixed once the live-estimate sidebar's request against it came back 400) will not do.
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

const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const JOB = '20261013-000000-advice';
const rep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
rep.spec.model_path = MODEL;
fs.mkdirSync(path.join(runs, JOB));
fs.writeFileSync(path.join(runs, JOB, 'report.json'), JSON.stringify(rep));
fs.writeFileSync(path.join(runs, JOB, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
fs.writeFileSync(path.join(runs, JOB, 'job.yaml'), [`metadata: {name: measured}`, `model: {type: llm, path: ${MODEL}}`, 'quant: {recipe: [w4a16, w4a8]}', `output_dir: ${runs}`, ''].join('\n'));
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
  const bad = await fetch(`${base}/api/v1/recommend`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: MODEL, objective: 'fast' }) });
  check('an unknown objective is a 400, not silently the default', bad.status === 400, String(bad.status));

  const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const rows = async () => (await page.getByTestId('advice-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  const generate = async (objective) => {
    await page.getByTestId('advice-objective').selectOption(objective);
    const resp = page.waitForResponse((r) => r.url().endsWith('/api/v1/recommend') && r.request().method() === 'POST', { timeout: 15000 });
    await page.getByTestId('advice-run').click();
    const body = await (await resp).json();
    await page.waitForFunction((n) => document.querySelectorAll('[data-testid="advice-table"] tbody tr').length === n && !document.querySelector('[data-testid="advice-run"]')?.disabled, body.top.length, { timeout: 10000 });
    return rows();
  };

  await page.goto(`${base}/new`, { waitUntil: 'load' });
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('advice-card').waitFor({ timeout: 10000 });
  check('the 部署建议 card is in step ①, before the algorithm', true);

  const tp = await generate('throughput');
  const conc = (r) => Number(r.split(' ').filter((x) => /^\d+$/.test(x)).slice(-1)[0]);
  const concs = tp.map(conc);
  check('throughput: ordered by how many sequences fit, most first', concs.every((c, i) => i === 0 || c <= concs[i - 1]) && concs[0] === Math.max(...concs), concs.join(','));

  const bal = await generate('balanced');
  const riskOf = (r) => (r.includes(' 高 ') ? 'high' : r.includes(' 中 ') ? 'medium' : r.includes(' 低 ') ? 'low' : '?');
  const firstHigh = bal.findIndex((r) => riskOf(r) === 'high');
  check('balanced: no high-risk option comes before a usable one', firstHigh === -1 || bal.slice(firstHigh).every((r) => riskOf(r) === 'high'), bal.map(riskOf).join(','));
  const w4a16 = bal.find((r) => r.includes('W4A16') && r.includes('vllm'));
  check('W4A16 shows the measured +12.74% PPL from the finished job, as medium risk', !!w4a16 && w4a16.includes('实测：PPL +12.74%') && riskOf(w4a16) === 'medium', w4a16);
  const w4a8 = bal.find((r) => r.includes('W4A8'));
  check('W4A8 is high risk because it measured +21.15% (its bit widths alone would say medium)', !!w4a8 && w4a8.includes('+21.15%') && riskOf(w4a8) === 'high', w4a8);
  check('a recipe never run on this model says it is a prior', bal.some((r) => r.includes('经验值')), '');
  const link = page.getByTestId('advice-table').locator(`a[href="/jobs/${JOB}"]`).first();
  const nLinks = await page.getByTestId('advice-table').locator(`a[href="/jobs/${JOB}"]`).count();
  check('every measured row links to the job it came from', (await link.count()) === 1 && nLinks === bal.filter((r) => r.includes('实测：')).length, `${nLinks} links`);

  const acc = await generate('accuracy');
  check('accuracy: a low-risk option comes first', riskOf(acc[0]) === 'low', acc[0]);
  check('balanced on this small model: memory is not the limit, so a low-risk recipe beats W4A16 (+12.7% PPL, 1% more sequences)', riskOf(bal[0]) === 'low', bal[0]);

  // adopt the measured W4A16 + vllm row
  const all = await rows();
  const idx = all.findIndex((r) => r.includes('W4A16') && r.includes('vllm'));
  check('accuracy still lists W4A16 on vLLM', idx >= 0, all.join(' // ').slice(0, 300));
  await page.getByTestId(`advice-adopt-${idx}`).click();
  await page.waitForFunction(() => { try { const j = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return j?.quant?.recipe?.length === 1 && j?.quant?.recipe?.[0] === 'w4a16'; } catch { return false; } }, null, { timeout: 10000 });
  const j = await preview();
  check('采用 sets the job to just that recipe and its algorithm', JSON.stringify(j.quant.recipe) === '["w4a16"]' && j.quant.algorithm === 'gptq', JSON.stringify(j.quant));
  await page.waitForFunction(() => { try { const j = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return j?.deploy?.vllm !== undefined && j?.export?.formats?.[0] === 'awq'; } catch { return false; } }, null, { timeout: 10000 }).catch(() => {});
  const j2 = await preview();
  check('and turns on vLLM deployment', j2.deploy.vllm !== undefined, JSON.stringify(j2.deploy));
  check('with the export format the planner suggests for W4A16 on vLLM (awq)', JSON.stringify(j2.export?.formats) === '["awq"]', JSON.stringify(j2.export));
  check('and the target hardware of that option', j2.deploy.hardware === 'nvidia/rtx4090', j2.deploy.hardware);

  await page.getByTestId('advice-backend-vllm').click();
  const hfOnly = await generate('balanced');
  check('deselecting vLLM leaves only hf_transformers options', hfOnly.length > 0 && hfOnly.every((r) => r.includes('hf_transformers') && !r.includes(' vllm ')), hfOnly.slice(0, 2).join(' // '));
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

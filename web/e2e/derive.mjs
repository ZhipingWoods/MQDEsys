// UI test for 作业派生: a finished job's page offers 换评估数据 / 换设备 / 换推理框架 / 换配方 / 复制全部参数, each opening the
// wizard with every field of that job's job.yaml carried over (including mixed-precision rules the wizard has no
// control for, in their order) and the chosen step in focus; the submitted job records parent_job.
//
//   node web/e2e/derive.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const yaml = (o, ind = '') => Object.entries(o).map(([k, v]) => Array.isArray(v)
  ? `${ind}${k}:\n` + v.map((x) => typeof x === 'object' ? `${ind}  - ` + yaml(x, ind + '    ').trimStart() : `${ind}  - ${JSON.stringify(x)}`).join('\n')
  : v && typeof v === 'object' ? `${ind}${k}:\n${yaml(v, ind + '  ')}` : `${ind}${k}: ${JSON.stringify(v)}`).join('\n');
const norm = (rules) => JSON.stringify((rules ?? []).map((r) => [r.match, r.bits ?? null, !!r.skip]));
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const LLM = '20261013-000000-parent', VLM = '20261013-000000-vparent', CANON = '20261013-000000-cparent';
const llmJob = {
  domain: 'llm', metadata: { name: 'qwen-p' }, model: { source: 'local', path: MODEL, type: 'llm' },
  quant: { recipe: ['w4a16'], algorithm: 'gptq', act_static: null, kv: { format: 'int', bits: 4, symmetric: false, method: 'naive' },
    mixed_precision: { rules: [{ match: 'self_attn\\.o_proj$', bits: 8 }, { match: 'block:0', bits: 8 }, { match: 'block:-1', bits: 8 }] } },
  quant_device: 'cpu', calibration: { dataset: 'wikitext-2-raw-v1:train', n_samples: 48, seq_len: 256 },
  evaluation: { dataset: 'wikitext-2-raw-v1:test', seq_len: 256, max_windows: 12, device: 'gpu' },
  deploy: { hardware: 'nvidia/rtx4090', device: 'gpu', vllm: { gpu_memory_utilization: 0.4, lm_eval: { tasks: ['gsm8k'], limit: 20, num_fewshot: 3 } } },
  export: { formats: ['gptq'] }, analysis: { sensitivity: 'block', sensitivity_windows: 8 },
};
const skipRules = [{ match: '^model\\.vision_tower\\.encoder\\.layers\\.1\\.self_attn\\.v_proj$', skip: true }, { match: '^model\\.language_model\\.layers\\.4\\.linear_attn\\.out_proj$', skip: true }];
const vlmJob = {
  domain: 'vlm', metadata: { name: 'minicpmv-p' }, model: { source: 'local', path: MODEL, type: 'vlm' },
  quant: { recipe: ['w4a16'], algorithm: 'gptq', mixed_precision: { rules: skipRules } },
  calibration: { dataset: 'synthetic', n_samples: 16 }, evaluation: { dataset: 'synthetic', n_samples: 18, max_new_tokens: 20, attribution: true },
  deploy: { hardware: 'nvidia/rtx4090' },
};
const canonJob = { ...llmJob, metadata: { name: 'qwen-c' }, quant: { ...llmJob.quant, mixed_precision: { rules: [{ match: 'block:0', bits: 8 }, { match: 'block:-1', bits: 8 }, { match: 'mlp\\.down_proj$', skip: true }] } } };
for (const [id, job, rep] of [[LLM, llmJob, 'llm_full.json'], [VLM, vlmJob, 'vlm_dq.json'], [CANON, canonJob, 'llm_full.json']]) {
  const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'job.yaml'), yaml({ apiVersion: 'mqde/v1', kind: 'Job', ...job, output_dir: runs }) + '\n');
  fs.writeFileSync(path.join(d, 'report.json'), fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports', rep)));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
}
const DEP = '20261013-000000-deployd';
{
  const d = path.join(runs, DEP); fs.mkdirSync(d);
  const real = 20.41190233277122, served = 20.45;
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify({
    job_id: DEP, kind: 'llm_deploy', verdict: 'succeeded',
    spec: { name: 'qwen-p-deploy', model_path: MODEL, artifact: { job: LLM, recipe: 'W4A16', format: 'awq' }, eval_seq_len: 256, eval_windows: 12, device: 'cuda:0' },
    stages: ['validate', 'eval_prepare', 'deploy_sglang', 'report'].map((stage) => ({ stage, status: 'succeeded', seconds: 1 })),
    artifact: { job: LLM, recipe: 'W4A16', format: 'awq', algorithm: 'gptq', path: `${runs}/${LLM}/export/w4a16-awq`, bytes: 449000000, parent_real_ppl: real, parent_baseline_ppl: 13.2 },
    data: { eval: { dataset: 'wikitext-2 test', windows: 12, seq_len: 256 } },
    deploy: { sglang: { status: 'succeeded', startup_s: 41, served_ppl: served, served_vs_real_ppl_pct: 100 * (served / real - 1), engine: { kv_cache_tokens: 512000, weights_gib: 0.42 },
      load: { ok: 16, spec: { requests: 16, concurrency: 4 }, output_tokens_per_s: 1850, ttft_ms: { p50: 31, p99: 60 }, tpot_ms: { p50: 2.1 } } } },
    env: { device_name: 'NVIDIA GeForce RTX 4090' },
  }));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), yaml({ apiVersion: 'mqde/v1', kind: 'Job', metadata: { name: 'qwen-p-deploy', parent_job: LLM }, model: { type: 'llm', path: MODEL },
    artifact: { job: LLM, recipe: 'W4A16', format: 'awq' }, deploy: { device: 'gpu', sglang: { mem_fraction_static: 0.3 } }, output_dir: runs }) + '\n');
}
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
  const parsed = (await (await fetch(`${base}/api/v1/jobs/${LLM}`)).json()).job;
  check('fixture: the server reads the parent job.yaml back as written', norm(parsed.quant.mixed_precision.rules) === norm(llmJob.quant.mixed_precision.rules), JSON.stringify(parsed.quant.mixed_precision));
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const waitJob = (fn) => page.waitForFunction(fn, null, { timeout: 10000 });

  // ---- LLM: 换评估数据 ----
  await page.goto(`${base}/jobs/${LLM}`, { waitUntil: 'load' });
  await page.getByTestId('derive-actions').waitFor({ timeout: 15000 });
  check('a finished job offers the five derive actions', (await page.getByTestId('derive-actions').locator('button').count()) === 5);
  await page.getByTestId('derive-eval').click();
  await page.waitForURL(/\/new\?from=/, { timeout: 5000 });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.metadata?.parent_job; } catch { return false; } });
  await page.waitForTimeout(800); // let the export-suggestion round trip land: it must NOT replace the parent's gptq
  let j = await preview();
  check('the banner names the parent job and says quantization will be rerun', (await page.getByTestId('derive-banner').innerText()).includes(LLM) && (await page.getByTestId('derive-banner').innerText()).includes('重新运行'));
  check('the evaluation card is the one in focus', ((await page.locator('#step-eval').getAttribute('class')) ?? '').includes('derive-focus'));
  check('name and parent_job', j.metadata.name === 'qwen-p-eval' && j.metadata.parent_job === LLM, JSON.stringify(j.metadata));
  check('model, recipe, algorithm carried over', j.model.path === MODEL && JSON.stringify(j.quant.recipe) === '["w4a16"]' && j.quant.algorithm === 'gptq', JSON.stringify(j.quant).slice(0, 120));
  check('KV cache spec carried over (INT4, naive)', j.quant.kv?.bits === 4 && j.quant.kv?.method === 'naive', JSON.stringify(j.quant.kv));
  check('mixed-precision rules carried over in their order (first match wins)', norm(j.quant.mixed_precision?.rules) === norm(llmJob.quant.mixed_precision.rules), JSON.stringify(j.quant.mixed_precision));
  check('calibration and evaluation settings', j.calibration.n_samples === 48 && j.calibration.seq_len === 256 && j.evaluation.max_windows === 12, JSON.stringify([j.calibration, j.evaluation]));
  check('the device split (quantize on CPU, evaluate on GPU)', j.quant_device === 'cpu' && j.evaluation.device === 'gpu' && j.deploy.device === 'gpu', JSON.stringify([j.quant_device, j.evaluation.device, j.deploy.device]));
  check('vLLM deployment with its lm-eval settings', j.deploy.vllm?.gpu_memory_utilization === 0.4 && j.deploy.vllm?.lm_eval?.limit === 20 && j.deploy.vllm?.lm_eval?.num_fewshot === 3, JSON.stringify(j.deploy.vllm));
  check('the parent\'s own export format (gptq), not the auto-suggestion for vLLM (awq)', JSON.stringify(j.export?.formats) === '["gptq"]', JSON.stringify(j.export));
  check('the sensitivity analysis', j.analysis?.sensitivity === 'block', JSON.stringify(j.analysis));
  check('rules the wizard cannot express are shown, and its own two controls are disabled', (await page.getByTestId('inherited-rules').count()) === 1);
  await page.locator('#step-eval input[type=number]').nth(2).fill('20');  // PPL 窗口数
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.max_windows === 20; } catch { return false; } });
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL((u) => u.pathname.startsWith('/jobs/') && !u.pathname.endsWith(LLM), { timeout: 15000 });
  const child = page.url().split('/jobs/')[1];
  const cj = (await (await fetch(`${base}/api/v1/jobs/${child}`)).json()).job;
  check('the derived job was submitted with the one change and everything else intact',
    cj.evaluation.max_windows === 20 && cj.metadata.parent_job === LLM && cj.export.formats[0] === 'gptq' && cj.quant.mixed_precision.rules.length === 3 && cj.deploy.vllm.lm_eval.tasks[0] === 'gsm8k', JSON.stringify(cj).slice(0, 200));
  await page.getByTestId('derived-from').waitFor({ timeout: 10000 });
  check('its page links back to the parent', (await page.getByTestId('derived-from').locator(`a[href="/jobs/${LLM}"]`).count()) === 1);

  // ---- B2: reuse the parent's exported checkpoint instead of quantizing again ----
  await page.goto(`${base}/new?from=${LLM}&step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('artifact-select').waitFor({ timeout: 10000 });
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.artifact; } catch { return false; } });
  const opts = await page.getByTestId('artifact-select').locator('option').allInnerTexts();
  check('the parent\'s exports are offered: 4 HF checkpoints and 3 GGUF files, plus "不复用"', opts.length === 8 && opts.some((o) => o.includes('W4A16 · awq') && o.includes('20.412')) && opts.some((o) => o.includes('GGUF q4_k')), opts.join(' | '));
  j = await preview();
  check('换推理框架 reuses an export by default (the first one listed)', JSON.stringify(j.artifact) === JSON.stringify({ job: LLM, recipe: 'W4A16', format: 'awq' }) && (await page.getByTestId('artifact-select').inputValue()) === 'W4A16/awq', JSON.stringify(j.artifact));
  check('the job serves it and quantizes nothing: no quant, export, analysis or calibration', !j.quant && !j.export && !j.analysis && !j.calibration, JSON.stringify(Object.keys(j)));
  check('it keeps the parent\'s vLLM settings and its evaluation windows', j.deploy.vllm?.gpu_memory_utilization === 0.4 && j.deploy.vllm?.lm_eval?.limit === 20 && j.evaluation.max_windows === 12 && j.evaluation.seq_len === 256 && j.deploy.device === 'gpu', JSON.stringify([j.deploy, j.evaluation]));
  check('②③ are hidden and say they come from the parent', !(await page.getByTestId('quant-steps').isVisible()) && (await page.getByTestId('artifact-quant-note').isVisible()));
  await page.getByTestId('artifact-select').selectOption('gguf/q4_k');
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.llamacpp; } catch { return false; } });
  j = await preview();
  check('a GGUF goes to llama.cpp, and only llama.cpp is offered', j.artifact.format === 'gguf' && j.artifact.type === 'q4_k' && !j.deploy.vllm && JSON.stringify(await page.getByTestId('deploy-backend').locator('option').allInnerTexts()) === '["llama.cpp（GGUF）"]', JSON.stringify(j.deploy));
  check('on the GPU pool (the parent evaluated on GPU): every layer offloaded', j.deploy.device === 'gpu' && j.deploy.llamacpp.gpu_layers === 99, JSON.stringify(j.deploy));
  await page.getByTestId('artifact-device').selectOption('cpu');
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.device === 'cpu'; } catch { return false; } });
  check('换设备 to CPU: llama.cpp keeps every layer on the CPU', (await preview()).deploy.llamacpp.gpu_layers === 0);
  await page.getByTestId('artifact-select').selectOption('W4A8/awq');
  await waitJob(() => { try { const x = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return x?.artifact?.recipe === 'W4A8' && x?.deploy?.vllm; } catch { return false; } });
  check('back to an HF checkpoint: vLLM / SGLang only, on the GPU', JSON.stringify(await page.getByTestId('deploy-backend').locator('option').allInnerTexts()) === '["vLLM","SGLang"]' && (await preview()).deploy.device === 'gpu');
  await page.getByTestId('deploy-backend').selectOption('sglang');
  await waitJob(() => { try { const x = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return x?.deploy?.sglang && !x?.deploy?.vllm; } catch { return false; } });
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL((u) => u.pathname.startsWith('/jobs/') && !u.pathname.endsWith(LLM), { timeout: 15000 }).catch(async (e) => {
    console.log('submit error:', await page.locator('.error-box').allInnerTexts(), JSON.stringify(await preview()).slice(0, 600));
    throw e;
  });
  const dj = page.url().split('/jobs/')[1];
  const djy = (await (await fetch(`${base}/api/v1/jobs/${dj}`)).json()).job;
  check('the server accepted the deploy job as written', djy.artifact.recipe === 'W4A8' && djy.artifact.format === 'awq' && djy.deploy.sglang.mem_fraction_static === 0.3 && !djy.export, JSON.stringify(djy.artifact));
  await page.locator('.pipeline').waitFor({ timeout: 10000 });
  const pl = await page.locator('.pipeline').innerText();
  check('its page shows the deploy stages for the one backend it uses', pl.includes('校验产物') && pl.includes('SGLang 部署') && !pl.includes('vLLM 部署') && !pl.includes('逐块量化'), pl.replace(/\s+/g, ' '));
  const nope = await fetch(`${base}/api/v1/jobs`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ ...djy, artifact: { job: 'nope', recipe: 'W4A16', format: 'awq' }, output_dir: undefined, apiVersion: undefined, kind: undefined }) });
  check('an artifact from a job that does not exist is refused before it is queued', nope.status === 400 && (await nope.text()).includes('not found'));
  await page.goto(`${base}/new?from=${LLM}&step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('artifact-select').waitFor({ timeout: 10000 });
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.artifact; } catch { return false; } });
  await page.getByTestId('artifact-select').selectOption('');
  await waitJob(() => { try { const x = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return !x?.artifact && x?.export; } catch { return false; } });
  check('"不复用" goes back to quantizing again with the parent\'s full settings', (await page.getByTestId('quant-steps').isVisible()) && JSON.stringify((await preview()).export.formats) === '["gptq"]');

  // ---- the deploy job's report page ----
  await page.goto(`${base}/jobs/${DEP}`, { waitUntil: 'load' });
  await page.getByTestId('deploy-table').waitFor({ timeout: 15000 });
  const dt = (await page.getByTestId('deploy-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('the report shows SGLang\'s served PPL against the parent\'s real-weight PPL', dt.length === 1 && dt[0].includes('SGLang') && dt[0].includes('20.450') && dt[0].includes('+0.19%') && dt[0].includes('1850') && dt[0].includes('31 / 60 ms') , dt[0]);
  const tiles = (await page.getByTestId('deploy-tiles').innerText()).replace(/\s+/g, ' ');
  check('tiles name the reused artifact and its source job, with the parent\'s PPL', tiles.includes('W4A16 · awq') && tiles.includes(LLM) && tiles.includes('20.412'), tiles);
  check('the artifact links to the job it came from', (await page.getByTestId('deploy-artifact').locator(`a[href="/jobs/${LLM}"]`).count()) === 1);

  // deriving from a deploy job keeps serving the same artifact, even for a step that does not reuse by default
  await page.goto(`${base}/new?from=${DEP}&step=eval`, { waitUntil: 'load' });
  await page.getByTestId('artifact-select').waitFor({ timeout: 10000 });
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.artifact; } catch { return false; } });
  check('a job derived from a deploy job serves that same artifact (from the original job), with its backend', (await page.getByTestId('artifact-select').inputValue()) === 'W4A16/awq' && (await preview()).artifact.job === LLM && !!(await preview()).deploy.sglang, JSON.stringify((await preview()).artifact));

  // ---- LLM: 换推理框架 focuses ⑤ ----
  await page.goto(`${base}/new?from=${LLM}&step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  check('换推理框架 puts ⑤ in focus', ((await page.locator('#step-deploy').getAttribute('class')) ?? '').includes('derive-focus'));
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.metadata?.name === 'qwen-p-deploy'; } catch { return false; } });
  await page.getByTestId('deploy-backend').selectOption('sglang');
  await waitJob(() => { try { const x = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return x?.deploy?.sglang && !x?.deploy?.vllm; } catch { return false; } });
  check('switching the backend there works as in a fresh job (SGLang on, vLLM off)', true);

  // ---- LLM whose rules are exactly what the wizard's two controls write: they map back onto those controls ----
  await page.goto(`${base}/new?from=${CANON}&step=recipe`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.metadata?.name === 'qwen-c-recipe'; } catch { return false; } });
  j = await preview();
  check('canonical rules come back as the 首尾块 8 bit + down_proj FP controls, not as an opaque list',
    (await page.getByTestId('inherited-rules').count()) === 0 && norm(j.quant.mixed_precision?.rules) === norm(canonJob.quant.mixed_precision.rules), JSON.stringify(j.quant.mixed_precision));
  check('换配方 puts ③ in focus', ((await page.locator('#step-recipe').getAttribute('class')) ?? '').includes('derive-focus'));

  // ---- VLM: 换设备, inherited skip list ----
  await page.goto(`${base}/jobs/${VLM}`, { waitUntil: 'load' });
  await page.getByTestId('derive-actions').waitFor({ timeout: 15000 });
  check('for a VLM job 换推理框架 is disabled (no serving backend for VLM yet)', await page.getByTestId('derive-deploy').isDisabled());
  await page.getByTestId('derive-device').click();
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.model?.type === 'vlm'; } catch { return false; } });
  j = await preview();
  check('VLM: domain, data, answer length and parent_job carried over', j.metadata.parent_job === VLM && j.metadata.name === 'minicpmv-p-dev' && j.domain === 'vlm' && j.evaluation.n_samples === 18 && j.evaluation.max_new_tokens === 20 && j.calibration.dataset === 'synthetic', JSON.stringify(j.evaluation));
  check('VLM: the dynamic-quant skip list is carried over', norm(j.quant.mixed_precision?.rules) === norm(skipRules), JSON.stringify(j.quant.mixed_precision));
  check('VLM: 换设备 explains that VLM runs on the GPU pool', (await page.getByTestId('derive-device-note').count()) === 1);
  await page.getByTestId('drop-inherited-rules').click();
  await waitJob(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant?.mixed_precision === undefined; } catch { return false; } });
  check('去掉这些规则 removes them from the job', (await preview()).quant.mixed_precision === undefined);

  const bad = await (await page.goto(`${base}/new?from=no-such-job&step=eval`, { waitUntil: 'load' })) && page.locator('.error-box').first();
  await bad.waitFor({ timeout: 10000 });
  check('an unknown parent job is an error on the page, not a silent blank wizard', (await bad.innerText()).length > 0, await bad.innerText());
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

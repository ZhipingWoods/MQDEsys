// UI test for T-1304 (slice 1 of the new-job wizard rework): the genuinely new pieces that
// ui_features.mjs/newjob_live_estimate.mjs do not touch -- the domain registry drives the domain tabs and
// blocks domains with no real pipeline, the "不量化" toggle reuses the real bf16 recipe, picking a deploy
// backend calls the real GET /api/v1/export-suggestion and pre-selects chips from its answer, and the
// catalog-backed model picker (grouped by category/family_group, T-1303) resolves a picked entry to a
// local path via the real download machinery (network-intercepted here so the test stays offline).
//
//   node web/e2e/newjob_wizard.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/newjob_wizard.mjs
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
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());

  // ---- 1. domain registry drives the tabs; a domain with no real pipeline is blocked, not silently accepted ----
  await page.goto(`${base}/new`, { waitUntil: 'load' });
  const realDomains = await (await fetch(`${base}/api/v1/domains`)).json();
  check('the server has all 5 registered domains', realDomains.map((d) => d.id).sort().join(',') === 'ewm,llm,vla,vlm,world');
  for (const d of realDomains) await page.getByTestId(`domain-tab-${d.id}`).waitFor({ timeout: 10000 });
  check('every registered domain gets a tab', true);

  await page.getByTestId('domain-tab-vlm').click();
  await page.getByTestId('domain-blocked').waitFor({ timeout: 10000 });
  check('vlm (no real pipeline) shows the blocked notice', true);
  check('submit is disabled for a blocked domain', await page.getByRole('button', { name: /该领域尚未支持提交/ }).isDisabled());

  await page.getByTestId('domain-tab-llm').click();
  check('switching back to llm (a real pipeline) removes the blocked notice', (await page.getByTestId('domain-blocked').count()) === 0);
  check('submit is enabled again', await page.getByRole('button', { name: '提交作业' }).isEnabled());

  // ---- 2. "高级参数": group_size override (T-1304), omitted from the job unless actually touched. Before
  // the "不量化" step below: that step leaves `picked` on bf16 (not per_group), which would make this
  // section's own "enabled" check fail for an unrelated reason.
  await page.getByTestId('model-path').fill(MODEL);
  check('advanced params panel is closed by default', (await page.getByTestId('group-size-override').count()) === 0);
  await page.getByTestId('advanced-params-toggle').click();
  await page.getByTestId('group-size-override').waitFor({ timeout: 5000 });
  check('group_size override is enabled (the default LLM recipes are all per_group)', await page.getByTestId('group-size-override').isEnabled());
  check('and untouched, so the job has no quant.group_size yet', (await preview()).quant.group_size === undefined);
  await page.getByTestId('group-size-override').fill('64');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant?.group_size === 64; } catch { return false; } }, null, { timeout: 10000 });
  check('typing a value adds quant.group_size as a number', (await preview()).quant.group_size === 64);
  await page.getByTestId('group-size-override').fill('');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant?.group_size === undefined; } catch { return false; } }, null, { timeout: 10000 });
  check('clearing it removes the field again', (await preview()).quant.group_size === undefined);

  // ---- 2c. "不量化" forces the bf16 recipe, matching data/recipes/bf16.yaml's real "no quantization" meaning ----
  await page.getByTestId('opt-no-quant').check();
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant?.recipe?.[0] === 'bf16'; } catch { return false; } }, null, { timeout: 10000 });
  check('checking "不量化" sets quant.recipe to just ["bf16"]', JSON.stringify((await preview()).quant.recipe) === '["bf16"]');
  check('bf16 is not per_group, so the override is now disabled', await page.getByTestId('group-size-override').isDisabled());
  await page.getByTestId('opt-no-quant').uncheck();
  check('unchecking it leaves the recipe chips selectable again', !(await page.getByTestId('opt-no-quant').isChecked()));
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant?.recipe?.length > 1; } catch { return false; } }, null, { timeout: 10000 });
  check('and restores the recipes that were picked before ticking it (not stuck on bf16)', JSON.stringify((await preview()).quant.recipe) === '["w4a16","w4a8","w2a16"]', JSON.stringify((await preview()).quant.recipe));

  // ---- 3. picking a deploy backend calls the real export-suggestion endpoint and pre-selects chips ----
  await page.getByTestId('deploy-backend').selectOption('llamacpp');
  await page.waitForFunction(() => document.querySelector('[data-testid="export-suggestion"]')?.textContent?.includes('gguf'), null, { timeout: 10000 });
  check('backend=llamacpp suggests format gguf (from the real planner, not a frontend guess)', (await page.getByTestId('export-suggestion').innerText()).includes('gguf'));
  const j1 = await preview();
  check('a GGUF type chip was pre-selected', Array.isArray(j1.deploy.llamacpp?.types) && j1.deploy.llamacpp.types.length > 0, JSON.stringify(j1.deploy.llamacpp));
  check('vLLM deployment is off for llamacpp', j1.deploy.vllm === undefined);

  await page.getByTestId('deploy-backend').selectOption('vllm');
  // wait on the actual effect (deploy.vllm appearing in the submitted job), not the suggestion text: a
  // stale "native" verdict from the previous (llamacpp) suggestion can still be on screen the instant
  // after switching, before the new vllm fetch resolves, which would make a text-only wait resolve early.
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm !== undefined; } catch { return false; } }, null, { timeout: 10000 });
  const j2 = await preview();
  check('backend=vllm turns on vLLM deployment', j2.deploy.vllm !== undefined, JSON.stringify(j2.deploy.vllm));
  check('and pre-selects awq (the real planner answer for a 4-bit recipe), not just any format', JSON.stringify(j2.export?.formats) === '["awq"]', JSON.stringify(j2.export));

  await page.getByTestId('deploy-backend').selectOption('sglang');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.sglang !== undefined; } catch { return false; } }, null, { timeout: 10000 });
  const jS = await preview();
  check('backend=sglang turns on SGLang deployment (and turns vLLM off)', jS.deploy.sglang?.mem_fraction_static === 0.3 && jS.deploy.vllm === undefined, JSON.stringify([jS.deploy.sglang, jS.deploy.vllm]));
  check('and pre-selects awq, which SGLang really loads (measured), not "unsupported"', JSON.stringify(jS.export?.formats) === '["awq"]' && (await page.getByTestId('export-suggestion').innerText()).includes('awq'), JSON.stringify(jS.export));
  await page.getByTestId('opt-sglang-mem').fill('0.45');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.sglang?.mem_fraction_static === 0.45; } catch { return false; } }, null, { timeout: 10000 });
  check('the static memory fraction input reaches the job', (await preview()).deploy.sglang.mem_fraction_static === 0.45);

  await page.getByTestId('deploy-backend').selectOption('none');
  const j3 = await preview();
  check('switching back to "不部署" clears all of them', j3.deploy.vllm === undefined && j3.deploy.llamacpp === undefined && j3.deploy.sglang === undefined, JSON.stringify([j3.deploy.vllm, j3.deploy.llamacpp, j3.deploy.sglang]));

  // ---- 3b. "仅量化不评测" (evaluation.skip, T-1304/T-1306 follow-up) strips export/vllm/llamacpp/analysis
  // outright -- even if they were set before the checkbox was ticked -- matching planner::validate_jobspec's
  // own rejection list exactly, so the UI can never produce a job the server would 400.
  await page.getByTestId('deploy-backend').selectOption('vllm');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm !== undefined; } catch { return false; } }, null, { timeout: 10000 });
  await page.getByTestId('opt-sglang').check(); // both backends on at once is allowed (compare them)
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.sglang !== undefined; } catch { return false; } }, null, { timeout: 10000 });
  await page.getByTestId('opt-quantize-only').check();
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.skip === true; } catch { return false; } }, null, { timeout: 10000 });
  const j4 = await preview();
  check('quantize-only sets evaluation.skip', j4.evaluation.skip === true);
  check('and strips deploy.vllm even though it was set moments ago', j4.deploy.vllm === undefined, JSON.stringify(j4.deploy.vllm));
  check('and strips deploy.sglang too', j4.deploy.sglang === undefined, JSON.stringify(j4.deploy.sglang));
  check('and strips export.formats', j4.export === undefined, JSON.stringify(j4.export));
  check('and strips deploy.llamacpp', j4.deploy.llamacpp === undefined, JSON.stringify(j4.deploy.llamacpp));
  check('the deploy-backend select is disabled while quantize-only is on', await page.getByTestId('deploy-backend').isDisabled());
  check('the vLLM checkbox is disabled too', await page.getByTestId('opt-vllm').isDisabled());
  check('and the SGLang one', await page.getByTestId('opt-sglang').isDisabled());
  await page.getByTestId('opt-quantize-only').uncheck();
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.skip === undefined; } catch { return false; } }, null, { timeout: 10000 });
  check('unchecking it brings deploy.vllm back', (await preview()).deploy.vllm !== undefined);
  await page.getByTestId('deploy-backend').selectOption('none');

  // ---- 3c. quant_device/eval_device split (T-1304/T-1306 follow-up): omitted when equal (today's
  // default, unchanged job shape), appear only once they differ, resolved for real end to end by
  // JobRunner (see job_runner_test.cpp / docs/PROGRESS.md for the server-side half of this) ----
  const j5a = await preview();
  check('both devices default to cpu and add no new fields', j5a.quant_device === undefined && j5a.evaluation.device === undefined && j5a.deploy.device === 'cpu', JSON.stringify({ quant_device: j5a.quant_device, eval_device: j5a.evaluation.device, deploy_device: j5a.deploy.device }));
  await page.getByTestId('quant-device-select').selectOption('gpu');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant_device === 'gpu'; } catch { return false; } }, null, { timeout: 10000 });
  const j5b = await preview();
  check('quant_device=gpu, eval_device still cpu: both fields appear once they differ', j5b.quant_device === 'gpu' && j5b.evaluation.device === 'cpu', JSON.stringify({ quant_device: j5b.quant_device, eval_device: j5b.evaluation.device }));
  check('deploy.device reflects "either one wants gpu"', j5b.deploy.device === 'gpu');
  await page.getByTestId('quant-device-select').selectOption('cpu');
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.quant_device === undefined; } catch { return false; } }, null, { timeout: 10000 });
  check('setting it back to cpu (matching eval_device again) removes both fields', (await preview()).quant_device === undefined);

  // ---- 4. catalog-backed model picker (grouped by category/family_group) resolves to a local path ----
  // Network-intercepted so this stays offline like every other e2e test in this suite: the real frontend
  // code path (list groups -> start download -> poll status -> fill the path) runs against fake responses
  // shaped like the real server's, not a mocked React component.
  let dlCalls = 0;
  await page.route('**/api/v1/download', async (route) => {
    if (route.request().method() !== 'POST') return route.continue();
    dlCalls++;
    await route.fulfill({ json: { id: 'dl-1', repo: 'Qwen/Qwen2.5-0.5B-Instruct', group: 'safetensors', dir: MODEL, state: 'running', error: '', bytes_done: 0, bytes_total: 100, files_done: 0, files_total: 1 } });
  });
  let polls = 0;
  await page.route('**/api/v1/download/dl-1/status', async (route) => {
    polls++;
    const done = polls >= 2;
    await route.fulfill({ json: { id: 'dl-1', repo: 'Qwen/Qwen2.5-0.5B-Instruct', group: 'safetensors', dir: MODEL, state: done ? 'succeeded' : 'running', error: '', bytes_done: done ? 100 : 40, bytes_total: 100, files_done: done ? 1 : 0, files_total: 1 } });
  });
  await page.getByText('从模型目录选择').click();
  await page.getByTestId('catalog-model-select').waitFor({ timeout: 10000 });
  await page.waitForFunction(() => (document.querySelector('[data-testid="catalog-model-select"]')?.querySelectorAll('option').length ?? 0) > 1, null, { timeout: 10000 });
  const optionCount = await page.getByTestId('catalog-model-select').locator('option').count();
  check('the catalog dropdown has real entries grouped under optgroups', optionCount > 1, `${optionCount} options`);
  const qwenOption = await page.getByTestId('catalog-model-select').locator('option', { hasText: 'Qwen/Qwen2.5-0.5B-Instruct' }).first().getAttribute('value').catch(() => null);
  if (qwenOption) {
    await page.getByTestId('catalog-model-select').selectOption(qwenOption);
    await page.getByTestId('catalog-download').click();
    await page.waitForFunction(() => document.querySelector('[data-testid="catalog-download"]')?.textContent === '已下载', null, { timeout: 10000 });
    check('after the (intercepted) download succeeds, the model path is filled', (await preview()).model.path === MODEL);
    check('the real download endpoints were actually called, not skipped', dlCalls === 1 && polls >= 1, `dlCalls=${dlCalls} polls=${polls}`);
  } else {
    check('Qwen/Qwen2.5-0.5B-Instruct is in the local catalog fixture (data/models/catalog.json)', false);
  }

  // ---- 5. a real VLA submission still produces the same JobSpec shape as before the rework ----
  await page.getByTestId('domain-tab-vla').click();
  await page.getByTestId('model-path').waitFor({ timeout: 10000 }); // back to the custom-path field (no catalog toggle for vla)
  await page.getByTestId('model-path').fill(MODEL);
  // the default calibration/evaluation dataset is a real LeRobot path outside this test's --allow sandbox
  // (a 403 server-side, correctly -- that is the same path-containment check every job submission gets);
  // "synthetic" is the server's own exempted special value (server.cpp's path_allowed), used here for
  // exactly the reason the real UI offers it: a job that does not need real data on disk.
  await page.getByTestId('dataset-path').fill('synthetic');
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL(/\/jobs\/\d{8}-\d{6}-[0-9a-f]{6}$/, { timeout: 15000 });
  const vlaId = page.url().split('/').pop();
  const vlaStored = (await (await fetch(`${base}/api/v1/jobs/${vlaId}`)).json()).job;
  check(
    'the submitted VLA job keeps the pre-rework shape plus the new, optional domain field',
    vlaStored?.domain === 'vla' && vlaStored?.model?.type === 'vla' && vlaStored?.model?.modality === 'vla' && vlaStored?.model?.source === 'local' && Array.isArray(vlaStored?.quant?.recipe),
    JSON.stringify({ domain: vlaStored?.domain, model: vlaStored?.model, recipe: vlaStored?.quant?.recipe }),
  );

  // ---- 6. quantize-only forces off VLA attribution (it needs the BF16 baseline this also skips) and a
  // real submission with it on still goes through the server's real validate_jobspec, not just the preview ----
  await page.goto(`${base}/new`, { waitUntil: 'load' });
  await page.getByTestId('domain-tab-vla').click();
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('dataset-path').fill('synthetic');
  await page.getByTestId('opt-quantize-only').check();
  await page.waitForFunction(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.skip === true; } catch { return false; } }, null, { timeout: 10000 });
  const j5 = await preview();
  check('quantize-only forces VLA attribution off', j5.evaluation.attribution === false, JSON.stringify(j5.evaluation));
  check('and sets evaluation.skip', j5.evaluation.skip === true);
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL(/\/jobs\/\d{8}-\d{6}-[0-9a-f]{6}$/, { timeout: 15000 });
  const qoId = page.url().split('/').pop();
  const qoStored = (await (await fetch(`${base}/api/v1/jobs/${qoId}`)).json()).job;
  check(
    'the server accepted the quantize-only VLA job (planner::validate_jobspec, not just the frontend preview)',
    qoStored?.evaluation?.skip === true && qoStored?.evaluation?.attribution === false,
    JSON.stringify(qoStored?.evaluation),
  );

  // switching domains resets the model path to a hardcoded default (outside this test's --allow sandbox),
  // so the live-estimate sidebar's own request against it 403s once -- same deliberate, tolerated 403 the
  // other new-job e2e tests already expect (see newjob_live_estimate.mjs).
  const unexpected = errors.filter((e) => !/403 \(Forbidden\)/.test(e));
  check('no unexpected console or page errors the whole run', unexpected.length === 0, unexpected.slice(0, 2).join(' | '));
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

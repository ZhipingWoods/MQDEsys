// The visible frontend rework of 2026-10-13: the new-job page is a real step-by-step wizard (one step on screen,
// ①…⑥ navigation, summaries of finished steps, prev/next with per-step validation, a single-page toggle), there is a
// home overview (GPU pool, running/recent jobs, backend environments), and the job list has search + filters.
//
//   node web/e2e/wizard_steps.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const mk = (id, name, kind, state, parent) => {
  const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: Date.now() / 1000 - 30 }, { type: 'job_state', state, seq: 2, ts: Date.now() / 1000 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), [`metadata: {name: ${name}${parent ? `, parent_job: ${parent}` : ''}}`, `model: {type: ${kind}, path: ${MODEL}}`, 'quant: {recipe: [w4a16]}', 'deploy: {device: gpu}', `output_dir: ${runs}`, ''].join('\n'));
};
mk('20261013-000001-aaaaaa', 'alpha-llm', 'llm', 'SUCCEEDED');
mk('20261013-000002-bbbbbb', 'beta-vlm', 'vlm', 'FAILED');
mk('20261013-000003-cccccc', 'gamma-llm', 'llm', 'SUCCEEDED', '20261013-000001-aaaaaa');

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
  const current = async () => { for (let n = 1; n <= 6; n++) if ((await page.getByTestId(`step-tab-${n}`).getAttribute('aria-current')) === 'step') return n; return 0; };
  const visible = (id) => page.getByTestId(id).isVisible();

  // ---- the wizard ----
  await page.goto(`${base}/new`, { waitUntil: 'load' });
  await page.getByTestId('stepper').waitFor({ timeout: 15000 });
  check('the page opens on step ① with the six steps on top', (await current()) === 1 && (await page.locator('[data-testid^="step-tab-"]').count()) === 6);
  check('only step ① is on screen: the model path yes, the recipes / evaluation / deploy / confirm no', (await visible('model-path')) && !(await visible('opt-quantize-only')) && !(await visible('quant-device-select')) && !(await visible('deploy-backend')) && !(await visible('step-confirm')));
  await page.getByTestId('model-path').fill('');
  check('下一步 is disabled until the model path is filled', await page.getByTestId('step-next').isDisabled());
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('step-next').click();
  check('② 量化算法 after ①', (await current()) === 2 && !(await visible('model-path')));
  const s1 = await page.getByTestId('step-tab-1').innerText();
  check('a finished step shows a ✓ and its summary (domain · model)', s1.includes('✓') && s1.includes('LLM') && s1.includes(path.basename(MODEL)), s1.replace(/\s+/g, ' '));
  await page.getByTestId('step-next').click();
  check('③ 量化配方', (await current()) === 3 && (await visible('opt-quantize-only')));
  // untick every recipe: cannot continue
  const chips = page.locator('#step-recipe ~ .chips .chip.on');  // the recipe chips, not the hidden advice panel's backend chips
  const n = await chips.count();
  for (let i = 0; i < n; i++) await page.locator('#step-recipe ~ .chips .chip.on').first().click();
  check('with no recipe picked 下一步 is disabled and the step says so', (await page.getByTestId('step-next').isDisabled()) && (await page.getByTestId('step-tab-3').innerText()).includes('请先完成本步'));
  await page.locator('#step-recipe ~ .chips .chip', { hasText: 'W4A16' }).first().click();
  await page.getByTestId('step-next').click();
  check('④ 评估数据 + 设备', (await current()) === 4 && (await visible('quant-device-select')));
  await page.getByTestId('eval-device-select').selectOption('gpu');
  await page.getByTestId('step-next').click();
  check('⑤ 部署与导出', (await current()) === 5 && (await visible('deploy-backend')));
  const s4 = await page.getByTestId('step-tab-4').innerText();
  check('the ④ summary reflects the device choice', s4.includes('量化 CPU') && s4.includes('评估 GPU'), s4.replace(/\s+/g, ' '));
  await page.getByTestId('deploy-backend').selectOption('vllm');
  await page.getByTestId('step-next').click();
  check('⑥ 确认提交 shows the summary table and the JobSpec, and no 下一步', (await current()) === 6 && (await visible('step-summary')) && (await page.getByTestId('step-next').count()) === 0);
  const sum = (await page.getByTestId('step-summary').innerText()).replace(/\s+/g, ' ');
  check('the summary names model, recipe, devices and backend', sum.includes(MODEL) && sum.includes('W4A16') && sum.includes('量化 CPU · 评估 GPU') && sum.includes('vllm'), sum.slice(0, 200));
  const j = JSON.parse(await page.locator('pre.log').innerText());
  check('the JobSpec built across the steps is complete', j.quant.recipe[0] === 'w4a16' && j.evaluation.device === 'gpu' && j.deploy.vllm && j.deploy.device === 'gpu', JSON.stringify(j).slice(0, 200));
  await page.getByTestId('step-tab-3').click();
  check('clicking a step in the nav goes back to it', (await current()) === 3 && (await visible('opt-quantize-only')));
  await page.getByTestId('step-prev').click();
  check('上一步 works', (await current()) === 2);
  await page.getByTestId('toggle-single-page').click();
  await page.waitForFunction(() => location.search.includes('mode=all'), null, { timeout: 5000 });
  check('单页显示全部 puts every step on one page (what the older tests use) and keeps the state', (await visible('model-path')) && (await visible('deploy-backend')) && (await visible('step-confirm')) && JSON.stringify((JSON.parse(await page.locator('pre.log').innerText())).quant.recipe) === '["w4a16"]');
  await page.getByTestId('toggle-single-page').click();
  check('and back to steps', !(await visible('deploy-backend')));
  await page.goto(`${base}/new?step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('stepper').waitFor({ timeout: 15000 });
  check('?step=deploy (the derive links) opens on ⑤', (await current()) === 5);
  await page.goto(`${base}/new`, { waitUntil: 'load' });
  await page.getByTestId('domain-tab-world').click();
  check('a domain without a pipeline keeps the blocked notice and blocks 下一步', (await visible('domain-blocked')) && (await page.getByTestId('step-next').isDisabled()));

  // ---- home ----
  await page.goto(`${base}/`, { waitUntil: 'load' });
  await page.getByTestId('home-tiles').waitFor({ timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="home-jobs"] tbody tr').length >= 3, null, { timeout: 15000 });
  const tiles = (await page.getByTestId('home-tiles').innerText()).replace(/\s+/g, ' ');
  check('the home page is the overview: GPU pool, running, last-24h results, backends', tiles.includes('GPU 池') && tiles.includes('1 / 1 空闲') && tiles.includes('2 成功 / 1 失败') && tiles.includes('推理后端'), tiles);
  const hj = (await page.getByTestId('home-jobs').innerText()).replace(/\s+/g, ' ');
  check('recent jobs with type, state and derivation', hj.includes('gamma-llm') && hj.includes('派生自 aaaaaa') && hj.includes('VLM') && hj.includes('失败'), hj.slice(0, 200));
  check('GPU 0 of the pool is listed as free', (await page.getByTestId('home-gpu-0').innerText()).includes('空闲'));
  await page.getByTestId('home-new-job').click();
  await page.waitForURL(/\/new/, { timeout: 5000 });
  check('＋ 新建评估作业 goes to the wizard', (await page.getByTestId('stepper').count()) === 1);
  check('the sidebar marks 总览 only on /', (await page.locator('nav a.active', { hasText: '总览' }).count()) === 0);

  // ---- job list filters ----
  await page.goto(`${base}/jobs`, { waitUntil: 'load' });
  await page.getByTestId('jobs-counts').waitFor({ timeout: 15000 });
  await page.waitForFunction(() => document.querySelectorAll('tr[data-testid^="job-row-"]').length === 3, null, { timeout: 15000 });
  await page.getByTestId('jobs-kind').selectOption('llm');
  check('type filter', (await page.locator('tr[data-testid^="job-row-"]').count()) === 2);
  await page.getByTestId('jobs-state').selectOption('FAILED');
  check('type + state filters compose (no llm failed)', (await page.locator('tr[data-testid^="job-row-"]').count()) === 0 && (await page.locator('td.empty').innerText()).includes('没有符合'));
  await page.getByTestId('jobs-kind').selectOption('');
  check('state filter alone finds the failed VLM job', (await page.locator('tr[data-testid^="job-row-"]').count()) === 1 && (await page.locator('tr[data-testid^="job-row-"]').innerText()).includes('beta-vlm'));
  await page.getByTestId('jobs-state').selectOption('');
  await page.getByTestId('jobs-search').fill('gamma');
  await page.waitForFunction(() => document.querySelectorAll('tr[data-testid^="job-row-"]').length === 1, null, { timeout: 5000 });
  const row = await page.locator('tr[data-testid^="job-row-"]').innerText();
  check('search by name; the derived job shows its parent', row.includes('gamma-llm') && row.includes('↳') && row.includes('aaaaaa'), row.replace(/\s+/g, ' '));
  check('the counts line says 1 / 3 and 2 succeeded / 1 failed', /1 \/ 3/.test(await page.getByTestId('jobs-counts').innerText()) && (await page.getByTestId('jobs-counts').innerText()).includes('失败 1'));
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

// Evaluation dataset catalogue + focused evaluate/quantize forms.
//
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/eval_datasets.mjs
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
const { chromium } = require('playwright');
const bin = path.resolve(repo, process.env.MQDE_BIN ?? 'build/dev/bin/mqde');
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });

const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-ds-model-'));
fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify({ model_type: 'qwen2', architectures: ['Qwen2ForCausalLM'] }));
const QUANT = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-ds-q4-'));
fs.writeFileSync(path.join(QUANT, 'config.json'), JSON.stringify({ model_type: 'qwen2', quantization_config: { quant_method: 'gptq' } }));
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-ds-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
process.on('exit', () => { try { server.kill(); } catch { /* gone */ } });
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 120; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not yet */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());

  await page.goto(`${base}/new/eval`, { waitUntil: 'load' });
  await page.getByTestId('quantized-path').waitFor({ timeout: 15000 });
  await page.getByTestId('quantized-path').fill(QUANT);
  await page.getByTestId('benchmark-suite').waitFor({ timeout: 10000 });
  check('evaluate-only page uses a suite dropdown', (await page.getByTestId('benchmark-suite').locator('option').count()) >= 2);
  check('dataset category dropdown exposes the offline groups', (await page.getByTestId('benchmark-group').locator('option').allInnerTexts()).includes('基础语言建模'));
  const info = await page.getByTestId('benchmark-info').innerText();
  check('dataset detail shows metric and sample count', info.includes('PPL') && info.includes('指标'));
  await page.getByTestId('benchmark-suite').selectOption('standard');
  await page.waitForFunction(() => {
    const rows = document.querySelectorAll('[data-testid="benchmark-selected"] .benchmark-pill');
    return rows.length === 3;
  }, null, { timeout: 5000 });
  const j = await preview();
  check('standard suite is written to evaluation.benchmarks', ['wikitext2_ppl', 'smoke_qa', 'smoke_mcq'].every((x) => (j.evaluation?.benchmarks ?? []).includes(x)), JSON.stringify(j.evaluation?.benchmarks));

  await page.getByTestId('focused-eval-deploy').locator('summary').click();
  await page.getByTestId('eval-deploy-backend').selectOption('vllm');
  await page.getByTestId('opt-evalscope').check();
  await page.getByTestId('evalscope-benchmark-search').fill('ceval');
  await page.getByTestId('evalscope-benchmark-select').selectOption('ceval');
  await page.getByTestId('evalscope-benchmark-add').click();
  await page.waitForFunction(() => {
    const pre = document.querySelector('pre.log');
    try { return JSON.parse(pre?.textContent ?? 'null')?.deploy?.vllm?.evalscope?.datasets?.includes('ceval'); } catch { return false; }
  }, null, { timeout: 10000 });
  check('EvalScope attachment carries ceval into deploy.vllm.evalscope', true);

  await page.goto(`${base}/new/quantize`, { waitUntil: 'load' });
  await page.getByTestId('focused-quant-model').waitFor({ timeout: 15000 });
  check('quantize page offers a primary-recipe dropdown, not a chip wall', (await page.getByTestId('recipe-primary').count()) === 1 && (await page.getByTestId('recipe-selected').count()) === 1);
  await page.getByTestId('recipe-add').selectOption('w8a16');
  await page.waitForFunction(() => document.querySelectorAll('[data-testid="recipe-selected"] .benchmark-pill').length >= 2, null, { timeout: 5000 });
  check('adding a comparison recipe updates the compact selected list', true);
  await page.getByTestId('export-format').selectOption('awq');
  const jq = await preview();
  check('export format is selected from a dropdown and reaches the JobSpec', (jq.export?.formats ?? []).includes('awq'), JSON.stringify(jq.export?.formats));

  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  failed = results.includes(false);
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${String(e).split('\n')[0]}`);
  console.log(e.stack ?? '');
} finally {
  await browser.close();
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed ? 1 : 0);

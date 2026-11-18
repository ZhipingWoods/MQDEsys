// Hardware cards + dedicated quantize/evaluate job entries.
//
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/hardware_cards.mjs
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

const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-hw-model-'));
fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify({
  architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 896, intermediate_size: 4864,
  num_hidden_layers: 2, num_attention_heads: 14, num_key_value_heads: 2, vocab_size: 151936,
  max_position_embeddings: 32768, tie_word_embeddings: true, torch_dtype: 'bfloat16',
}));
const QUANT = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-hw-q4-'));
fs.writeFileSync(path.join(QUANT, 'config.json'), JSON.stringify({ model_type: 'qwen2', quantization_config: { quant_method: 'gptq' } }));

const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-hw-runs-'));
const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* already gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 120; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up yet */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const untilPreview = (src) => page.waitForFunction((body) => {
    try { return new Function('j', `return (${body});`)(JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')); } catch { return false; }
  }, src, { timeout: 10000 });

  // ---- hardware catalog is card-based and includes the requested NVIDIA families ----
  await page.goto(`${base}/hardware`, { waitUntil: 'load' });
  await page.getByTestId('hardware-picker').waitFor({ timeout: 15000 });
  for (const id of ['nvidia/jetson-agx-orin-64g', 'nvidia/jetson-orin-nx-16g', 'nvidia/jetson-orin-nano-8g', 'nvidia/jetson-agx-thor', 'nvidia/dgx-spark-gb10']) {
    check(`hardware card exists: ${id}`, (await page.getByTestId(`hw-card-${id}`).count()) === 1);
  }
  await page.getByTestId('hw-search').fill('dspark');
  check('the common "dspark" spelling finds DGX Spark', (await page.getByTestId('hw-card-nvidia/dgx-spark-gb10').count()) === 1);
  await page.getByTestId('hw-search').fill('');
  await page.getByTestId('hw-filter-edge').click();
  check('edge filter keeps Jetson Orin and Thor visible', (await page.getByTestId('hw-card-nvidia/jetson-agx-orin-64g').count()) === 1 && (await page.getByTestId('hw-card-nvidia/jetson-agx-thor').count()) === 1);
  await page.getByTestId('hw-filter-workstation').click();
  check('workstation filter exposes DGX Spark', (await page.getByTestId('hw-card-nvidia/dgx-spark-gb10').count()) === 1);

  await page.getByTestId('hw-filter-edge').click();
  await page.getByTestId('hw-card-nvidia/jetson-agx-thor').click();
  await page.getByTestId('hw-selected-bar').waitFor({ timeout: 5000 });
  check('clicking a card selects it as the deployment target', (await page.getByTestId('hw-card-nvidia/jetson-agx-thor').getAttribute('aria-pressed')) === 'true' && (await page.getByTestId('hw-selected-bar').innerText()).includes('Jetson AGX Thor'));

  // ---- one click from a hardware card carries the target into the evaluate-only job ----
  await page.getByTestId('hw-use-eval').click();
  await page.waitForURL(/\/new\/eval\?hw=nvidia%2Fjetson-agx-thor/, { timeout: 10000 });
  await page.getByTestId('job-mode-eval').waitFor({ timeout: 10000 });
  check('hardware card -> new evaluate job keeps the target id', (await page.getByTestId('hw-select').inputValue()) === 'nvidia/jetson-agx-thor');
  check('new evaluate job preselects evaluate-only', (await page.getByTestId('job-mode-eval').getAttribute('class')).includes('on'));
  await page.getByTestId('quantized-path').fill(QUANT);
  await untilPreview('j.stages?.[0] === "evaluate" && j.deploy?.hardware === "nvidia/jetson-agx-thor"');
  const je = await preview();
  check('evaluate-only JobSpec: stages [evaluate], quantized_path, chosen hardware', JSON.stringify(je.stages) === '["evaluate"]' && je.model.quantized_path === QUANT && je.deploy.hardware === 'nvidia/jetson-agx-thor', JSON.stringify({ stages: je.stages, hw: je.deploy.hardware }));
  check('evaluate-only hides the quantize section and keeps offline benchmarks', je.quant === undefined && (je.evaluation?.benchmarks ?? []).includes('wikitext2_ppl') && (await page.getByTestId('offline-benchmarks').count()) === 1);

  // ---- dedicated quantize entry, no evaluation/deploy backend section ----
  await page.goto(`${base}/new/quantize?mode=all&hw=nvidia/jetson-agx-orin-64g`, { waitUntil: 'load' });
  await page.getByTestId('job-mode-quantize').waitFor({ timeout: 10000 });
  await page.getByTestId('model-path').fill(MODEL);
  check('new quantize job preselects quantize-only', (await page.getByTestId('job-mode-quantize').getAttribute('class')).includes('on'));
  await untilPreview('j.stages?.includes("quantize") && j.evaluation === undefined');
  const jq = await preview();
  check('quantize-only JobSpec: [quantize, export], no evaluation', JSON.stringify(jq.stages) === '["quantize","export"]' && jq.evaluation === undefined, JSON.stringify(jq.stages));
  check('quantize-only uses the hardware selected from the card', jq.deploy.hardware === 'nvidia/jetson-agx-orin-64g');
  check('quantize-only has no serving cards', (await page.getByTestId('backend-cards').count()) === 0);

  // ---- navigation exposes both entries directly ----
  await page.goto(`${base}/`, { waitUntil: 'load' });
  const sidebar = page.getByTestId('sidebar');
  check('sidebar has a dedicated evaluate-job entry', (await sidebar.getByRole('link', { name: '新建评测作业' }).count()) === 1);
  check('sidebar has a dedicated quantize-job entry', (await sidebar.getByRole('link', { name: '新建量化作业' }).count()) === 1);

  check('no page errors', errors.length === 0, errors.join(' | '));
  if (results.includes(false)) failed = true;
} finally {
  await browser.close();
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed ? 1 : 0);

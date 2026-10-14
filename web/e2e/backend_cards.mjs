// The deploy step shows the three inference backends -- llama.cpp, vLLM, SGLang -- as cards: what each serves, whether it is
// installed on this server, a switch, and its own options. Several can be on at once (to compare them); a reused artifact
// goes to exactly one.
//
//   node web/e2e/backend_cards.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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

const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-cards-')));
const runs = path.join(root, 'runs'); fs.mkdirSync(runs);
const MODEL = path.join(root, 'model'); fs.mkdirSync(MODEL);
{ const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000, max_position_embeddings: 512, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const header = { 'model.embed_tokens.weight': { dtype: 'BF16', shape: [1000, 64], data_offsets: [0, 128000] } };
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb])); }
// a server where vLLM and llama.cpp are installed and SGLang is not
const fake = path.join(root, 'fake_setup.py');
fs.writeFileSync(fake, `#!/usr/bin/env python3
import json, sys
if sys.argv[1:2] == ["status"]:
    print(json.dumps([
        {"id": "vllm", "installed": True, "version": "0.30.0", "path": "/envs/vllm", "source": "managed", "managed": True, "envs_dir": "/envs"},
        {"id": "sglang", "installed": False, "version": None, "path": None, "source": None, "managed": True, "envs_dir": "/envs"},
        {"id": "llamacpp", "installed": True, "version": "b4000", "path": "/envs/llamacpp", "source": "managed", "managed": True, "envs_dir": "/envs"}]))
else:
    sys.exit(1)
`);
fs.chmodSync(fake, 0o755);
// a finished job with exported checkpoints: reusing it is the B2 flow (one backend)
const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
const mkjob = (id, report, yaml) => { const d = path.join(runs, id); fs.mkdirSync(d, { recursive: true });
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify({ ...report, job_id: id }));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), yaml.join('\n') + '\n'); };
const PARENT = '20261014-000000-cardsp';
{ const rep = JSON.parse(JSON.stringify(fixture)); rep.recipes.W4A16.export = { awq: { status: 'exported', bytes: 1, real_ppl: 20.4 } };
  fs.mkdirSync(path.join(runs, PARENT, 'export', 'w4a16-awq'), { recursive: true }); fs.writeFileSync(path.join(runs, PARENT, 'export', 'w4a16-awq', 'config.json'), '{}');
  mkjob(PARENT, rep, ['metadata: {name: cards-parent}', `model: {type: llm, path: ${MODEL}}`, 'quant: {recipe: [w4a16]}', `output_dir: ${runs}`]); }

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', root, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'], env: { ...process.env, MQDE_SETUP_BACKEND_BIN: fake } });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1300 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const until = (fn, arg) => page.waitForFunction(fn, arg, { timeout: 10000 });
  const text = async (id) => (await page.getByTestId(id).innerText()).replace(/\s+/g, ' ').trim();
  const on = async (id) => (await page.getByTestId(`backend-toggle-${id}`).isChecked());
  const deploy = async () => (await preview()).deploy ?? {};

  // ---------------------------------------------------------------- the cards
  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });  // every step on one page
  await page.getByTestId('domain-tab-llm').click();
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('backend-cards').waitFor({ timeout: 10000 });
  for (const id of ['llamacpp', 'vllm', 'sglang']) await page.getByTestId(`backend-card-${id}`).waitFor({ timeout: 5000 });
  check('the deploy step shows all three backends, each as its own card', (await page.locator('[data-testid^="backend-card-"]').count()) === 3);
  const names = await page.locator('.backend-name').allInnerTexts();
  check('llama.cpp, vLLM and SGLang, in that order', JSON.stringify(names) === '["llama.cpp","vLLM","SGLang"]', JSON.stringify(names));
  await until(() => document.querySelector('[data-testid="backend-status-vllm"]')?.textContent?.includes('0.30.0'));
  check('vLLM is installed here, and the card says which version', /已安装 0\.30\.0/.test(await text('backend-status-vllm')), await text('backend-status-vllm'));
  check('llama.cpp is installed, with its version', /已安装 b4000/.test(await text('backend-status-llamacpp')));
  check('SGLang is not installed here, and the card says so', /未安装/.test(await text('backend-status-sglang')));
  const facts = await page.getByTestId('backend-card-vllm').innerText();
  check('each card says what it serves, where it runs and what it measures', /AWQ/.test(facts) && /GPU/.test(facts) && /TTFT/.test(facts) && /GGUF/.test(await page.getByTestId('backend-card-llamacpp').innerText()));
  check('nothing is on to begin with, and the page says that means no real deployment', !(await on('vllm')) && !(await on('sglang')) && !(await on('llamacpp')) && /还没有选部署后端/.test(await text('backend-none')) && !('vllm' in await deploy()));

  // ---------------------------------------------------------------- several at once
  await page.getByTestId('backend-toggle-vllm').check();
  await until(() => { try { return !!JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm; } catch { return false; } });
  let d = await deploy();
  check('switching vLLM on puts deploy.vllm in the job and nothing else', !!d.vllm && !d.sglang && !d.llamacpp, JSON.stringify(Object.keys(d)));
  await page.getByTestId('backend-toggle-llamacpp').check();
  await until(() => { try { return !!JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.llamacpp; } catch { return false; } });
  d = await deploy();
  check('switching llama.cpp on as well keeps vLLM: both are in the job, llama.cpp with a default GGUF type', !!d.vllm && JSON.stringify(d.llamacpp.types) === '["q4_k"]', JSON.stringify(d));
  check('a backend that is not installed does not warn while it is switched off', (await page.getByTestId('backend-missing-sglang').count()) === 0);
  await page.getByTestId('backend-toggle-sglang').check();
  await until(() => { try { return !!JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.sglang; } catch { return false; } });
  d = await deploy();
  check('and SGLang too: all three backends are in one job, to compare them', !!d.vllm && !!d.sglang && !!d.llamacpp, JSON.stringify(Object.keys(d)));
  check('the card that is on is marked, the others are not', (await page.getByTestId('backend-card-sglang').getAttribute('data-on')) === '1');
  check('switching on a backend that is not installed warns, with a link to where it is installed', /没有 SGLang/.test(await text('backend-missing-sglang')) && (await page.getByTestId('backend-missing-sglang').locator('a[href="/hardware"]').count()) === 1);
  check('an installed one does not warn', (await page.getByTestId('backend-missing-vllm').count()) === 0);
  const sum = await page.getByTestId('backend-cards').innerText();
  check('each backend that is on shows its own options inside its card (GGUF types, memory share, lm-eval)', /Q4_K/i.test(sum) && (await page.getByTestId('opt-vllm-util').count()) === 1 && (await page.getByTestId('opt-sglang-mem').count()) === 1 && (await page.getByTestId('opt-lmeval').count()) === 1);
  const fm = (await preview()).export?.formats ?? [];
  check('with vLLM and SGLang both on it is one export (awq), not the default gptq plus the suggestion', JSON.stringify(fm) === '["awq"]', JSON.stringify(fm));
  check('the export formats the vLLM and SGLang runs need are already chosen: the job names at least one, and its chip is shown as on', fm.length >= 1 && (await Promise.all(fm.map((f) => page.getByTestId(`export-chip-${f}`).first().evaluate((e) => e.classList.contains('on'))))).every(Boolean), JSON.stringify(fm));
  await page.getByTestId('backend-toggle-vllm').uncheck();
  await until(() => { try { return !JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm; } catch { return false; } });
  d = await deploy();
  check('switching one off removes only that one', !d.vllm && !!d.sglang && !!d.llamacpp, JSON.stringify(Object.keys(d)));
  await page.getByTestId('backend-toggle-sglang').uncheck();
  await page.getByTestId('backend-toggle-llamacpp').uncheck();
  await until(() => { try { const j = JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null'); return j && !j.deploy?.llamacpp && !j.deploy?.sglang && !j.deploy?.vllm; } catch { return false; } });
  check('switching all off is "no real deployment" again', /还没有选部署后端/.test(await text('backend-none')) && !(await on('vllm')));

  // ---------------------------------------------------------------- the options reach the job
  await page.getByTestId('backend-toggle-vllm').check();
  await page.getByTestId('opt-vllm-util').fill('0.4');
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm?.gpu_memory_utilization === 0.4; } catch { return false; } });
  check('the vLLM options in its card reach deploy.vllm', (await deploy()).vllm.gpu_memory_utilization === 0.4);
  await page.getByTestId('backend-toggle-vllm').uncheck();

  // ---------------------------------------------------------------- quantize-only
  await page.getByTestId('opt-quantize-only').check();
  check('"quantize only" greys out all three cards and says why', (await page.getByTestId('backend-toggle-vllm').isDisabled()) && (await page.getByTestId('backend-toggle-llamacpp').isDisabled()) && (await page.getByTestId('backend-toggle-sglang').isDisabled()) && /仅量化不评测已开启/.test(await text('backend-section')));
  await page.getByTestId('opt-quantize-only').uncheck();

  // ---------------------------------------------------------------- a reused artifact goes to exactly one backend
  await page.goto(`${base}/new?from=${PARENT}&step=deploy`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await page.getByTestId('backend-cards').waitFor({ timeout: 10000 });
  await until(() => document.querySelector('[data-testid="backend-card-vllm"]') !== null);
  const radios = await page.locator('[data-testid^="backend-toggle-"]').evaluateAll((els) => els.map((e) => e.type));
  check('reusing an exported checkpoint: the backends are a single choice (radio buttons), not a set', radios.length === 3 && radios.every((t) => t === 'radio'), JSON.stringify(radios));
  check('an HF checkpoint (AWQ) cannot be served by llama.cpp: that card is not selectable, and says why', await page.getByTestId('backend-toggle-llamacpp').isDisabled() && /不能用这个后端/.test(await text('backend-card-llamacpp')));
  check('a reused artifact has already been exported: no export-format chips offered', (await page.getByTestId('export-chip-awq').count()) === 0 && (await page.getByTestId('export-chip-gptq').count()) === 0);
  await page.getByTestId('backend-toggle-sglang').check();
  await until(() => { try { return !!JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.sglang; } catch { return false; } });
  d = await deploy();
  check('choosing SGLang for it puts deploy.sglang in the deploy job, and no vLLM', !!d.sglang && !d.vllm && !!(await preview()).artifact, JSON.stringify(Object.keys(d)));
  await page.getByTestId('backend-toggle-vllm').check();
  await until(() => { try { return !!JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.deploy?.vllm; } catch { return false; } });
  d = await deploy();
  check('choosing vLLM instead replaces it', !!d.vllm && !d.sglang, JSON.stringify(Object.keys(d)));

  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  failed = results.some((r) => !r);
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${String(e.message).split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
  fs.rmSync(root, { recursive: true, force: true });
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed || results.some((r) => !r) ? 1 : 0);

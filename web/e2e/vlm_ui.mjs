// UI test for the VLM domain: the wizard offers a real VLM form (not the "no pipeline" notice) and submits a job whose
// shape vlm_job.py reads; the job page, the standalone error-analysis page and the recipe comparison render a vlm report
// (a real MiniCPM-V-4.6 run, python/tests/fixtures/reports/vlm_full.json) with its numbers.
//
//   node web/e2e/vlm_ui.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const rep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/vlm_full.json'), 'utf8'));
const JOB = '20261012-000000-vlmfix';
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const MODEL = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-vlm-'));
{ // a header-only model so the live-estimate sidebar has something to read
  const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000, max_position_embeddings: 512, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const t = { 'model.embed_tokens.weight': [1000, 64] };
  let off = 0; const header = {};
  for (const [k, shape] of Object.entries(t)) { const n = shape.reduce((a, b) => a * b, 1) * 2; header[k] = { dtype: 'BF16', shape, data_offsets: [off, off + n] }; off += n; }
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb]));
}
const src = path.join(runs, JOB);
fs.mkdirSync(src);
fs.writeFileSync(path.join(src, 'report.json'), JSON.stringify(rep));
fs.writeFileSync(path.join(src, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
fs.writeFileSync(path.join(src, 'job.yaml'), ['metadata: {name: vlm-fixture}', `model: {type: vlm, path: ${MODEL}}`, 'quant: {recipe: [w2a16, w4a8]}', `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));

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
  const page = await browser.newPage({ viewport: { width: 1500, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const until = (fn, arg) => page.waitForFunction(fn, arg, { timeout: 10000 });

  // ---- wizard ----
  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  await page.getByTestId('domain-tab-vlm').click();
  await page.getByTestId('vlm-note').waitFor({ timeout: 10000 });
  check('the VLM tab is a form, not the "no pipeline" notice', (await page.getByTestId('domain-blocked').count()) === 0);
  check('and submitting is enabled', await page.getByRole('button', { name: '提交作业' }).isEnabled());
  await page.getByTestId('model-path').fill(MODEL);
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.model?.type === 'vlm'; } catch { return false; } });
  let j = await preview();
  check('the job is domain vlm / model.type vlm', j.domain === 'vlm' && j.model.type === 'vlm', JSON.stringify([j.domain, j.model]));
  check('default recipes are the two cheap ones, synthetic data, 24 answer tokens, attribution on',
    JSON.stringify(j.quant.recipe) === '["w8a16","w4a16"]' && j.calibration.dataset === 'synthetic' && j.evaluation.dataset === 'synthetic' && j.evaluation.max_new_tokens === 24 && j.evaluation.attribution === true, JSON.stringify(j));
  check('no LLM / VLA-only fields leak in (PPL windows, concurrency, action head, kv)', j.evaluation.max_windows === undefined && j.evaluation.concurrency === undefined && j.quant.also_quantize === undefined && j.quant.kv === undefined && j.quant.act_static === undefined);
  await page.getByTestId('vlm-max-new-tokens').fill('8');
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.max_new_tokens === 8; } catch { return false; } });
  check('max answer tokens is editable and reaches the job', (await preview()).evaluation.max_new_tokens === 8);
  await page.getByTestId('opt-quantize-only').check();
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.skip === true; } catch { return false; } });
  j = await preview();
  check('quantize-only sets evaluation.skip and switches attribution off', j.evaluation.skip === true && j.evaluation.attribution === false, JSON.stringify(j.evaluation));
  await page.getByTestId('opt-quantize-only').uncheck();
  await page.getByTestId('domain-tab-llm').click();
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.model?.type === 'llm'; } catch { return false; } });
  check('switching to LLM and back resets to that domain\'s own defaults', JSON.stringify((await preview()).quant.recipe) === '["w4a16","w4a8","w2a16"]');
  await page.getByTestId('domain-tab-vlm').click();
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('vlm-max-new-tokens').waitFor({ timeout: 5000 });
  await page.getByRole('button', { name: '提交作业' }).click();
  await page.waitForURL(/\/jobs\//, { timeout: 15000 });
  const jobId = page.url().split('/jobs/')[1];
  const yamlText = fs.readFileSync(path.join(runs, jobId, 'job.yaml'), 'utf8');
  check('the server accepted it and stored model.type vlm', /type: vlm/.test(yamlText) && /max_new_tokens: 24/.test(yamlText), yamlText.split('\n').slice(0, 6).join(' | '));
  const list = await (await fetch(`${base}/api/v1/jobs`)).json();
  check('the job list tags it as VLM', list.find((x) => x.job_id === jobId)?.kind === 'vlm' || list.find((x) => x.job_id === JOB)?.kind === 'vlm', JSON.stringify(list.map((x) => [x.job_id, x.kind])));

  // ---- job page ----
  await page.goto(`${base}/jobs/${JOB}`, { waitUntil: 'load' });
  await page.getByTestId('vlm-tiles').waitFor({ timeout: 20000 });
  const tiles = (await page.getByTestId('vlm-tiles').innerText()).replace(/\s+/g, ' ');
  check('tiles show the unquantized answer perplexity and ground-truth accuracy', tiles.includes(rep.baseline.answer_ppl.toFixed(4)) && tiles.includes('100%'), tiles.slice(0, 160));
  check('and the sample counts', tiles.includes(`${rep.data.calib.n} 校准 / ${rep.data.eval.n} 评测`), tiles.slice(0, 200));
  const stagesText = await page.locator('.pipeline').innerText();
  check('the pipeline shows the VLM stages (原模型答案, 模态归因), not the LLM/VLA ones', stagesText.includes('原模型答案') && stagesText.includes('模态归因') && !stagesText.includes('PPL'), stagesText.replace(/\s+/g, ' ').slice(0, 120));
  const rows = (await page.getByTestId('vlm-summary-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  const s2 = rep.recipes.W2A16.summary, s4 = rep.recipes.W4A8.summary;
  check('one summary row per recipe', rows.length === 2 && rows[0].startsWith('W2A16') && rows[1].startsWith('W4A8'), rows.join(' // '));
  check('W2A16 row: KL, 0.0% top-1, 0% exact, 0% correct', rows[0].includes(s2.kl.toExponential(2)) && rows[0].includes('0.0%') && rows[0].includes('0%'), rows[0]);
  check('W4A8 row: KL, 96.1% top-1, 38.9% exact, 69.0% prefix, 100% correct', rows[1].includes(s4.kl.toExponential(2)) && rows[1].includes('96.1%') && rows[1].includes('38.9%') && rows[1].includes('69.0%') && rows[1].includes('100%'), rows[1]);
  check('the worst modality is named (语言模型 for W4A8)', rows[1].includes('语言模型'), rows[1]);

  await page.getByText('W4A8 误差分析').first().click();
  await page.getByTestId('vlm-attribution').waitFor({ timeout: 10000 });
  const att = (await page.getByTestId('vlm-attribution').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('attribution lists vision and language with their KL', att.length === 2 && att[0].includes('视觉塔') && att[0].includes(rep.recipes.W4A8.attribution[0].kl.toExponential(2)) && att[1].includes('语言模型'), att.join(' // '));
  const ex = (await page.getByTestId('vlm-examples').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('the answers table shows 5 rows with the question and both answers', ex.length === 5 && ex[0].includes('What color is the circle') && ex[0].includes('blue'), ex[0]);
  check('"显示全部" reveals every example the report kept', await (async () => { await page.getByRole('button', { name: /显示全部/ }).click(); return (await page.getByTestId('vlm-examples').locator('tbody tr').count()) === rep.recipes.W4A8.eval.examples.length; })());
  check('the layer heatmap is there', (await page.getByText('逐层量化误差热力图').count()) > 0);
  await page.getByTestId('open-analysis-link').click();
  await page.waitForURL(/\/analysis\//, { timeout: 5000 });
  await page.getByText('逐层量化误差热力图').first().waitFor({ timeout: 10000 });
  check('the standalone error-analysis page renders the same heatmap for a VLM job', page.url().includes('recipe=W4A8'));

  // ---- compare ----
  await page.goto(`${base}/compare?tab=vlm`, { waitUntil: 'load' });
  await page.getByTestId('vlm-compare-table').waitFor({ timeout: 15000 });
  const cr = (await page.getByTestId('vlm-compare-table').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('the VLM comparison lists both recipes, the gentler one (lower KL) first', cr.length === 2 && cr[0].includes('W4A8') && cr[1].includes('W2A16'), cr.join(' // '));
  await page.getByTestId('compare-tab-vlm').waitFor();
  await page.goto(`${base}/compare?tab=vla`, { waitUntil: 'load' });
  await page.waitForTimeout(800);
  const sub = await page.locator('p.subtitle').innerText();
  check('a VLM job is counted as VLM, not as VLA, and the VLA tab does not choke on it', /VLM 1 个作业 · VLA 0 个作业/.test(sub) && errors.length === 0, sub);
  check('no page errors anywhere', errors.length === 0, errors.slice(0, 2).join(' | '));
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
}
const bad = results.filter((x) => !x).length;
console.log(`\n${results.length - bad}/${results.length} checks passed`);
process.exit(failed || bad ? 1 : 0);

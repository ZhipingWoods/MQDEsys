// The Compare page's matrix: runs and recipes against a baseline of your choice, each change coloured by whether it is better or worse
// for THAT metric (lower PPL is better, higher throughput is better), and only runs of the same model under the same evaluation
// protocol in one table.
//
//   node web/e2e/score_matrix.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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

const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-matrix-')));
const runs = path.join(root, 'runs'); fs.mkdirSync(runs);
const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));

const MIB = 2 ** 20;
const recipe = (name, algo, ppl, extra = {}) => {
  const r = JSON.parse(JSON.stringify(fixture.recipes.W4A16));
  r.recipe = name; r.algorithm = algo;
  r.eval = { fake_quant_wo_kv: { ppl, tokens: 1, windows: 1, seq_len: 1, seconds: 1 }, ...(extra.pplKv ? { fake_quant: { ppl: extra.pplKv, tokens: 1, windows: 1, seq_len: 1, seconds: 1 } } : {}) };
  delete r.sensitivity; delete r.kv; delete r.export;
  if (extra.export) r.export = extra.export;
  return r;
};
const serve = (served, tps) => ({ status: 'succeeded', served_ppl: served, load: { ok: 16, failed: 0, spec: { concurrency: 4, requests: 16, prompt_len: 256, n_predict: 64 }, output_tokens_per_s: tps } });
const job = (id, model, windows, seq, base, recipes) => {
  const rep = JSON.parse(JSON.stringify(fixture));
  rep.job_id = id; rep.spec = { ...rep.spec, name: id, model_path: `/m/${model}` };
  rep.baseline = { ppl: base, tokens: 1, windows, seq_len: seq, seconds: 1 };
  rep.data = { calib: { dataset: 'wikitext-2 train', n: 1, seq_len: 1 }, eval: { dataset: 'wikitext-2 test', windows, seq_len: seq } };
  rep.recipes = Object.fromEntries(recipes.map((r) => [r.recipe, r]));
  delete rep.llamacpp;
  const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(rep));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), `metadata: {name: ${id}}\nmodel: {type: llm, path: /m/${model}}\nquant: {recipe: [w4a16]}\noutput_dir: ${runs}\n`);
};
const A = '20261014-000000-mxa111', B = '20261014-000000-mxb222', E = '20261014-000000-mxe555', C = '20261014-000000-mxc333', D = '20261014-000000-mxd444', F = '20261014-000000-mxf666';
job(A, 'qwen', 40, 512, 18.10, [recipe('W4A16', 'gptq', 20.40, { pplKv: 20.35 }), recipe('W4A8', 'gptq', 21.80)]);
job(B, 'qwen', 40, 512, 18.10, [recipe('W4A16', 'awq', 20.60, { export: { awq: { status: 'exported', bytes: 448 * MIB, real_ppl: 20.65, vllm: serve(20.70, 900) } } })]);
job(E, 'qwen', 40, 512, 18.10, [recipe('W4A16', 'awq', 20.55, { export: { awq: { status: 'exported', bytes: 440 * MIB, real_ppl: 20.58, sglang: serve(20.60, 1500) } } })]);
// F differs from B by 0.005% in PPL: a change that small is neither better nor worse
job(F, 'qwen', 40, 512, 18.10, [recipe('W4A16', 'rtn', 20.601)]);
job(C, 'llama', 40, 512, 8.50, [recipe('W4A16', 'gptq', 9.00)]);
job(D, 'qwen', 4, 256, 17.00, [recipe('W4A16', 'gptq', 19.00)]);
const key = (id, r) => `${id}/${r}`;

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', root, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1400 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const cell = (id, r, m) => page.getByTestId(`cell-${key(id, r)}-${m}`);
  const txt = async (loc) => (await loc.innerText()).replace(/\s+/g, ' ').trim();
  const tone = (id, r, m) => cell(id, r, m).getAttribute('data-tone');
  const tokenColor = (name) => page.evaluate((n) => { const e = document.createElement('span'); e.style.color = `var(${n})`; document.body.appendChild(e); const c = getComputedStyle(e).color; e.remove(); return c; }, name);
  const colorOf = (loc) => loc.evaluate((e) => getComputedStyle(e).color);
  const pct = (x, b) => ((x / b - 1) * 100);
  const signed = (v, d = 2) => `${v > 0 ? '+' : ''}${v.toFixed(d)}%`;

  await page.goto(`${base}/compare`, { waitUntil: 'load' });
  await page.getByTestId('score-matrix').waitFor({ timeout: 20000 });

  // ---------------------------------------------------------------- groups: only comparable runs together
  const opts = await page.getByTestId('matrix-group').locator('option').allInnerTexts();
  check('three groups: same model + protocol (5 columns), another model, same model under another protocol', opts.length === 3 && /qwen · 40 窗口 × 512（5 列）/.test(opts[0]) && opts.some((o) => /llama · 40 窗口 × 512（1 列）/.test(o)) && opts.some((o) => /qwen · 4 窗口 × 256（1 列）/.test(o)), JSON.stringify(opts));
  const cols = await page.locator('[data-testid^="matrix-col-"]').allInnerTexts();
  check('the biggest group is shown first, with its five columns and no column for the other models or protocol', cols.length === 5 && !cols.some((c) => /llama|4 窗口/.test(c)), JSON.stringify(cols));
  check('the baseline defaults to the unquantized model', /原模型（未量化）/.test(await page.getByTestId('matrix-baseline').locator('option:checked').innerText()));

  // ---------------------------------------------------------------- deltas against the unquantized model
  const exp1 = pct(20.40, 18.10);
  check('PPL change against the original: +12.71% for A W4A16, shown with its sign', (await txt(cell(A, 'W4A16', 'ppl'))) === signed(exp1), await txt(cell(A, 'W4A16', 'ppl')));
  check('a PPL that went up is "bad"', (await tone(A, 'W4A16', 'ppl')) === 'bad');
  check('the best (lowest) PPL in the row is marked: A W4A16', (await cell(A, 'W4A16', 'ppl').getAttribute('data-best')) === '1' && (await cell(B, 'W4A16', 'ppl').getAttribute('data-best')) === '0' && (await cell(E, 'W4A16', 'ppl').getAttribute('data-best')) === '0');
  check('against the original, a metric the original does not have shows "–", not a number', (await txt(cell(B, 'W4A16', 'real_ppl'))) === '–' && (await txt(cell(E, 'W4A16', 'tps'))) === '–');
  check('metrics that no column has are not rows at all (no KV-cache PPL for B/E, but A has one)', (await page.getByTestId('matrix-row-ppl_kv').count()) === 1);
  await page.getByTestId('matrix-mode-abs').click();
  check('absolute mode prints the numbers with their units: 20.400, 900 tok/s, 448 MiB (bytes converted to MiB)', (await txt(cell(A, 'W4A16', 'ppl'))) === '20.400' && (await txt(cell(B, 'W4A16', 'tps'))) === '900 tok/s' && (await txt(cell(B, 'W4A16', 'size_mib'))) === '448 MiB', `${await txt(cell(B, 'W4A16', 'tps'))} | ${await txt(cell(B, 'W4A16', 'size_mib'))}`);
  check('and nothing is coloured in absolute mode', (await tone(A, 'W4A16', 'ppl')) === 'none');
  await page.getByTestId('matrix-mode-delta').click();

  // ---------------------------------------------------------------- another baseline, and the direction of each metric
  await page.getByTestId('matrix-baseline').selectOption(key(B, 'W4A16'));
  const good = await tokenColor('--good-text'), bad = await tokenColor('--critical');
  const dPpl = pct(20.40, 20.60);
  check('baseline B W4A16: A W4A16 changed by -0.97% in PPL, and a lower PPL is "good"', (await txt(cell(A, 'W4A16', 'ppl'))) === signed(dPpl) && (await tone(A, 'W4A16', 'ppl')) === 'good', await txt(cell(A, 'W4A16', 'ppl')));
  const dTps = pct(1500, 900);
  check('throughput: E 1500 vs B 900 is +66.67%, and a HIGHER throughput is "good"', (await txt(cell(E, 'W4A16', 'tps'))) === signed(dTps) && (await tone(E, 'W4A16', 'tps')) === 'good', await txt(cell(E, 'W4A16', 'tps')));
  check('size: E is 440 MiB against B\'s 448, -1.79%, smaller is "good"', (await txt(cell(E, 'W4A16', 'size_mib'))) === signed(pct(440, 448)) && (await tone(E, 'W4A16', 'size_mib')) === 'good');
  check('served PPL and real-weight PPL: E is lower than B, "good"', (await tone(E, 'W4A16', 'served_ppl')) === 'good' && (await tone(E, 'W4A16', 'real_ppl')) === 'good');
  check('W4A8 is worse than B on PPL: "bad"', (await tone(A, 'W4A8', 'ppl')) === 'bad');
  check('a change under 0.05% (F against B: +0.005%) is flat: not coloured good or bad', (await tone(F, 'W4A16', 'ppl')) === 'flat' && (await colorOf(cell(F, 'W4A16', 'ppl'))) === (await colorOf(cell(B, 'W4A16', 'ppl'))) && (await colorOf(cell(F, 'W4A16', 'ppl'))) !== good && (await colorOf(cell(F, 'W4A16', 'ppl'))) !== bad, `${await txt(cell(F, 'W4A16', 'ppl'))} ${await tone(F, 'W4A16', 'ppl')}`);
  check('a column with no value for a metric shows "–" (W4A8 never ran a server)', (await txt(cell(A, 'W4A8', 'tps'))) === '–' && (await tone(A, 'W4A8', 'tps')) === 'flat');
  check('the baseline column itself shows its absolute value, uncoloured', (await txt(cell(B, 'W4A16', 'ppl'))) === '20.600' && (await tone(B, 'W4A16', 'ppl')) === 'none');
  check('the header marks which column is the baseline', /（基线）/.test(await txt(page.getByTestId(`matrix-col-${key(B, 'W4A16')}`))));
  check('"good" and "bad" are drawn in the design tokens\' green and red', (await colorOf(cell(E, 'W4A16', 'tps'))) === good && (await colorOf(cell(A, 'W4A8', 'ppl'))) === bad && good !== bad);

  // ---------------------------------------------------------------- throughput from different engines is flagged
  const note = await txt(page.getByTestId('matrix-engine-note'));
  check('B served by vLLM and E by SGLang: the page says throughput across engines is not comparable', /vllm/.test(note) && /sglang/.test(note) && /不能直接比/.test(note), note);

  // ---------------------------------------------------------------- the other groups
  await page.getByTestId('matrix-group').selectOption({ label: 'qwen · 4 窗口 × 256（1 列）' });
  const cols2 = await page.locator('[data-testid^="matrix-col-"]').allInnerTexts();
  check('another protocol is another table: one column, baseline back to the unquantized model, PPL +11.76% against 17.00', cols2.length === 1 && /原模型/.test(await page.getByTestId('matrix-baseline').locator('option:checked').innerText()) && (await txt(cell(D, 'W4A16', 'ppl'))) === signed(pct(19, 17)), await txt(cell(D, 'W4A16', 'ppl')));
  check('and the note names its protocol', /4 窗口 × 256/.test(await txt(page.getByTestId('score-matrix'))));
  check('no engine note when no two engines are in the table', (await page.getByTestId('matrix-engine-note').count()) === 0);
  await page.getByTestId('matrix-group').selectOption({ label: 'llama · 40 窗口 × 512（1 列）' });
  check('another model: its own baseline 8.50, PPL +5.88%, and none of the qwen numbers', (await txt(cell(C, 'W4A16', 'ppl'))) === signed(pct(9.0, 8.5)) && (await page.locator('[data-testid^="matrix-col-"]').count()) === 1);
  const yMatrix = (await page.getByTestId('score-matrix').boundingBox()).y, yDetail = (await page.getByRole('heading', { name: '各作业配方明细' }).boundingBox()).y;
  check('the card is on the Compare page\'s LLM tab, above the per-job detail', yMatrix < yDetail, `${Math.round(yMatrix)} < ${Math.round(yDetail)}`);

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

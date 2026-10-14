// The LLM job report: a verdict per recipe against the project's accuracy budget, a per-window review (which texts lose the
// most), a model -> block -> module sunburst of the error, and a banner when a run did not finish cleanly.
//
//   node web/e2e/report_review.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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

const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-rep-')));
const runs = path.join(root, 'runs'); fs.mkdirSync(runs);
const MODEL = path.join(root, 'model'); fs.mkdirSync(MODEL);
fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify({ architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000, max_position_embeddings: 512, torch_dtype: 'bfloat16' }));

// ---- a report with per-window losses whose answers are known
const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
const N = 40, SPIKE = 7, BETTER = 21, ZERO = 33;
const base = Array.from({ length: N }, (_, i) => 2.8 + 0.05 * Math.sin(i));
// the typical change grows a little with the window index, so the median is not just one repeated number; one window is exactly unchanged
const shifted = (d, spike) => base.map((b, i) => b + (i === SPIKE ? spike : i === BETTER ? -0.02 : i === ZERO ? 0 : d + 0.0005 * i));
const deltas = (d, spike) => shifted(d, spike).map((q, i) => q - base[i]);
const medianOf = (v) => { const t = [...v].sort((a, b) => a - b); return t.length % 2 ? t[(t.length - 1) / 2] : (t[t.length / 2 - 1] + t[t.length / 2]) / 2; };
const signed4 = (x) => `${x >= 0 ? '+' : ''}${x.toFixed(4)}`;
const ppl = (v) => Math.exp(v.reduce((a, b) => a + b, 0) / v.length);
const previews = Array.from({ length: N }, (_, i) => (i === SPIKE ? 'The quick brown fox jumps over the lazy dog' : `window ${i} of an ordinary article about nothing in particular`));
const recipe = (name, algo, d, spike) => {
  const q = shifted(d, spike);
  const r = JSON.parse(JSON.stringify(fixture.recipes.W4A16));
  r.recipe = name; r.algorithm = algo;
  r.eval = { fake_quant_wo_kv: { ppl: ppl(q), tokens: 20440, windows: N, seq_len: 512, seconds: 1, window_nll: q } };
  r.summary = { ...r.summary, ppl: ppl(q), ppl_delta_pct: (ppl(q) / ppl(base) - 1) * 100 };
  delete r.export; delete r.sensitivity; delete r.kv;
  return r;
};
const full = JSON.parse(JSON.stringify(fixture));
full.baseline = { ppl: ppl(base), tokens: 20440, windows: N, seq_len: 512, seconds: 1, window_nll: base };
full.data = { calib: { dataset: root + '/corpus/notes', n: 32, seq_len: 512, files: 3, chars: 9000, tokens: 2400, truncated: false }, eval: { dataset: 'wikitext-2 test', windows: N, seq_len: 512, previews } };
full.recipes = { W4A16: recipe('W4A16', 'gptq', 0.025, 0.5), W4A8: recipe('W4A8', 'gptq', 0.08, 0.7), W2A16: recipe('W2A16', 'rtn', 0.9, 2.0), W8A16: recipe('W8A16', 'rtn', -0.01, -0.01), FP8: recipe('FP8', 'rtn', 0.0149, 0.0149) };
// an output head outside the blocks (embeddings / lm_head live there in a real model)
for (const r of Object.values(full.recipes)) r.layers.push({ ...r.layers[0], name: 'lm_head', block: -1, shape: [100, 64], out_sqnr_db: 45, w_sqnr_db: 40 });
delete full.llamacpp;
const old = JSON.parse(JSON.stringify(fixture));  // as written before per-window losses existed
const broken = JSON.parse(JSON.stringify(full));
broken.stages = broken.stages.map((s) => (s.stage === 'report' ? { ...s, status: 'failed', error: 'disk full' } : s)).concat([{ stage: 'deploy_vllm', status: 'failed', error: 'engine died' }]);
broken.recipes.W4A16.export = { awq: { status: 'exported', vllm: { status: 'degraded', reason: 'smoke test failed' }, sglang: { status: 'succeeded' } } };

const mkjob = (id, report, name) => { const d = path.join(runs, id); fs.mkdirSync(d);
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify({ ...report, job_id: id }));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), `metadata: {name: ${name}}\nmodel: {type: llm, path: ${MODEL}}\nquant: {recipe: [w4a16]}\noutput_dir: ${runs}\n`); };
const NEW = '20261014-000000-repnew', OLD = '20261014-000000-repold', BAD = '20261014-000000-repbad';
mkjob(NEW, full, 'rep-new'); mkjob(OLD, old, 'rep-old'); mkjob(BAD, broken, 'rep-bad');

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', root, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base_ = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base_}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1400 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const text = async (id) => (await page.getByTestId(id).innerText()).replace(/\s+/g, ' ').trim();

  // ---------------------------------------------------------------- the verdicts
  await page.goto(`${base_}/jobs/${NEW}`, { waitUntil: 'load' });
  await page.getByTestId('verdict-strip').waitFor({ timeout: 15000 });
  const status = async (r) => page.getByTestId(`verdict-${r}`).getAttribute('data-status');
  check('W4A16 at about +3.8% is within its 5% budget: pass', (await status('W4A16')) === 'pass' && /达标/.test(await text('verdict-badge-W4A16')), await text('verdict-delta-W4A16'));
  check('W4A8 at about +10% is over its 8% budget: fail, and says so', (await status('W4A8')) === 'fail' && /超出预算/.test(await text('verdict-badge-W4A8')), await text('verdict-delta-W4A8'));
  check('W2A16 has no gate: recorded only, never a failure however bad', (await status('W2A16')) === 'record' && /仅记录/.test(await text('verdict-badge-W2A16')));
  const fp8 = await text('verdict-FP8');
  check('FP8 at about +1.4% is over its 1% budget: fail (a looser budget would let it through)', (await status('FP8')) === 'fail' && /≤ 1%/.test(fp8) && /超出预算/.test(fp8), fp8);
  check('W4A16 and W4A8 show their own budgets, 5% and 8%', /≤ 5%/.test(await text('verdict-W4A16')) && /≤ 8%/.test(await text('verdict-W4A8')));
  check('a recipe the table does not name has no rule rather than an invented one', (await status('W8A16')) === 'none' && /没有门槛/.test(await text('verdict-badge-W8A16')));
  const p416 = await text('verdict-delta-W4A16');
  check('the change is printed with its sign and two decimals', /^\+\d+\.\d{2}%$/.test(p416) && Number(p416.replace(/[+%]/g, '')) > 3 && Number(p416.replace(/[+%]/g, '')) < 5, p416);
  const colorOf = async (loc) => loc.evaluate((e) => getComputedStyle(e).color);
  const nameColor = await colorOf(page.getByTestId('verdict-W4A8').locator('td b'));
  const improved = await text('verdict-delta-W8A16');
  const goodColor = await colorOf(page.getByTestId('verdict-delta-W8A16'));
  // what the two design tokens resolve to in this browser, so "good" and "bad" are checked as themselves, not just as "different"
  const tokenColor = (name) => page.evaluate((n) => { const e = document.createElement('span'); e.style.color = `var(${n})`; document.body.appendChild(e); const c = getComputedStyle(e).color; e.remove(); return c; }, name);
  check('a recipe whose PPL went DOWN is shown with a minus sign, in the "good" colour', /^-\d/.test(improved) && goodColor === (await tokenColor('--good-text')) && goodColor !== nameColor, `${improved} ${goodColor}`);
  check('one whose PPL went UP is in the "critical" colour, never the good one', (await colorOf(page.getByTestId('verdict-delta-W4A8'))) === (await tokenColor('--critical')) && (await tokenColor('--critical')) !== (await tokenColor('--good-text')));
  check('a PPL that went up (worse) is coloured as a warning, not like ordinary text', (await colorOf(page.getByTestId('verdict-delta-W4A8'))) !== nameColor && (await colorOf(page.getByTestId('verdict-delta-W4A16'))) === (await colorOf(page.getByTestId('verdict-delta-W4A8'))));
  check('the strip names the section of the architecture the budgets come from', /§11\.4/.test(await text('verdict-strip')));

  const calibTile = (await page.getByText('校准', { exact: true }).first().locator('xpath=..').innerText()).replace(/\s+/g, ' ');
  check('the calibration tile says what was read from a user-named corpus: where, how many files, how many tokens', /notes/.test(calibTile) && /3 个文件/.test(calibTile) && /2400 token/.test(calibTile), calibTile);

  // ---------------------------------------------------------------- the window review
  await page.getByTestId('window-review').waitFor({ timeout: 15000 });
  await page.getByTestId('window-recipe-W4A16').click();  // the chips follow the report's own recipe order, which puts unlisted recipes first
  const d416 = deltas(0.025, 0.5);
  const worse416 = d416.filter((x) => x > 0).length;
  check(`every window is counted: ${worse416} of 40 got worse for W4A16 (window ${BETTER} improved, window ${ZERO} did not change at all, so it is not "worse")`, (await text('window-worse')) === `${worse416} / 40` && worse416 === 38, await text('window-worse'));
  check('the median change is the middle of the sorted list (the mean of the two middle windows), not dragged up by the spike', (await text('window-median')) === signed4(medianOf(d416)), `${await text('window-median')} vs ${signed4(medianOf(d416))}`);
  check('the worst window is the spike, +0.5000', /^\+0\.5000$/.test(await text('window-worst')) && /窗口 7/.test(await text('window-tiles')), await text('window-worst'));
  const rows = await page.getByTestId('window-worst-table').locator('tbody tr').allInnerTexts();
  check('the table lists the 8 windows that changed most, the spike first with its own text', rows.length === 8 && /^7\s/.test(rows[0]) && /quick brown fox/.test(rows[0]), rows[0]);
  check('and the one that got better is not among the worst', !rows.some((r) => /^21\s/.test(r)));
  check('the spike row also shows the PPL that change alone would make: exp(0.5) - 1 = +64.9%', /\+64\.9%/.test(rows[0]) && /\+0\.5000/.test(rows[0]), rows[0]);
  check('the row also carries the two losses it is the difference of', new RegExp(`${base[SPIKE].toFixed(4)}\\s+${shifted(0.025, 0.5)[SPIKE].toFixed(4)}`).test(rows[0]), rows[0]);
  await page.getByTestId('window-recipe-W4A8').click();
  check('choosing another recipe re-draws the review: the spike is now +0.7000', /^\+0\.7000$/.test(await text('window-worst')), await text('window-worst'));
  check('the review tiles and the table show the same recipe (header names it)', /W4A8/.test(await text('window-worst-table')));
  check('both charts are drawn (the page renders charts as SVG)', (await page.getByTestId('window-review').locator('svg').count()) >= 2);

  // ---------------------------------------------------------------- the sunburst
  await page.goto(`${base_}/jobs/${NEW}?tab=W4A16`, { waitUntil: 'load' });
  await page.getByTestId('layer-sunburst').waitFor({ timeout: 15000 });
  const sum = await text('sunburst-summary');
  check('the sunburst counts what it drew: 169 layers, 24 blocks, and the 1 layer outside the blocks', /169 个量化层/.test(sum) && /24 个块/.test(sum) && /块外 1 层/.test(sum), sum);
  const worstBefore = await text('sunburst-worst');
  check('it names the single worst module (lowest output SQNR) in model-relative form', /^layers\.\d+\.\S+$/.test(worstBefore), worstBefore);
  const expectWorst = full.recipes.W4A16.layers.filter((l) => l.out_sqnr_db != null).reduce((a, b) => (b.out_sqnr_db < a.out_sqnr_db ? b : a)).name.replace('model.', '');
  check('and that is the true minimum of the report', worstBefore === expectWorst, `${worstBefore} vs ${expectWorst}`);
  await page.getByTestId('sunburst-metric-w_sqnr_db').click();
  const expectW = full.recipes.W4A16.layers.reduce((a, b) => (b.w_sqnr_db < a.w_sqnr_db ? b : a)).name.replace('model.', '');
  check('switching the colouring to weight SQNR re-derives the worst module', (await text('sunburst-worst')) === expectW, `${await text('sunburst-worst')} vs ${expectW}`);
  check('the sunburst is drawn, with sectors in it', (await page.getByTestId('layer-sunburst').locator('svg path').count()) > 50);
  check('the existing per-block SQNR analysis is still on the page, below the sunburst', (await page.getByText('各块平均输出 SQNR').count()) === 1 && (await page.getByTestId('worst-row').count()) > 0);

  // ---------------------------------------------------------------- an older report, and an unfinished run
  await page.goto(`${base_}/jobs/${OLD}`, { waitUntil: 'load' });
  await page.getByTestId('window-review-missing').waitFor({ timeout: 15000 });
  check('a report from before per-window losses says so instead of drawing nothing', (await page.getByTestId('window-review').count()) === 0 && /重跑/.test(await text('window-review-missing')));
  check('the verdicts still work on an old report', (await page.getByTestId('verdict-strip').count()) === 1 && (await status('W4A16')) !== null);
  check('a clean run has no "did not finish" banner', (await page.getByTestId('run-incomplete').count()) === 0);
  await page.goto(`${base_}/jobs/${BAD}`, { waitUntil: 'load' });
  await page.getByTestId('run-incomplete').waitFor({ timeout: 15000 });
  check('a run with a failed stage and a failed deployment says exactly which', (await page.getByTestId('incomplete-stage-report').count()) === 1 && (await page.getByTestId('incomplete-stage-deploy_vllm').count()) === 1 && /disk full/.test(await text('incomplete-stage-report')), await text('run-incomplete'));
  check('a degraded deployment is listed with its reason, a succeeded one is not', (await page.getByTestId('incomplete-deploy').count()) === 1 && /smoke test failed/.test(await text('incomplete-deploy')) && !/sglang/.test(await text('incomplete-deploy')), await text('incomplete-deploy'));
  check('and it warns that the numbers cover only the part that worked', /不要把它当作完整结果/.test(await text('run-incomplete')));

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

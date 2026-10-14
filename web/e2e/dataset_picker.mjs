// Calibration / evaluation data in the new-job wizard: pick a built-in dataset or give a path, and learn right away whether the
// server can read it. LLM calibration text (file or directory), VLA / VLM calibration and evaluation data separately, derive
// from an earlier job, and the server's own checks (/datasets, /datasets/inspect, submit-time validation).
//
//   node web/e2e/dataset_picker.mjs      # needs: built web/dist, built mqde, Playwright (see demo/)
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

// everything lives under one temp root, which is the only directory the server may read
const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-ds-')));
const runs = path.join(root, 'runs'); fs.mkdirSync(runs);
const hfCache = path.join(root, 'hf-empty'); fs.mkdirSync(hfCache);  // no wikitext in it: the built-in is "not on this server"
const MODEL = path.join(root, 'model'); fs.mkdirSync(MODEL);
{ const cfg = { architectures: ['Qwen2ForCausalLM'], model_type: 'qwen2', hidden_size: 64, intermediate_size: 128, num_hidden_layers: 2, num_attention_heads: 4, num_key_value_heads: 2, vocab_size: 1000, max_position_embeddings: 512, torch_dtype: 'bfloat16' };
  fs.writeFileSync(path.join(MODEL, 'config.json'), JSON.stringify(cfg));
  const header = { 'model.embed_tokens.weight': { dtype: 'BF16', shape: [1000, 64], data_offsets: [0, 128000] } };
  const hb = Buffer.from(JSON.stringify(header)); const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(hb.length));
  fs.writeFileSync(path.join(MODEL, 'model.safetensors'), Buffer.concat([len, hb])); }
const CORPUS = path.join(root, 'corpus'); fs.mkdirSync(path.join(CORPUS, 'sub'), { recursive: true });
fs.writeFileSync(path.join(CORPUS, 'a.txt'), 'calibration text '.repeat(50));
fs.writeFileSync(path.join(CORPUS, 'sub', 'b.jsonl'), JSON.stringify({ text: 'more text '.repeat(50) }) + '\n');
fs.writeFileSync(path.join(CORPUS, 'weights.bin'), Buffer.alloc(100));
const ONEFILE = path.join(root, 'notes.md'); fs.writeFileSync(ONEFILE, '# notes\n' + 'word '.repeat(100));
const BINFILE = path.join(root, 'blob.bin'); fs.writeFileSync(BINFILE, Buffer.alloc(10));
const EMPTYFILE = path.join(root, 'empty.txt'); fs.writeFileSync(EMPTYFILE, '');
const LEROBOT = path.join(root, 'lerobot'); fs.mkdirSync(path.join(LEROBOT, 'meta'), { recursive: true }); fs.mkdirSync(path.join(LEROBOT, 'data', 'chunk-000'), { recursive: true });
fs.writeFileSync(path.join(LEROBOT, 'meta', 'info.json'), '{}'); fs.writeFileSync(path.join(LEROBOT, 'data', 'chunk-000', 'file-000.parquet'), 'pq');
const NOTLEROBOT = path.join(root, 'not-lerobot'); fs.mkdirSync(NOTLEROBOT);
const OUTSIDE = '/etc';

const mkjob = (id, yaml) => { const d = path.join(runs, id); fs.mkdirSync(d);
  const rep = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(rep));
  fs.writeFileSync(path.join(d, 'events.jsonl'), JSON.stringify({ type: 'job_state', state: 'SUCCEEDED', seq: 1, ts: 1 }) + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), yaml.join('\n') + '\n'); };
const LLMJOB = '20261014-000000-dsllm', VLMJOB = '20261014-000000-dsvlm', VLMGONE = '20261014-000000-dsgone';
mkjob(LLMJOB, ['metadata: {name: ds-llm}', `model: {type: llm, path: ${MODEL}}`, 'quant: {recipe: [w4a16]}', `calibration: {dataset: ${CORPUS}, n_samples: 8, seq_len: 128}`,
  'evaluation: {dataset: wikitext-2-raw-v1:test, seq_len: 128, max_windows: 4}', `output_dir: ${runs}`]);
mkjob(VLMJOB, ['metadata: {name: ds-vlm}', `model: {type: vlm, path: ${MODEL}}`, 'quant: {recipe: [w4a16]}', 'calibration: {dataset: synthetic, n_samples: 8}',
  `evaluation: {dataset: ${LEROBOT}, n_samples: 6, max_new_tokens: 12}`, `output_dir: ${runs}`]);
mkjob(VLMGONE, ['metadata: {name: ds-gone}', `model: {type: vlm, path: ${MODEL}}`, 'quant: {recipe: [w4a16]}', 'calibration: {dataset: synthetic, n_samples: 8}',
  `evaluation: {dataset: ${path.join(root, 'gone')}, n_samples: 6, max_new_tokens: 12}`, `output_dir: ${runs}`]);

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', root, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'], env: { ...process.env, HF_HUB_CACHE: hfCache } });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }
const api = (p) => fetch(`${base}/api/v1${p}`).then(async (r) => ({ status: r.status, body: await r.json() }));
const post = (p, body) => fetch(`${base}/api/v1${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }).then(async (r) => ({ status: r.status, body: await r.json() }));

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  // ---------------------------------------------------------------- the server's side
  { const r = await api('/datasets');
    const wiki = r.body.builtin.find((b) => b.id === 'wikitext-2-raw-v1:train'), syn = r.body.builtin.find((b) => b.id === 'synthetic');
    check('GET /datasets lists the built-ins, and says wikitext-2 is not in this server\'s cache', r.status === 200 && wiki?.available === false && /cache/.test(wiki.reason) && syn?.available === true, JSON.stringify(r.body.builtin));
    check('it also tells the UI which directories may be used', JSON.stringify(r.body.allowed_roots) === JSON.stringify([root]), JSON.stringify(r.body.allowed_roots)); }
  { const r = await api(`/datasets/inspect?path=${encodeURIComponent(CORPUS)}`);
    check('inspect a directory: readable files and their formats, other files not counted', r.body.usable === true && r.body.files === 2 && r.body.formats['.txt'] === 1 && r.body.formats['.jsonl'] === 1 && r.body.kind === 'dir', JSON.stringify(r.body)); }
  { const r = await api(`/datasets/inspect?path=${encodeURIComponent(BINFILE)}`);
    check('inspect an unsupported file: not usable, and it names what is supported', r.body.usable === false && /\.bin/.test(r.body.reason) && /\.parquet/.test(r.body.reason), JSON.stringify(r.body)); }
  { const r = await api(`/datasets/inspect?path=${encodeURIComponent(path.join(root, 'nope'))}`);
    check('inspect a missing path', r.body.usable === false && r.body.kind === 'missing' && /does not exist/.test(r.body.reason)); }
  { const r = await api(`/datasets/inspect?path=${encodeURIComponent(OUTSIDE)}`);
    check('a path outside the allowed roots is refused (403), and no metadata about it leaks', r.status === 403 && !('files' in r.body), JSON.stringify(r.body)); }
  { const r = await api(`/datasets/inspect?path=${encodeURIComponent(root + '/../etc')}`);
    check('a ../ path that leaves the root is refused as well', r.status === 403, `${r.status}`); }
  { const r = await api(`/datasets/inspect?path=${encodeURIComponent(LEROBOT)}&kind=lerobot`), r2 = await api(`/datasets/inspect?path=${encodeURIComponent(NOTLEROBOT)}&kind=lerobot`);
    check('inspect as LeRobot: a real layout passes, an empty directory says what is missing', r.body.usable === true && r.body.files === 1 && r2.body.usable === false && /meta\/info\.json/.test(r2.body.reason), JSON.stringify([r.body, r2.body])); }
  { const r = await api('/datasets/inspect?path=synthetic'), r2 = await api(`/datasets/inspect?path=${encodeURIComponent(CORPUS)}&kind=zip`), r3 = await api('/datasets/inspect');
    check('built-ins pass without a path check; an unknown kind and a missing path are 400', r.body.builtin === true && r2.status === 400 && r3.status === 400, JSON.stringify([r.body, r2.status, r3.status])); }
  const llmJob = (cal) => ({ model: { type: 'llm', path: MODEL, source: 'local' }, quant: { recipe: ['w4a16'] }, calibration: { dataset: cal, n_samples: 4, seq_len: 64 }, evaluation: { dataset: 'wikitext-2-raw-v1:test', seq_len: 64, max_windows: 2 }, deploy: { device: 'cpu' } });
  { const r = await post('/jobs?dry_run=1', llmJob(BINFILE));
    check('submit-time check: an LLM job whose calibration file cannot be read is a 400 that names the field', r.status === 400 && r.body.problems?.[0]?.field === 'calibration.dataset' && /unsupported file type/.test(r.body.message), JSON.stringify(r.body)); }
  { const r = await post('/jobs?dry_run=1', llmJob(EMPTYFILE)); check('an empty calibration file is refused too', r.status === 400 && /empty/.test(r.body.message), JSON.stringify(r.body)); }
  { const r = await post('/jobs?dry_run=1', llmJob(CORPUS)), r2 = await post('/jobs?dry_run=1', llmJob('wikitext-2-raw-v1:train'));
    check('a readable directory passes, and the built-in is not checked at submit (the executor resolves it)', r.status === 200 && r.body.valid === true && r2.status === 200, JSON.stringify([r.body, r2.body])); }
  { const r = await post('/jobs?dry_run=1', llmJob(OUTSIDE)); check('a calibration path outside the sandbox is still 403 (the existing rule)', r.status === 403); }
  const vJob = (type, cal, ev) => ({ model: { type, path: MODEL, source: 'local' }, quant: { recipe: ['w4a16'] }, calibration: { dataset: cal, n_samples: 4 }, evaluation: { dataset: ev, n_samples: 4 }, deploy: {} });
  { const r = await post('/jobs?dry_run=1', vJob('vlm', 'synthetic', NOTLEROBOT));
    check('VLM: an evaluation directory that is not LeRobot is a 400 on evaluation.dataset only', r.status === 400 && r.body.problems.length === 1 && r.body.problems[0].field === 'evaluation.dataset', JSON.stringify(r.body)); }
  { const r = await post('/jobs?dry_run=1', vJob('vla', NOTLEROBOT, NOTLEROBOT));
    check('VLA: both fields are reported, together', r.status === 400 && r.body.problems.map((p) => p.field).sort().join() === 'calibration.dataset,evaluation.dataset', JSON.stringify(r.body)); }
  { const r = await post('/jobs?dry_run=1', vJob('vlm', 'synthetic', LEROBOT)); check('VLM with synthetic calibration and a LeRobot evaluation set is valid', r.status === 200, JSON.stringify(r.body)); }

  // ---------------------------------------------------------------- the wizard
  const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  const preview = async () => JSON.parse(await page.locator('pre.log').innerText());
  const submit = page.getByRole('button', { name: '提交作业' });
  const until = (fn, arg) => page.waitForFunction(fn, arg, { timeout: 10000 });
  const status = async (id) => (await page.getByTestId(id).innerText()).replace(/\s+/g, ' ');

  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('calib-dataset-select').waitFor({ timeout: 10000 });
  check('LLM: there is a calibration data choice, wikitext-2 by default', (await page.getByTestId('calib-dataset-select').inputValue()) === 'wikitext-2-raw-v1:train' && (await preview()).calibration.dataset === 'wikitext-2-raw-v1:train');
  await page.getByTestId('calib-dataset-unavailable').waitFor({ timeout: 10000 });
  check('and it warns that wikitext-2 is not on this server, instead of failing later in the job', /找不到/.test(await status('calib-dataset-unavailable')) && await submit.isEnabled());
  const opts = await page.getByTestId('calib-dataset-select').locator('option').allInnerTexts();
  check('the other choice is a local file or directory', opts.length === 2 && /本地文件或目录/.test(opts[1]), JSON.stringify(opts));

  await page.getByTestId('calib-dataset-select').selectOption({ label: '本地文件或目录…' });
  check('choosing it shows a path box (empty) and says to fill it; submitting is blocked', (await page.getByTestId('calib-dataset').inputValue()) === '' && /填一个路径/.test(await status('calib-dataset-status')) && await submit.isDisabled());
  check('it names the directories the server allows', (await page.getByTestId('calib-dataset-picker').innerText()).includes(root));
  await page.getByTestId('calib-dataset').fill(CORPUS);
  await until(() => /✓/.test(document.querySelector('[data-testid="calib-dataset-status"]')?.textContent ?? ''));
  const ok = await status('calib-dataset-status');
  check('a directory with text in it: ✓, with the number of readable files, size and formats (the .bin is not counted)', /目录 · 2 个可读文件/.test(ok) && /\.txt×1/.test(ok) && /\.jsonl×1/.test(ok), ok);
  check('submitting is enabled and the JobSpec carries the path', await submit.isEnabled() && (await preview()).calibration.dataset === CORPUS);
  await page.getByTestId('calib-dataset').fill(ONEFILE);
  await until(() => /✓ 文件/.test(document.querySelector('[data-testid="calib-dataset-status"]')?.textContent ?? ''));
  check('a single text file works too', await submit.isEnabled() && (await preview()).calibration.dataset === ONEFILE);

  for (const [what, p, re] of [['an unsupported file', BINFILE, /✗.*\.bin.*supported/], ['an empty file', EMPTYFILE, /✗.*empty/], ['a path that does not exist', path.join(root, 'nope'), /✗.*does not exist/], ['a path outside the allowed directories', OUTSIDE, /不在服务器允许的目录/]]) {
    await page.getByTestId('calib-dataset').fill(p);
    // right after typing, before the server has answered, the previous path's ✓ must not be trusted for the new one
    check(`${what}: submitting is blocked at once, while the new path is still being checked`, await submit.isDisabled() && /检查中/.test(await status('calib-dataset-status')), await status('calib-dataset-status'));
    await until((s) => (document.querySelector('[data-testid="calib-dataset-status"]')?.textContent ?? '') !== s && !/检查中/.test(document.querySelector('[data-testid="calib-dataset-status"]')?.textContent ?? ''), await status('calib-dataset-status'));
    check(`${what}: the reason is shown and submitting is blocked`, re.test(await status('calib-dataset-status')) && await submit.isDisabled(), await status('calib-dataset-status'));
  }
  await page.getByTestId('calib-dataset-select').selectOption('wikitext-2-raw-v1:train');
  check('going back to the built-in re-enables submitting and puts it back in the JobSpec', await submit.isEnabled() && (await preview()).calibration.dataset === 'wikitext-2-raw-v1:train' && (await page.getByTestId('calib-dataset').count()) === 0);

  // submit with a real path: the server accepts it and keeps it in job.yaml
  await page.getByTestId('calib-dataset-select').selectOption({ label: '本地文件或目录…' });
  await page.getByTestId('calib-dataset').fill(CORPUS);
  await until(() => /✓/.test(document.querySelector('[data-testid="calib-dataset-status"]')?.textContent ?? ''));
  await submit.click();
  await page.waitForURL(/\/jobs\//, { timeout: 15000 });
  const jid = page.url().split('/jobs/')[1];
  const y = fs.readFileSync(path.join(runs, jid, 'job.yaml'), 'utf8');
  check('the server accepted the job and stored the chosen calibration path', y.includes(`dataset: ${CORPUS}`) && /type: llm/.test(y), y.split('\n').filter((l) => /dataset/.test(l)).join(' | '));

  // ---- VLM: calibration and evaluation data are chosen separately
  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  await page.getByTestId('domain-tab-vlm').click();
  await page.getByTestId('model-path').fill(MODEL);
  await page.getByTestId('dataset-path-select').waitFor({ timeout: 10000 });
  await page.getByTestId('eval-dataset-path-select').waitFor({ timeout: 10000 });
  let j = await preview();
  check('VLM: two choices, both synthetic by default', (await page.getByTestId('dataset-path-select').inputValue()) === 'synthetic' && (await page.getByTestId('eval-dataset-path-select').inputValue()) === 'synthetic' && j.calibration.dataset === 'synthetic' && j.evaluation.dataset === 'synthetic');
  await page.getByTestId('eval-dataset-path-select').selectOption({ label: 'LeRobot v3 目录…' });
  await page.getByTestId('eval-dataset-path').fill(NOTLEROBOT);
  await until(() => /✗/.test(document.querySelector('[data-testid="eval-dataset-path-status"]')?.textContent ?? ''));
  check('a directory that is not LeRobot says what is missing and blocks submitting', /meta\/info\.json/.test(await status('eval-dataset-path-status')) && await submit.isDisabled(), await status('eval-dataset-path-status'));
  await page.getByTestId('eval-dataset-path').fill(LEROBOT);
  await until(() => /✓/.test(document.querySelector('[data-testid="eval-dataset-path-status"]')?.textContent ?? ''));
  j = await preview();
  check('a real LeRobot directory is accepted; only the evaluation set changed, calibration stays synthetic', j.calibration.dataset === 'synthetic' && j.evaluation.dataset === LEROBOT && await submit.isEnabled(), JSON.stringify([j.calibration, j.evaluation]));
  await submit.click();
  await page.waitForURL(/\/jobs\//, { timeout: 15000 });
  const y2 = fs.readFileSync(path.join(runs, page.url().split('/jobs/')[1], 'job.yaml'), 'utf8');
  check('the server stored different calibration and evaluation data', /calibration:[\s\S]*dataset: synthetic/.test(y2) && y2.includes(`dataset: ${LEROBOT}`), y2.split('\n').filter((l) => /dataset/.test(l)).join(' | '));

  // ---- VLA: the default data is a path on the dev server, which this server cannot read
  await page.goto(`${base}/new?mode=all`, { waitUntil: 'load' });
  await page.getByTestId('domain-tab-vla').click();
  await page.getByTestId('dataset-path-fellback').waitFor({ timeout: 10000 });
  await page.getByTestId('eval-dataset-path-fellback').waitFor({ timeout: 10000 });
  j = await preview();
  check('VLA: a default this server cannot read falls back to synthetic for both, says so, and the form is usable',
    /用不了/.test(await status('dataset-path-fellback')) && /synthetic/.test(await status('dataset-path-fellback')) && j.calibration.dataset === 'synthetic' && j.evaluation.dataset === 'synthetic' && await submit.isEnabled(), await status('dataset-path-fellback'));
  await page.getByTestId('dataset-path-select').selectOption({ label: 'LeRobot v3 目录…' });
  await page.getByTestId('dataset-path').fill(OUTSIDE);
  await until(() => /不在服务器允许的目录/.test(document.querySelector('[data-testid="dataset-path-status"]')?.textContent ?? ''));
  check('VLA: a path the user typed is never replaced by the fallback, even though it is unreadable: it is shown, and blocks submitting', (await page.getByTestId('dataset-path').inputValue()) === OUTSIDE && await submit.isDisabled() && (await page.getByTestId('dataset-path-fellback').count()) === 0);
  await page.getByTestId('dataset-path-select').selectOption('synthetic');
  check('choosing synthetic again unblocks it', await submit.isEnabled());
  await page.getByTestId('eval-dataset-path-select').selectOption({ label: 'LeRobot v3 目录…' });
  await page.getByTestId('eval-dataset-path').fill(LEROBOT);
  await until(() => /✓/.test(document.querySelector('[data-testid="eval-dataset-path-status"]')?.textContent ?? ''));
  j = await preview();
  check('VLA: the evaluation data can differ from the calibration data and reaches evaluation.dataset only', j.calibration.dataset === 'synthetic' && j.evaluation.dataset === LEROBOT, JSON.stringify([j.calibration, j.evaluation]));

  // ---- derive keeps what the parent used
  await page.goto(`${base}/new?mode=all&from=${LLMJOB}&step=eval`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await page.getByTestId('calib-dataset').waitFor({ timeout: 10000 });
  await until(() => /✓/.test(document.querySelector('[data-testid="calib-dataset-status"]')?.textContent ?? ''));
  check('deriving from a job that calibrated on a directory shows that directory, checked, and keeps it', (await page.getByTestId('calib-dataset').inputValue()) === CORPUS && (await preview()).calibration.dataset === CORPUS && await submit.isEnabled());
  await page.goto(`${base}/new?mode=all&from=${VLMJOB}&step=eval`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await until(() => { try { return JSON.parse(document.querySelector('pre.log')?.textContent ?? 'null')?.evaluation?.dataset; } catch { return false; } });
  j = await preview();
  check('deriving a VLM job keeps calibration and evaluation data apart', j.calibration.dataset === 'synthetic' && j.evaluation.dataset === LEROBOT, JSON.stringify([j.calibration, j.evaluation]));
  await page.goto(`${base}/new?mode=all&from=${VLMGONE}&step=eval`, { waitUntil: 'load' });
  await page.getByTestId('derive-banner').waitFor({ timeout: 10000 });
  await page.getByTestId('eval-dataset-path').waitFor({ timeout: 10000 });
  await until(() => /✗/.test(document.querySelector('[data-testid="eval-dataset-path-status"]')?.textContent ?? ''));
  check('deriving a job whose data has since gone: the missing path is shown and blocks submitting, not silently swapped for synthetic',
    (await page.getByTestId('eval-dataset-path').inputValue()) === path.join(root, 'gone') && await submit.isDisabled() && (await page.getByTestId('eval-dataset-path-fellback').count()) === 0, await status('eval-dataset-path-status'));

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

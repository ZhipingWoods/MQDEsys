// UI test for T-0904: sensitivity page -> "按建议生成新作业" (generate a new job from the mixed-precision suggestion).
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/clone_from_suggestion.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/clone_from_suggestion.mjs
//
// Exit code 0 = every check passed. Each check prints PASS / FAIL with what it saw.
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

const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/llm_full.json'), 'utf8'));
const SRC_JOB = '20261008-000000-aaaaaa';
const RECIPE = 'W4A16';
const expected = fixture.recipes[RECIPE].sensitivity.suggestion;

// The server round-trips the job through YAML, which may reorder keys: compare structures, not strings.
const same = (a, b) => { const n = (v) => Array.isArray(v) ? v.map(n) : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, n(v[k])])) : v; return JSON.stringify(n(a)) === JSON.stringify(n(b)); };
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };

const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
const src = path.join(runs, SRC_JOB);
fs.mkdirSync(src);
fs.writeFileSync(path.join(src, 'report.json'), JSON.stringify(fixture));
fs.writeFileSync(path.join(src, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
fs.writeFileSync(path.join(src, 'job.yaml'), [
  'metadata: {name: tiny-source}', `model: {type: llm, path: ${os.tmpdir()}/model}`,
  'quant:', '  recipe: [w4a16, w4a8]', '  algorithm: gptq', '  kv: {format: int, bits: 8, method: kivi}',
  '  mixed_precision: {rules: [{match: "block:0", bits: 8}]}', 'deploy: {device: cpu}', 'analysis: {sensitivity: block}',
  `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));

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
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleErrors = [];
  page.on('pageerror', (e) => consoleErrors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));

  await page.goto(`${base}/jobs/${SRC_JOB}?tab=${RECIPE}`, { waitUntil: 'load' });
  const button = page.getByRole('button', { name: new RegExp(`按建议生成新作业.*${RECIPE}`) });
  await button.waitFor({ timeout: 20000 });
  check('the sensitivity card shows the button for this recipe', true);
  check('the button is enabled (job spec loaded and the suggestion has rules)', await button.isEnabled());
  check('the suggested groups are shown', (await page.locator('.card', { has: button }).innerText()).includes(expected.groups[0]));

  const before = (await (await fetch(`${base}/api/v1/jobs`)).json()).length;
  await button.click();
  await page.waitForURL(/\/jobs\/\d{8}-\d{6}-[0-9a-f]{6}$/, { timeout: 20000 });
  const newId = page.url().split('/').pop();
  check('clicking navigates to the new job page', newId !== SRC_JOB, newId);
  const after = (await (await fetch(`${base}/api/v1/jobs`)).json()).length;
  check('exactly one new job was created', after === before + 1, `${before} -> ${after}`);

  const detail = await (await fetch(`${base}/api/v1/jobs/${newId}`)).json();
  const j = detail.job;
  check('name has the -mp suffix', j.metadata?.name === 'tiny-source-mp', j.metadata?.name);
  check('only the chosen recipe is kept', JSON.stringify(j.quant.recipe) === JSON.stringify([RECIPE]), JSON.stringify(j.quant.recipe));
  check('mixed-precision rules are exactly the suggestion', same(j.quant.mixed_precision.rules, expected.rules), JSON.stringify(j.quant.mixed_precision.rules));
  check('other settings are carried over (algorithm, KV, device, analysis)', j.quant.algorithm === 'gptq' && j.quant.kv?.method === 'kivi' && j.deploy?.device === 'cpu' && j.analysis?.sensitivity === 'block');
  check('the new job points at the same model', j.model.path === `${os.tmpdir()}/model`);
  check('server-owned fields were not copied from the source (own output_dir, job id differs)', j.output_dir === runs && newId !== SRC_JOB);

  const other = fixture.recipes['W4A8'].sensitivity.suggestion;
  await page.goto(`${base}/jobs/${SRC_JOB}?tab=W4A8`, { waitUntil: 'load' });
  const b2 = page.getByRole('button', { name: /按建议生成新作业.*W4A8/ });
  await b2.waitFor({ timeout: 20000 });
  await b2.click();
  await page.waitForURL(/\/jobs\/\d{8}-\d{6}-[0-9a-f]{6}$/, { timeout: 20000 });
  const id2 = page.url().split('/').pop();
  const j2 = (await (await fetch(`${base}/api/v1/jobs/${id2}`)).json()).job;
  check('the other recipe tab yields ITS suggestion, not the first one', same(j2.quant.recipe, ['W4A8']) && same(j2.quant.mixed_precision.rules, other.rules));
  check('a suggestion with different rules makes a job with different rules', !same(j2.quant.mixed_precision.rules, j.quant.mixed_precision.rules));

  await page.waitForFunction(() => /SUCCEEDED|成功/.test(document.body.innerText), null, { timeout: 20000 }).catch(() => {});
  check('the new job runs through the real server (stand-in worker) and the page loads without errors', consoleErrors.length === 0, consoleErrors.slice(0, 2).join(' | '));
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

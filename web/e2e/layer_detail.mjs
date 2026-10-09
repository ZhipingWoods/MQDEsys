// UI test for T-0903: clicking a worst-layer row opens its detail card with metrics and ranks.
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/layer_detail.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/layer_detail.mjs
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


const layers = fixture.recipes.W4A16.layers;
const rankOf = (l, key, lowIsWorse) => { const xs = layers.map((x) => x[key]).filter((v) => typeof v === 'number'); return xs.filter((x) => (lowIsWorse ? x < l[key] : x > l[key])).length + 1; };
const worst = [...layers].filter((l) => l.out_sqnr_db != null).sort((a, b) => a.out_sqnr_db - b.out_sqnr_db);
const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`${base}/jobs/${SRC_JOB}?tab=W4A16`, { waitUntil: 'load' });
  await page.getByTestId('worst-row').first().waitFor({ timeout: 20000 });
  check('no detail card before anything is clicked', (await page.getByTestId('layer-detail').count()) === 0);
  for (const idx of [0, 3]) {
    const want = worst[idx];
    await page.getByTestId('worst-row').nth(idx).click();
    const card = page.getByTestId('layer-detail');
    await card.waitFor({ timeout: 5000 });
    const text = (await card.innerText()).replace(/\s+/g, ' ');
    const short = want.name.replace(/^model\./, ''); // the LLM page strips the leading "model."
    check(`row ${idx}: the card is for that layer`, text.includes(short), short);
    check(`row ${idx}: output SQNR shown`, text.includes(want.out_sqnr_db.toFixed(2) + ' dB'), want.out_sqnr_db.toFixed(2));
    const rk = rankOf(want, 'out_sqnr_db', true);
    check(`row ${idx}: its rank by output SQNR is ${rk} / ${layers.filter((x) => x.out_sqnr_db != null).length}`, text.includes(`${rk} / ${layers.filter((x) => x.out_sqnr_db != null).length}`));
    const wr = rankOf(want, 'w_sqnr_db', true);
    check(`row ${idx}: its weight-SQNR rank is ${wr}`, text.includes(`${wr} / ${layers.length}`));
  }
  check('the worst layer is rank 1 by output SQNR', rankOf(worst[0], 'out_sqnr_db', true) === 1);
  await page.getByRole('button', { name: '关闭' }).click();
  check('the close button removes the card', (await page.getByTestId('layer-detail').count()) === 0);
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
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

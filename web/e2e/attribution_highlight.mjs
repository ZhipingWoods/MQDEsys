// UI test for T-0903: the VLA job page names, per recipe, the modality whose quantization hurts the action most.
//
// Starts its own `mqde serve` on a temp runs dir with a stand-in worker (fake_worker.py), so clicking the button submits
// through the real REST API but never touches a GPU or a model. The finished job it opens is a real report fixture.
//
//   node web/e2e/attribution_highlight.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
//   MQDE_BIN=build/dev/bin/mqde PLAYWRIGHT_DIR=demo node web/e2e/attribution_highlight.mjs
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

const fixture = JSON.parse(fs.readFileSync(path.join(repo, 'python/tests/fixtures/reports/vla_attribution.json'), 'utf8'));
const JOB = '20260924-000000-cccccc';
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));
// Two finished jobs: the plain fixture (an old report without distributions) and a copy whose worst layer carries them.

const src = path.join(runs, JOB);
fs.mkdirSync(src);
fs.writeFileSync(path.join(src, 'report.json'), JSON.stringify(fixture));
fs.writeFileSync(path.join(src, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
fs.writeFileSync(path.join(src, 'job.yaml'), ['metadata: {name: vla-fixture}', `model: {type: vla, path: ${os.tmpdir()}/model}`, 'quant: {recipe: [w4a16]}', `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));
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
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`${base}/jobs/${JOB}`, { waitUntil: 'load' });
  await page.getByTestId('attribution-largest').waitFor({ timeout: 20000 });
  const text = (await page.getByTestId('attribution-largest').innerText()).replace(/\s+/g, ' ');
  for (const [name, rec] of Object.entries(fixture.recipes)) {
    const big = [...rec.attribution].sort((a, b) => b.action_rel_l2 - a.action_rel_l2)[0];
    const pct = Math.round((100 * big.action_rel_l2) / rec.eval.action_rel_l2);
    const item = [...text.matchAll(new RegExp(`${name}：[^。]*。`, 'g'))].map((m) => m[0])[0] ?? '';
    check(`${name}: names ${big.modality} as the largest, ${big.action_rel_l2.toFixed(4)}, ${pct}% of the all-quantized error`,
      item.includes(`量化 ${big.modality} 的误差最大`) && item.includes(big.action_rel_l2.toFixed(4)) && item.includes(`${pct}%`), item);
  }
  const mods = new Set(Object.values(fixture.recipes).map((r) => [...r.attribution].sort((a, b) => b.action_rel_l2 - a.action_rel_l2)[0].modality));
  check('the fixture is not trivial: both modalities are the largest for some recipe', mods.size === 2, [...mods].join(','));
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

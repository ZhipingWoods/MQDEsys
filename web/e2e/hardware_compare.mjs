// UI test for T-1308: Compare.tsx's "按硬件对比" section only appears once the same recipe has been run
// on more than one real hardware (distinguished by env.eval_device_name, not just a submitted label),
// and stays hidden otherwise rather than showing a one-row "comparison" that compares nothing.
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
const results = [];
const check = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };
const freePort = () => new Promise((res) => { const s = net.createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));

const writeJob = (id, report) => {
  const dir = path.join(runs, id);
  fs.mkdirSync(dir);
  fs.writeFileSync(path.join(dir, 'report.json'), JSON.stringify({ ...report, job_id: id }));
  fs.writeFileSync(path.join(dir, 'events.jsonl'), [{ type: 'job_state', state: 'RUNNING', seq: 1, ts: 1 }, { type: 'job_state', state: 'SUCCEEDED', seq: 2, ts: 2 }].map((e) => JSON.stringify(e)).join('\n') + '\n');
  fs.writeFileSync(path.join(dir, 'job.yaml'), [
    'metadata: {name: hw-compare-fixture}', `model: {type: llm, path: ${os.tmpdir()}/model}`,
    'quant:', '  recipe: [w4a16]', '  algorithm: gptq', 'deploy: {device: cpu}', `output_dir: ${runs}`, 'apiVersion: mqde/v1', 'kind: Job', ''].join('\n'));
};

// Job A: the fixture as-is (env.device only, the pre-T-1308 shape -- falls back to spec.device="cpu").
const JOB_CPU = '20261008-000010-dddddd';
writeJob(JOB_CPU, fixture);

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
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  // ---- 1. only one hardware among the finished jobs so far: the section must stay hidden ----
  await page.goto(`${base}/compare`, { waitUntil: 'load' });
  await page.getByText('各作业配方明细').waitFor({ timeout: 10000 });
  check('hidden when every job ran on the same hardware', (await page.getByTestId('hardware-compare').count()) === 0);

  // ---- 2. add a second job, same recipe, a different *real measured* device name ----
  const JOB_GPU = '20261008-000011-eeeeee';
  const onGpu = JSON.parse(JSON.stringify(fixture));
  onGpu.env = { ...onGpu.env, eval_device_name: 'NVIDIA GeForce RTX 4090', eval_device: 'cuda:0' };
  for (const s of onGpu.stages) if (s.stage === 'quantize' || s.stage === 'eval_fake_wo_kv') s.seconds = s.seconds / 8; // GPU: faster
  writeJob(JOB_GPU, onGpu);
  await page.goto(`${base}/compare`, { waitUntil: 'load' });
  await page.getByTestId('hardware-compare').waitFor({ timeout: 10000 });
  check('appears once the same recipe has 2 distinct hardwares', true);
  const rowCpu = await page.getByTestId('hw-compare-row-W4A16-cpu').innerText();
  const rowGpu = await page.getByTestId(`hw-compare-row-W4A16-NVIDIA GeForce RTX 4090`).innerText();
  check('cpu row is labelled cpu', rowCpu.includes('cpu'), rowCpu.replace(/\s+/g, ' '));
  check('gpu row is labelled with the real device name', rowGpu.includes('NVIDIA GeForce RTX 4090'), rowGpu.replace(/\s+/g, ' '));
  check('PPL is the same (error is hardware-independent for the same fake-quant run)', rowCpu.match(/[\d.]+$/)?.[0] === rowGpu.match(/[\d.]+$/)?.[0], `${rowCpu} vs ${rowGpu}`);
  check('quantize seconds differ (throughput IS hardware-dependent)', !rowCpu.includes(onGpu.stages.find((s) => s.stage === 'quantize').seconds.toFixed(2)));

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

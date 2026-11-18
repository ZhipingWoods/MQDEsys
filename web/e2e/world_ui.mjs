// World-model / EWM report UI: PSNR overview, recipe table, attribution, per-clip scores, per-layer
// analysis, dynamic skip suggestion, and the WM tab on the comparison page.
//
//   node web/e2e/world_ui.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-world-'));
const model = path.join(runs, 'TinyDiT');

const layer = (name, component, modality, block, sqnr, kurtosis) => ({
  name, component, modality, block, shape: [128, 128],
  w_sqnr_db: 22.5, w_rel_l2: 0.075, w_cosine: 0.997,
  out_sqnr_db: sqnr, out_rel_l2: 0.18, out_cosine: 0.985,
  act_absmax: 4.2, act_kurtosis: kurtosis, act_outlier_ratio: 0.0021, w_kurtosis: 3.1,
  algorithm: 'rtn', algo_info: { bits: 4, act: 'fp16' },
});
const layers = (base) => [
  layer('transformer_blocks.0.attn.q_proj', 'transformer', 'world', 0, base + 9, 8.1),
  layer('transformer_blocks.0.mlp.fc1', 'transformer', 'world', 0, base + 12, 4.2),
  layer('transformer_blocks.1.attn.k_proj', 'transformer', 'world', 1, base + 6, 11.3),
  layer('transformer_blocks.1.mlp.fc2', 'transformer', 'world', 1, base + 14, 3.8),
];

function report(id, kind) {
  const r = {};
  for (const [name, psnr, mse, base, ap] of [['W8A16', 81.81, 6.6e-9, 27.0, 24.0], ['W4A16', 57.49, 1.8168e-6, 21.0, 18.0]]) {
    r[name] = {
      recipe: name, algorithm: 'rtn', maturity: 'stable',
      weight: { bits: name === 'W8A16' ? 8 : 4, granularity: 'per_group', group_size: 128, symmetric: true, enabled: true },
      act: { bits: 16, granularity: 'per_token', symmetric: true, enabled: true },
      layers: layers(base),
      attribution: (kind === 'ewm' ? ['world', 'action'] : ['world']).map((m, i) => ({
        modality: m, units: m === 'action' ? 2 : 4, psnr_vs_ref: psnr - i * 2.4, mse_vs_ref: mse * (1 + i), psnr_vs_gt: base - 12 + i,
      })),
      eval: {
        psnr_vs_ref: psnr, mse_vs_ref: mse, psnr_vs_gt: base - 12, mse_vs_gt: 0.04,
        psnr_drop_db: 4.96 - (base - 12), mse_ratio_vs_gt: 1.1,
        lpips_vs_ref: null, latency_ms_per_video: 41.2, latency_ms_per_frame: 10.3,
        peak_mem_bytes: 14364672, peak_mem_note: 'CUDA max_memory_allocated',
        per_sample: [0, 1, 2, 3].map((i) => ({ psnr: psnr - i * 0.3, mse: mse * (1 + i * 0.1), psnr_vs_gt: base - 12 - i * 0.2, mse_vs_gt: 0.04, seed_id: i })),
      },
      summary: {
        psnr_vs_ref: psnr, mse_vs_ref: mse, psnr_vs_gt: base - 12,
        psnr_drop_db: 4.96 - (base - 12), mse_ratio_vs_gt: 1.1, lpips_vs_ref: null,
        psnr_over_noise_floor: psnr / 5.4924, below_noise_floor: true, worst_stage: 'world',
        worst_layers: [{ name: 'transformer_blocks.1.attn.k_proj', out_sqnr_db: base + 6, act_kurtosis: 11.3 }],
        component_mean_out_sqnr_db: { transformer: base + 10 },
        suggestion: {
          rules: [{ match: 'transformer_blocks.1.attn.k_proj', skip: true }],
          layers: [{ name: 'transformer_blocks.1.attn.k_proj', component: 'transformer', out_sqnr_db: base + 6, median_db: base + 12 }],
          extra_weight_bytes: 65536, basis: '低于同组件中位数 6 dB',
        },
      },
    };
  }
  return {
    apiVersion: 'mqde/v1', kind, job_id: id, verdict: 'succeeded',
    spec: {
      name: kind === 'ewm' ? 'tiny-ewm-demo' : 'tiny-dit-world-demo', model_path: model,
      recipes: ['w8a16', 'w4a16'], algorithm: 'rtn', device: 'cuda:0', frames: 4, sample_steps: 8,
      action_conditioned: kind === 'ewm', action_dim: kind === 'ewm' ? 128 : 0, quantize_only: false,
    },
    stages: [
      ['validate', 'succeeded'], ['inspect', 'succeeded'], ['calib_prepare', 'succeeded'],
      ['eval_baseline', 'succeeded'], ['attribution', 'succeeded'], ['quantize', 'succeeded'],
      ['eval_quantized', 'succeeded'], ['report', 'succeeded'],
    ].map(([stage, status]) => ({ stage, status, seconds: 1.2 })),
    model: {
      family: kind === 'ewm' ? 'dit_ewm' : 'dit_world', weights_source: 'model.safetensors', arch: 'tiny_dit',
      frames: 4, action_conditioned: kind === 'ewm', action_dim: kind === 'ewm' ? 128 : 0, n_linears: 16,
      components: { patch_embed: { modality: 'vision', linears: 1, quantizable: 0, blocks: 0 }, transformer: { modality: 'world', linears: 14, quantizable: 12, blocks: 2 } },
    },
    data: {
      calib: { source: 'synthetic-video', n: 8, frames: 4 },
      eval: { source: 'synthetic-video', n: 4, frames: 4, has_gt: true, action_conditioned: kind === 'ewm', action_dim: kind === 'ewm' ? 128 : 0, prompts: ['a red square moves right'] },
    },
    baseline: {
      psnr_vs_gt: 4.9575, mse_vs_gt: 0.207, noise_floor: { psnr: 5.4924, mse: 0.282 }, latency_ms_per_video: 47.57,
      latency_ms_per_frame: 11.89, peak_mem_bytes: 14364672, peak_mem_note: 'CUDA max_memory_allocated',
      frames: 4, image_size: 32, sample_steps: 8, action_conditioned: kind === 'ewm', action_dim: kind === 'ewm' ? 128 : 0, lpips_vs_gt: null,
    },
    recipes: r,
    env: { device_name: 'NVIDIA GeForce RTX 4090', device: 'cuda:0', torch: '2.11.0', cuda: '12.8' },
  };
}

const stages = ['validate', 'inspect', 'calib_prepare', 'eval_baseline', 'attribution', 'quantize', 'eval_quantized', 'report'];
function mkJob(id, kind) {
  const d = path.join(runs, id);
  fs.mkdirSync(d);
  const now = Date.now() / 1000;
  const ev = [];
  let seq = 1;
  for (const [i, stage] of stages.entries()) {
    ev.push({ type: 'stage_started', stage, seq: seq++, ts: now - 30 + i });
    ev.push({ type: 'stage_finished', stage, status: 'succeeded', seconds: 1.2, seq: seq++, ts: now - 29 + i });
  }
  ev.push({ type: 'job_state', state: 'SUCCEEDED', seq: seq++, ts: now, msg: path.join(d, 'report.json') });
  fs.writeFileSync(path.join(d, 'events.jsonl'), ev.map((e) => JSON.stringify(e)).join('\n') + '\n');
  fs.writeFileSync(path.join(d, 'job.yaml'), [
    `metadata: {name: ${kind === 'ewm' ? 'tiny-ewm-demo' : 'tiny-dit-world-demo'}}`,
    `model: {type: ${kind}, path: ${model}}`, 'quant: {recipe: [w8a16, w4a16], algorithm: rtn}',
    'calibration: {dataset: synthetic-video, n_samples: 8, frames: 4}',
    `evaluation: {dataset: synthetic-video, n_samples: 4, frames: 4, action_conditioned: ${kind === 'ewm'}}`,
    'deploy: {device: gpu}', `output_dir: ${runs}`, '',
  ].join('\n'));
  fs.writeFileSync(path.join(d, 'report.json'), JSON.stringify(report(id, kind), null, 2));
}
const WORLD = '20261109-000001-world01';
const EWM = '20261109-000002-ewm001';
mkJob(WORLD, 'world');
mkJob(EWM, 'ewm');

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', 'python3', '--gpus', '0', '--allow', os.tmpdir(), '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }

const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));

  await page.goto(`${base}/jobs/${WORLD}`, { waitUntil: 'load' });
  await page.getByTestId('world-tiles').waitFor({ timeout: 15000 });
  const tiles = (await page.getByTestId('world-tiles').innerText()).replace(/\s+/g, ' ');
  check('WM overview shows BF16-vs-GT, the seed noise floor and the best recipe', tiles.includes('BF16 对真实未来 PSNR') && tiles.includes('采样噪声下限') && tiles.includes('最佳保真 W8A16'), tiles);
  check('the world-only job renders the WM stage strip', (await page.locator('.pipeline').innerText()).includes('BF16 采样基线') && (await page.locator('.pipeline').innerText()).includes('量化后生成质量'));

  const summary = (await page.getByTestId('world-summary-table').innerText()).replace(/\s+/g, ' ');
  check('recipe table has both recipes with PSNR/MSE and attribution source', summary.includes('W8A16') && summary.includes('W4A16') && summary.includes('81.81') && summary.includes('57.49') && summary.includes('去噪主干'), summary.slice(0, 240));
  check('the noise-floor comparison is shown as a pass', summary.includes('✓ 是'));

  await page.getByText('W4A16 误差分析').click();
  await page.getByTestId('world-attribution').waitFor({ timeout: 10000 });
  check('attribution table names the denoising backbone', (await page.getByTestId('world-attribution').innerText()).includes('去噪主干'));
  check('per-clip table lists every evaluation clip', (await page.locator('[data-testid="world-samples"] tbody tr').count()) === 4);
  check('dynamic skip suggestion names the worst layer', (await page.getByTestId('world-suggest-table').innerText()).includes('transformer_blocks.1.attn.k_proj'));
  check('per-layer heatmap and Top-15 table are rendered', (await page.getByText('逐层量化误差热力图').count()) === 1 && (await page.getByText('误差最大的 15 层').count()) === 1);

  await page.goto(`${base}/jobs/${EWM}`, { waitUntil: 'load' });
  await page.getByTestId('world-tiles').waitFor({ timeout: 15000 });
  await page.getByText('W4A16 误差分析').click();
  await page.getByTestId('world-attribution').waitFor({ timeout: 10000 });
  const attr = (await page.getByTestId('world-attribution').innerText()).replace(/\s+/g, ' ');
  check('EWM attributes the video backbone and the action encoder separately', attr.includes('去噪主干') && attr.includes('动作编码器'), attr);
  check('EWM summary says it is action-conditioned', (await page.getByTestId('world-tiles').innerText()).includes('动作条件'));

  await page.goto(`${base}/compare?tab=world`, { waitUntil: 'load' });
  await page.getByTestId('world-compare-table').waitFor({ timeout: 15000 });
  const cmp = (await page.getByTestId('world-compare-table').innerText()).replace(/\s+/g, ' ');
  check('WM compare tab shows the world job with both recipes', cmp.includes('tiny-dit-world-demo') && (await page.locator('[data-testid="world-compare-table"] tbody tr').count()) === 2, cmp.slice(0, 200));
  await page.getByTestId('compare-tab-ewm').click();
  await page.waitForFunction(() => document.querySelector('[data-testid="world-compare-table"]')?.textContent?.includes('tiny-ewm-demo'), null, { timeout: 10000 });
  const cmpEwm = (await page.getByTestId('world-compare-table').innerText()).replace(/\s+/g, ' ');
  check('EWM compare tab shows the action-conditioned job', cmpEwm.includes('tiny-ewm-demo') && (await page.locator('[data-testid="world-compare-table"] tbody tr').count()) === 2, cmpEwm.slice(0, 200));

  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
}
console.log(`\n${results.filter(Boolean).length}/${results.length} passed`);
process.exit(failed || results.some((x) => !x) ? 1 : 0);

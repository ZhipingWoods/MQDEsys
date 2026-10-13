// T-R05 UI: the model-structure page reads a GGUF file's header -- file list from the allowed roots, tiles, the quantization
// type table, per-block sizes/types, the filterable tensor table and metadata -- and refuses what is not a GGUF.
//
//   node web/e2e/gguf_ui.mjs          # needs: built web/dist, built mqde, Playwright (see demo/)
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
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-gguf-'));
const runs = fs.mkdtempSync(path.join(os.tmpdir(), 'mqde-e2e-runs-'));

// ---- a byte-exact GGUF writer (little-endian, v3) ----
const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b; };
const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; };
const str = (s) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);
const kvStr = (k, v) => Buffer.concat([str(k), u32(8), str(v)]);
const kvU32 = (k, v) => Buffer.concat([str(k), u32(4), u32(v)]);
const kvStrArr = (k, items) => Buffer.concat([str(k), u32(9), u32(8), u64(items.length), ...items.map(str)]);
const TYPES = { F32: [0, 1, 4], F16: [1, 1, 2], Q8_0: [8, 32, 34], Q4_K: [12, 256, 144], Q6_K: [14, 256, 210] };
function gguf(file, kvs, tensors, align = 32) {
  let off = 0;
  const infos = tensors.map(([name, type, ne]) => {
    const [id, blck, bytes] = TYPES[type]; const numel = ne.reduce((a, b) => a * b, 1); const nb = (numel / blck) * bytes;
    const b = Buffer.concat([str(name), u32(ne.length), ...ne.map(u64), u32(id), u64(off)]); off += nb; return b;
  });
  let head = Buffer.concat([Buffer.from('GGUF'), u32(3), u64(tensors.length), u64(kvs.length), ...kvs, ...infos]);
  head = Buffer.concat([head, Buffer.alloc((align - (head.length % align)) % align)]);
  fs.writeFileSync(file, Buffer.concat([head, Buffer.alloc(off, 0x7f)]));
}
// a 3-block "Q4_K_M"-like file: block 1's down_proj is Q6_K, which only the per-block view shows
const toks = Array.from({ length: 20 }, (_, i) => `tok${i}`);
const sampleKv = [kvStr('general.architecture', 'llama'), kvStr('general.name', 'Tiny GGUF'), kvU32('llama.block_count', 3), kvU32('llama.context_length', 4096), kvStrArr('tokenizer.ggml.tokens', toks)];
const sampleTensors = [['token_embd.weight', 'Q6_K', [256, 8]], ['output_norm.weight', 'F32', [256]]];
for (let b = 0; b < 3; b++) sampleTensors.push([`blk.${b}.attn_q.weight`, 'Q4_K', [256, 4]], [`blk.${b}.ffn_down.weight`, b === 1 ? 'Q6_K' : 'Q4_K', [256, 4]], [`blk.${b}.attn_norm.weight`, 'F32', [256]]);
fs.mkdirSync(path.join(root, 'tiny-GGUF'));
gguf(path.join(root, 'tiny-GGUF', 'tiny-q4_k_m.gguf'), sampleKv, sampleTensors);
gguf(path.join(root, 'plain-f16.gguf'), [kvStr('general.architecture', 'llama')], [['blk.0.attn_q.weight', 'F16', [64, 4]]]);
fs.writeFileSync(path.join(root, 'broken.gguf'), 'GGUF\x03\x00');
fs.writeFileSync(path.join(root, 'not-a-model.txt'), 'hello');

const port = await freePort();
const server = spawn(bin, ['serve', '--repo', repo, '--runs', runs, '--web', path.join(repo, 'web/dist'), '--python', path.join(here, 'fake_worker.py'),
  '--gpus', '0', '--allow', root, '--port', String(port)], { stdio: ['ignore', 'ignore', 'ignore'] });
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 100; i++) { try { if ((await fetch(`${base}/api/v1/health`)).ok) break; } catch { /* not up */ } await new Promise((r) => setTimeout(r, 100)); }
const post = (p, b) => fetch(`${base}/api/v1${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b) });

const browser = await chromium.launch({ channel: 'chrome' });
let failed = false;
try {
  // ---- API ----
  const list = await (await fetch(`${base}/api/v1/gguf`)).json();
  check('the file list finds .gguf files at the root and one directory down, nothing else', list.map((f) => f.name).sort().join(',') === 'broken.gguf,plain-f16.gguf,tiny-q4_k_m.gguf', JSON.stringify(list.map((f) => f.name)));
  check('each entry has its size and directory', list.find((f) => f.name === 'tiny-q4_k_m.gguf').dir === 'tiny-GGUF' && list.find((f) => f.name === 'plain-f16.gguf').bytes === fs.statSync(path.join(root, 'plain-f16.gguf')).size);
  const sum = await (await post('/models:gguf', { path: path.join(root, 'tiny-GGUF', 'tiny-q4_k_m.gguf') })).json();
  check('the summary reads architecture, name, block count and tensor count from the header', sum.architecture === 'llama' && sum.name === 'Tiny GGUF' && sum.block_count === 3 && sum.n_tensors === 11, JSON.stringify([sum.architecture, sum.name, sum.block_count, sum.n_tensors]));
  check('payload bytes add up to the file: tensor_bytes + data_offset == file_bytes', sum.tensor_bytes + sum.data_offset === sum.file_bytes, `${sum.tensor_bytes} + ${sum.data_offset} vs ${sum.file_bytes}`);
  const bad = await post('/models:gguf', { path: path.join(root, 'broken.gguf') });
  check('a truncated GGUF is a 4xx with the reason, not a crash', bad.status >= 400 && bad.status < 500 && (await bad.text()).includes('truncated'));
  check('a path outside the allowed roots is 403', (await post('/models:gguf', { path: '/etc/passwd' })).status === 403);
  check('a non-.gguf file inside the roots is refused', (await post('/models:gguf', { path: path.join(root, 'not-a-model.txt') })).status === 400);
  check('without tensors the response leaves the table out', (await (await post('/models:gguf', { path: path.join(root, 'plain-f16.gguf'), tensors: false })).json()).tensors === undefined);

  // ---- page ----
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && !/Failed to load resource/.test(m.text()) && errors.push(m.text()));
  await page.goto(`${base}/graph`, { waitUntil: 'load' });
  await page.getByTestId('graph-tab-gguf').click();
  await page.waitForURL(/view=gguf/, { timeout: 5000 });
  await page.getByTestId('gguf-select').waitFor({ timeout: 15000 });
  await page.getByTestId('gguf-error').waitFor({ timeout: 15000 });  // the first file alphabetically is the broken one: opened by default, so the error is what shows
  check('the first file opens by default, and a broken one is an error on the page rather than a blank', (await page.getByTestId('gguf-error').innerText()).includes('truncated'));
  const opts = await page.getByTestId('gguf-select').locator('option').allInnerTexts();
  check('the picker lists the files with their directory and size', opts.length === 3 && opts.some((o) => o.includes('tiny-GGUF/tiny-q4_k_m.gguf')), opts.join(' | '));
  await page.getByTestId('gguf-select').selectOption({ label: opts.find((o) => o.includes('tiny-q4_k_m')) });
  await page.waitForFunction(() => document.querySelector('[data-testid="gguf-tiles"]')?.textContent?.includes('Tiny GGUF'), null, { timeout: 10000 });
  const tiles = (await page.getByTestId('gguf-tiles').innerText()).replace(/\s+/g, ' ');
  check('tiles: architecture, name, GGUF version, 11 tensors / 3 blocks, quantization mix', tiles.includes('llama') && tiles.includes('Tiny GGUF · GGUF v3') && tiles.includes('11 个张量') && tiles.includes('3 块') && tiles.includes('Q4_K'), tiles);
  const trow = async (t) => (await page.getByTestId(`gguf-type-${t}`).innerText()).replace(/\s+/g, ' ');
  check('the type table counts tensors per type (Q4_K 5, Q6_K 2, F32 4)', (await trow('Q4_K')).includes(' 5 ') && (await trow('Q6_K')).includes(' 2 ') && (await trow('F32')).includes(' 4 '), `${await trow('Q4_K')} // ${await trow('Q6_K')} // ${await trow('F32')}`);
  check('and bits per weight: Q4_K is 4.5, Q6_K 6.56, F32 32', (await trow('Q4_K')).includes('4.50') && (await trow('Q6_K')).includes('6.56') && (await trow('F32')).includes('32.00'));
  const blocks = (await page.getByTestId('gguf-blocks').locator('tbody tr').allInnerTexts()).map((t) => t.replace(/\s+/g, ' '));
  check('three block rows; only block 1 has a Q6_K', blocks.length === 3 && !blocks[0].includes('Q6_K') && blocks[1].includes('Q6_K') && !blocks[2].includes('Q6_K'), blocks.join(' // '));
  check('tensors outside blocks are named', (await page.getByTestId('gguf-global').innerText()).includes('token_embd.weight') && (await page.getByTestId('gguf-global').innerText()).includes('output_norm.weight'));
  check('the tensor table has all 11 rows', (await page.getByTestId('gguf-tensors').locator('tbody tr').count()) === 11);
  await page.getByTestId('gguf-tensor-filter').fill('ffn_down');
  check('filtering by name', (await page.getByTestId('gguf-tensors').locator('tbody tr').count()) === 3);
  await page.getByTestId('gguf-tensor-filter').fill('FFN_Down');
  check('the name filter ignores case', (await page.getByTestId('gguf-tensors').locator('tbody tr').count()) === 3);
  await page.getByTestId('gguf-tensor-type').selectOption('Q6_K');
  check('and by type: ffn_down in Q6_K is block 1 only', (await page.getByTestId('gguf-tensors').locator('tbody tr').allInnerTexts()).join('|').includes('blk.1.ffn_down.weight') && (await page.getByTestId('gguf-tensors').locator('tbody tr').count()) === 1);
  const metaRows = (await page.getByTestId('gguf-meta').locator('tbody tr').allInnerTexts()).join('\n');
  check('metadata shows scalars and summarises the token array (20 entries, first few)', metaRows.includes('llama.context_length') && metaRows.includes('4096') && /string\[20\]/.test(metaRows) && metaRows.includes('"tok0"'), metaRows.slice(0, 300));
  await page.getByTestId('gguf-meta-filter').fill('TOKENIZER');
  check('so does the metadata filter', (await page.getByTestId('gguf-meta').locator('tbody tr').count()) === 1);
  await page.getByTestId('gguf-meta-filter').fill('tokenizer');
  check('metadata filter', (await page.getByTestId('gguf-meta').locator('tbody tr').count()) === 1);
  await page.getByTestId('gguf-path').fill(path.join(root, 'broken.gguf'));
  await page.getByTestId('gguf-open').click();
  await page.getByTestId('gguf-error').waitFor({ timeout: 10000 });
  check('opening a broken file shows the reason and clears the old view', (await page.getByTestId('gguf-error').innerText()).includes('truncated') && (await page.getByTestId('gguf-tiles').count()) === 0);
  await page.getByTestId('graph-tab-hf').click();
  await page.waitForFunction(() => !location.search.includes('view=gguf'), null, { timeout: 5000 });
  check('the HF structure tab is still there', (await page.locator('h1', { hasText: '模型结构' }).count()) === 1);
  check('no page errors', errors.length === 0, errors.slice(0, 2).join(' | '));
} catch (e) {
  failed = true;
  console.log(`FAIL  unexpected error  — ${e.message.split('\n')[0]}`);
} finally {
  await browser.close();
  stop();
}
const nbad = results.filter((x) => !x).length;
console.log(`\n${results.length - nbad}/${results.length} checks passed`);
process.exit(failed || nbad ? 1 : 0);

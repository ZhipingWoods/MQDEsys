// MQDEsys demo recorder.  Usage: node record.mjs --lang zh|en [--base URL] [--dry] [--voice NAME] [--rate WPM]
import { chromium } from 'playwright';
import { scenes, cards } from './scenes.mjs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => { const i = process.argv.indexOf('--' + n); return i < 0 ? d : (process.argv[i + 1]?.startsWith('--') || i + 1 >= process.argv.length ? true : process.argv[i + 1]); };
const lang = arg('lang', 'zh');
if (!['zh', 'en'].includes(lang)) throw new Error('--lang must be zh or en');
const BASE = arg('base', 'http://localhost:18080');
const DRY = arg('dry', false) === true;
const voice = arg('voice', lang === 'zh' ? 'Tingting' : 'Samantha');
const rate = String(arg('rate', lang === 'zh' ? '225' : '190'));
const GAP = 0.25, SCENE_PAUSE = 0.5;
const W = 1440, H = 900;
const work = path.join(here, 'work'), out = path.join(here, 'out');
const audioDir = path.join(work, 'audio', `${lang}-${voice}-${rate}`);
for (const d of [audioDir, out]) fs.mkdirSync(d, { recursive: true });
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- narration audio (macOS `say`) and duration (ffprobe) ----
function render(text) {
  const f = path.join(audioDir, createHash('sha1').update(text).digest('hex').slice(0, 16) + '.aiff');
  if (!fs.existsSync(f)) execFileSync('say', ['-v', voice, '-r', rate, '-o', f, '--', text]);
  const d = parseFloat(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f]).toString());
  return { file: f, dur: d };
}
for (const sc of scenes) for (const s of sc.sentences) Object.assign(s, render(s[lang]));
const planned = scenes.map(sc => sc.sentences.reduce((a, s) => a + s.dur + GAP, 0) + SCENE_PAUSE);
console.log(`[${lang}] voice=${voice} narration total ${planned.reduce((a, b) => a + b, 0).toFixed(1)} s`);
scenes.forEach((sc, i) => console.log(`  ${sc.id.padEnd(9)} ${planned[i].toFixed(1)} s`));
if (DRY) process.exit(0);

// ---- in-page overlay: privacy rewrite, caption bar, highlight boxes, safety check ----
const OVERLAY = `(() => {
  if (window.__ov) return; window.__ov = true;
  const RX = /\\/cache\\/(?:[^\\/\\s"'<>]+\\/)*([^\\/\\s"'<>]+)/g;
  const fix = s => s.replace(RX, '<models>/$1');
  const walk = root => { const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n;
    while ((n = w.nextNode())) if (n.nodeValue.includes('/cache/')) n.nodeValue = fix(n.nodeValue); };
  const masks = new Map();
  const maskInputs = () => document.querySelectorAll('input,textarea').forEach(el => {
    if (!/\\/cache\\//.test(el.value)) { const m = masks.get(el); if (m) { m.remove(); masks.delete(el); el.style.color = ''; } return; }
    let m = masks.get(el);
    if (!m) { m = document.createElement('div'); m.dataset.mask = '1'; document.body.appendChild(m); masks.set(el, m); }
    const cs = getComputedStyle(el), r = el.getBoundingClientRect();
    el.style.color = 'transparent';
    Object.assign(m.style, { position: 'absolute', zIndex: 99990, pointerEvents: 'none', left: (r.left + scrollX) + 'px', top: (r.top + scrollY) + 'px', width: r.width + 'px', height: r.height + 'px',
      boxSizing: 'border-box', padding: cs.padding, font: cs.font, background: cs.backgroundColor, color: getComputedStyle(document.body).color, lineHeight: cs.lineHeight, display: 'flex', alignItems: 'center', overflow: 'hidden', borderRadius: cs.borderRadius });
    m.textContent = fix(el.value);
  });
  const tick = () => { if (document.body) { walk(document.body); maskInputs(); } requestAnimationFrame(tick); };
  const capEl = () => { let c = document.getElementById('__cap');
    if (!c) { c = document.createElement('div'); c.id = '__cap'; document.body.appendChild(c);
      Object.assign(c.style, { position: 'fixed', left: 0, right: 0, bottom: 0, zIndex: 99999, padding: '14px 90px 18px', background: 'rgba(8,10,18,.88)', color: '#fff',
        font: '600 25px/1.4 -apple-system,"PingFang SC","Helvetica Neue",sans-serif', textAlign: 'center', pointerEvents: 'none', minHeight: '40px', borderTop: '2px solid #7c6cff' }); }
    return c; };
  const hl = document.createElement('style');
  hl.textContent = '@keyframes __p{0%,100%{box-shadow:0 0 0 3px rgba(255,204,0,.35),0 0 14px 2px rgba(255,204,0,.55)}50%{box-shadow:0 0 0 6px rgba(255,204,0,.2),0 0 26px 6px rgba(255,204,0,.8)}} .__hl{position:absolute;z-index:99980;pointer-events:none;border:3px solid #ffcc00;border-radius:8px;animation:__p 1.4s ease-in-out infinite}';
  const boot = () => { document.head.appendChild(hl); requestAnimationFrame(tick); capEl().textContent = (window.name || '').startsWith('CAP:') ? window.name.slice(4) : ''; capEl().style.visibility = capEl().textContent ? 'visible' : 'hidden'; };
  if (document.body) boot(); else document.addEventListener('DOMContentLoaded', boot);
  window.__setCap = t => { window.name = 'CAP:' + t; const c = capEl(); c.textContent = t; c.style.visibility = t ? 'visible' : 'hidden'; };
  window.__clearHl = () => document.querySelectorAll('.__hl').forEach(e => e.remove());
  window.__drawHl = rs => { window.__clearHl(); rs.forEach(r => { const d = document.createElement('div'); d.className = '__hl';
    Object.assign(d.style, { left: (r.x - 5) + 'px', top: (r.y - 4) + 'px', width: (r.w + 10) + 'px', height: (r.h + 8) + 'px' }); document.body.appendChild(d); }); };
  window.__scrollTo = y => window.scrollTo({ top: y, behavior: 'smooth' });
  window.__check = () => { const bad = /\\/cache\\/|MiniCPM-o|o-5\\b|\\b\\d{1,3}(?:\\.\\d{1,3}){3}\\b|\\/home\\/|\\/Users\\//i;
    const t = document.body.innerText.replace(/<models>\\/[^\\s]+/g, '');
    const m = t.match(bad); const inp = [...document.querySelectorAll('input,textarea')].filter(e => !masks.has(e)).map(e => e.value).join(' ').match(bad);
    return (m && m[0]) || (inp && inp[0]) || ''; };
})();`;

function cardHtml(c) {
  return `<!doctype html><meta charset=utf-8><style>html,body{margin:0;height:100%;background:radial-gradient(circle at 30% 20%,#2b2566,#0c0e1a 60%);color:#fff;font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif}
  .w{height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:22px;text-align:center;padding-bottom:90px}
  h1{font-size:120px;margin:0;letter-spacing:4px;background:linear-gradient(90deg,#9d8cff,#5ee0ff);-webkit-background-clip:text;color:transparent}
  h2{font-size:40px;font-weight:600;margin:0}p{font-size:26px;color:#aab;margin:0}</style>
  <div class=w><h1>${c[0]}</h1><h2>${c[1]}</h2><p>${c[2]}</p></div>`;
}

// ---- locating highlight targets ----
function locate(page, s) {
  let l;
  if (s.card) {
    l = page.locator('.card', { has: page.locator('h2', { hasText: s.card }) }).first();
    if (s.row !== undefined || s.rowIdx !== undefined) l = s.row !== undefined ? l.locator('tbody tr', { hasText: s.row }).first() : l.locator('tbody tr').nth(s.rowIdx);
  } else if (s.css) {
    l = page.locator(s.css).first();
    if (s.row !== undefined || s.rowIdx !== undefined) l = s.row !== undefined ? l.locator('tbody tr', { hasText: s.row }).first() : l.locator('tbody tr').nth(s.rowIdx);
  } else if (s.text !== undefined) {
    l = (s.text instanceof RegExp ? page.getByText(s.text) : page.getByText(s.text, { exact: true })).first();
    for (let i = 0; i < (s.up || 0); i++) l = l.locator('xpath=..');
    return l;
  }
  if (s.col !== undefined) l = l.locator('td').nth(s.col);
  return l;
}
async function rectOf(loc) {
  const h = await loc.elementHandle({ timeout: 5000 });
  return h.evaluate(el => { const r = el.getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; });
}

// ---- recording ----
const browser = await chromium.launch({ channel: 'chrome' });
const vdir = path.join(work, `video-${lang}`);
fs.rmSync(vdir, { recursive: true, force: true });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: vdir, size: { width: W, height: H } }, locale: lang === 'zh' ? 'zh-CN' : 'en-US' });
await ctx.addInitScript(OVERLAY);
const page = await ctx.newPage();
const T0 = Date.now(); const now = () => (Date.now() - T0) / 1000;
const consoleLog = [];
page.on('console', m => { if (m.type() === 'error') consoleLog.push('console.error: ' + m.text().slice(0, 300)); });
page.on('pageerror', e => consoleLog.push('pageerror: ' + String(e).slice(0, 300)));
page.on('requestfailed', r => consoleLog.push('requestfailed: ' + r.url().replace(BASE, '') + ' ' + r.failure()?.errorText));
page.on('response', r => { if (r.status() >= 400) consoleLog.push(`http ${r.status()}: ` + r.url().replace(BASE, '')); });

const events = [];
async function openScene(sc) {
  await page.evaluate(() => window.__setCap && window.__setCap('')).catch(() => {});
  if (sc.card) {
    // Cards are loaded from a file, not page.setContent(): a setContent page leaves the bottom ~85 px of the recorded
    // video as flat grey (measured), which is where the caption bar sits. A file:// page records the full frame.
    const f = path.join(work, `card-${sc.card}-${lang}.html`);
    fs.writeFileSync(f, cardHtml(cards[sc.card][lang]));
    await page.goto('file://' + f, { waitUntil: 'load' });
    await page.evaluate(OVERLAY); await sleep(300); return;
  }
  await page.goto(BASE + sc.url, { waitUntil: 'load' });
  await page.evaluate(OVERLAY);
  if (sc.wait) await page.locator(sc.wait).first().waitFor({ timeout: 20000 });
  await sleep(2000);
  if (sc.click) { await page.getByText(sc.click, { exact: true }).first().click(); await sleep(1800); }
  await page.evaluate(() => window.scrollTo(0, 0));
}
async function applyHl(specs) {
  if (!specs?.length) { await page.evaluate(() => window.__clearHl()); return; }
  const rects = [];
  for (const s of specs) { try { rects.push(await rectOf(locate(page, s))); } catch (e) { console.warn('  highlight miss', JSON.stringify(s), String(e).split('\n')[0]); } }
  if (!rects.length) return;
  const top = Math.min(...rects.map(r => r.y));
  await page.evaluate(([rs, y]) => { window.__drawHl(rs); window.__scrollTo(y); }, [rects, Math.max(0, top - 150)]);
}

for (const sc of scenes) {
  console.log('scene', sc.id, now().toFixed(1));
  await openScene(sc);
  for (let i = 0; i < sc.sentences.length; i++) {
    const s = sc.sentences[i]; const start = now();
    await page.evaluate(t => window.__setCap(t), s[lang]);
    await applyHl(s.hl);
    const bad = await page.evaluate(() => window.__check());
    if (bad) throw new Error(`privacy check failed in scene ${sc.id}: "${bad}"`);
    events.push({ scene: sc.id, start, dur: s.dur, text: s[lang], file: s.file });
    const end = start + s.dur + (i === sc.sentences.length - 1 ? GAP + SCENE_PAUSE : GAP);
    await sleep(Math.max(0, (end - now()) * 1000));
  }
  await page.evaluate(() => { window.__clearHl(); window.__setCap(''); }).catch(() => {});
}
const total = now() + 0.5; await sleep(500);
await page.close(); await ctx.close(); await browser.close();
const webm = fs.readdirSync(vdir).filter(f => f.endsWith('.webm')).map(f => path.join(vdir, f))[0];

// ---- subtitles + mux ----
const ts = t => { const ms = Math.round(t * 1000), p = (n, w = 2) => String(n).padStart(w, '0'); return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`; };
const srt = events.map((e, i) => `${i + 1}\n${ts(e.start)} --> ${ts(e.start + e.dur)}\n${e.text}\n`).join('\n');
const base = path.join(out, `mqde-demo-${lang}`);
fs.writeFileSync(base + '.srt', srt);
fs.writeFileSync(path.join(work, `timeline-${lang}.json`), JSON.stringify({ total, events: events.map(({ file, ...r }) => r) }, null, 1));
const inputs = ['-i', webm], filt = [];
events.forEach((e, i) => { inputs.push('-i', e.file); const ms = Math.round(e.start * 1000); filt.push(`[${i + 1}:a]aresample=44100,aformat=channel_layouts=mono,adelay=${ms}:all=1[a${i}]`); });
filt.push(events.map((_, i) => `[a${i}]`).join('') + `amix=inputs=${events.length}:normalize=0:dropout_transition=0,apad[aout]`);
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...inputs, '-filter_complex', filt.join(';'), '-map', '0:v', '-map', '[aout]', '-t', total.toFixed(2),
  '-c:v', 'libx264', '-crf', '28', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-r', '25', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', base + '.mp4'], { stdio: 'inherit' });
fs.writeFileSync(path.join(work, `console-${lang}.log`), [...new Set(consoleLog)].join('\n'));
console.log(`done ${base}.mp4  (${total.toFixed(1)} s)  console issues: ${new Set(consoleLog).size}`);

# MQDEsys demo video / 演示视频

Scripted, reproducible screen recording of the web UI, with captions and narration in Chinese and English.
脚本化、可复现的网页界面录屏，带字幕和旁白，中英文各一版。

## 中文

### 重新录制
```bash
cd demo
./build.sh          # 两种语言；也可 ./build.sh zh 或 ./build.sh en
```
产物在 `demo/out/`（已被 git 忽略）：`mqde-demo-zh.mp4`、`mqde-demo-en.mp4`（H.264 + AAC，CRF 28）以及同名 `.srt` 字幕。
`BURN_SUBS=1 ./build.sh` 另外生成硬字幕版（`*.burned.mp4`，需要 ffmpeg 带 libass）。画面里本来就有字幕条，通常不需要。

### 前置条件
- macOS（旁白用系统自带 `say`），Node.js，Google Chrome（Playwright 以 `channel: "chrome"` 启动，使用 Playwright 默认的隔离临时配置，不碰你的 Chrome 资料）。
- `brew install ffmpeg`（需要 `ffmpeg` 与 `ffprobe`）。首次运行会 `npm install`，并需要 `npx playwright install ffmpeg`（Playwright 录屏用的小型编码器）。
- Web 界面可在 `http://localhost:18080` 访问（例如 SSH 隧道）；其他地址用 `DEMO_BASE=http://host:port ./build.sh`。
- 只发 GET 请求，不提交作业，不用 GPU。
- 演示用到的作业 ID 写在 `scenes.mjs`；它们必须仍存在于服务器的 `runs/` 里。

### 时间轴如何确定
`scenes.mjs` 是分镜脚本：每个场景由若干句旁白组成，每句有中英文文本和要高亮的元素。
`record.mjs` 先用 `say -o` 把每句旁白渲染成 aiff，用 `ffprobe` 量出时长，然后按这个时长停留（句间 0.25 s，场景间 0.5 s）。字幕与高亮在句子开始时切换。录完后把每句音频按真实开始时间用 `adelay` 叠加到视频上。所以画面与声音同步，换声音或语速后时间轴自动变化。渲染好的 aiff 缓存在 `work/audio/`（按文本哈希命名）。`--dry` 只打印各场景时长，不录制。

### 换声音
```bash
DEMO_VOICE_ZH=Sinji DEMO_VOICE_EN=Daniel ./build.sh
DEMO_RATE_ZH=200 DEMO_RATE_EN=170 ./build.sh     # 语速（词/分钟，say -r）
```
`say -v '?'` 列出可用声音。若要换成其他 TTS：改 `record.mjs` 里的 `render()`，只要它返回一个音频文件和其时长即可。

### 隐私处理（仅外观）
页面可能显示服务器路径。`record.mjs` 注入的脚本会把可见文字里的 `/cache/...` 路径改写为 `<models>/<basename>`（文本节点直接改写；输入框用遮罩覆盖）。这只是**显示层面的美化**，不改变服务器数据。每句话开始时还会做一次检查：可见文字里若出现 `/cache/`、`MiniCPM-o`、`o-5`、IPv4 地址、`/home/`、`/Users/`，录制立即失败。
没有使用「模型目录」页面和含内部检查点的页面；`/graph` 页面含 `o-5` 字样，已跳过。

### 修改分镜
编辑 `scenes.mjs`。数字必须和页面显示完全一致（修改后请重新核对）。高亮规格：`{card:'卡片标题', row:'行内文字', col:列号}`、`{css:'table', rowIdx:行号, col:列号}`、`{text:'文字'}`。

## English

### Re-record
```bash
cd demo
./build.sh          # both languages; or ./build.sh zh / ./build.sh en
```
Output goes to `demo/out/` (git-ignored): `mqde-demo-zh.mp4`, `mqde-demo-en.mp4` (H.264 + AAC, CRF 28) and matching `.srt` files. `BURN_SUBS=1 ./build.sh` also writes burned-in copies (`*.burned.mp4`, needs ffmpeg with libass). The video already has a caption bar, so this is rarely needed.

### Prerequisites
- macOS (narration uses the built-in `say`), Node.js, Google Chrome (launched by Playwright with `channel: "chrome"` and Playwright's default isolated temporary profile, never your own profile).
- `brew install ffmpeg` (needs `ffmpeg` and `ffprobe`). First run does `npm install`; you also need `npx playwright install ffmpeg` (the small encoder Playwright uses for screen capture).
- The web UI must answer at `http://localhost:18080` (for example through an SSH tunnel). Use `DEMO_BASE=http://host:port ./build.sh` for another address.
- Only GET requests: no jobs are submitted, no GPU is used.
- The job IDs used are in `scenes.mjs`; they must still exist in the server's `runs/`.

### How the timing works
`scenes.mjs` is the storyboard: each scene is a list of sentences with Chinese and English text plus the elements to highlight. `record.mjs` first renders every sentence with `say -o` to an aiff file and measures it with `ffprobe`; the scene then holds for exactly that duration (0.25 s between sentences, 0.5 s between scenes). Caption and highlight change when a sentence starts. After recording, every audio clip is placed at its real start time with `adelay` and mixed onto the video, so picture and sound stay in sync and the timeline adapts automatically to a different voice or rate. Rendered aiff files are cached in `work/audio/` (named by text hash). `--dry` prints the per-scene durations without recording.

### Swap the voice
```bash
DEMO_VOICE_ZH=Sinji DEMO_VOICE_EN=Daniel ./build.sh
DEMO_RATE_ZH=200 DEMO_RATE_EN=170 ./build.sh     # words per minute (say -r)
```
`say -v '?'` lists voices. To use another TTS engine, change `render()` in `record.mjs`; it only has to return an audio file and its duration.

### Privacy handling (cosmetic only)
Pages may show server filesystem paths. An injected script rewrites visible `/cache/...` paths to `<models>/<basename>` (text nodes are rewritten; input fields are covered by an overlay). This is **cosmetic, display-level only**; no server data changes. At the start of every sentence a check runs: if the visible text contains `/cache/`, `MiniCPM-o`, `o-5`, an IPv4 address, `/home/` or `/Users/`, the recording fails at once. The model catalog page and any page listing the internal checkpoint are not used; `/graph` contains an `o-5` string and is skipped.

### Editing the storyboard
Edit `scenes.mjs`. Numbers must match the page exactly (re-check after edits). Highlight specs: `{card:'card title', row:'row text', col:N}`, `{css:'table', rowIdx:N, col:N}`, `{text:'text'}`.

Files: `scenes.mjs` (storyboard), `record.mjs` (recorder), `build.sh` (driver), `storyboard.md` (readable storyline), `work/` (cache, intermediates, `console-*.log`; ignored), `out/` (results; ignored).

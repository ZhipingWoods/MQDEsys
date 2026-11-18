# MQDEsys static showcase

A self-contained bilingual product page. No CDN, no build step.

```bash
cd /path/to/MQDEsys
python3 -m http.server 19090
# open http://127.0.0.1:19090/site/
```

Files:

- `index.html` — Chinese/English content, feature sections, screenshots, architecture and quick start;
- `styles.css` — dark/light tokens, responsive layout and screenshot frames;
- `main.js` — language/theme persistence, screenshot tabs, lightbox, reveal animations and code copy;
- `assets/` — real UI screenshots captured from this repository's running service;
- `assets/mqde-system-short.webm` — 15-second MQDEsys console recording;
- `assets/mqde-system-poster.png` — poster frame for the video.

When the page is served from `localhost` or `127.0.0.1`, the local-service buttons point at
`http://127.0.0.1:18081` and `http://127.0.0.1:7860`. On a public host they become local-startup
guidance instead of linking visitors to a nonexistent service. Documentation links use the local
Sphinx build during local preview and fall back to the public source documents on GitHub elsewhere.

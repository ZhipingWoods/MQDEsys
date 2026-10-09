# web/e2e — browser tests against a real server

`clone_from_suggestion.mjs` (T-0904): on a finished job's sensitivity card, the button "按建议生成新作业" must create one
new job whose `job.yaml` has the chosen recipe only, the suggested mixed-precision rules, a `-mp` name and everything else
carried over. It starts its own `mqde serve` on a temp directory with `fake_worker.py` as the "python", so the click goes
through the real REST API but no GPU, model or quantization is involved. The finished job it opens is a real report
(`python/tests/fixtures/reports/llm_full.json`).

`newjob_live_estimate.mjs` (T-0304): the new-job page's live estimate sidebar must show the C++ estimator's numbers (weights, peak, feasibility, capacity) for every picked recipe, follow recipe / hardware changes, send few requests while typing (300 ms debounce) and show the server's error for a bad path. It builds a header-only Qwen2-style model directory in the temp dir; no GPU or weights needed. Run it the same way: `node web/e2e/newjob_live_estimate.mjs`. Checked on macOS + Chrome: 10/10, and 2 mutations of the page (wrong column, no debounce) each turn checks red (3 and 2 of them).

```bash
cd web && npm run build && cd ..          # the test serves web/dist
(cd demo && npm install)                  # Playwright; Google Chrome must be installed
node web/e2e/clone_from_suggestion.mjs
```

Exit code: 0 all checks passed, 1 a check failed, 2 binary or `web/dist` missing, 3 Playwright missing (nothing ran).
`MQDE_BIN` and `PLAYWRIGHT_DIR` override the server binary and the directory Playwright is installed in.

Status: verified on macOS with Chrome (14/14, and 2 of them fail when the button is broken on purpose). The dev server has
neither Playwright nor Chrome, so this is **not** part of `scripts/check_all.sh`; wiring it in changes the gate (L2, see
docs/PROGRESS.md D-10).

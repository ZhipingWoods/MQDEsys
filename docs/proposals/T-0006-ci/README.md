# 方案：T-0006 CI 流水线（L2，等待评审，未启用）

日期 2026-10-13 · 作者 Claude（Sonnet 5.5） · 状态 **Proposed** · 触及：`.github/workflows/**`、工具链配置（L2）

## 1. 现状

- 本地门禁 `scripts/check_all.sh`（T-0005，DONE）已经是"与 CI 等价"的：依赖规则 → HAL ABI 布局 → 算法注册表同步 → 作业 schema → clang-format → 构建 → ctest → clang-tidy → ruff → mypy → pytest → 前端构建。缺的工具会报 SKIP 而不是失败。
- **仓库没有 `.github/workflows/`**（只有 `docs/site/pages-workflow.proposal.yml` 这份文档站的方案）。所以现在 `main` 的"始终可构建、测试全绿"（AGENTS.md 第 5.1 节）完全靠人自觉。
- 开发机的工具链来自 conda-forge（`conda env export --from-history -n mqde`），不是 apt。关键一点：**必须 GCC 12**，系统 g++ 11 编不过 `std::async`（T-0704 的实测教训）；`-Werror` 对 GCC 与 Clang 的行为不同（本轮就碰到 GCC 的 `-Wdangling-else`、`-Wmaybe-uninitialized` 而 Clang 不报）——所以 CI 里必须有 GCC，不能只有 macOS。

## 2. 方案

本目录两个文件：

- `ci.proposal.yml`：启用时复制为 `.github/workflows/ci.yml`。两个任务：
  - `gate`（ubuntu-latest）：用 `environment.ci.yml` 建 conda 环境 → 缓存 `build/dev` → `cmake` 配置 → `npm ci`（前端门禁需要 `web/node_modules`）→ **`scripts/check_all.sh`** → 全部 Playwright e2e（`web/e2e/*.mjs`，任一失败即失败）。
  - `macos`（macos-14）：用 brew 装依赖，构建并跑 ctest（编 Metal 插件）。
- `environment.ci.yml`：门禁需要的工具链（GCC 12、cmake、ninja、clang-format、clang-tools、nodejs 20、C++ 依赖、CPU 版 torch、pytest/ruff/mypy）。建议启用时放到仓库根目录，也供 T-0008 的 `dev_setup.sh` 复用同一份，避免两处依赖清单。

**不做的**：Windows（任务书说"仅编译最小集"，仓库里没有 Windows 构建，先不承诺）、`nightly.yml`（GPU 夜间回归需要自托管 GPU runner，是 T-1206）、`release.yml`（等 T-1201/1202）、vcpkg 缓存（本仓库用 conda 而不是 vcpkg，任务书的这一项已经过时）、PR 覆盖率评论。

## 3. 验证：做了什么，没做什么

见本文件第 4 节（"实测"）。**不能在 GitHub 上运行**，所以以下几点没有验证，评审时请留意：

1. `ci.proposal.yml` 的 GitHub 专属语法与动作版本（`conda-incubator/setup-miniconda@v3`、`actions/cache@v4`），只做了 YAML 解析检查。
2. e2e 用的是系统 Google Chrome（`chromium.launch({channel: 'chrome'})`）。GitHub 托管的 ubuntu runner 镜像预装了它，但**我没有确认**；若没有，需要加 `npx playwright install chrome` 一步。
3. 冷构建时间、缓存命中时间（任务书的 30 / 10 分钟目标）要在真实 runner 上量。

## 4. 实测（4090，干净 conda 环境）

*待填：见下方。*

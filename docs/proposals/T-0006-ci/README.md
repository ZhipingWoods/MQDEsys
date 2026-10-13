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

环境：4090 服务器，按 `environment.ci.yml` 新建的 `mqde-ci`（Python 3.11，GCC 12，CPU 版 torch，无 GPU 依赖），`BUILD_DIR=build/ci bash scripts/check_all.sh`。

**第一次在干净环境跑门禁：不通过**（开发环境 `mqde` 里一直是绿的）。原因都是"开发环境碰巧掩盖的"：

| 项 | 现象 | 处理 |
|---|---|---|
| clang-format | 23 个 C++ 文件、555 处不合格式 | 用固定版本重排 |
| 构建 | `-O3` 下 GCC 12 的 `-Wformat-truncation`（`hal_loader.c`） | 检查 `snprintf` 返回值并标注截断 |
| ruff | 146 处（新版规则） | 修复；`vlm_job.py` 加 B023 的单文件豁免（闭包只在循环内同步调用） |
| mypy | 13 处（`minicpmv.py` 的 HF 桩类型等） | 修复 |
| pytest | 19 个失败：CPU 版 torch 上 `get_device_name` 抛错；GGUF 测试需要 pyarrow / wikitext 数据集 | 新增 `pipeline/envinfo.device_name`（不抛错）；`test_deploy_job` 桩掉 `wikitext_text`；环境文件加 `pyarrow` |

修复后：C++ 344 项 ctest 通过，pytest 583 通过 / 18 跳过，mypy 55 个文件无问题，前端构建通过；clang-tidy 仍是 SKIP（该工具链的系统头文件它解析不了）。工具版本已在环境文件里固定：clang-format 23.1.2、ruff 0.16.8、mypy 2.3.1（clang-format 23.1.3、ruff 0.17.0 也各自验证过）。torch / transformers 在干净环境里是 2.14.1+cpu / 5.19.0，开发环境是 2.11 / 5.7.0，测试两边都过。

顺带的两处小的工具链改动，属 L2，评审时请看：`cpp/CMakeLists.txt`（`modelspec` 的 include 目录）、`python/pyproject.toml`（上面那条 B023 豁免）。

**仍未验证**：见第 3 节（GitHub 专属语法、Chrome、冷/热构建时间）；`ci.proposal.yml` 里的 e2e 步骤没有在干净环境里跑过（那一步需要服务 + Chrome）。

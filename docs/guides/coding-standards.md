# 编码规范与本地门禁（T-0005）

规范的**可执行形式**就是 `scripts/check_all.sh`：它与 CI 等价，任何“规范”条目都应当有对应的自动检查，没有检查的条目只是建议。

```bash
scripts/check_all.sh            # 全部检查
scripts/check_all.sh --fast     # 跳过 C++ 重新构建与 pytest
SKIP=web,cpp scripts/check_all.sh   # 跳过指定组：deps,format,cpp,python,web
BUILD_DIR=build/rel scripts/check_all.sh
```

输出末尾是一张汇总表，每项为 PASS / FAIL / SKIP。**SKIP 只表示该工具在当前环境不可用**（例如没装 clang-format、`web/node_modules` 不存在），不代表通过。

| 检查 | 工具 | 配置 |
|------|------|------|
| 模块依赖方向 | `scripts/check_deps.py` | 脚本内 `ALLOWED` 表（改动 = 架构变更，L2） |
| C++ 格式 | `clang-format --dry-run --Werror` | `.clang-format`（Google，列宽 110） |
| C++ 构建 + 单测 | `cmake --build` / `ctest` | `CMakePresets.json` |
| C++ 静态检查 | `clang-tidy -p build/dev` | `.clang-tidy`（只开缺陷类检查） |
| Python lint | `ruff check python` | `python/pyproject.toml` |
| Python 类型 | `mypy`（在 `python/` 下运行） | 同上，`disallow_untyped_defs` 开启 |
| Python 单测 | `pytest` | 同上 |
| 前端 | `npm run build`（tsc -b + vite） | `web/tsconfig.json` |

提交前也可以用 `pre-commit`（`.pre-commit-config.yaml`）：

```bash
pip install pre-commit && pre-commit install     # 之后每次 git commit 自动跑
pre-commit run --all-files
```

---

## C++

- C++20/23，`namespace mqde::<module>`；文件 `snake_case`，类型 `PascalCase`，函数/变量 `snake_case`，常量 `kPascalCase`，成员变量后缀 `_`。
- 模块边界不抛异常，返回 `mqde::Result<T>`；错误信息带上下文（哪个模型、哪一层、哪个参数）。
- **格式由 clang-format 决定**，不要手工对齐。本仓库允许单行守卫（`if (!r) return std::unexpected(r.error());`），因此 `.clang-tidy` 关掉了 `readability-braces-around-statements` 一类风格检查。
- **模块依赖方向**（`scripts/check_deps.py` 强制）：

  ```
  common ← modelspec, quant ← estimator ← fit, hub ← server ← apps / bindings / tests
  ```

  `#include <mqde/x/...>` 只能指向本模块或上表允许的模块。新增一条边属于架构变更，先走 L2 评审再改 `ALLOWED` 表。
- 数值核心 `cpp/src/quant/` 编译时关闭 FMA 收缩与 fast-math（`-ffp-contract=off -fno-fast-math`），这是与 Python 逐位一致的前提，不要改。
- 厂商 SDK 一律 `dlopen`，不得成为编译期硬依赖；日志走 `MQDE_LOG_*`，不直接 `printf/std::cout`（CLI 的用户输出除外）。

## Python

- Python ≥ 3.10，**全部函数必须有类型注解**（mypy `disallow_untyped_defs`）。
- torch 的 `nn.Module.__getattr__` 返回 `Tensor | Module`，因此注册的 buffer 要在类体里声明类型（见 `FakeQuantLinear`），而不是到处 `cast`；HF 的 config 等动态对象用 `cast(Any, ...)` 并写明原因。
- 行宽 140；`ruff` 规则集 `E,F,W,I,UP,B,SIM,NPY,RUF`。已知例外都写在 `pyproject.toml` 的 `ignore` / `per-file-ignores` 里并附原因（例如作业流水线的 `B023`：闭包由 `_stage()` 在本轮循环内立即调用）。
- Worker 的 stdout 只能输出协议帧，日志一律 `logging` 到 stderr。
- 数值语义以 `mqde._core`（C++）为准；新增 Torch 快速路径必须有 golden 对拍测试。
- 新算法/模型/导出器通过注册表注册（`ALGO_REGISTRY` / `KV_REGISTRY` / `EXPORTER_REGISTRY`），不改主流程。

## 前端

- TypeScript strict；图表组件只接收纯数据 props；不要在组件里拼 API URL，统一走 `web/src/api.ts`。
- 后端字段先在 `api.ts` 里声明类型，再在页面里用；不要用 `as any` 绕过（`LayerTrace.algo_info` 这类可选字段就声明成可选）。

## 通用

- 不提交凭证、模型权重、大数据集（pre-commit 限制 512 KB）；token 不得出现在日志、报告、事件、数据库中。
- 默认值要安全：`trust_remote_code=false`，server 只监听 `127.0.0.1`。
- 测试默认离线：需要网络的用 `MQDE_ONLINE_TESTS=1` 显式开启；本地 HTTP 桩见 `cpp/tests/unit/hub_remote_test.cpp`。
- 不要为了让检查通过而删测试、放宽阈值或改 golden（AGENTS.md §9）。

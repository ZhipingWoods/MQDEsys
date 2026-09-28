# AGENTS.md —— MQDEsys Agent 工作手册

> **任何 Agent（或新成员）接手本项目时，先完整阅读本文件。**
> 它回答三个问题：**干到哪儿了**、**怎么干的**、**接下来如何干**。
>
> 文档级别：**L2**（修改需人工评审）。

---

## 0. 先读这个

动手前先读 [docs/REFERENCES.md](docs/REFERENCES.md)：每个子系统模仿哪个开源项目、照搬什么、不照搬什么。任务清单附录 R 列出了剩余的对齐任务。

## 1. 项目一句话

**MQDEsys（Model Quantization & Deployment Evaluation System）**：指定模型（HuggingFace 等）、量化配方（W4A8 / W4A16 / W2A16 / W2A8 / FP8）、评估数据/类型、部署硬件，系统自动完成 **预估 → 量化 → 评估 → 部署 → 压测 → 反馈**，并在 Web 前端可视化执行过程、量化误差（定位到层与转换环节）以及模型占用、KV Cache、单/多并发资源预估。

- 后端主干：**C++20**（控制面、预估、探针、数值核心、部署适配、API 服务）
- 执行器：**Python**（PyTorch 相关的量化算法、评测、导出），由 C++ 以子进程调度
- 前端：**React + TypeScript**
- 代码管理：**Git**（GitHub：`zhipingwoods/MQDEsys`）

---

## 2. 必读文档与阅读顺序

| 顺序 | 文档 | 什么时候读 | 你能从中得到什么 |
|------|------|-----------|-----------------|
| 1 | `AGENTS.md`（本文件） | 每次接手 | 规则、流程、禁止事项 |
| 2 | [`docs/PROGRESS.md`](docs/PROGRESS.md) | 每次接手 | **当前干到哪儿了**、上一位留下的交接信息、阻塞项 |
| 3 | [`docs/TASKS.md`](docs/TASKS.md) | 选任务时 | 全部任务、依赖、状态、验收与校验方法 |
| 4 | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 做任务前读相关章节 | 设计意图、接口、公式、数据流 |
| 5 | [`docs/STRUCTURE.md`](docs/STRUCTURE.md) | 新增/修改文件前 | 文件放哪、谁能改、改了要谁审（L0–L3） |
| 6 | `docs/adr/` | 涉及架构取舍时 | 已做出的决策及其原因 |
| 7 | `docs/guides/` | 做对应类型任务时 | 如何新增算法/模型/后端/硬件插件、编码规范 |

原始需求与设想：`../模型量化部署评估系统/模型量化部署评估系统.md`（仓库外，供参考；以本仓库文档为准）。

---

## 3. 标准工作流程（每个任务都按这个走）

```
┌─ 1. 同步 ────── git checkout main && git pull
├─ 2. 了解现状 ── 读 PROGRESS.md 最新 3 条 + TASKS.md 总览表
├─ 3. 选任务 ──── 选一个 状态=TODO 且 依赖全部 DONE 的任务（优先编号小、在关键路径上的）
├─ 4. 认领 ────── 在 TASKS.md 把状态改为 `DOING @你的名字 feat/T-xxxx-slug`，单独提交并推送
│                  （提交信息：`chore(tasks): claim T-xxxx`）—— 防止多个 Agent 撞车
├─ 5. 建分支 ──── git checkout -b feat/T-xxxx-short-slug
├─ 6. 确认级别 ── 对照 STRUCTURE.md 判断会碰到的最高级别路径：
│                  L0/L1 → 直接做；L2 → 先在 PR 草稿/Issue 写接口方案等人工确认；L3 → 先写 ADR 停下等待
├─ 7. 实现 ────── 小步提交；先写/改测试再写实现（数值、预估、协议类必须先有 golden/契约测试）
├─ 8. 自验 ────── 逐条执行任务的「校验方法」+ scripts/check_all.sh；保存输出
├─ 9. 更新文档 ── TASKS.md 状态改为 `REVIEW #PR号`；PROGRESS.md 追加交接记录；相关 guides/README 同步
├─ 10. 提 PR ──── 按 PR 模板填写：任务 ID、级别、变更摘要、校验输出、未完成事项
└─ 11. 合并后 ─── TASKS.md 状态改为 `DONE <sha>`（可由合并者或下一位 Agent 完成）
```

**会话中断/上下文将满时**：立即执行第 9 步中的 PROGRESS.md 更新（即使任务未完成），写清“做到哪一步、下一步做什么、哪些文件改了一半”，并把当前工作提交到特性分支（提交信息加 `wip:` 前缀）。**任何时候都不要让进度只存在于你的上下文里。**

---

## 4. 变更级别速查（详见 STRUCTURE.md 第 1 节）

| 级别 | 典型路径 | 你能做什么 |
|------|---------|-----------|
| **L0** 自由 | `cpp/src/**` 内部实现、单元测试、`examples/` | 直接改 |
| **L1** 常规评审 | 新文件、新插件实现、新数据条目、前端页面 | 改完提 PR，1 名评审 |
| **L2** 人工评审 | `cpp/include/**`、`api/**`、`schemas/**`、HAL ABI、Worker 协议、DB 迁移、`data/compat/**`、`data/golden/**`、构建/CI、本文件及 ARCHITECTURE/STRUCTURE | **先提方案，获批再改** |
| **L3** 架构决策 | 新第三方依赖、许可证、安全、破坏性变更、删除功能 | **只写 ADR，然后停下等待负责人** |

需要人工决策时：在 PROGRESS.md 的「待人工决策」区登记，并将任务标记为 `BLOCKED: 等待 xxx 决策`，然后去做其他不受阻的任务。

---

## 5. Git 规范

### 5.1 分支

- `main`：受保护，始终可构建、测试全绿；只能通过 PR 合并（squash merge）。
- 特性分支：`feat/T-xxxx-slug`；修复：`fix/T-xxxx-slug` 或 `fix/issue-123-slug`；文档：`docs/slug`；ADR：`adr/NNNN-slug`。
- 分支生命周期尽量 < 3 天；超过则拆分任务。
- 发布：在 `main` 上打 tag `vMAJOR.MINOR.PATCH`（SemVer）；需要热修时从 tag 拉 `release/vX.Y` 分支。

### 5.2 提交信息（Conventional Commits + 任务 ID）

```
<type>(<scope>): <简述，祈使句，≤ 72 字符>

<为什么改、怎么改，可选>

Refs: T-0208
```

- `type`：`feat` `fix` `refactor` `perf` `test` `docs` `build` `ci` `chore`
- `scope`：模块名，如 `estimator` `quant` `hub` `web` `worker` `api` `tasks`
- 破坏性变更：`feat(api)!: ...` 并在正文写 `BREAKING CHANGE: ...`

### 5.3 PR 要求

- 一个 PR 只做一个任务（或其子任务）；非生成代码 ≤ 800 行，超过请拆。
- PR 描述必须包含：任务 ID、级别、变更摘要、**校验方法的实际执行输出**、截图（前端）、未完成事项/已知问题。
- 生成代码（`web/src/api/generated/**` 等）与源契约同一 PR 提交。
- 禁止 `git push --force` 到 `main`；特性分支上可以 `--force-with-lease`。

---

## 6. 编码约定（摘要，完整见 `docs/guides/coding-standards.md`）

### C++
- C++20，`namespace mqde::<module>`；文件 `snake_case`，类型 `PascalCase`，函数/变量 `snake_case`，常量 `kPascalCase`，成员变量后缀 `_`。
- 模块边界不抛异常，返回 `mqde::Result<T>`；错误信息包含上下文（哪个模型、哪一层、哪个参数）。
- 公共头文件只放接口与必要类型；实现细节放 `src/`。
- 遵守模块依赖方向（STRUCTURE.md 3.5），`scripts/check_deps.py` 会检查。
- 厂商 SDK 一律 `dlopen` 动态加载，**不能**成为编译期硬依赖。
- 日志使用 `MQDE_LOG_*` 宏，不直接 `printf/std::cout`（CLI 的用户输出除外）。

### Python
- Python ≥ 3.10，类型注解必写（mypy 检查），ruff 格式化。
- Worker 的 stdout 只能输出协议帧；任何 `print` 都会破坏协议 —— 日志一律用 `logging`（输出到 stderr）。
- 数值语义以 `mqde._core` 为准；新增 Torch 快速路径必须通过 golden 测试。
- 新算法/模型/导出器通过注册表注册，不修改主流程。

### 前端
- TypeScript strict；API 类型只能来自生成代码；图表组件只接收纯数据 props。
- 所有用户可见文本走 i18n（zh-CN / en-US）。

### 通用
- 不提交凭证、模型权重、大数据集；token 不得出现在日志、报告、事件、数据库中。
- 默认值要安全：`trust_remote_code=false`、server 只监听 `127.0.0.1`。
- 测试用 `tests/fixtures/tiny-llama` 等离线夹具，CI 默认不访问网络；需要网络的测试用 `MQDE_ONLINE_TESTS=1` 显式开启。

---

## 7. 如何验证（验收与校验）

1. **每个任务的验收标准与校验方法写在 TASKS.md 中**，必须逐条执行并在 PR 中贴出输出。
2. **本地全量门禁**：`scripts/check_all.sh`（与 CI 等价）。在 T-0005 完成前，用任务自身的校验命令代替。
3. **常用命令**：
   ```bash
   # C++
   cmake --preset dev && cmake --build --preset dev && ctest --preset dev
   ctest --preset dev -R estimator_          # 只跑某模块
   # Python
   pip install -e ".[dev,quant]" && pytest python/tests -q
   # Web
   cd web && pnpm lint && pnpm typecheck && pnpm test && pnpm e2e
   # 契约
   scripts/gen_api.sh --check && scripts/contract_test.sh
   # 端到端（CPU，tiny 模型）
   pytest tests/integration/test_e2e_tiny.py
   ```
4. **Golden 数据**（`data/golden/**`）是数值正确性的锚点：测试失败时**先怀疑代码，不要改 golden**。确需更新 golden，必须在 PR 中说明数值依据并走 L2 评审。
5. **需要 GPU 的验收**：若当前环境无 GPU，完成 CPU 可验证部分后，将 GPU 校验项在 PR 与 PROGRESS.md 中标注为「待 GPU 验证」，任务状态保持 `REVIEW` 而不是 `DONE`。

---

## 8. 常见任务的做法指引

| 我要…… | 去看 | 核心步骤 |
|--------|------|---------|
| 新增量化算法 | `docs/guides/add-algorithm.md` | 继承 `BaseBlockwiseQuantization` → 实现 `block_opt` → `@ALGO_REGISTRY.register` → tiny 模型测试 → 更新兼容矩阵（L2） |
| 新增模型架构 | `docs/guides/add-model-adapter.md` | 优先写 `data/model_families/<family>.yaml` → 加 golden ModelSpec → 必要时写 Python `BaseModel` 子类 |
| 新增部署后端 | `docs/guides/add-backend.md` | C++ `IDeployBackend` + Python `Exporter` + `data/backends/<name>.yaml` + 兼容矩阵 + 集成测试 |
| 新增硬件 | `docs/guides/add-hardware-plugin.md` | 规格进 `data/hardware_db`；需要探测/微基准则复制 `cpp/plugins/template` |
| 新增 API | `api/README.md` | 先改 `openapi.yaml`（L2）→ `scripts/gen_api.sh` → 实现路由 → 契约测试 → 前端 |
| 新增配方 | `docs/guides/add-recipe.md` | `data/recipes/<name>.yaml` → 兼容矩阵 → 预估用例 |

> 上述 guides 由 P0/P5/P6/P7/P11 中的任务逐步产出；若尚不存在，参照 ARCHITECTURE.md 对应章节执行，并在完成任务时顺手补齐该 guide。

---

## 9. 禁止事项（红线）

1. 未读 PROGRESS.md 和 TASKS.md 就开始写代码。
2. 未认领就做任务；同时持有多个 DOING 任务。
3. 越级修改：未获批就改 L2 路径；未写 ADR 就引入新依赖或做破坏性变更。
4. 为了让测试通过而删除/跳过测试、放宽阈值、修改 golden。
5. 手改生成代码；修改已合并的数据库迁移。
6. 提交凭证、权重、大文件；在日志中打印 token。
7. 在任务范围外“顺手”大规模重构或改名（发现问题请登记到 PROGRESS.md 的「发现的问题」区或开 Issue）。
8. 在 PR 中声称“已验证”但未贴出校验输出；把 GPU 未验证的任务标记为 DONE。
9. 结束会话前不更新 PROGRESS.md。

---

## 10. 交接模板（写入 docs/PROGRESS.md 顶部“会话日志”）

```markdown
### YYYY-MM-DD HH:MM · @agent-name · T-xxxx <任务标题>
- **状态**：DOING / REVIEW #PR / DONE <sha> / BLOCKED
- **本次完成**：
  - …
- **校验结果**：`<命令>` → 通过/失败（关键输出摘要）
- **未完成 / 下一步**（下一位 Agent 从这里接着做）：
  1. …
- **改了一半的文件**：`path/to/file`（说明状态）
- **发现的问题 / 风险**：…
- **需要人工决策**：…（同时登记到“待人工决策”区）
```

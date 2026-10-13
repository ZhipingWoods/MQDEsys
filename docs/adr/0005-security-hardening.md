# ADR-0005: 安全加固（跨站请求、认证、凭证、trust_remote_code）

| 字段 | 内容 |
|------|------|
| 状态 | **Proposed**（未批准；本文档只起草，**不实现任何代码**，见第 6 节） |
| 日期 | 2026-10-13 |
| 作者 | Claude（Sonnet 5.5） |
| 审核人 | 负责人（zhipingwu）—— 待定 |
| 依赖 | T-0512（作业端点）已 PARTIAL |
| 影响任务 | T-1204（L3，本 ADR 由其起草）；批准后会派生出 L2 子任务（server 中间件、JobSpec 字段、前端） |

---

## 1. 背景

T-1204 要求：凭证存储与脱敏审计、`allowed_paths` 沙箱、server token 认证（对外监听时强制）、`trust_remote_code` 二次确认。
本文先列出**代码里实际是什么样**（含一个可复现的问题），再给选项和推荐。以下事实取自 `80bcd20` 之后的主干，复现命令见第 2.1 节。

## 2. 现状（代码事实）

### 2.1 服务端对来自网页的请求没有任何防护 —— **已复现**

- 默认只监听 `127.0.0.1`（`ServerOptions.host`），可改；**没有任何认证**，改成非回环地址后同样没有。
- **不校验 `Origin`，不校验 `Host`，不检查 `Content-Type`**：`parse_body` 把任何 POST 体当 JSON 解析。没有任何 CORS 头，`OPTIONS` 返回 404。
- 后果 1（跨站请求伪造）：浏览器对 `Content-Type: text/plain` 的跨站 POST 是"简单请求"，**不做预检**。服务端照单全收，所以用户开着服务时打开任意网页，该网页就能让服务执行下表里的动作（页面读不到响应，但副作用发生）。
- 后果 2（DNS rebinding）：没有 `Host` 白名单，rebinding 到 `127.0.0.1` 的域名能连响应一起读走（作业报告、模型路径、硬件信息）。

会改变状态的端点（全部受上述影响）：

| 端点 | 副作用 |
|------|--------|
| `POST /api/v1/jobs` | 占用 GPU、启动 Python 子进程 |
| `POST /api/v1/jobs/<id>:cancel` | 终止别人提交的作业 |
| `POST /api/v1/backend-envs/<id>/install` | 联网下载并安装 vLLM / SGLang / llama.cpp（运行 uv / pip / cmake） |
| `POST /api/v1/download` 及 `:cancel` | 从 Hugging Face 下载到允许目录，占磁盘与带宽 |
| `POST /api/v1/hardware/profile` | 在 GPU 上跑微基准 |
| 其余 `POST`（`estimate`、`fit`、`models:*`、`recommend`） | 只读计算，但会读允许目录下的文件 |

复现（本地起 `mqde serve`，把端口换成自己的）：

```bash
# 1) 陌生 Origin 的 text/plain POST 被接受并按 JSON 解析 → HTTP 200 {"valid":true}
curl -s -X POST "localhost:18090/api/v1/jobs?dry_run=1" -H "content-type: text/plain" -H "Origin: http://evil.example" \
     --data '{"domain":"llm","model":{"path":"/path/to/model","type":"llm"},"quant":{"recipe":["w4a16"],"algorithm":"rtn"},"deploy":{"device":"gpu"}}'
# 2) 预检被 404，但简单请求本来就不需要预检
curl -s -o /dev/null -w "%{http_code}\n" -X OPTIONS localhost:18090/api/v1/jobs -H "Origin: http://evil.example" -H "Access-Control-Request-Method: POST"
# 3) 任意 Host 头都被接受
curl -s -o /dev/null -w "%{http_code}\n" -H "Host: evil.example:18090" localhost:18090/api/v1/health
```

2026-10-13 在 Mac 与 4090 同一份代码上的结果：200 / 404 / 200。

### 2.2 路径沙箱

`path_allowed()`（`server.cpp`）用 `weakly_canonical` 展开符号链接后做"根目录 + `/`"前缀比较，覆盖模型路径、`models:*`、`models:gguf`。作业 id 与 `artifact.job` 用正则限制为 `[A-Za-z0-9_-]+`，所以不能通过它们穿越目录。**现有测试没有专门覆盖符号链接逃逸、`..`、同前缀兄弟目录（`/root` 与 `/root-evil`）这几种情形**，行为靠代码阅读而不是测试保证。

### 2.3 凭证

`HF_TOKEN` 只从环境变量读取（`hub/hf.cpp`），错误信息里不带 token（代码注释写明）。系统不把它写盘，前端也不接收它。但：

- Python 执行器由 `JobRunner` 直接 `fork + exec`，**继承服务进程的完整环境**，包括 `HF_TOKEN`；`fetch_model` 在 C++ 里做，Python worker 其实不需要它。
- `worker.log` 收集子进程 stdout/stderr。目前没有代码打印环境，但**没有测试保证**日志、报告、事件里不含 token。

### 2.4 `trust_remote_code`

`trust_remote_code=True` 会执行模型目录自带的 Python 文件。AGENTS.md 第 6 节规定默认必须是 false。

| 适配器 | 现状 |
|--------|------|
| `hf_llm.py`（LLM） | `False` |
| `minicpmv.py`（VLM） | **曾硬编码 `True`**；本次已改为参数、默认 `False`（该模型是 transformers 原生架构，目录里没有 `auto_map` 与 `.py`），真机验证 VLM 作业不需要它 |
| `minicpm_vla.py`（VLA） | `True`——该模型目录自带 `modeling_minicpm_vla.py`，**必须开**；没有任何用户可见的确认，JobSpec 里也无法表达 |

## 3. 决策点与选项

### D1 跨站请求与 DNS rebinding

| 选项 | 内容 | 评价 |
|------|------|------|
| **A（推荐）** | 所有会改状态的 `POST` 要求 `Content-Type: application/json`（否则 415），并拒绝 `Origin` 不在白名单的请求（403）；所有请求校验 `Host` 在白名单（`localhost`、`127.0.0.1`、`[::1]`、`--host` 指定值，否则 421）。无 `Origin` 的请求（curl、脚本）照常通过 | 几行中间件；不改前端（前端 `fetch` 本来就发 JSON、同源）；关掉伪造与 rebinding |
| B | 只加 A 的 Content-Type 要求 | 能挡住简单请求，但挡不住 rebinding |
| C | 不改，在文档里写"不要在浏览器开着服务时访问不可信网页" | 不可接受：用户无从得知 |

### D2 认证

| 选项 | 内容 |
|------|------|
| **A（推荐）** | 绑定非回环地址时**必须**配置 token（`--token-file` 或 `MQDE_TOKEN`），否则拒绝启动；请求带 `Authorization: Bearer`，常量时间比较；回环地址上可选。前端首次访问提示输入，存 `sessionStorage`，不写日志与报告 |
| B | 回环地址也强制 token | 对本机单用户开发是负担，且仍挡不住本机其它进程，收益小 |
| C | 不做认证，要求部署者放反向代理 | 与"对外监听时强制"的任务书不符 |

### D3 凭证

推荐：保持"只读环境变量、不落盘"；`JobRunner` 启动 worker 时**去掉 `HF_TOKEN`**（需要时再显式传）；加一个 grep 测试，跑一次完整作业后断言 `runs/<id>/` 下所有文件（日志、报告、事件、job.yaml）不含该 token 的值。

### D4 `trust_remote_code` 二次确认

推荐：JobSpec 新增 `model.trust_remote_code`（布尔，默认 false，**L2：`schemas/`**）。`validate_jobspec` 对"目录里有 `auto_map` 或自带 `.py`"的模型，在未显式为 true 时拒绝并说明会执行哪些文件；前端在作业向导里显示这些文件名和明确警告，要求勾选。适配器只信任 JobSpec 传下来的值。现有的 MiniCPM-RobotManip VLA 作业要显式勾选。

### D5 路径沙箱

保持现有实现，补测试：符号链接逃逸、`..` 段、`/root` 对 `/root-evil`、不存在的路径、大小写（macOS）。

## 4. 不做的事

- 不做 TLS（默认回环；对外暴露的部署由反向代理终结）。
- 不做多用户 / RBAC / 审计日志（超出任务书范围）。
- 不改变 `allowed_roots` 的语义。

## 5. 影响与迁移

- D1-A 的唯一不兼容：不带 `Content-Type: application/json` 的写请求会被 415。仓库内的前端、e2e、Python 客户端（`bench_reference.py`）都发 JSON；外部脚本需要补这个头。
- D4 增加 JobSpec 字段（向后兼容：缺省即 false），但会让现有 VLA 作业在不勾选时被拒——这是预期行为。

## 6. 验收（批准后实现时用）

`tests/integration/test_security.py`（对真实 `mqde serve`）：

1. 陌生 `Origin` 的写请求 → 403；`text/plain` 的写请求 → 415；非白名单 `Host` → 421；同源与无 `Origin` 的请求照常。
2. 非回环地址且无 token → 启动失败；有 token 时无/错 token → 401，正确 → 200。
3. 完整作业后 `runs/<id>/**` 不含 `HF_TOKEN` 的值；worker 的环境里没有 `HF_TOKEN`。
4. 路径沙箱的 5 种逃逸尝试全部 403。
5. 需要远程代码的模型在未设 `model.trust_remote_code` 时 400（带文件名），设了才放行；前端 e2e 覆盖警告与勾选。
6. 每条断言都做 mutation：删掉对应检查，测试必须变红。

## 7. 待批准

请负责人对 D1–D5 逐项选择（推荐为各自的 A）并把状态改为 Accepted。批准前，**不要**实现 D1–D4；本 ADR 起草时唯一改的代码是 MiniCPM-V 适配器的默认值（第 2.4 节），因为那是把代码恢复到 AGENTS.md 已规定的默认，不是新的安全功能。

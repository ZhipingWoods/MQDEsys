# 方案：T-0007 契约与 Schema（L2，等待评审，未实现）

日期 2026-10-13 · 作者 Claude（Sonnet 5.5） · 状态 **Proposed** · 触及路径：`api/**`、`schemas/**`（L2）

## 1. 现状（取自代码，不是取自任务书）

| 任务书要求 | 现在 |
|---|---|
| `api/openapi.yaml`，覆盖 A§6.14 全部端点 | **不存在**（没有 `api/` 目录）。服务端实际注册了 **38 个路由**（`server.cpp`：GET 21、POST 17，含 `/metrics`），其中不少是任务书之后才出现的：`models:gguf`、`backend-envs`、`recommend`、`export-suggestion`、`jobs?dry_run=1`…… |
| `api/events.schema.json` | 不存在。真实事件流（4090 的 14 个作业、8563 条事件）里出现了 8 种 `type`：`job_state`、`stage_started`、`stage_finished`、`metric`、`log`、`progress`、`trace`、`layer`（后两种各占约 1/3，`layer` 的 `trace` 是逐层误差对象）。前端 `web/src/api.ts` 里手写了对应的 TypeScript 类型 |
| `api/worker_rpc.schema.json` | 不存在，见 `T-0501-worker-rpc.md`（建议废止） |
| `schemas/*.schema.json`（A§7 全部结构） | 只有 `job_spec.schema.json`（T-1305 做的，`additionalProperties: true`，所有 pipeline 都用它）。其余结构（`Report`、`HardwareProfile`、`Recipe`…）没有 schema，只有 C++ 加载器与 `data/*/schema` 的零散校验 |
| `examples/jobs/*.yaml` ≥5 个，覆盖五种配方 | 3 个（`minicpm_vla_libero`、`minicpm_vla_smoke`、`qwen25_05b_wikitext`）；`scripts/validate_job_schema.py` 已校验它们 |
| `scripts/gen_api.sh --check` + `openapi-typescript` 生成类型 | 不存在；前端类型是手写的，**已经发生过漂移**：本轮发现的 `llm_deploy` 作业种类、`/jobs` 的 `device_name`/`parent_job` 字段都是前端和服务端各自加的 |

## 2. 建议：先做"防漂移"，再做"生成"

不一次性写一个 38 路由、几十个结构的 OpenAPI（那是最容易与代码脱节的做法），而是三步，每步都是小而可验证的：

1. **`api/events.schema.json`（最小，价值最高）**：按真实的 8 种事件类型写 JSON Schema（`type` 做判别字段），校验方式是**把 `runs/*/events.jsonl` 里的全部真实事件过一遍**（今天 8563 条）并要求 0 条不符；同时对 `python/mqde/pipeline/events.py` 的发射函数做一次反向检查：任何新增的事件类型没进 schema 就失败。
2. **`api/openapi.yaml` 只列路由 + 请求体引用 `schemas/`，响应先写 `type: object`**，配一个**路由漂移测试**：从 `server.cpp` 抽出全部 `svr.Get/Post(...)` 路径，与 OpenAPI 的 `paths` 双向对比，多一条少一条都失败（这和已有的 `check_algo_registry_sync.py` 同一思路）。逐个端点再把响应收紧，优先前端实际用到的。
3. **前端类型生成**（`openapi-typescript`）放在 1、2 之后，且只在响应收紧后才有意义；在此之前手写类型保持现状。这一步还会新增一个前端依赖（`web/package.json`，L3 要 ADR），所以单独拆出去。

## 3. 需要负责人决定

- 是否同意**把 T-0007 的范围从"一次性全覆盖"改为上述三步**，并把任务书里的"所有示例 ≥5 个覆盖五种配方"改为"在 `examples/jobs` 补到每个配方一个"（两项都不难，但与任务书不同，所以请确认）。
- `schemas/job_spec.schema.json` 目前 `additionalProperties: true`；要不要对 `evaluation`、`deploy` 收紧为 `false`？收紧会让拼错字段的作业在提交时就被拒绝，但也会让现有的、依赖多余字段的客户端失败，需要你拍板。

## 4. 验收（批准后）

- `events.schema.json` 通过 8563 条真实事件，且删掉任一类型后测试变红。
- 路由漂移测试：往 `server.cpp` 加一个路由不改 OpenAPI → 红；反之亦然。
- `scripts/check_all.sh` 把这两项纳入（已有同类检查的位置）。

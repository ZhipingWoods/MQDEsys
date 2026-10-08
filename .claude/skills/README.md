# MQDEsys skills / 项目技能

Project-level Claude Code skills. They are checked in so anyone who clones the repository (and their agents) gets the
same procedures. Each one is short; the details live in its `references/` folder and in `docs/`.
项目级 Claude Code 技能，随仓库提交，克隆即用。每个技能保持简短，细节放在各自的 `references/` 和 `docs/`。

| Skill | Use it when / 何时使用 |
|---|---|
| [`mqde-quantize-eval`](mqde-quantize-eval/SKILL.md) | estimate memory, pick a recipe and algorithm, run and read a quantization job / 预估显存、选配方与算法、跑并读量化作业 |
| [`mqde-deploy-benchmark`](mqde-deploy-benchmark/SKILL.md) | export, serve with vLLM or llama.cpp, load-test, served PPL, capacity boundary / 导出、部署、压测、部署后 PPL、容量边界 |
| [`mqde-extend`](mqde-extend/SKILL.md) | add an algorithm, exporter, backend, model family, recipe or KV method / 新增算法、导出格式、后端、模型家族、配方、KV 方法 |
| [`mqde-dev-gate`](mqde-dev-gate/SKILL.md) | sync, run the gate, scan before pushing, commit, update the task books / 同步、门禁、推送前扫描、提交、更新任务文档 |

Not skills: the long-form docs are in `docs/` (internal, Chinese) and `docs/site/` (user-facing, English and Chinese).
非技能的长文档在 `docs/`（内部，中文）与 `docs/site/`（面向用户，中英文）。

Conventions / 约定: front matter has `name`, a trigger-oriented `description` that also says when **not** to use it,
`user-invocable`, and a minimal `allowed-tools`. Commands in a skill must be runnable as written; paths of a particular
machine are written as placeholders (`<REPO>`, `<MODEL_DIR>`, `<SERVER>`).

# T-R21~R24 实现总结

## 已完成功能

### 1. InstalledIndex (T-R24)
- **核心功能**：`probe_installed()` 扫描 HF cache + Ollama models 目录
- **配置**：`data/fit/installed.yaml` 定义 provider 路径与匹配规则
- **查询**：`installed_providers(model_name)` 返回哪些 runtime 已安装该模型
- **API集成**：`/api/v1/fit` 的 `ModelFit` 包含 `installed: bool` 和 `installed_providers: []`
- **状态**：基础功能完成，但 **4个字段仅占位**（disk_size_gb=0, capability_ids=[], supports_tp=false, usable_context=0）

### 2. ConcurrencyEstimate (T-R21)
- **核心功能**：给定模型+硬件+配方+用户数/上下文，判断是否 fits
- **算法**：`memory_for_users(N)` = 权重 + KV(N × ctx) + 激活 + 元数据
- **返回**：`fits`, `max_users`, `max_context_for_users`, `utilization_pct`
- **API**：`POST /api/v1/fit:concurrency` 端点
- **状态**：简化版完成，**未对齐 llmfit §3.3**（KV cache复用、paged attention、chunked prefill）

### 3. ClaimRange (T-R23)
- **核心功能**：基于实测数据给出保守/乐观速度声明
- **算法**：当前占位（conservative = measured × 0.8, optimistic × 1.2）
- **API**：`POST /api/v1/fit:claim` 端点
- **状态**：端点完成，**未实现 llmfit §3.4** 置信区间/样本分布/多样本聚合

### 4. HF Catalog + Capability Filter (T-R25 部分)
- **功能**：`mqde catalog update` 抓取 HF trending models → `data/models/catalog.json`
- **搜索**：`mqde catalog search -n 300 --include-demoted` + provider/capability过滤
- **API**：`GET /api/v1/catalog?capability=text-generation&provider=hf_transformers`
- **Web页面**：`/catalog` 搜索 + provider dropdown + capability dropdown
- **状态**：基础完成，但 **4个字段缺失**（capability_ids[], downloads, likes, created_at 未从HF API提取）

## 缺失的 llmfit 对齐项

### InstalledIndex 需补充（优先级 HIGH）
1. **disk_size_gb**：扫描 `.safetensors` / `.bin` 累加文件大小
2. **usable_context**：读 `config.json` → `max_position_embeddings`
3. **capability_ids**：读 `config.json` 推断（text/vision/audio/vla），参考 llmfit `_classify_capabilities`
4. **supports_tp**：检查是否存在 TP shard（如 `model-00001-of-00008.safetensors`）

### HF Catalog 需补充（优先级 MEDIUM）
1. 调用 `https://huggingface.co/api/models/{id}` 完整响应
2. 提取 `pipeline_tag` → `capability_ids[]`（text-generation / image-text-to-text / etc）
3. 提取 `downloads`, `likes`, `createdAt`
4. 迁移 llmfit `_classify_capabilities` 逻辑到 C++

### ConcurrencyEstimate 需对齐（优先级 MEDIUM）
- llmfit §3.3：KV cache 复用、paged attention overhead、chunked prefill
- `kv_cache_dtype` 影响（FP16/FP8/INT4）

### ClaimRange 需实现（优先级 LOW）
- llmfit §3.4：置信区间计算、样本分布、多样本聚合

## 测试覆盖

### 已有测试（86 tests pass）
- `installed_test.cpp`：3个（load_config, is_installed, providers_lookup）
- `concurrency_test.cpp`：3个（single_user_fits, max_users, max_context）
- `hub_test.cpp`：5个（normalize, parse_size, card_json, ...）

### 缺失边界用例
- concurrency：显存不足、KV cache超限
- claim：多样本/置信度
- catalog：vision/audio/multimodal分类
- installed：多provider冲突、版本解析

## 前端集成状态

### 已完成
- `/catalog` 页面（搜索 + provider/capability dropdown）
- Design system应用（5级surface tonal ladder, fluid clamp, grid-first）

### 待完成
1. Catalog卡片点击 → 跳转 `/fit?model={id}`
2. Fit页面「详情」面板集成 `ClaimRange`（显示 conservative/optimistic）
3. 已安装模型高亮显示（通过 `installed: true` 判断）
4. Catalog 与 Fit 页面设计统一（当前风格略有差异）

## 下一步建议

### 短期（本周内）
1. **补全 InstalledIndex 4字段**（disk_size_gb / usable_context / capability_ids / supports_tp）
   - 优先 disk_size_gb 和 usable_context（影响 fit 评估）
2. **HF catalog 完整metadata**（pipeline_tag / downloads / likes / createdAt）
3. **前端集成**：catalog → fit 跳转 + claim显示

### 中期（下周）
1. ConcurrencyEstimate 对齐 llmfit §3.3
2. ClaimRange 实现 llmfit §3.4
3. 测试覆盖补全

### 长期
- Runtime探测（T-R24后半部分）：检测已安装的 vLLM/Ollama/SGLang 版本
- 并发反解（T-R22）：给定硬件+模型，反推最大并发/上下文

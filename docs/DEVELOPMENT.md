# 开发指南（DEVELOPMENT）

> 环境、构建、测试、调试、提交流程。文档级别 **L1**。
>
> 相关：[../AGENTS.md](../AGENTS.md)（工作规则）· [STRUCTURE.md](STRUCTURE.md)（文件放哪）·
> [TASKS.md](TASKS.md)（做什么）· [REFERENCES.md](REFERENCES.md)（参考什么）

---

## 1. 机器与分工

| 角色 | 位置 | 用途 |
|------|------|------|
| 本地 Mac | `~/modelbest/MQDEsys/MQDEsys` | 编辑代码、写文档、看截图。**不编译、不跑模型** |
| 开发机 | `ssh 4090` → `/cache/wuzhiping/MQDEsys` | 编译、跑测试、跑模型、起服务 |

**GPU 约束**：只能用 **GPU 2 和 GPU 3**。当前 GPU 2 被他人进程占用（约 21 GB），所以实际只用 **GPU 3**。
起服务时显式传 `--gpus 3`，跑 Python 时设 `CUDA_VISIBLE_DEVICES=3`。

**代码同步**（Mac → 开发机，单向）：

```bash
cd ~/modelbest/MQDEsys/MQDEsys
rsync -a --exclude .git --exclude build --exclude __pycache__ --exclude '*.so' \
      --exclude runs --exclude node_modules --exclude dist --exclude 3rdpart \
      ./ 4090:/cache/wuzhiping/MQDEsys/
```

`3rdpart/` 里的第三方参考库是 **git 子模块**：仓库只记录上游地址和固定提交，GitHub 上点击目录会跳转到对应上游仓库。
源码只在开发机检出，不要同步、不要把源码提交进本仓库。`runs/`、`build/`、模型权重和 `*.gguf` 同样不进 git（见 `.gitignore`）。

需要在新机器上拉取参考源码时：

```bash
git submodule update --init 3rdpart/llmfit          # 只拉需要的一个
git submodule update --init --depth 1 --recursive   # 或全部浅克隆
```

---

## 2. 环境

开发机上一律先激活 conda 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mqde
```

外网访问（HF、npm）需要代理，按开发机上已配置的环境变量走；`mqde catalog update` 这类需要联网的命令要确认代理生效。

模型位置：`/cache/wuzhiping/hub/`

| 目录 | 架构 | 特点 |
|------|------|------|
| `MiniCPM-RobotManip` | minicpm_vla | VLA，1.54B，BF16 |
| `Qwen3.8-27B-FP8` | qwen3_5 | VLM，27.78B，块级 FP8（128×128），带 MTP 头 |

---

## 3. 构建与测试

```bash
ssh 4090
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mqde
cd /cache/wuzhiping/MQDEsys

cmake --preset dev              # 首次或改了 CMakeLists 后
cmake --build --preset dev -j 32
ctest --preset dev
```

编译选项是 `-Wall -Wextra -Werror`，警告即错误。常踩的坑：

- `-Wmissing-field-initializers`：用了指定初始化器（`.name = x`）就必须初始化**所有**字段，否则改成先声明再赋值。
- 结构化绑定遍历 `nlohmann::json` 临时对象会悬空（`for (auto& [k,v] : j.value("x", Json::object()).items())`）——必须先存到命名变量再遍历。这个 bug 曾导致 `model inspect` 直接崩溃。

只跑某个模块的测试：

```bash
build/dev/cpp/mqde_unit_tests --gtest_filter="fit.*"
```

前端：

```bash
cd /cache/wuzhiping/MQDEsys/web && npm run build
```

---

## 4. 起服务与看页面

生产服务一律用 `scripts/serve_prod.sh`（flags 的唯一来源；GPU 池默认 4-7，可用 `MQDE_GPUS=…` 覆盖；导出 `MQDE_ENVS_DIR` 让 vLLM/SGLang/lm-eval 能被找到）。少带 `--runs/--gpus/--python` 会让作业列表变空、GPU 作业无法调度、后端环境页报错。下面是它展开后的样子：

```bash
pkill -x mqde; sleep 1
cd /cache/wuzhiping/MQDEsys
nohup build/dev/bin/mqde serve \
  --repo /cache/wuzhiping/MQDEsys \
  --runs /cache/wuzhiping/MQDEsys/runs \
  --web /cache/wuzhiping/MQDEsys/web/dist \
  --python /root/miniconda3/envs/mqde/bin/python \
  --gpus 3 \
  --allow /cache/wuzhiping/hub \
  --port 18080 > /tmp/mqde_server.log 2>&1 &
```

Mac 上建隧道后用浏览器访问 `http://localhost:18080`：

```bash
ssh -f -N -L 18080:127.0.0.1:18080 4090
```

**改了前端后页面没变** → 先确认三件事，再怀疑代码：

1. `rsync` 是否在**写完文件之后**执行（顺序错了会同步旧文件）。
2. `web/dist/` 的时间戳是否是最新构建。
3. 浏览器缓存 —— `Cmd+Shift+R` 硬刷新，或开隐私窗口。

验证服务端确实在发新代码：

```bash
curl -s localhost:18080/ | grep assets/          # 取到 bundle 文件名
curl -s localhost:18080/assets/index-XXXX.js | grep -c '某个新加的字符串'
```

截图（Mac 上，隧道已建立）：

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --hide-scrollbars \
  --window-size=1440,1400 --virtual-time-budget=20000 \
  --screenshot=/tmp/mqde_shots/x.png "http://localhost:18080/fit"
```

---

## 5. 常用 CLI

```bash
build/dev/bin/mqde model inspect /cache/wuzhiping/hub/MiniCPM-RobotManip
build/dev/bin/mqde --json model inspect <path>          # JSON 输出
build/dev/bin/mqde estimate -m <path> -r w4a16 --hw nvidia/rtx4090 --ctx 8192
build/dev/bin/mqde fit /cache/wuzhiping/hub --hw nvidia/rtx4090
build/dev/bin/mqde fit <lib> --hw <id> --use-case vla --sort tps
build/dev/bin/mqde list recipes|hardware|families
```

---

## 6. 调试

**崩溃定位**（`-Werror` 之外的运行时错误）：

```bash
gdb -batch -ex run -ex bt --args build/dev/bin/mqde model inspect <path> 2>&1 | grep -E "^#" | head -20
```

栈帧里找第一个出现我们自己文件路径的行，那就是现场。

**看模型内部结构**（不经过我们的代码，用来交叉验证）：

```bash
python3 -c "
import json, struct, glob, collections, re
d = '/cache/wuzhiping/hub/<model>'
c = json.load(open(d + '/config.json')); print(list(c.keys()))
pat = collections.Counter()
for f in glob.glob(d + '/*.safetensors'):
    fh = open(f, 'rb'); n = struct.unpack('<Q', fh.read(8))[0]
    h = json.loads(fh.read(n)); h.pop('__metadata__', None)
    for k, v in h.items():
        pat[(re.sub(r'\.\d+\.', '.N.', k), v['dtype'], tuple(v['shape']))] += 1
for (k, dt, sh), m in sorted(pat.items())[:40]: print(m, dt, sh, k)
"
```

**API 调试**：

```bash
curl -s -X POST localhost:18080/api/v1/fit -d '{"hardware":"nvidia/rtx4090"}' | python3 -m json.tool | head -40
```

---

## 7. 提交

**提交和推送在 Mac 上做**（Mac 的仓库是主仓库，远端 `origin` 为私有 GitHub 仓库）；4090 只拉取，避免两边分叉。

```bash
# Mac：改完并在 4090 上验证通过后
cd ~/modelbest/MQDEsys/MQDEsys
git add -A && git status --short            # 确认没有 runs/、权重、*.gguf、内网信息
git commit -m "..." && git push origin main

# 4090：对齐到远端（不要在 4090 上提交）
ssh 4090 'cd /cache/wuzhiping/MQDEsys && git pull --ff-only origin main'
```

4090 访问 GitHub 使用部署密钥 `~/.ssh/id_rsa_wzp`，只在本仓库配置（不改该账户的 `~/.ssh/config`）：

```bash
git config core.sshCommand "ssh -i ~/.ssh/id_rsa_wzp -o IdentitiesOnly=yes"
```

仓库作者统一为 `Zhiping Wu <zhipingwucn@gmail.com>`（Mac 上已在本仓库局部配置）。**提交前确认不含内网地址、同机其他用户的服务信息和未公开的内部检查点名。**

提交信息格式：

```
<type>(<scope>): <简短描述>

- 具体改动 1
- 具体改动 2

Refs: T-Rxx, T-Ryy
```

`type` 取 `feat` / `fix` / `refactor` / `docs` / `test` / `chore`。每个提交应对应 `TASKS.md` 里的任务 ID。

**提交前自查**：

- [ ] `ctest --preset dev` 全绿
- [ ] 前端改动 `npm run build` 通过
- [ ] 触及 L2/L3 路径（见 STRUCTURE.md §1）已获批
- [ ] 没有凭证、权重、大文件入库
- [ ] `TASKS.md` 状态已更新
- [ ] `docs/PROGRESS.md` 已追加本次会话记录

---

## 8. 会话结束前必做

在 `docs/PROGRESS.md` 顶部追加一条，包含：

- **完成**：哪些任务 ID、落在哪些文件
- **校验**：跑了什么命令、结果是什么（测试数、关键输出）
- **下一步**：按 TASKS.md 的顺序，下一个该做什么
- **坑**：踩到的问题与结论，避免下个会话重复踩

没有这条记录，下个 Agent 会从零开始摸索。

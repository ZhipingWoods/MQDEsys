# 测试与本地门禁

`scripts/check_all.sh` 是项目规则的可执行形式，目标是与 CI 等价（仓库里目前还没有 CI 工作流；`docs/site/pages-workflow.proposal.yml` 里有一份仅用于文档构建的提案）。

```bash
scripts/check_all.sh                     # 当前环境可用的全部检查
scripts/check_all.sh --fast              # 跳过 C++ 重新构建和 pytest
SKIP=web,cpp scripts/check_all.sh        # 跳过指定组：deps、format、cpp、python、web
BUILD_DIR=build/mac scripts/check_all.sh # 使用别的 C++ 构建目录（默认 build/dev）
```

| 检查 | 工具 |
|---|---|
| 模块依赖方向 | `scripts/check_deps.py` |
| C++ 格式 | `clang-format --dry-run --Werror`（`.clang-format`） |
| C++ 构建与单元测试 | `cmake --build`、`ctest` |
| C++ 静态分析 | `clang-tidy` |
| Python lint 与类型 | `ruff check python`、`mypy`（要求函数有类型注解） |
| Python 测试 | 在 `python/` 下运行 `pytest` |
| 前端 | `npm run build`（`tsc -b` + `vite build`） |

末尾的汇总表把每项标为 PASS、FAIL 或 SKIP。**SKIP 表示本机没有这个工具，不代表检查通过。** 脚本运行的是 `python3 -m ruff` 和 `python3 -m mypy`，所以它们必须装在 `python3` 所指的 Python 里；脚本不接受 Python 路径参数。

也可以直接运行子集：

```bash
ctest --preset dev -R estimator_                     # 某一个 C++ 模块
build/mac/cpp/mqde_unit_tests --gtest_filter="fit.*"   # 测试二进制在 <build>/cpp/ 下
PYTHONPATH=python python -m pytest python/tests -q   # Python（部分测试在没有 CUDA、vLLM 或 GGUF 夹具时会跳过）
```

测试使用离线夹具（项目规则是联网测试必须显式开启；`AGENTS.md` 里提到的 `MQDE_ONLINE_TESTS` 变量目前没有任何测试引用）。需要真实 llama.cpp 二进制或 GGUF 的测试读取环境变量（例如 `MQDE_TEST_GGUF`、`MQDE_TEST_NGL`），没有就跳过。`python/tests/test_numerics_golden.py` 把 Torch 数值与 C++ 核心对拍；它失败时，先怀疑代码。（设计文档里描述的 `data/golden/` 目录目前并不存在。）

推送之前，`scripts/privacy_scan.sh` 会扫描这次推送将要发布的提交：内网 IP、token 格式、二进制与大文件、作者身份。

## 构建本文档

```bash
python3 -m venv build/venv-docs && build/venv-docs/bin/pip install -r docs/site/requirements.txt
cd docs/site && make html            # 两种语言，警告视为错误
```

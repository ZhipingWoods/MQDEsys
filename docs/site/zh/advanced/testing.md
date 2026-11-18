# 测试与质量门禁

`scripts/check_all.sh` 汇总项目的全部本地检查，覆盖依赖方向、格式、构建、单元测试与前端构建。

```bash
scripts/check_all.sh                     # 当前环境可用的全部检查
scripts/check_all.sh --fast              # 跳过 C++ 重新构建和 pytest
SKIP=web,cpp scripts/check_all.sh        # 跳过指定组：deps、format、cpp、python、web
BUILD_DIR=build/mac scripts/check_all.sh # 使用其他 C++ 构建目录（默认 build/dev）
```

| 检查项 | 工具 |
|---|---|
| 模块依赖方向 | `scripts/check_deps.py` |
| C++ 格式 | `clang-format --dry-run --Werror`（`.clang-format`） |
| C++ 构建与单元测试 | `cmake --build`、`ctest` |
| C++ 静态分析 | `clang-tidy` |
| Python 代码检查与类型 | `ruff check python`、`mypy` |
| Python 测试 | 在 `python/` 目录下运行 `pytest` |
| 前端构建 | `npm run build`（`tsc -b` + `vite build`） |

脚本末尾输出汇总表，每项标记为 PASS、FAIL 或 SKIP。SKIP 表示本机未安装对应工具，不代表检查通过。脚本通过 `python3 -m ruff` 与 `python3 -m mypy` 调用工具，因此二者需安装在 `python3` 对应的环境中。

也可单独运行部分检查：

```bash
ctest --preset dev -R estimator_                      # 单个 C++ 模块
build/mac/cpp/mqde_unit_tests --gtest_filter="fit.*"  # 测试二进制位于 <build>/cpp/
PYTHONPATH=python python -m pytest python/tests -q    # Python 测试
```

测试默认使用离线夹具。联网测试需显式设置 `MQDE_ONLINE_TESTS=1`。依赖真实推理引擎或 GGUF 的测试通过环境变量（如 `MQDE_TEST_GGUF`、`MQDE_TEST_NGL`）启用，未设置时自动跳过。

`python/tests/test_numerics_golden.py` 将 PyTorch 数值实现与 C++ 核心对拍；该测试失败时应优先检查实现，而非修改 golden 数据。

提交前，`scripts/privacy_scan.sh` 会扫描待推送的提交，检查内网地址、凭证格式、二进制文件、大文件与作者信息。

## 构建文档

```bash
python3 -m venv build/venv-docs && build/venv-docs/bin/pip install -r docs/site/requirements.txt
cd docs/site && make html            # 构建中英文版本，警告视为错误
```

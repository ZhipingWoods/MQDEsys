# 开发环境搭建（T-0008）

目标：新人从零到 `scripts/check_all.sh` 全绿。机器分工、同步、起服务见 [DEVELOPMENT.md](../DEVELOPMENT.md)；这里只讲搭环境。

## Linux（conda）

```bash
git clone <repo> && cd MQDEsys
git submodule update --init --recursive          # 3rdparty/* 是子模块，不是内置拷贝
scripts/dev_setup.sh                             # 建 conda 环境 `mqde`、检查 Python 依赖、配置 build/dev、npm ci
conda activate mqde
BUILD_DIR=build/dev scripts/check_all.sh
```

- 工具链清单是 [`docs/proposals/T-0006-ci/environment.ci.yml`](../proposals/T-0006-ci/environment.ci.yml)（GCC 12、CMake、Ninja、C++ 依赖、CPU 版 torch、pytest/ruff/mypy、固定版本的 clang-format）。CI 方案启用后它会搬到仓库根目录，`ENV_FILE=...` 可覆盖路径。
- 必须 GCC 12：系统 g++ 11 编不过 `std::async`；`-Werror` 对 GCC 与 Clang 的行为不同，本地只用 Clang 会漏。
- `scripts/dev_setup.sh --check` 只报告会做什么，不改任何东西；重复运行是更新环境并复用构建目录。
- 带 GPU 的开发机再装 CUDA 版 torch 即可（`pip install torch --index-url https://download.pytorch.org/whl/cu1xx`）；没有 GPU 时 GPU 相关测试自动跳过。

## macOS

环境文件只有 Linux 的包（GCC 12）。C++ 依赖用 brew，其余同上：

```bash
brew install cmake ninja nlohmann-json fmt cli11 spdlog yaml-cpp cpp-httplib googletest pybind11 sqlite openssl curl node
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers safetensors pyarrow numpy pyyaml jsonschema pytest ruff mypy pillow
scripts/dev_setup.sh --no-env
```

macOS 的 GCC/`-Werror` 差异、Metal 插件见 DEVELOPMENT.md。

## 验证

`scripts/check_all.sh` 末尾的表里每一项都应是 PASS；clang-tidy 在工具链的系统头文件解析不了时是 SKIP，不算失败。在干净环境里的实测结果见 CI 方案第 4 节。

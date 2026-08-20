# Windows / RTX 5060 专用运行环境

本文件是根规格第10节的操作说明。项目只使用根目录 `.venv`，不使用系统 Python、Conda base 或其他项目环境。当前专用环境已经在本机创建并由 VS Code 固定选择。

## 已验证机器与固定版本

| 项目 | 要求 | 本机实测 |
|---|---|---|
| 操作系统 | Windows 11 x64 | Windows 11 build 26200 |
| Python | CPython 3.12.x x64 | 3.12.10 |
| GPU | 支持 CUDA 的 NVIDIA GPU；目标 RTX 5060 | RTX 5060 Laptop GPU，7.96 GiB |
| NVIDIA 驱动 | CUDA 13.x 需要 580 或更高 | 610.88 |
| PyTorch | 官方 Windows CUDA wheel；本项目当前验证版本 | 2.12.1+cu132 |
| PyTorch CUDA runtime | 随 wheel 提供 | 13.2 |
| GPU 架构 | wheel 必须含目标架构 | compute capability 12.0 / `sm_120` |
| NumPy / SciPy / PySide6 | 由 lock 固定 | 2.4.6 / 1.17.1 / 6.10.3 |

`nvidia-smi` 顶部显示的 `CUDA Version` 是驱动可支持的最高 CUDA 版本，不代表电脑安装了同版本 CUDA Toolkit。本项目正常开发和运行只需要 NVIDIA 驱动与 PyTorch 官方 `cu132` wheel；wheel 自带所需 CUDA runtime，因此不要另外安装 Toolkit，也不要把 Toolkit 的 DLL 手动复制到 `.venv`。

只有将来编译自定义 CUDA/C++ 扩展时，才额外安装 CUDA Toolkit 13.2、Visual Studio 2022 Build Tools 的“使用 C++ 的桌面开发”和匹配的 Windows SDK。自定义扩展必须另立实施项并通过 `sm_120` 构建与实测，不能成为当前主链路的隐含依赖。

## 第一次创建或重建

在 VS Code 中打开项目根目录后：

1. 打开“终端 → 运行任务”。
2. 运行“环境：创建或更新 RTX 5060 专用环境”。
3. 运行“环境：GPU 完整自检”。
4. 运行默认测试任务“测试：当前规格全部自动测试”。

也可以在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

需要彻底重建时使用 `-Recreate`。脚本只允许删除根目录中精确名为 `.venv` 的环境；它依次建立 Python 3.12 环境、按 `requirements.lock` 的 SHA-256 hash 安装、以 editable 模式安装本项目、检查依赖冲突，并强制执行 CUDA 管线自检。`.venv` 不能提交、复制或改名复用。

## VS Code 已配置内容

- `.vscode/settings.json` 固定解释器为 `.venv\Scripts\python.exe`，自动激活环境并配置 pytest、Ruff 和项目路径。
- `.vscode/project.env` 只保存编码和允许的部署绑定覆盖，不覆盖48 kHz、通道数、shape、几何、时间或算法配置；这些值只来自`config/config.yaml`。
- `pyproject.toml`把根`config/`列为正式Python包，并把`config.yaml`声明为package data；构建wheel后仍携带同一份唯一配置文件，不生成另一套默认值。`tests/test_packaging_contract.py`校验包发现范围与该资源清单。
- `.vscode/tasks.json` 提供环境创建、GPU检查、全部测试、L1服务及 Development Test UI 入口。
- `.vscode/launch.json` 提供相同入口的断点调试配置。
- `.vscode/extensions.json` 推荐 Python、Pylance、Python Environments、Debugpy 与 Ruff；本机均已安装。

当前 Development Test UI 已复用根规格定义的同一个`ApplicationRuntime`接入L1、Ingest/Window、L2定位、Layer 3方向增强音频及Layer 4 MarbleNet基准。Runtime以唯一WindowKey和冻结配置驱动有界L2/L3/L4跨窗口流水：同窗严格L2→L3→L4，稳态跨窗L2(n)/L3(n-1)/L4(n-2)并行；三层队列均latest-wins且只替换未开始旧任务。ResultJoiner和completion/backlog/commit路径均有硬容量限制，丢弃或pre-joiner拒绝仍以有序`error` DecisionRecord+watermark审计；UI只读公开`processing_status`。目标域模型校准和正式`app.main`入口尚未完成，后续仍必须复用同一个`ApplicationRuntime`。

Test UI的L4画面另消费Runtime公开的容量1 `latest_l4_dev_ui`：它只包含真正完成且L2/L3/L4同窗的帧，不影响正式Join/commit顺序。`processing_status`公开L4实际完成/丢弃/跳过、最近1秒完成Hz和显示邮箱深度/容量/覆盖数；有序丢弃/跳过帧不擦除最近有效CNN画面，超过配置的`stale_after_ms`才显示过期。

Runtime启动时先建RecordingStore session，再按`commit→L4→L3→L2`启动worker，最后启动设备pipeline和L1读取。停机时先停设备/L1，再按L2→L3→L4→commit传EOS并drain；超时窗口记录为`CANCELLED/error`，所有worker退出前不得关闭RecordingStore。这些是运行时生命周期门禁，与CUDA环境自检同等必须。

## 启动门禁

下列任一项失败时，不得把运行状态标记为 GPU 正常：

- 当前解释器不是根目录 `.venv\Scripts\python.exe`；
- Python 不是 3.12.x；
- `torch.cuda.is_available()` 为 false；
- GPU 不是预期设备，或 wheel 不含 `sm_120`；
- CUDA STFT 不能得到 complex64 `[7,513,17]`；
- complex64 `torch.linalg.solve` 产生 NaN/Inf；
- 实际MarbleNet批量波形 `[5,7680]` 不能完成16 kHz适配、前处理和GPU模型前向；
- `pip check` 报依赖冲突。

正式运行时如果 CUDA 突然不可用或 OOM，按根规格第10、13节降级并记录；环境自检任务使用 `--require-cuda`，不允许用 CPU fallback 把安装错误掩盖为成功。

当前`scripts/check_runtime_env.py`已验证CUDA设备、17帧STFT、complex64线性求解和实际MarbleNet 160 ms波形前向；L4 CPU/CUDA概率一致性由自动测试单独门禁。

## 历史320 ms CUDA逐窗性能基线

2026-08-18在NVIDIA GeForce RTX 5060 Laptop GPU、PyTorch `2.12.1+cu132`上测得的以下结果使用旧320 ms契约，仅作历史参考，不代表当前160 ms性能：L3单候选avg/P95 `7.49/11.13 ms`，双候选`13.38/17.39 ms`；L4单候选`2.84/3.85 ms`，双候选`3.57/4.61 ms`。160 ms配置必须重新运行`scripts/benchmark_l3_l4.py`后建立新基线。

该基准预生成输入、不计构造开销，并在专用CUDA stream上对每窗显式同步；它衡量的是**warm-cache逐窗stage计算**，不等于端到端延迟、正式有序commit频率或Test UI可见Hz。驱动、PyTorch、模型、算法或配置改变后必须重跑，不得将本次数字当作跨环境承诺。

## 笔记本运行要求

- 性能验收和长时间运行时连接电源，Windows 电源模式设为“最佳性能”。
- 在 Windows“图形设置”中把 VS Code 与 `.venv\Scripts\python.exe` 指定为高性能 NVIDIA GPU，避免混合显卡自动切到核显。
- 关闭会长期占用大量显存的程序；启动日志必须记录总显存和可用显存。
- 任何驱动、PyTorch 或 Python 版本变化后，都重新执行环境任务、GPU自检和全部测试；升级版本时同步更新 `requirements-lock.in`、`requirements.lock`、`requirements-vscode.txt` 与根规格。

## 依赖文件职责

- `pyproject.toml`：项目兼容范围与包定义；Windows GPU 主路径固定 `torch==2.12.1+cu132`。
- `requirements-vscode.txt`：人工维护的顶层精确版本，不单独用于生产安装。
- `requirements-lock.in`：lock 的输入与 CUDA wheel 索引。
- `requirements.lock`：Windows x64 + Python 3.12 的完整传递依赖和分发包 SHA-256；安装必须使用 `--require-hashes`。

更新依赖后，在同一 Windows/Python 3.12 平台重新生成 lock：

```powershell
uv pip compile requirements-lock.in --generate-hashes --emit-index-url --python .\.venv\Scripts\python.exe --output-file requirements.lock --index-strategy unsafe-best-match
```

生成后必须从干净 `.venv` 运行安装脚本并完成 GPU 自检，不能只验证依赖解析成功。

# v1.4.3 项目文档写作资料索引

## 1. 索引用途

本文是完整项目文档的写作底稿，记录 v1.4.3 当前分支的权威来源、运行链路、模块边界、测试证据、历史资料和待核实事项。它用于帮助后续写作定位事实来源，不替代代码、配置或测试。

本次遍历以提交 `d944640` 的 118 个 Git 跟踪文件为输入，覆盖 67 个 Python 文件、项目配置、环境与启动脚本、VS Code 配置、测试、Markdown 文档、校准 JSON、图片、PDF 及既有 DOCX/PDF 输出。忽略的本地运行数据、虚拟环境、缓存、日志和临时文件不属于项目文档的事实来源。

## 2. 权威来源顺序

写作出现冲突时，按以下顺序核实：

1. 当前分支的运行代码与 `config/config.yaml`；
2. `common/data_types.py`、`common/config.py` 等公共契约及相应自动化测试；
3. 根 `README.md` 与 `CHANGELOG.md` 的当前版本说明；
4. 各模块 README；
5. `docs/v1.4.3_existing_docs/` 中的历史说明与研究资料；
6. `output/` 中 v1.3.6 的既有 DOCX/PDF，仅作旧版成品参考。

外部研究报告提供设计依据和实验假设，不可写成 v1.4.3 已实现功能或已达到指标。

## 3. 当前版本定位

- 包名：`micarray-6plus1-voice-direction`。
- 版本：`1.4.3`。
- Python：`>=3.12,<3.13`。
- 当前运行范围：真实 Sipeed 8 通道输入、L1、窗口组装、L2、轻量声源数估计、方向 ID、Development Test UI 和有限轨迹日志。
- 当前排除范围：L3～L6、录音管理、Production UI、模型推理、模拟输入、HTTP 服务、PyTorch、CUDA、ONNX、CountNet 和声纹聚类。
- 旧完整 L1～L6 系统：不可变标签 `v1.3.6`。

## 4. 端到端运行链路

```text
Sipeed MA-USB8 / Windows WDM-KS
  -> LiveSipeedSource / AudioCapture
  -> 8ch PCM16 解码与逻辑通道映射
  -> 7 个物理麦校准，HardwareMix 保留
  -> IngestCoordinator 建立 session / epoch / absolute sample 时间轴
  -> Layer1Imcra 每 20 ms 输出 P1、P2、噪声 PSD
  -> 可选 ImcraWienerPreDenoiser
  -> WindowAssembler 形成 160 ms DecisionWindow
  -> 单一 L2 worker
       -> IncrementalGccPhatSourceCounter 持续估计 0/1/2
       -> ProbabilityGate 使用当前 20 ms P2
       -> Gate OPEN 时 RollingNormMusicScanner
       -> GlobalDirectionTracker 维护方向 ID
  -> L2DevUiSnapshot 原子组合快照
  -> Development Test UI 与有限轨迹日志
```

主入口是 `scripts/launch_dev_test_ui.ps1`，它使用 `.venv-v1.4`，在导入 NumPy/SciPy 前把 OpenBLAS 与 OMP 线程数限制为 1，随后运行 `pythonw -m gui.dev_test_ui.app`。GUI 创建 `ApplicationRuntime` 并控制开始、停止、参数开关及显示刷新。

## 5. 核心时间与数据契约

- 采样率：48,000 Hz。
- 设备通道：8；逻辑顺序为 7 个物理麦加 1 个 HardwareMix。
- 20 ms hop：960 samples。
- 160 ms上下文：7,680 samples，每 20 ms 发布一次窗口。
- 配置中的 `doa_window_samples`：1,920 samples。
- L1 IMCRA：960 点 FFT，0～10 kHz 输出 201 个 50 Hz 频点。
- L2 MUSIC：2～4 kHz，FFT 1024，窗长 960，步长 480，配置上下文 200 ms。
- 声源数估计：2～4 kHz、160 ms 增量 GCC-PHAT，结果限定为 `0/1/2`。
- 时间身份：`session_id`、`stream_epoch`、窗口 ID、绝对 sample 边界与 decision sample；跨 epoch 数据不得拼接。
- 数组契约：公共音频与谱数据要求 finite、`float32`、C-contiguous、只读，并执行严格 shape 校验。

公共数据类型应从 `common/data_types.py` 解释，配置字段与校验应从 `common/config.py` 和 `config/config.yaml` 解释。

## 6. 模块写作地图

| 模块 | 主要职责 | 写作时优先读取 |
| --- | --- | --- |
| `app/` | 运行时编排、自适应 L2 周期、性能窗口、轨迹日志 | `app/runtime.py`、`app/adaptive_rate.py`、`app/track_log.py`、`app/README.md` |
| `common/` | 严格配置、公共 DTO、角度、阵列几何、固定时间常量 | `common/config.py`、`common/data_types.py`、`common/geometry.py`、`common/timing.py` |
| `layer1_input/` | 设备采集、PCM 解码、通道映射、校准、连续性、IMCRA、预降噪、串口灯控 | `layer1_input/README.md`、`capture.py`、`sources.py`、`imcra.py`、`pre_denoise.py` |
| `ingest/` | 把采集块和健康事件统一到唯一时间轴 | `ingest/coordinator.py`、`ingest/fanout.py` |
| `windowing/` | 有界缓存与 160 ms `DecisionWindow` 组装 | `windowing/assembler.py`、`windowing/README.md` |
| `source_counting/` | 增量 GCC-PHAT 突出声源数估计 | `source_counting/counter.py`、`configuration.py`、`README.md` |
| `layer2_source_detection/` | P2 Gate、NormMUSIC、候选筛选、IMM-JPDA 方向跟踪 | `pipeline.py`、`music.py`、`global_tracker.py`、`probability_gate.py`、`README.md` |
| `gui/dev_test_ui/` | L1/L2 控制、原子快照、角度图、轨迹表、性能栏 | `app.py`、`contracts.py`、`srp_panel.py`、`README.md` |
| `scripts/` | 环境安装、环境边界检查、UI 启动、L1 校准工具 | 两个 PowerShell 脚本及两个 Python 脚本 |
| `tests/` | 当前架构的主要自动化证据 | 按下文测试矩阵读取 |

## 7. 关键算法与实现边界

### L1

- `LiveSipeedSource` 获取真实 8 通道 PCM16，映射为 7 个物理麦和 HardwareMix。
- `ChannelCalibrator` 只校准 7 个物理麦，使用增益、极性和整数延迟；HardwareMix 不校准。
- `Layer1Imcra` 每 20 ms 计算逐麦 P1、七麦中位数 P2 与逐麦噪声 PSD。
- P1 以 250～600、600～1600、1600～3400 Hz 三段加权；实现细节和边界权重以 `layer1_input/speech_spectrum.py` 为准。
- `ImcraWienerPreDenoiser` 默认关闭，启用时使用 50% 重叠 sqrt-Hann WOLA；不改变 HardwareMix。

### 声源数估计

- 对 21 个麦克风对计算 2～4 kHz PHAT 互谱和 360°空间图。
- 通过活动能量、主峰、残差峰、角距、逐帧共存和 3 次中 2 次稳定投票输出 `0/1/2`。
- Gate 关闭时仍持续推进；手动关闭时清空状态。
- MUSIC 阶数跟随开启时，预热、故障、计数 0 或 1 映射为 1 阶，计数 2 映射为 2 阶。

### L2

- `ProbabilityGate` 只使用当前 20 ms P2，默认阈值 `0.80`。
- Gate OPEN 后运行 2～4 kHz frequency-normalized MUSIC；方向扫描步长 1°，候选最小间距 50°，最多 3 个候选，有效阶数上限 2。
- `GlobalDirectionTracker` 使用 circular IMM-JPDA，支持 tentative、confirmed、coasting 生命周期、短时失联预测、重复轨迹合并和最多 4 条活动轨迹。
- 自适应调度以 20 ms 为基础输出时钟，过载时减少真实 L2 计算频率并复用最近结果，恢复后逐级回升。

### UI 与运行时状态

- UI 只展示 L1 和 L2，不保留 L3～L6 占位区域。
- L1 显示采用最近 10 个 20 ms 快照的 0.2 秒平均；角度图与 L2 状态读取 latest-only 原子组合快照。
- 性能事件只保留约 1 秒，可随时关闭并清空。
- 音频只在内存保留约 1 秒，不写 WAV；`tmp/l2_track_history.txt`只保存有限 ID、持续时间和稀疏轨迹。

## 8. 配置与环境来源

- `pyproject.toml`：包版本、Python 范围、直接依赖、UI/dev 可选依赖、pytest 与 Ruff 设置。
- `config/config.yaml`：硬件映射、校准、时间常量、L1、预降噪、L2、声源数、运行时和 UI 默认值。
- `requirements-lock.in`、`requirements-vscode.txt`、`requirements.lock`：v1.4 精简环境的直接依赖入口。
- `ENVIRONMENT.md`、`scripts/setup_vscode_env.ps1`、`scripts/check_runtime_env.py`：`.venv-v1.4` 创建与旧模型依赖排除规则。
- `.gitattributes`：音频、模型、研究 PDF/PNG 的 Git LFS 规则及校准 JSON 的 LF 规则。
- `.gitignore`：运行录音、数据集、日志、缓存、虚拟环境和临时文件边界。

## 9. 自动化测试证据地图

| 测试文件 | 主要证明内容 |
| --- | --- |
| `test_l1_v03.py` | 通道映射、校准、IMCRA、P1/P2、频谱平滑、时间轴与断流恢复 |
| `test_l1_pre_denoise.py` | WOLA 连续性、增益更新、epoch 隔离、逐麦 mask 与 HardwareMix 边界 |
| `test_l1_hardware_calibration_tool.py` | 校准刺激确定性、响度与削波、整数延迟和极性相关 |
| `test_ingest_windowing.py` | 不可变 block、唯一时间轴、任意分块一致性、断流与有界缓存 |
| `test_l2_gate_probability.py` | 当前 hop P2 Gate、阈值边界、缺失/错窗拒绝和 Gate 前 MUSIC 跳过 |
| `test_l2_music_tracking.py` | MUSIC 几何、频带、增量协方差、候选、阶数、性能及 IMM-JPDA 生命周期 |
| `test_l2_tracker_rescue_association.py` | 50°关联、跨 0°、ID 恢复、邻近轨迹抑制与合并 |
| `test_source_counting.py` | 0/1/2 估计、共存判定、增量更新、Gap、控制原子性与运行时线程边界 |
| `test_runtime_adaptive_rate.py` | 降频/恢复、复用快照重新定时、TTL、同窗原子快照和关闭尾部行为 |
| `test_dev_test_ui.py` | 控件联锁、显示过期、布局稳定、L1 平均、轨迹表和原子快照读取 |
| `test_track_log.py` | 有界日志、I/O 故障隔离和 epoch 重置 |
| `test_geometry.py` | 阵列物理方向与配置几何身份 |
| `test_runtime_thread_limits.py` | 数值库单线程环境在运行时导入前生效 |

完整项目文档引用“已验证”时，应同时给出测试名称或测试文件，并区分自动化验证、性能基准和真实麦克风实测。

本次遍历使用项目专用 `.venv-v1.4` 执行测试收集检查，共成功收集 160 项测试；该检查证明测试可被发现和导入，不等同于本次重新执行全部测试。

## 10. 资产与历史资料

- `docs/v1.4.3_existing_docs/`：本次整理前的全部 docs 内容。
- 两份 L1 校准 JSON：2026-08-21 与 2026-08-24；当前配置采用后者的校准版本与结果。
- `docs/v1.4.3_existing_docs/references/`：硬件官方资料、MUSIC/追踪/L3 研究报告和空间相关度图。
- `layer1_input/references/`：阵列图片和原理图。
- `tests/fixtures/audio/`：仅允许短小、审阅过、带 manifest 与哈希的回归音频；当前清单应从该目录实时核对。
- `output/documents/` 与 `output/pdf/`：v1.3.6 旧版完整说明成品，用于版式和章节参考。

Git LFS 管理二进制音频、模型资产及归档研究 PDF/PNG。运行录音、完整本地语料、缓存和日志不得进入 Git。

## 11. 写作前必须核实的历史残留

以下内容与当前 v1.4.3 代码存在偏差，写正文时不能直接沿用：

1. `ingest/README.md` 引用当前分支不存在的 `ARCHITECTURE_V0.3_TARGET.md`，并写有“当前代码仍是 v0.2”；实际 ingest 与当前时间轴契约需按代码和测试重写。
2. `.vscode/tasks.json` 与 `.vscode/launch.json` 仍包含旧 `.venv`、GPU/CUDA 自检、`layer1_input.api` 和 `gui.production_ui.app` 启动项；这些模块或依赖不属于当前 v1.4.3 精简运行范围。
3. `gui/dev_test_ui/README.md` 和部分模块标题仍写 `v1.4`，内容大体描述当前精简架构，最终文档应统一标明适用版本为 v1.4.3。
4. 历史研究资料提到 L3、L4、白化、DPD、神经网络、声纹和录音系统；当前分支未实现这些链路。
5. `config/config.yaml` 的 MUSIC 配置同时存在 160/200/240/320 ms 比较字段与当前 200 ms 主上下文；文档应清楚区分直接窗口、滚动协方差上下文和历史比较参数。

## 12. 建议的完整项目文档结构

1. 项目概览与 v1.4.3 范围
2. 硬件、阵列几何与通道映射
3. 环境安装、启动与退出
4. 全局架构与唯一时间轴
5. L1 采集、校准、IMCRA 与预降噪
6. 160 ms 窗口与有界内存
7. 突出声源数估计
8. L2 Gate、NormMUSIC 与方向候选
9. IMM-JPDA 方向 ID 跟踪
10. 运行时调度、自适应降频与故障隔离
11. Development Test UI 操作说明
12. 配置字段参考与推荐默认值
13. 数据结构、输入输出与跨模块契约
14. 自动化测试、性能测试与真实麦克风验证
15. 日志、数据边界、Git LFS 与隐私约束
16. 已知问题、历史迁移与 v1.3.6 恢复方法
17. 术语表、文件索引和参考资料

## 13. 后续写作工作流

每一章先从本索引找到权威代码与测试，再提取配置默认值和明确边界；完成技术正文后与 `CHANGELOG.md` 的最新记录交叉核对。所有性能数字注明硬件、输入、样本规模、统计口径和验证日期。尚未实机复测的结论使用“自动化验证”或“待实机验证”，避免写成正式验收。

# 6+1麦克风阵列 v1.4.3

v1.4.3 是实时方向定位精简版，只保留真实麦克风输入的L1与L2。完整L1～L6、录音管理、模型和示例音频保存在不可变标签`v1.3.6`。

## 文档导航

本文提供v1.4.3的总体架构。各部分的算法原理、输入输出、配置参数、代码位置和使用边界通过以下链接进入详细文档。

| 部分 | 详细文档 | 内容 |
| --- | --- | --- |
| 基本假设与适用范围 | [01 基本假设、模型与适用范围](docs/project_document/01-assumptions-models-and-scope.md) | 远场模型、声源条件、适用场景和使用限制 |
| 阵列与通道 | [02 阵列几何与通道映射](docs/project_document/02-array-geometry-and-channel-mapping.md) | 6+1几何、8通道映射、校准及更换阵列时的修改位置 |
| L1 IMCRA与P Gate | [03 IMCRA、预降噪与P Gate](docs/project_document/03-imcra-pre-denoise-and-probability-gate.md) | 底噪估计、SPP、P1/P2、频段加权、预降噪和0声源过滤 |
| L2方向检测 | [04 声源数估计与NormMUSIC DOA](docs/project_document/04-source-counting-and-normmusic-doa.md) | GCC-PHAT声源数、MUSIC阶数、空间谱、取峰和输出 |
| 方向ID | [05 ID合并、追踪与预测](docs/project_document/05-direction-id-tracking.md) | tentative/confirmed/coasting、JPDA关联、IMM平滑与预测 |
| 局限与不足 | [06 当前局限与待验证问题](docs/project_document/06-limitations-and-open-issues.md) | 物理、算法、硬件、实时性和验收边界 |
| 后续方向 | [07 波束形成与双人音频恢复](docs/project_document/07-future-beamforming-and-two-speaker-reconstruction.md) | 双人同时讲话时的波束形成和原始音频恢复方向 |
| 运行与验证 | [08 Runtime、Test UI与验证](docs/project_document/08-runtime-ui-and-verification.md) | 线程、队列、动态回退、界面操作、性能和测试 |
| 参考资料 | [参考资料索引](docs/project_document/references/README.md) | 论文原文、硬件资料和研究报告入口 |

## 总体架构

```text
Sipeed MA-USB8：48 kHz / 8ch / PCM16 / 960 samples
  -> AudioCapture：有界回调交接、sequence和健康事件
  -> LiveSipeedSource：PCM16解码、设备通道重排
  -> ChannelCalibrator：7物理麦增益/极性/整数延迟
  -> IngestCoordinator：session、epoch和唯一sample时间轴
  -> L1 IMCRA（20 ms）
       -> 逐频噪声PSD/SPP
       -> P1逐麦宽带概率 -> P2七麦中位数
       -> 可选IMCRA-Wiener预降噪（默认关闭）
  -> WindowAssembler：160 ms DecisionWindow，每20 ms发布
  -> 单一L2 worker
       -> 当前20 ms P2 Gate
       -> 持续增量GCC-PHAT突出声源数0/1/2
       -> 2–4 kHz Rolling NormMUSIC（1/2阶）
       -> Circular IMM-JPDA方向ID
  -> 同窗原子L2快照
  -> Development Test UI / 1秒性能统计 / 稀疏轨迹日志
```

## 关键运行契约

| 项目 | 当前值 |
| --- | --- |
| 物理模型 | 二维远场平面波，只估计0..359°方位角 |
| 物理阵列 | 半径4 cm的6外圈+1中心麦 |
| 逻辑通道 | MIC0..MIC5、Center、HardwareMix |
| 决策节拍 | 960 samples / 20 ms / 约50 Hz |
| 公开窗口 | 7,680 samples / 160 ms / 8通道 |
| IMCRA | 960点FFT，0..10 kHz输出，1 s预热 |
| P Gate | 当前20 ms P2，默认门限0.80 |
| 声源数 | 160 ms增量GCC-PHAT，输出0/1/2 |
| MUSIC | 2–4 kHz，初始15帧，连续后19帧/200 ms |
| 方向ID | tentative/confirmed/coasting，最长2 s滑行 |
| 持久化 | 不写WAV；逻辑音频只保留最近1 s |

- 音频只在内存中保留最近1秒，不写WAV、不进入录音管理系统。
- 实时入口在加载NumPy/SciPy前固定OpenBLAS/OMP为单线程，避免小矩阵工作负载建立大型线程池和产生调度抖动；该设置只作用于本项目进程，不修改Windows全局环境。
- `tmp/l2_track_history.txt`只保存ID、持续时间和稀疏轨迹。
- 性能监控只保留最近1秒，显示IMCRA、P、声源数估计、MUSIC、ID、总耗时、排队时间、输出/实算帧率及计数故障率。
- L2默认每20 ms实算一次；过载时按20 ms步长逐级降低完整实算频率，最高200 ms。未实算窗口沿用最近结果并按当前20 ms时间戳持续输出；稳定恢复后逐级回到可容纳当前负载的最小周期。
- Test UI只接收真实麦克风，仅显示L1与L2；左列上下排列L1和L2控制/轨迹表，右上保留正方形360°角度图，底部为横跨窗口的性能栏；L3～L6区域已删除。
- 突出声源数估计默认开启，在Gate关闭时也持续按每20 ms新增的两个STFT帧推进，第二候选的逐帧共存校验不额外执行FFT。Test UI右下角独立控制框可关闭估计或切换MUSIC阶数跟随；跟随关闭时Gate OPEN后的MUSIC固定2阶，开启后把计数`0/1`及预热映射为1阶、把`2`及以上映射为2阶。算法与边界见[`source_counting/README.md`](source_counting/README.md)。
- Test UI从单个L2组合快照读取同窗声源数、Gate、MUSIC及ID；组合DTO强制校验session、epoch、window ID和decision sample完全一致，避免独立latest-only邮箱刷新时把相邻窗口拼在一起。

## 快速开始

首次创建精简环境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_dev_test_ui.ps1
```

VS Code固定使用`.venv-v1.4`，不安装PyTorch、CUDA、ONNX、CountNet或L4～L6依赖。`data/`、`tmp/`、录音、日志、缓存和虚拟环境不提交。旧系统从`v1.3.6`恢复。

当前版本只接受真实麦克风输入。启动后点击“启动采集”，等待约1 s IMCRA预热，再观察P2、Gate、声源数、MUSIC角度和方向ID。完成后点击“停止采集”。

## 版本与历史资料

v1.4.1的完整架构、各层输入输出、关键参数、Test UI使用方法和1小时真实麦克风长测结果见 [历史文档](docs/v1.4.3_existing_docs/V1.4.1_ARCHITECTURE_USAGE_AND_ONE_HOUR_TEST.md)。

旧架构与迁移说明见 [历史迁移文档](docs/v1.4.3_existing_docs/V1.3.6_TO_V1.4_OVERVIEW_AND_USAGE.md)。

# 6+1麦克风阵列 v1.4.3

v1.4.3 是实时方向定位精简版，只保留真实麦克风输入的L1与L2。完整L1～L6、录音管理、模型和示例音频保存在不可变标签`v1.3.6`。

本仓库按公开GitHub项目维护，公开源以`Dwan-123456/micarray-6plus1-voice-direction`为准。`v1.4.3`按用户决定保持现有公开范围；本次发布不新增代码许可证，也不调整仓库内论文PDF资产。

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
| 历史完整音频链 | [07 v1.3.6波束形成与双人音频恢复](docs/project_document/07-future-beamforming-and-two-speaker-reconstruction.md) | v1.3.6已实现的L3–L6、精简原因及选择性恢复边界 |
| 运行与验证 | [08 Runtime、Test UI与验证](docs/project_document/08-runtime-ui-and-verification.md) | 线程、队列、动态回退、界面操作、性能和测试 |
| 参考资料 | [参考资料索引](docs/project_document/references/README.md) | 论文原文、硬件资料和研究报告入口 |

## 总体架构

```text
Sipeed MA-USB8实时输入：48 kHz、8通道、每20 ms一块
  ↓
① 采集并整理音频
  - 接收设备数据，发现丢块或异常
  - 把8个设备通道排成项目规定顺序
  - 校准7个物理麦克风的音量、极性和整数延迟
  - 为连续音频建立统一的session、epoch和sample时间轴
    （AudioCapture → LiveSipeedSource → ChannelCalibrator → IngestCoordinator）
  ↓
② L1：估计环境噪声和“当前是否有明显声音”
  - IMCRA每20 ms更新各麦克风、各频率的噪声和声音概率
  - P1表示每个物理麦把各频率SPP按当前频段权重汇总后的宽带声音概率
  - P2取7个加权后P1的中位数，降低单个异常麦克风的影响
  - 可选预降噪默认关闭
  ↓
③ 组装用于方向判断的音频窗口
  - 每个窗口包含最近160 ms音频
  - 每20 ms生成一个新窗口
    （WindowAssembler / DecisionWindow）
  ↓
④ L2：判断声源数量、方向并维持方向ID
  - P2 Gate先过滤0声源或声音证据不足的窗口
  - GCC-PHAT持续估计当前有0、1或2个突出方向声源
  - NormMUSIC在2–4 kHz判断1或2个声源的角度
  - IMM-JPDA把相邻窗口的方向合并成连续ID，并在短时无观测时预测
  ↓
⑤ 发布和显示结果
  - 同一个窗口的Gate、声源数、角度和ID组成一份完整快照
  - Development Test UI显示电平、概率、方向图和ID状态
  - 底部性能栏显示上一秒耗时、帧率、排队、故障和drop
  - 稀疏轨迹日志只记录ID和少量方向点，不保存音频
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

## 各阶段输入、输出与功能

| 阶段 | 主要输入 | 主要输出 | 功能 |
| --- | --- | --- | --- |
| 采集与解码 | MA-USB8 8通道PCM16 | 逻辑`float32 [N,8]` | 有界交接、编号、通道重排 |
| 校准与时间轴 | 逻辑8通道、校准参数 | `IngestedAudioBlock` | 校准7麦，建立session/epoch/sample边界 |
| L1 IMCRA | 7路物理麦20 ms音频 | PSD、SPP、P1、P2 | 估计底噪和宽带声源证据 |
| Windowing | 连续8通道block | `DecisionWindow [7680,8]` | 形成160 ms上下文，每20 ms发布 |
| P Gate | 当前20 ms P2 | OPEN/CLOSED等状态 | 过滤0声源和宽带证据不足窗口 |
| 声源数 | 160 ms、21对PHAT互谱 | 0/1/2 | 持续估计突出方向数量，决定MUSIC阶数 |
| NormMUSIC | 7麦滚动协方差、1/2阶 | 360°空间谱、方向候选 | 2–4 kHz方位角扫描和取峰 |
| IMM-JPDA | 候选方向、历史轨迹 | `track_id`、滤波角、状态 | ID关联、平滑、预测、coasting和过期 |
| Runtime/UI | 同窗L1/L2快照、性能事件 | Test UI与诊断 | 动态回退、原子发布和实时显示 |

- 音频只在内存中保留最近1秒，不写WAV、不进入录音管理系统。
- 实时入口在加载NumPy/SciPy前固定OpenBLAS/OMP为单线程，避免小矩阵工作负载建立大型线程池和产生调度抖动；该设置只作用于本项目进程，不修改Windows全局环境。
- `tmp/l2_track_history.txt`只保存ID、持续时间和稀疏轨迹。
- 性能监控只保留最近1秒，显示IMCRA、P、声源数估计、MUSIC、ID、总耗时、排队时间、输出/实算帧率及计数故障率。
- L2默认每20 ms实算一次；过载时按20 ms步长逐级降低完整实算频率，最高200 ms。未实算窗口沿用最近结果并按当前20 ms时间戳持续输出；稳定恢复后逐级回到可容纳当前负载的最小周期。
- Test UI只接收真实麦克风，仅显示L1与L2；左列上下排列L1和L2控制/轨迹表，右上保留正方形360°角度图，底部为横跨窗口的性能栏；L3～L6区域已删除。
- 突出声源数估计默认开启，在Gate关闭时也持续按每20 ms新增的两个STFT帧推进，第二候选的逐帧共存校验不额外执行FFT。Test UI右下角独立控制框可关闭估计或切换MUSIC阶数跟随；跟随关闭时Gate OPEN后的MUSIC固定2阶，开启后把计数`0/1`及预热映射为1阶、把`2`及以上映射为2阶。算法与边界见[`source_counting/README.md`](source_counting/README.md)。
- Test UI从Runtime单一L2 worker发布的一个组合快照读取声源数、Gate、MUSIC及ID，避免从多个latest-only邮箱自行拼接相邻窗口。`L2DevUiSnapshot`强制校验声源数快照的session、epoch、window ID和decision sample；Gate、MUSIC与方向对象的同窗关系由上游`Layer2PipelineResult`校验和Runtime顺序组装保证。

## 快速开始

### 最简单的使用方法

下载项目并解压后：

1. 双击根目录的 **`一键安装环境.cmd`**，等待提示安装完成；
2. 连接Sipeed MA-USB8和麦克风阵列；
3. 双击根目录的 **`启动Test UI.cmd`**；
4. 在界面中点击“启动采集”，等待约1 s IMCRA预热；
5. 使用完成后点击“停止采集”。

首次使用建议保持全部默认参数。灯光可手动开关；预降噪默认关闭。界面底部显示上一秒耗时、帧率、L2周期、排队、故障和drop。

### 开发者：从GitHub克隆

在PowerShell中运行：

```powershell
git clone https://github.com/Dwan-123456/micarray-6plus1-voice-direction.git
cd .\micarray-6plus1-voice-direction
git lfs pull
```

仓库默认分支就是当前v1.4.3。`git lfs pull`用于下载文档中的论文和图片。

### 开发者：手动创建运行环境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

脚本会自动创建项目专用Python 3.12环境、安装依赖并执行环境检查。

### 开发者：手动启动Test UI

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_dev_test_ui.ps1
```

启动后可观察P2、Gate、声源数、MUSIC角度和方向ID。声源数、MUSIC阶数跟随、ID追踪和预降噪可在Test UI中调整。

> v1.4.3实机1小时测试结果：系统能够保持实时和稳定，未观察到内存泄漏，进程内存占用约200 MB。

## 版本与历史资料

v1.4.1的完整架构、各层输入输出、关键参数、Test UI使用方法和1小时真实麦克风长测结果见 [历史文档](docs/v1.4.3_existing_docs/V1.4.1_ARCHITECTURE_USAGE_AND_ONE_HOUR_TEST.md)。

旧架构与迁移说明见 [历史迁移文档](docs/v1.4.3_existing_docs/V1.3.6_TO_V1.4_OVERVIEW_AND_USAGE.md)。

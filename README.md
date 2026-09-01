# 6+1麦克风阵列 v1.4.3

v1.4.3 是实时方向定位精简版，只保留真实麦克风输入的 L1 与 L2。完整 L1～L6、录音管理、模型和示例音频保存在不可变标签 `v1.3.6`。

## 项目文档

[进入 v1.4.3 完整项目文档](docs/project_document/README.md)

```text
Sipeed 8通道输入 -> 校准 -> L1 IMCRA（20 ms）
  -> P1逐麦概率 -> P2七麦中位数 -> 可选预降噪
  -> 160 ms窗口（每20 ms更新）
     -> L2 P2 Gate
        -> 持续增量GCC-PHAT突出声源数0/1/2（默认开启，可手动关闭）
        -> 2～4 kHz加权NormMUSIC（Gate OPEN时固定2阶或跟随估计）
        -> IMM-JPDA方向ID -> 时间、track_id、角度、状态
```

- 音频只在内存中保留最近1秒，不写WAV、不进入录音管理系统。
- 实时入口在加载NumPy/SciPy前固定OpenBLAS/OMP为单线程，避免小矩阵工作负载建立大型线程池和产生调度抖动；该设置只作用于本项目进程，不修改Windows全局环境。
- `tmp/l2_track_history.txt`只保存ID、持续时间和稀疏轨迹。
- 性能监控只保留最近1秒，显示IMCRA、P、声源数估计、MUSIC、ID、总耗时、排队时间、输出/实算帧率及计数故障率。
- L2默认每20 ms实算一次；排队或任一处理环节超过20 ms时，自动按40、60、80、100 ms逐级降低实算频率，未实算窗口沿用最近结果并按当前20 ms时间戳持续输出；稳定恢复后逐级回到20 ms。
- Test UI只接收真实麦克风，仅显示L1与L2；左列上下排列L1和L2控制/轨迹表，右上保留正方形360°角度图，底部为横跨窗口的性能栏；L3～L6区域已删除。
- 突出声源数估计默认开启，在Gate关闭时也持续按每20 ms新增的两个STFT帧推进，第二候选的逐帧共存校验不额外执行FFT。Test UI右下角独立控制框可关闭估计或切换MUSIC阶数跟随；跟随关闭时Gate OPEN后的MUSIC固定2阶，开启后把计数`0/1`及预热映射为1阶、把`2`及以上映射为2阶。算法与边界见[`source_counting/README.md`](source_counting/README.md)。
- Test UI从单个L2组合快照读取同窗声源数、Gate、MUSIC及ID；组合DTO强制校验session、epoch、window ID和decision sample完全一致，避免独立latest-only邮箱刷新时把相邻窗口拼在一起。

首次创建精简环境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_dev_test_ui.ps1
```

VS Code固定使用`.venv-v1.4`，不安装PyTorch、CUDA、ONNX、CountNet或L4～L6依赖。`data/`、`tmp/`、录音、日志、缓存和虚拟环境不提交。旧系统从`v1.3.6`恢复。

v1.4.1的完整架构、各层输入输出、关键参数、Test UI使用方法和1小时真实麦克风长测结果见 [历史文档](docs/v1.4.3_existing_docs/V1.4.1_ARCHITECTURE_USAGE_AND_ONE_HOUR_TEST.md)。

旧架构与迁移说明见 [历史迁移文档](docs/v1.4.3_existing_docs/V1.3.6_TO_V1.4_OVERVIEW_AND_USAGE.md)。

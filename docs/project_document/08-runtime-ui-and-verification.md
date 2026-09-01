# 08 Runtime、Development Test UI与验证

本文属于**操作指南 + 技术参考**。它说明v1.4.3如何启动、停止和调度各层，UI控件怎样影响同一L2 worker，以及开发者如何解释性能和故障状态。

## 1. 运行入口

项目要求Python 3.12和专用`.venv-v1.4`。

首次创建或更新环境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
```

脚本会：

1. 创建`.venv-v1.4`；
2. 安装锁定的直接依赖；
3. 以editable模式安装项目；
4. 执行`pip check`；
5. 检查必需模块；
6. 拒绝torch、onnxruntime、safetensors和spectralcluster等旧L4–L6依赖。

启动UI：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_dev_test_ui.ps1
```

启动脚本在导入NumPy/SciPy前设置：

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
```

项目大量运算是7×7小矩阵，多线程BLAS的线程池调度开销和抖动通常大于并行收益。

## 2. 主要线程和回调

| 执行单元 | 名称/来源 | 职责 |
| --- | --- | --- |
| PortAudio callback | sounddevice内部 | 复制原始PCM块、编号、入有界handoff |
| capture worker | `l1-capture` | 解码、校准、Coordinator、IMCRA、预降噪、窗口组装 |
| L2 worker | `l2-music-id` | Gate、声源数、MUSIC、ID和原子快照 |
| track log worker | `l2-track-log` | 低优先级稀疏轨迹TXT写入 |
| CDC reader | SerialDevice daemon | 串口读取、16×16 hotmap兼容解析和LED控制 |
| Qt main thread | QApplication | 消费latest-only邮箱并绘制UI |

声源数没有独立worker，它与Gate、MUSIC和ID在同一个L2线程顺序执行，保证同窗因果关系。

## 3. 有界队列和邮箱

| 容器 | 容量 | 满时行为 |
| --- | ---: | --- |
| capture numbered handoff | 500块，约10 s | 丢最旧块，记录并合并handoff健康事件 |
| L2窗口队列 | 100窗，约2 s | 丢最旧窗，增加`processing_drops` |
| `latest_l1` | 1 | 替换旧快照 |
| `latest_l2_dev_ui` | 1 | 替换旧快照 |
| `latest_source_count` | 1 | 兼容邮箱，UI不使用它拼接 |
| `latest_windows` | 1 | 替换旧窗口 |
| track log submit | 1 | 替换旧日志任务 |

latest-only邮箱保证慢UI不会反压实时算法。丢失中间显示帧是允许行为；算法主队列丢窗则进入诊断和时间连续性处理。

## 4. `ApplicationRuntime.start()`

启动时：

1. 拒绝在已有worker仍活动时重复启动；
2. 清空stop/input-done事件和上轮错误；
3. 重建Coordinator、Assembler、IMCRA、预降噪、L2、计数器和几何；
4. 清空1 s逻辑音频环、性能事件、L2队列和旧计数邮箱；
5. 启动真实输入Pipeline和轨迹日志线程；
6. 尝试关闭阵列灯光；
7. 启动capture和L2线程。

每次采集拥有新的`session_id`，旧session的UI快照和计数会通过时间戳/启动时间检查被隐藏。

## 5. Capture worker

每次读取一个`DecodedAudio`：

```text
pipeline.read
  -> IngestCoordinator.ingest
  -> Layer1Imcra.process
  -> attach aligned ImcraHopSnapshot
  -> ImcraWienerPreDenoiser.process
  -> choose raw or denoised block
  -> WindowAssembler.add
  -> publish L1 snapshot and DecisionWindow
```

底层源耗尽时设置`input_exhausted`。任意未处理异常写入`last_error`，finally阶段只在预降噪链确实拥有未发布尾块时flush，并始终通知L2输入结束。

## 6. L2 worker顺序

对队列中的每个窗口：

1. 计算排队时间；
2. 在control lock下原子读取当前Gate、MUSIC、ID和声源数控制revision；
3. 应用待处理ID reset；
4. 从窗口最后两个IMCRA hop构造P2对象；
5. 评估当前Gate和连续OPEN计数；
6. 根据控制变化和自适应周期决定实算或复用；
7. 实算时先运行声源数，再决定MUSIC阶数；
8. 运行MUSIC/ID或只推进预测；
9. 记录阶段耗时和自适应负载；
10. 发布一个同窗`L2DevUiSnapshot`。

控制revision进入force条件。用户改变Gate阈值、候选阈值、ID开关或计数状态后，下一窗强制实算，避免复用与新控制不一致的结果。

## 7. 自适应计算周期

基础输出时钟固定20 ms。完整L2实算周期可为：

```text
20, 40, 60, ..., 200 ms
```

### 7.1 升档

- queue wait超过20 ms，或
- 任一计算阶段耗时超过当前实算周期，或
- L2发生处理故障。

每次只增加20 ms。阶段包括IMCRA、预降噪、概率、声源数、MUSIC、ID和L2总耗时。

### 7.2 稳定

一个24 ms MUSIC计算在20 ms档属于过载，会升到40 ms；到40 ms后它处于预算内，不会继续级联到200 ms。

### 7.3 恢复

负载连续健康5 s后每次降低20 ms。恢复门槛为下一档留约8 ms余量，20 ms基础档使用12 ms健康门槛。

## 8. 跳过窗怎样处理

不进行完整实算的窗口仍按当前20 ms身份发布：

- Gate重新使用当前P2评估；
- 声源数沿用上一稳定结果并重新绑定当前身份；
- Gate开启且阶数兼容时，MUSIC只推进滚动协方差，不做EVD/取峰；
- 方向输出改为未观测预测/coasting；
- TTL按当前绝对sample过滤；
- 标记`reused_output=True`和当前`processing_period_ms`。

复用不能伪装成新观测，不能增加ID确认次数，也不能越过两秒TTL。

## 9. 故障降级

L2窗口处理异常时：

1. 记录`L2 processing <type>: <message>`；
2. 自适应周期升一档；
3. 若上一快照属于同stream、Gate/阶数兼容，则安全复用；
4. 否则发布Gate `UNAVAILABLE`、无MUSIC/方向的fault快照；
5. 记录每秒fault计数。

声源数异常单独进入`source_count_last_error`和计数故障率。阶数跟随开启时安全回退1阶；固定阶数模式继续2阶；它不升级为主链错误。

轨迹日志I/O异常只进入`track_log_last_error`。

## 10. 停止和关闭

`stop()`：

1. 设置stop事件；
2. 停止输入Pipeline；
3. 在配置的10 s优雅关闭期限内join capture/L2；
4. 停止轨迹日志；
5. 若worker仍存活，保留线程引用并报告timeout。

保留超时线程引用可以阻止用户在旧L2仍运行时重新启动第二套worker。`close()`随后尝试关闭串口。

## 11. UI布局

### L1区域

- 8通道RMS电平，显示下限-60 dBFS；
- 7个P1和阵列P2；
- IMCRA状态；
- 预降噪开关和平均增益；
- 启动/停止采集；
- LED开关。

UI每20 ms消费L1邮箱，但L1文字/电平每200 ms更新一次，使用最近10个hop。dBFS先还原线性功率平均，再转回dB。

### L2左下

- Gate门限滑块；
- Candidate threshold滑块；
- ID Tracking开关；
- 3行方向表：ID、观测角、输出角、score、状态、新建、观测。

### 右侧

- 稳定正方形360°NormMUSIC图；
- formal ID使用稳定颜色，tentative为中性灰色；
- Gate关闭时隐藏旧MUSIC曲线，只绘制仍活跃的方向状态；
- 右下overlay放声源数和阶数控制，不参与角度图尺寸协商。

### 底部

- 性能监控开关；
- 上一秒各阶段平均耗时；
- 计数/MUSIC/ID/总耗时；
- 输出/实算fps；
- 当前L2周期、排队、fault和drop。

## 12. 声源数与阶数控制

两个开关存在原子约束：

- 关闭声源数会同时关闭阶数跟随并回到固定2阶；
- 未启用声源数时不能开启阶数跟随；
- 重新开启声源数不会自动重新开启跟随；
- 单独切换阶数跟随不会重置连续计数状态；
- 关闭/重开声源数会改变control revision并重置计数器。

UI显示计数前还检查：enabled、非None、属于当前采集、晚于控制变更、年龄不超过500 ms。

## 13. 性能快照

性能事件窗口固定1 s、最多512项。公开字段包括：

- `pre_denoise_ms`；
- `imcra_ms`；
- `probability_ms`；
- `source_count_ms/fps/faults`；
- `music_ms`；
- `id_tracking_ms`；
- `total_ms`；
- `queue_wait_ms`；
- output/compute/reuse fps；
- faults；
- adaptive period/stride/reason。

关闭性能监控会立即清空窗口，不保存历史。

## 14. Processing status

`processing_status`用于程序化诊断：

- L2队列深度和容量；
- L1/L2/source-count worker存活；
- 完成计数和drop；
- 平均耗时与L2 Hz；
- 自适应周期和原因；
- 主错误、计数错误、日志错误；
- 当前控制状态和MUSIC阶数。

这些是进程内当前状态，不是持久化监控系统。

## 15. 一次标准实机操作

1. 连接MA-USB8、阵列和可选CDC串口；
2. 确认Windows WDM-KS提供48 kHz、8通道；
3. 启动Development Test UI；
4. 点击“启动采集”；
5. 等待IMCRA从WARMING进入READY；
6. 检查8通道电平，无固定满量程、断通道或明显异常增益；
7. 观察P2和Gate；
8. 在已知方向播放宽带语音，检查角度方向；
9. 检查计数、MUSIC阶数和ID状态；
10. 观察底栏是否接近50 fps、queue/drop/fault是否正常；
11. 点击“停止采集”；
12. 再关闭窗口。

## 16. 常见故障

### 找不到麦克风

- 检查设备名包含`MicArray`；
- 检查Host API为`Windows WDM-KS`；
- 检查设备支持至少8输入通道；
- 检查采样率实际协商为48 kHz。

### 一直WARMING

- 确认连续收到20 ms块；
- 检查sequence/timestamp是否反复重置epoch；
- 检查校准身份是否变化；
- 查看`last_error`和capture健康事件。

### Gate一直关闭

- 检查P1/P2；
- 检查输入通道和校准；
- 确认声源包含250–3400 Hz宽带证据；
- 临时调整门限只用于诊断，不能替代标定。

### Gate打开但没有ID

- 前10个连续OPEN hop禁止出生；
- MUSIC可能没有超过score/prominence；
- ID开关可能关闭；
- 新候选可能位于已有轨迹50°范围；
- adaptive周期会延长10次真实观测确认时间。

### 周期升到较高档

- 查看`adaptive_last_overload_reason`；
- 区分queue wait和具体stage；
- 检查系统其他进程；
- 预降噪、Gate开启率和轨迹数会改变负载；
- 40 ms稳定运行可以是正常回退结果。

## 17. 自动验证

全量测试：

```powershell
.\.venv-v1.4\Scripts\python.exe -m pytest -q
```

静态检查：

```powershell
.\.venv-v1.4\Scripts\python.exe -m ruff check .
git diff --check
```

MUSIC CPU基准：

```powershell
.\.venv-v1.4\Scripts\python.exe .\benchmark_l2.py --frames 200
```

自动测试不访问真实麦克风；实机验收需要单独记录硬件、房间、声源位置、配置、运行时长和真值。

## 18. 数据边界

不得提交：

- `.venv/`和`.venv-v1.4/`；
- `data/`运行录音和本地语料；
- `tmp/`轨迹日志和临时输出；
- cache、coverage和build；
- 密钥、token和本地代理配置。

需要进入自动测试的短音频必须放入`tests/fixtures/audio/`，更新manifest、SHA-256和消费测试，并由Git LFS管理。

## 19. 权威文件

| 内容 | 文件 |
| --- | --- |
| Runtime | `app/runtime.py` |
| 自适应周期 | `app/adaptive_rate.py` |
| 轨迹日志 | `app/track_log.py` |
| UI | `gui/dev_test_ui/app.py` |
| UI DTO | `gui/dev_test_ui/contracts.py` |
| 角度图/表 | `gui/dev_test_ui/srp_panel.py` |
| 启动脚本 | `scripts/launch_dev_test_ui.ps1` |
| 环境脚本 | `scripts/setup_vscode_env.ps1` |
| Runtime测试 | `tests/test_runtime_adaptive_rate.py` |
| UI测试 | `tests/test_dev_test_ui.py` |
| 线程限制测试 | `tests/test_runtime_thread_limits.py` |

[上一章：未来波束形成](07-future-beamforming-and-two-speaker-reconstruction.md) · [返回项目总导航](../../README.md)

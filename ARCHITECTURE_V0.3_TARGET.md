# 6+1 麦克风阵列 v0.3 目标架构与迁移契约

> 本文件保留为项目1.0.1历史迁移契约。1.1.1的MUSIC、永久方向ID与跨层公开ID架构见[`ARCHITECTURE_V1.1_TARGET.md`](ARCHITECTURE_V1.1_TARGET.md)；当前实现状态以1.1.1架构、配置和代码为准。

状态：**主链迁移、L1可切换预降噪、L3三档对照模式及Test UI连续试听sidecar已完成；相关自动化门禁通过；实机动态参数标定待执行**  
适用范围：Layer 1～Layer 5、Application Runtime、Development Test UI、RecordingStore 与数据资产。
覆盖规则：本文件是当前主链的权威契约；与 v0.2 执行规格及各目录旧说明冲突时，以本文件为准。当前`config/config.yaml`和Python主链已经实现v0.3契约；仍留在源码树但不再由主链导入的v0.2模块只供迁移比对，不得据此改变当前架构或完成度。

## 1. 目标架构图

```text
Sipeed R6+1 + MA-USB8：48 kHz native HostAudio [N,8]
    Host顺序：CH0..CH5=MIC0..MIC5，CH6=HardwareMix，CH7=Center
                            ↓
Layer 1：解码、校准、逻辑重排、连续性guard
    LogicalAudio [N,8]：MIC0..MIC5、Center、HardwareMix
       ├── PhysicalAudio [N,7]：参与几何、SRP与波束形成
       └── HardwareMix [N]：只作预留接口、显示、录制与后续实验
                            ↓
IngestCoordinator：建立唯一session/epoch/绝对sample时间轴
                            ↓ 原始IngestedAudioBlock
    IMCRA：对7个物理麦逐20 ms更新，PSD覆盖80～8000 Hz
       ├── 每麦宽频噪声PSD、SPP及噪声特征
       └── 从500～4000 Hz聚合array_source_probability_20ms ∈ [0,1]
                            ↓
    可切换IMCRA预降噪：每麦独立Wiener增益
       ├── 40 ms sqrt-Hann分析窗、20 ms hop、50% WOLA连续重建
       ├── 80～8000 Hz抑噪，最低增益-18 dB；其余频率与HardwareMix直通
       └── OFF输出原音频；ON等待对应降噪hop完成后替换LogicalAudio前7路
                            ↓ 选定的IngestedAudioBlock → WindowAssembler
       ├── RecordingStore异步订阅原始/选定音频
       └── DecisionWindow：320 ms，每20 ms生成一次
                            ↓
    WindowWorkItem：不可变DecisionWindow + 窗口边界配置快照
    唯一WindowKey = (session_id, stream_epoch, window_id, decision_sample)
                            ↓ 有界L2 latest-wins队列
Layer 2 worker：Probability Gate → SRP-PHAT → 私有ID/圆周卡尔曼
       ├── Gate关闭：显式L2完成，L3/L5标记SKIPPED
       └── Gate开启：Raw SpatialResponse + Smoothed CandidateDirection[0..2]
                            ↓ 有界L3 latest-wins队列
Layer 3 worker：LogicalAudio [15360,8] + 候选角度
       ├── optimized（正式默认）：复用滚动STFT、IMCRA统计、协方差及有界静态查表
       ├── ds_baseline（对照）：固定7麦Delay-and-Sum
       ├── loaded_mvdr_baseline（对照）：全频diagonal-loaded MVDR、失败频点回退DAS
       └── 每候选输出theta_deg + EnhancedAudio 48 kHz mono [15360]
                            ↓ 有界L5 latest-wins队列
Layer 5 worker：增强音频副本 + 对齐IMCRA概率 → 响度补偿 → 降采样 → CNN
       ├── COMPLETED → latest_l5_dev_ui容量1：完整同窗L2/L3/L5帧，仅供即时显示
       └── 正式L5StageResult：每方向Voice / Non-Voice概率
                            ↓
ResultJoiner：按WindowKey合并L2/L3/L5终态，按唯一时间轴有序提交
       ├── Development Test UI：正式有序审计快照；L5即时帧不改变提交顺序
       │      └── 仅限试听sidecar：Center Mic参考 + L2私有ID元数据 + 有界连续音频缓存
       └── RecordingStore：有序DecisionRecord → ResultWatermark；不保存L2内部ID

跨窗口流水并行：稳态时 L2(n) || L3(n-1) || L5(n-2)
同一窗口仍严格依赖：L2(n) → L3(n) → L5(n)，不能把缺失角度或音频伪造成并行输入
每层过载都只替换本层尚未开始的最旧任务；已被worker取走的计算不取消、不抢占
被替换或接纳前拒绝的窗口仍按window_id生成error DecisionRecord并推进ResultWatermark
缓存全部有界：CPU ComputeCache按L2/L3/L5分区，L3 GPU prepared context固定小容量，完成即回收
```

## 2. 板级映射、观察面与算法坐标

### 2.1 资料边界

Sipeed 官方资料只定义阵列板的麦克风与 I²S 数据线：`MIC_D0`承载 MIC0/1，`MIC_D1`承载 MIC2/3，`MIC_D2`承载 MIC4/5，`MIC_D3`承载中央麦。当前项目统一按设备实际摆放方式，从**灯面正上方向下观察**；灯面图中的物理编号必须直接对应算法几何，不再换算到另一观察面。

MA-USB8 桥接后的 Host 通道语义来自本项目随附桥接资料：CH0～CH5 对应 MIC0～MIC5，CH6 是阵列内部合成音频，CH7 是中央麦。两类资料必须分开记录，禁止把 USB 通道顺序冒充成 Sipeed 原始阵列板的官方 USB 定义。

### 2.2 唯一算法观察面

- 阵列实际使用时**灯面朝上**。
- 从灯面正上方向下看，中央麦为原点。
- 从Center指向MIC0的方向定义为物理`+x`，即`theta_deg=0°`。
- 从灯面观察时角度逆时针增加，范围为`[0,360)`，外圈依次经过MIC5、MIC4、MIC3、MIC2、MIC1。

### 2.3 外圈坐标

设外圈半径为 `r`（当前硬件标称值仍由唯一配置给出），目标几何固定为：

| 逻辑通道 | Host通道 | 角度 | 坐标 `(x,y)` |
|---|---:|---:|---|
| MIC0 | CH0 | 0° | `(r, 0)` |
| MIC1 | CH1 | 300° | `(r/2, -sqrt(3)r/2)` |
| MIC2 | CH2 | 240° | `(-r/2, -sqrt(3)r/2)` |
| MIC3 | CH3 | 180° | `(-r, 0)` |
| MIC4 | CH4 | 120° | `(-r/2, +sqrt(3)r/2)` |
| MIC5 | CH5 | 60° | `(r/2, +sqrt(3)r/2)` |
| Center | CH7 | 不适用 | `(0, 0)` |
| HardwareMix | CH6 | 无物理坐标 | 不得加入几何 |

因此从灯面观察，算法正角方向的编号次序是**MIC0→MIC5→MIC4→MIC3→MIC2→MIC1**。L1的逻辑通道标签、`common.geometry`、L2 steering delay、L3 steering vector、UI角度和录音manifest必须引用同一个几何版本，不能各自维护坐标副本。

## 3. Layer 1 目标契约

### 3.1 八通道音频

L1仍读取native Host顺序 `[CH0,CH1,CH2,CH3,CH4,CH5,CH6,CH7]`，对外逻辑顺序改为：

```text
[MIC0, MIC1, MIC2, MIC3, MIC4, MIC5, Center, HardwareMix]
 = [CH0, CH1, CH2, CH3, CH4, CH5, CH7, CH6]
```

正式音频为 `float32 [N,8]`、C-contiguous、只读且finite。前7路是物理阵列，最后1路是硬件合成总声音。需要物理阵列的算法必须显式读取前7路；不得用HardwareMix替代Center，也不得把HardwareMix加入麦对。

### 3.2 IMCRA

IMCRA从L2迁入L1，在校准后的7路物理音频上按20 ms hop连续更新。噪声PSD、SPP及噪声特征的目标有效频带为**80～8000 Hz**：覆盖人声主要能量、低音和辅音清晰度，同时排除直流/低频结构振动及8 kHz以上对当前L3/L5价值较低的频率。每个hop至少发布：

正式实现采用[Israel Cohen 2003年IMCRA论文](https://doi.org/10.1109/TSA.2003.811544)的双迭代时频平滑、最小值跟踪、先验语音缺失概率、后验SPP和递归噪声估计流程，算法版本固定为`cohen_imcra_2003_l1_v1`；不得以能量阈值或平滑阶跃函数替代论文概率模型。

每个物理麦独立执行同一递归。第一轮按式(14)～(16)得到`S`和`S_min`，再按式(18)、(21)形成粗语音缺失指示；第二轮按式(26)～(28)得到条件平滑谱和条件最小值。式(28)中的`gamma_min_tilde`使用瞬时功率除以`B_min * S_min_tilde`，`zeta_tilde`的分子必须是第一轮平滑谱`S`，不得误用第二轮条件平滑谱。随后按式(29)得到`q_hat`，按式(7)得到后验SPP，按式(32)～(33)更新Decision-Directed先验SNR和`G_H1`，最后按式(10)～(12)更新噪声PSD。

版本参数固定为`w=1、alpha_s=0.9、U=8、V=15、D=120、B_min=1.66、gamma_0=4.6、gamma_1=3、zeta_0=1.67、alpha=0.92、alpha_d=0.85、beta=1.47`。这些数值来自论文表I；表I原始实验为16 kHz，本项目保持其递归参数但使用48 kHz、960-sample hop、2048点FFT和80～8000 Hz发布频带，属于明确记录的前端适配。若重新标定`B_min`或改变窗、hop、FFT，必须更新算法版本和回归基线。

- 与音频相同的session、epoch和绝对sample区间；
- 7路80～8000 Hz噪声PSD及其338点频率轴；
- 第一轮/第二轮平滑谱与局部最小量、`q_hat`、后验SPP、先验/后验SNR等可复现状态摘要；
- 每麦噪声特征；
- 从500～4000 Hz证据子带聚合、finite且位于`[0,1]`的`array_source_probability_20ms`。

IMCRA内部采用“宽频估计、窄频门控”：80～8000 Hz结果供L1诊断、L3增强、L5数据分析和存储使用；500～4000 Hz参与L2 Gate概率聚合。L2的SRP定位频带独立固定为2000～4000 Hz。概率必须由版本化适配器定义并记录算法版本；不能由L2重新估计，也不能把未校准的能量启发式分数冒充该概率。HardwareMix可保留独立诊断，但不参与7物理麦的阵列概率聚合。

本版本的概率适配器先对每个物理麦在500～4000 Hz内对SPP作算术平均，得到`source_probability_per_mic[7]`，再取7麦中位数作为`array_source_probability_20ms`。这是阵列级工程聚合，不是论文单通道IMCRA公式本身；修改聚合方法同样必须升级算法版本。

连续性断裂或epoch变化时，L1清空IMCRA状态并重新预热；预热不完整时概率状态为`WARMING_UP`，L2不得将缺失概率当作0。

### 3.3 可切换IMCRA预降噪

`imcra_wiener_wola_v1`使用每个物理麦自己的`prior_snr`与`SPP`形成频率增益：`G_W=xi/(1+xi)`，`G=SPP+(1-SPP)G_W`。增益限制在`[-18,0] dB`并在频率、时间上平滑。算法使用两个20 ms hop组成40 ms平方根Hann窗，以20 ms步长执行50%重叠相加；每次仍发布960点连续音频，固定墙钟延迟为一个hop。

IMCRA必须先读取原始音频并完成当前hop状态更新。预降噪开关关闭时，下游接收未经修改的LogicalAudio；开关开启时，Runtime等待对应降噪hop完成，将`IngestedAudioBlock.samples`前7路替换为降噪结果后再交给WindowAssembler、L2及后续层。HardwareMix和`native_samples`不修改。IMCRA预热、无效或缺失时增益为1，不得将首段输入当作噪声强制抑制。开关可在Development Test UI的L1区域动态切换并持久化。

## 4. Runtime与窗口对齐

ApplicationRuntime同时管理8通道音频、对应IMCRA hop及预降噪开关。开启预降噪后先等待一个20 ms hop完成WOLA，再发布具有原绝对sample边界的替换音频；因此算法时间轴不平移，但墙钟处理延迟增加20 ms。`DecisionWindow`仍为320 ms、每20 ms发布一次；每个窗口的末尾40 ms恰好覆盖两个完整20 ms IMCRA结果：

```text
gate_probability_40ms =
    (array_source_probability_20ms[t-1]
     + array_source_probability_20ms[t]) / 2
```

只有两个概率都属于同一session/epoch、连续sample区间且状态有效时才可发布正式40 ms概率。缺失、跨epoch或预热时，Gate状态为`WARMING_UP/UNAVAILABLE`，不得拼接旧概率或伪造正式结果。

### 4.1 唯一窗口身份与冻结配置

每个正式窗口进入算法前只构造一次不可变`WindowWorkItem`。其唯一身份为：

```text
WindowKey = (session_id, stream_epoch, window_id, decision_sample)
```

四个字段必须在L2、L3、L5、Joiner、UI及RecordingStore中逐项相等；`context/doa`的sample边界继续由同一个`DecisionWindow`携带，禁止后续阶段重新推导。入队时同时冻结该窗口使用的Gate、SRP、方向平滑、L3模式、L5阈值、几何版本和config hash。运行中修改参数只影响之后接纳的窗口，不能让一个窗口的不同阶段读到不同revision。

### 4.2 跨窗口流水并行与逐层latest-wins

L2、L3、L5各有一个保持状态顺序的独立worker和有界等待队列。它们不是对同一窗口无依赖地同时计算，而是在稳态处理不同窗口：

```text
时间片 k：  L2(window n)  ||  L3(window n-1)  ||  L5(window n-2)

同一窗口：  L2(n)  ───────→  L3(n)  ─────────→  L5(n)
```

L2的Gate、私有ID和卡尔曼必须按worker真正取走的窗口顺序推进；L3滚动STFT/噪声协方差也保持单worker顺序；L5对已完成的L3结果推理。下游队列只承接已满足依赖的不可变StageResult。

三个阶段的等待队列都实施latest-wins：新任务到达且本层队列已满时，只移除队列中**尚未被本层worker取走**的最旧任务，再放入新任务。已经开始的SRP、BF或CNN不被强制取消、不做线程抢占。终态记录按丢弃发生层级保留已有成果：

- L2队列溢出：L2/L3/L5均为`DROPPED`；
- L3队列溢出：保留已完成L2，L3/L5为`DROPPED`；
- L5队列溢出：保留已完成L2/L3，L5为`DROPPED`。

这些窗口全部继续进入ResultJoiner，最终按全局`window_id`生成一条`status=error`的DecisionRecord和同窗ResultWatermark。阶段失败、超时、Gate跳过和停机取消分别记录为`FAILED/TIMED_OUT/SKIPPED/CANCELLED`，不能静默消失。`SKIPPED`本身不是计算失败；任一阶段为`FAILED/TIMED_OUT/DROPPED/CANCELLED`时整窗DecisionRecord必须为`error`。算法成功产生完整输出但使用了声明的回退路径时为`degraded`；不得用`degraded`包装计算失败或半截结果。

CUDA运行时L3与L5各自使用独立CUDA stream；只同步本阶段stream，不执行阻塞整个设备的全局同步。这使GPU资源允许时L3和上一窗口的L5可重叠执行；实际加速比例仍以实机负载与显存门禁为准。

### 4.3 有界复用与内存约束

- Runtime CPU `ComputeCache`按L2/L3/L5分区，并同时受窗口数、单分区字节数和全局`compute_cache_max_bytes`硬限制；缓存只是优化，StageResult才是正确性来源。
- L2复用窗、频率轴、频带mask、统一RFFT/PHAT前端和有界steering；逐窗GCC等大中间量用后即释放。
- L3相邻320 ms窗口复用29/33个STFT帧，只计算新4帧；若latest-wins造成1～15个20 ms hop跳跃，仍按绝对sample复用`31-2N`个重叠内部帧，并只搬运新增N个IMCRA hop、滚动更新对应协方差贡献。达到320 ms无重叠、非hop对齐、时间倒退或身份/配置变化才完整重建。频率轴、Hann窗、频带mask、steering及空间可分度查询有界复用；GPU上的`PreparedL3Context`不得进入CPU ComputeCache，固定只保留最近2个窗口。
- L5只复用不可变模型artifact和适配器；候选音频、响度补偿数组和推理batch按窗口释放。
- session、epoch、时间连续性、配置、几何、设备或算法版本改变时，对应缓存必须整体失效。已提交窗口由Joiner完成后立即从跨层缓存退休，禁止被重新发布或复活。

默认队列容量为L2=10000、L3=10000、L5=10000、completion=8，最大Joiner在途窗口30003（覆盖三层等待队列及每层1个正在执行的窗口），全局CPU计算缓存64 MiB。所有可配置值来自唯一`config/config.yaml`并由schema限制。按50窗/s计算，单层10000窗约对应200秒等待工作，端到端累计等待可能更长。大队列不会预分配窗口内存，但窗口数上限对应的原始8通道float32音频下限已约13.7 GiB，IMCRA快照、StageResult和对象开销还会继续增加；ResultJoiner的独立字节硬限可能更早拒绝接纳。它只提供有界过载缓冲，不代表持续算力不足已经解决，也不得保留CUDA张量或形成无界缓存。

ResultJoiner在窗口数和估算字节两个维度均有硬上限。若在正式注册前已无法接纳新窗口，Runtime不保留其320 ms音频，只在有界范围表中压缩保留session/epoch、首尾window/sample及原因。commit遇到该window_id时，再逐条展开为轻量`error` DecisionRecord（L2/L3/L5均`DROPPED`）和ResultWatermark。这是**pre-joiner容量拒绝审计**，不是无记录丢弃，也不能为它保留波形或空间谱。

Join后提交路径也全部有界：主completion队列容量为配置的`completion_queue_windows`，后备completion backlog使用相同硬上限；两者都满时停止新窗口接纳，已注册结果由Joiner的有界在途集合保留待重试。commit乱序等待表在`max_inflight_windows`达到软阈值时拒绝新接纳，硬容量固定派生为`2*max_inflight_windows + 2*completion_queue_windows`。任一层拥塞都只会丢弃/拒绝可测窗口，不得将无界内存作为缓冲。

### 4.4 有序Join、原子Recording水位与启停

`ResultJoiner`允许三个阶段乱序完成，但只在同一`WindowKey`的所有阶段均达到明确终态后生成一个`JoinedWindowResult`。commit worker再按全局单调`window_id`提交`DecisionRecord`。Runtime优先调用RecordingStore的`append_result_with_watermark(record, watermark)`，以一条有界队列命令原子接纳结果与同窗水位；队列溢出时两者均不接纳，且生产者水位不假前进。epoch切换前必须封闭旧epoch所有已接纳窗口，迟到旧结果不能污染新epoch。

commit先完成正式`DecisionRecord`与水位提交，再生成Development Test UI的有序审计投影。除此之外，L5 worker只在正式`L5StageResult=COMPLETED`后、提交Joiner前，把包含同一`WindowKey`下完整L2空间响应、L3预览和L5结果的`DevUiFrame`写入容量1的`latest_l5_dev_ui`。该latest-only邮箱满时只覆盖旧显示帧并累计覆盖次数；它不发布失败帧，不改变StageResult、DecisionRecord、RecordingStore或有序commit，也不得成为正式结果来源。UI过时、窗口关闭或绘制异常只记录为诊断，不得阻断录音水位、缓存回收或后续窗口。

录音结果schema升级为`decision_record_v3`，保留原有候选、空间谱、增强音频和CNN字段，新增`stage_statuses`、`stage_timings_ms`、`stage_queue_wait_ms`与`terminal_reason`。`DROPPED/CANCELLED`同时写入水位审计，但同一窗口在结果JSONL只保留一条正式终态记录。

启动顺序固定为：重置时间轴/缓存/队列 → 启动RecordingStore session → 按`commit → L5 → L3 → L2`启动消费worker → 启动音频设备pipeline → 启动L1读入线程。因此首个音频块不会赶在下游消费者之前到达。任一启动步失败时按反向停止pipeline、唤醒并join已启动线程，再封闭失败的录音session，不得留下半启动worker。

正常停机采用drain而不是清空队列：先停止设备输入，L1刷出已延迟的预降噪hop并声明输入结束 → L2消费完并向L3传EOS → L3消费完并向L5传EOS → L5消费完并向completion传EOS → commit耗尽主队列、后备backlog和Joiner待重试结果，完成最终原子结果/水位写入 → 关闭RecordingStore与释放设备。全过程受`graceful_shutdown_timeout_seconds`限制；超时时将所有已注册未完成窗口终止为`CANCELLED/error`并唤醒worker。只有所有处理worker真正退出后才允许关闭RecordingStore；若仍有worker存活，必须报错并保留资源所有权，不能假装停机成功。

## 5. Layer 2 目标契约

L2删除Noise Estimation职责及NE后端选择，只保留：

```text
AlignedDecisionWindow + gate_probability_40ms
    → Probability Gate
    → SRP-PHAT
    → Raw SpatialResponse + Raw CandidateDirection[0..2]
    → Internal DirectionSmoother
    → Raw SpatialResponse + Smoothed CandidateDirection[0..2]
```

Gate概率来自L1的**500～4000 Hz**聚合；SRP-PHAT定位独立固定为**2000～4000 Hz**。Gate初始阈值为`0.60`，通常判断为`probability >= threshold`。阈值属于运行时可变参数：Test UI提供`0.00～1.00`滑动条，修改后在下一个完整窗口边界生效并产生新的`config_revision`；数值原子保存到Test UI设置并在下次启动恢复，但不改写`config/config.yaml`。当ID追踪和卡尔曼同时开启且至少一个正式ID仍存活时，任何有效P值均被强制判为Gate开启；最后一个正式ID失效后的下一窗口恢复阈值判断。概率预热、缺失或无效仍安全阻断。Gate关闭时跳过SRP并输出空候选；Gate开启时，SRP仅读取40 ms、7路物理音频的2000～4000 Hz频点。

L2正式候选上限固定为2，不提供运行时调节。通过threshold、prominence和45°圆周NMS的峰按normalized score降序、同分角度升序排列，只构造前两个正式`CandidateDirection`；45°专指同一窗口内两个声源点之间的最小角差，不限制单个ID的移动或稳定性。圆周距离按`min(|a-b|,360-|a-b|)`计算，359°与2°相距3°。方向平滑后的两个公开声源点也必须保持至少45°，否则本窗口回退原始SRP候选。不能先输出更多候选再由Runtime或L3静默截断。完整360°`SpatialResponse`仍保留，诊断记录限制前有效峰数量及是否触发候选上限。L3只允许接收0、1或2个正式候选。

现有SRP-PHAT扫描、Robust-Z归一化、圆周峰值、NMS及候选排序语义可以保留，但必须用第2节新几何重做无镜像、无固定旋转的方向测试。

SRP候选筛选后固定按“私有ID追踪→按ID圆周卡尔曼”编排。ID追踪先按圆周距离全局关联候选；卡尔曼以分配后的私有ID为状态键，避免rank切换导致串轨。两者均默认关闭并支持运行时持久化切换，但卡尔曼依赖ID追踪；ID关闭时卡尔曼不可开启，关闭ID会同步关闭卡尔曼。公共类型仍是原`CandidateDirection`，不增加ID字段；数量、rank、时间字段及分数继承原始SRP候选。L2内部结果可另带与候选逐项对齐的私有ID、预测、正式和首次分配标志；唯一允许的跨模块消费者是本机Runtime的Test UI诊断投影，包括右上SRP候选身份显示和左下试听sidecar。

新建的临时ID从首次建立起按绝对sample观察2秒，并在这首个2秒内累计归并至少5次自然Gate窗口候选且至少1个同窗角度被L5识别为人声后，才转为正式ID。临时阶段的人声证据只参与转正确认；正式化时获得3秒语音租约。SRP角度匹配、卡尔曼校正、预测和强制Gate均只改变位置或观测状态，不续命。L5把同窗人声与非人声分类证据送回L2，L2按同窗历史与20°圆周距离自动匹配；正式化后唯一匹配到仍存活正式ID的人声结果，才把截止sample滑动到该人声点之后3秒。无历史、歧义、错流或过期一律拒绝且不得复活。低P强制窗口可以更新已有正式ID位置，预测后首个重匹配使用2倍测量可信度，但不能创建或晋升新ID。租约到期后在Gate判断与关联前删除ID及卡尔曼状态，最后一个ID删除后恢复按P门控。公共候选结构不变，预测不得突破Top-3和45°圆周间距。

内部ID不得进入公共`CandidateDirection`、L3、L5模型、DecisionRecord、RecordingStore或数据集标签。唯一例外是本机Runtime可将L2内部结果中的逐候选私有元数据投影给Test UI：右上SRP面板以灰色表示临时点、红/绿交替表示正式ID，并以大/小点区分当前观测与卡尔曼预测；左下试听sidecar只从正式ID开始缓存已有L3音频。production不启用这些诊断旁路，私有元数据不得反向影响任何正式算法或持久化结果。L5反馈仍不携带ID；L2保留有界同窗角度-ID历史完成自动匹配。概率无效或无`SpatialResponse`时公共候选为空；租约仍有效且有当前响应时才可预测。寿命严格按48 kHz绝对decision sample推进，L2/L5丢窗不改变3秒定义。session/epoch变化立即清空状态、历史和反馈邮箱。

`SpatialResponse`始终保留原始360°扫描。平滑后不得重新执行threshold、prominence、NMS、Top-3或排序；如果后验角发生冲突，较低rank候选本窗口回退原始角，不得随机扰动。详细迁移和测试计划见[`L2_INTERNAL_DIRECTION_SMOOTHER_PLAN.md`](L2_INTERNAL_DIRECTION_SMOOTHER_PLAN.md)。

## 6. Layer 3 目标契约

L3公共输入为同一320 ms的8通道48 kHz音频及L2平滑后的候选角度；内部波束形成只读取前7个物理通道。L3不接收内部ID、不做第二次方向滤波，并保持候选数量和rank。每个候选公共输出固定为：

```text
theta_deg
enhanced_audio: float32 [15360]  # 48 kHz mono
时间身份字段：session/epoch/window/decision_sample
算法与降级诊断
```

L3方向增强的目标有效频带为**80～8000 Hz**。内部仍可使用STFT、DAS、MVDR及频带融合，但复数STFT不再暴露为跨层主契约。删除`SpectrogramFeature [33,169]`及其FeatureExtractor主链路；L3播放器只能处理副本，不能改变交给L5或RecordingStore的正式波形。

### 6.1 全局空间可分度p表

双候选BF使用预计算空间可分度`p`选择逐频点分离策略。这里的`p`严格表示两个steering vector的归一化空间相关度，即前述公式中的`rho`；它不是IMCRA的语音存在概率`SPP(k,t)`。二者不得复用字段名、数组或更新时间轴。

表属于项目全局公共资源，不归属L1～L5任何一层：

```text
spatial_separability/
├── spatial_separability_p.npy  # 仅保存float32 p值
└── p_table.py                  # 公共只读访问、轴定义及上下文校验
```

表的固定形状为`[169,60,360]`，各轴依次为：

- 169个BF STFT频点：48 kHz、FFT 1024下的bin 2～170，即93.75～7968.75 Hz；
- `theta_A mod 60°`：利用6个环形麦克风的旋转对称性，同时保留绝对朝向影响；
- A到B的有符号圆周角差：整数`-180°..179°`。

公共访问方法如下，任何层都必须通过该包访问，禁止直接打开`.npy`、修改映射内容或在各层维护副本：

```python
from spatial_separability import (
    P_FREQUENCIES_HZ,
    load_p_table,
    lookup_p,
    validate_p_table_context,
)

# 可选：启动或建立处理器时验证当前STFT和阵列与静态表一致。
validate_p_table_context(
    sample_rate=48_000,
    n_fft=1_024,
    frequency_min_hz=80.0,
    frequency_max_hz=8_000.0,
    geometry=geometry,
)

# 双候选的常规访问；返回只读float32 [169]，与P_FREQUENCIES_HZ逐项对应。
p_f = lookup_p(theta_a_deg, theta_b_deg)

# 仅在确需遍历整表时使用；返回只读float32 [169,60,360]内存映射。
p_table = load_p_table()
```

`lookup_p`先把两个有限角度规范化到`0°..359°`并量化到最近1°，随后对A/B作对称规范化，因此交换候选顺序必须得到逐bit相同的`p_f`。L3有两个候选时必须把查得的`p_f`写入完整513-bin控制向量的bin 2～170，据此选择Dual LCMV、soft-null loaded MVDR或loaded MVDR；禁止在L3运行时重新根据steering vector计算空间相关度，双候选权重求解器缺少查表结果时必须拒绝执行。0个候选不运行BF，1个候选继续使用单约束loaded MVDR，不查询也不计算双源可分度。IMCRA的SPP、噪声PSD和噪声协方差仍按窗口动态进入权重求解，不能被静态`p`表替代。

表只适用于`r6plus1_led_face_mic0_posx_ccw_54321_v2`、4 cm半径、343 m/s声速、48 kHz/FFT 1024及80～8000 Hz BF配置。上下文不匹配必须明确拒绝，不能静默使用错误索引。修改麦克风坐标、角度定义、声速、FFT或BF频带后，必须同步升级`P_TABLE_VERSION`并运行：

```powershell
.\.venv\Scripts\python.exe scripts\generate_spatial_separability_p_table.py
```

重新生成的`.npy`仍只能包含`p`数组；频率轴、角度轴、几何版本和校验规则由公共模块管理。

### 6.2 Test UI可切换三档L3模式

Development Test UI提供`优化算法 / DS基线 / Loaded MVDR基线`三档循环切换按键，启动采集前和采集运行中均可操作。默认始终为`optimized`，production入口不主动切换；运行中修改以L3开始处理某个新窗口时读取到的模式为准，不中断采集，不重算已经完成的窗口。

- `optimized`：保留本节定义的IMCRA噪声统计、全局`p`查表、Dual LCMV、soft-null loaded MVDR、loaded MVDR及逐频点DAS降级。
- `ds_baseline`：使用同一320 ms窗口、同一前7个物理麦、同一候选角、同一STFT和steering vector，仅执行7通道Delay-and-Sum；不读取IMCRA、不查询`p`表、不应用自适应噪声权重。输出算法标识固定为`ds_baseline`。
- `loaded_mvdr_baseline`：以同窗IMCRA噪声协方差对全部有效频点执行diagonal-loaded MVDR；数值不安全频点回退DAS。旧固定30°与五频段模式已经删除。

被选模式产生的正式L3音频继续进入L5、正式320 ms预览和Test UI试听侧路，因此三种模式比较的是完整下游输入。切换后仅后续窗口采用新模式；界面必须停止当前播放、清空旧预览，并重置Test UI专用ID音频缓存。试听缓存还必须按输出模式自检，禁止把不同模式的hop追加到同一条ID音轨。该切换不改变`config.yaml`中的正式默认后端，也不删除或替换优化算法。

Test UI试听sidecar在正式结果提交后消费现成候选与L3预览。存在L2私有ID时以其为首选关联键；ID换号只在3秒等待期内、20°以内且只有一个可续接旧轨时合并，同时出现的两个近角ID不得合并。每条音轨按48 kHz绝对decision sample拼接20 ms位置：可从当前320 ms预览恢复的跳窗补回真实音频，更老的缺口补等时静音，不压缩时间；连续边界交叉淡化。界面首行固定提供预降噪前LogicalAudio第7路Center Mic原音参考，其余方向轨累计至少2秒才显示，并按缓存时长降序排列。Gate暂时`UNAVAILABLE/WARMING_UP`以及同一session内的epoch连续性恢复只关闭当前方向段，不得删除已缓存文件或界面行；旧epoch轨道转为`ENDED`归档，新epoch重复的L2私有ID必须分配会话内唯一的Test UI试听ID，避免覆盖历史音频。播放时对稳定快照作一次目标`-28 dBFS`、最大`18 dB`的试听归一化；该处理不得改写L3正式波形、L5输入或录音。缓存按10秒分段、最多保留3段和8条已结束方向轨；新session、L3模式切换或关闭UI时才按职责重置或删除。

项目尚未定义或实现L2公共方向不确定度输出，也未实现由不确定度驱动的L3动态波束宽度。

## 7. Layer 5 目标契约

L5对每个候选直接接收L3的48 kHz、320 ms单声道增强音频及其平滑角度，同时接收与context中16个20 ms区间严格对齐的L1 IMCRA `array_source_probability_20ms`。L5不使用L2的40 ms平均概率。它先创建仅供CNN使用的音频副本并执行`imcra_probability_rms_v1`响度补偿，之后模型适配器内部降采样到16 kHz，因此保留的目标有效频带为**80～8000 Hz**（实际最高有效频率受抗混叠滤波和16 kHz Nyquist限制）。随后CNN输出`[0,1]` Voice / Non-Voice概率。L5不接收内部ID或`[33,169]`特征，不修改角度，不做跨窗口ID聚合，也不反馈改变L2 Gate或L3音频。

响度补偿将320 ms副本切为16个960-sample片段，目标RMS为`-23.0 dBFS`。概率`p<=0.30`时增益为0，`p>=0.80`时使用完整RMS补偿，中间线性加权；片段高于目标RMS时不放大。新增增益以`-3 dBFS`为峰值上限，相邻片段在20 ms内进行dB域线性过渡并再次执行逐样本峰值保护。算法只限制新增增益，不衰减原始输入：若输入本身已超过`-3 dBFS`，应用0 dB并保留原峰值。概率缺失/未预热和低于`-100 dBFS`的静音片段均为0 dB；NaN/Inf拒绝进入正式CNN。L3原始输出、Test UI试听和RecordingStore增强音频均保持不变。

实现状态：L5已使用独立`Layer5AudioSegment`接收finite、只读`float32[15360]`波形及16个对齐概率槽；L5包不导入L3或GUI预览DTO。ApplicationRuntime从同一DecisionWindow按`context_start_sample + i*960`建立概率槽，非ready或缺失hop写为`None`。补偿诊断逐片段记录概率、补偿前后RMS/峰值及请求/限制/应用增益，逐窗口记录最大/平均增益、补偿片段数、峰值保护次数和算法版本。

L2 Gate已开启且空间响应有效、但没有候选峰时，L3直接返回`COMPLETED`的空`Layer3Output`，不创建320 ms STFT/协方差prepared context；L5仍对空batch执行公共接口并返回`COMPLETED`空结果。该窗口的三阶段都是成功终态，正式记录中的增强音频和Voice方向为空，不得伪装成`SKIPPED`或`error`。

## 8. Development Test UI

- 左上：显示MIC0～MIC5、Center、HardwareMix共8路电平；显示IMCRA预热、每麦噪声摘要及20 ms/40 ms声源概率。
- 右上：删除NE后端选择；增加L2 Gate阈值滑动条，默认0.60，显示当前值、配置值、revision、40 ms概率及开/关状态；360°圆环显示原始`SpatialResponse`，候选点显示平滑角，因此允许点不严格位于峰顶。
- 左下：显示/试听正式平滑候选对应的48 kHz增强音频，并可实时循环切换`optimized / ds_baseline / loaded_mvdr_baseline`。连续试听首行是预降噪前Center Mic原音参考；方向轨使用L2私有ID元数据优先关联，累计至少2秒后显示，缺失后等待3秒再结束，按绝对sample补洞并维护有界磁盘缓存。它不得滤波/改写角度、生成预测方向、触发额外L3波束形成或进入正式记录；`[33,169]`频谱不再是公共契约。
- 右下：显示L5逐方向CNN概率及Voice判断；L5阈值滑条与L2 Gate滑条必须明确分开。它优先消费容量1的`latest_l5_dev_ui`完成帧，以免前序窗口的有序commit等待压低可见刷新率；该帧本身仍是一个完整同窗快照，不能把新L5结果拼到其他窗口的L2/L3数据上。
- L2/L3及正式终态诊断消费Joiner有序提交的ApplicationRuntime快照。L5即时显示是唯一例外且只属于UI side channel；正式结果、录音和watermark仍只认Joiner/commit。后续有序`DROPPED/SKIPPED`帧不得立即清除上一份有效CNN画面；只有超过`dev_test_ui.stale_after_ms`仍未收到新的L5完成帧时才显示`STALE`。
- 全局状态栏只通过Runtime公开只读`processing_status`读取L2/L3/L5/completion队列深度与容量、worker存活、在途窗口、缓存字节、完成数和错误数；L5另显示`l5_actual_completed`、`l5_dropped`、`l5_skipped`、最近1秒`l5_actual_hz`以及显示邮箱深度/容量/覆盖数。UI不得再次访问`_processing_windows`等私有队列。该状态只用于诊断，不参与调度。

## 9. 录音与数据管理

RecordingStore异步保存同一时间轴上的：

- native Host 8ch（原始通道顺序）；
- logical 8ch（MIC0～MIC5、Center、HardwareMix）；
- 可选physical 7ch兼容资产，但manifest必须注明它是派生视图；
- 每20 ms IMCRA的80～8000 Hz特征与7路噪声PSD/SPP，以及由500～4000 Hz聚合的阵列概率和预热/错误状态；
- 每40 ms Gate聚合概率、阈值、配置revision及Gate结果；
- 原始SRP空间谱、平滑候选方向、L3增强音频和L5结果；不保存L2内部ID。

所有sidecar必须使用绝对sample区间关联，不能按文件到达时间猜测。算法结果只能由ResultJoiner按`WindowKey`合并并有序提交；RecordingStore不得直接订阅某个stage的未合并结果。manifest必须记录Host映射、logical映射、几何版本、灯面朝上观察规则、方向平滑器版本、完整配置与config hash。由于ID不持久化，离线复现平滑角必须从同一epoch起点顺序重放全部窗口。写盘失败仍不得反压实时采集。

当前RecordingStore按以下有界事务边界实施：

- 唯一配置schema严格要求`chunk_seconds`、`audio_queue_seconds`、`result_queue_capacity`、`retention_days`和`max_storage_gb`大于0，`min_free_storage_gb`非负且严格小于`max_storage_gb`；`result_queue_capacity`默认为256且硬上限为256。非法容量或存储预算在启动前拒绝，不能退化为零容量队列或无限制存储。`config/config.yaml`作为`config`包的package data进入wheel，分发包不维护第二份业务默认值。
- Runtime对每个有序窗口调用`append_result_with_watermark`；结果和水位以单条命令一起进入结果队列，队列溢出时记录gap且两者都不接纳。
- 结果按录制chunk范围`start_sample < decision_sample <= end_sample`聚合。音频chunk已封闭且writer watermark到达其尾端后，立即写出JSONL/NPZ/sidecar并从内存释放该chunk结果；不等到整个session结束才统一累积。若水位停滞，待保留结果仍受硬数量上限保护。
- `event`模式同时使用按2秒sample裁剪的音频环和结果pre-roll；未触发事件的旧结果按当前epoch最新decision sample持续裁剪，不保留无界增强波形。相同epoch内，新触发的pre-roll起点不晚于当前事件post-roll终点时合并到同一事件段；manifest的`event_triggers`每个合并段只保留一条有界审计，包含首/末`window_id`、首/末decision sample、事件起止sample和`trigger_count`。跨epoch或不重叠触发建立新段。容量扫描只在新段前执行，合并触发不重复扫描；若触发发生在旧post-roll结束之后但其pre-roll仍相交，则从2秒音频环补写间隙，保证事件资产连续。`off`及未录制的`manual`窗口在复制大数组前直接丢弃。
- Hotmap按CDC sequence去重后直接流式写入`hotmaps.jsonl.partial`，不在session内累积16×16矩阵列表；停机时flush/fsync后原子改名并写入hash与count。
- 普通chunk的WAV、NPY、noise NPZ与IMCRA NPZ作为同一批资产提交：首次final改名前先持久化`chunk_asset_commit_<stem>.json`。崩溃恢复时，若manifest已以匹配hash完整索引全批资产则只清理journal；否则整批partial、已改名但未完整索引的final及journal统一进入quarantine。未正常封存的open session manifest另行校验已索引资产后恢复为可审计的incomplete状态。
- 一旦对应音频区间已写入，320 ms增强波形就立即写入WAV `.partial`并从结果内存释放。session封存时先写`enhanced_asset_commit.json`事务journal，再改名WAV、原子写manifest、最后删除journal。崩溃恢复时，manifest已完整索引的资产视为已提交；其他journal中的partial或已改名但未进manifest的WAV全部进入quarantine，不冒充正式资产。

## 10. 必须新增或调整的测试

- 官方MIC/I²S关系、Host通道、logical通道三层映射分别测试。
- 从灯面观察：MIC0=0°，逆时针依次为MIC5、MIC4、MIC3、MIC2、MIC1。
- 新7麦坐标、21麦对、0/60/120/180/240/300°及全角度无镜像测试。
- `[N,8]`音频契约、HardwareMix保留且不进入SRP/MVDR测试。
- Cohen IMCRA每20 ms更新、表I参数锁定、两轮平滑/最小值状态、式(28)分子来源、式(29)/(7)概率边界、80～8000 Hz的338点PSD覆盖、500～4000 Hz概率聚合、连续性重置、finite/只读及预热状态测试。
- 两个连续20 ms概率算术平均得到40 ms Gate概率；跨epoch、缺帧和预热拒绝测试。
- Gate默认0.60、包含等于门限、UI动态revision及关闭时不运行SRP测试。
- 单次、迭代及迭代回退路径均最多输出3个候选；超过上限时排序、诊断和跨层拒绝测试。
- L2内部圆周卡尔曼覆盖0°边界、静止降抖、移动跟随、双候选关联、漏检/跳窗、Gate阻断、epoch重置和异常回退。
- 真实候选平滑前后rank、时间身份和两个score逐项相等，仅`theta_deg`允许变化；成熟ID预测候选从当前响应取score，最终仍受Top-3与45°间距约束；公共DTO、L3/L5和正式记录均不得出现内部ID。
- Runtime移除Test UI预测方向L3旁路；可选试听sidecar只消费正式L3已完成预览及其对齐的私有ID元数据。测试覆盖Center Mic原音参考、2秒显示门槛、3秒coasting、ID唯一换号续接/近角双ID隔离、跳窗等时补洞、模式切换清缓存及关闭删除；L3/L5只继承一次平滑后的候选角。
- L2不再实例化Noise Estimation；L3仅输出音频；L5内部降采样测试。
- RecordingStore的8ch、IMCRA、Gate、方向和结果sample级对齐与恢复测试。
- `WindowKey`和窗口配置快照不可变测试；任一阶段身份不一致、跨revision混用必须拒绝。
- L2/L3/L5跨窗口同时在运行、同窗依赖不越级、每阶段单worker状态顺序和输出确定性测试。
- 分阶段队列、在途窗口、CPU缓存字节、L3 GPU prepared context与静态LRU全部达到硬上限时仍不无界增长；缓存命中/未命中结果逐元素一致。
- 阶段乱序完成、Gate跳过、阶段失败、入口丢窗、epoch切换和停机drain均生成明确终态；Joiner提交顺序、DecisionRecord顺序与Recording水位严格递增。
- L2/L3/L5三个latest-wins队列分别验证“只替换未开始最旧任务”，已开始worker不取消；每一层的丢弃都产生对应阶段`DROPPED`终态、一条有序`error` DecisionRecord及同窗watermark。
- pre-joiner容量拒绝不保留波形，范围审计结构、completion主队列/backlog、Joiner在途字节/窗口和commit乱序表全部在压力下不超过硬上限。
- RecordingStore验证result+watermark原子接纳、队列溢出不假推水位、逐chunk写出/释放、event pre-roll硬上限、同epoch触发合并审计/跨epoch分段/容量扫描节流/环形缓存间隙补写、hotmap流写入及增强音频partial+journal崩溃恢复。
- 录音chunk/音频队列/结果队列为0或负数、保留期/最大容量非法、`min_free_storage_gb >= max_storage_gb`均在配置加载时失败；wheel包清单包含唯一`config/config.yaml`。
- 任一`FAILED/TIMED_OUT/DROPPED/CANCELLED`阶段必须使DecisionRecord为`error`；完整成功但使用明确回退算法才为`degraded`。
- Test UI只读取公开`processing_status`，并能显示每层队列、worker、缓存、完成与错误诊断；缺少该API时仅显示telemetry unavailable，不探测私有队列。
- L5即时UI邮箱固定容量1且latest-only；只发布`COMPLETED`的完整同窗帧，覆盖不改变正式结果。有序`DROPPED/SKIPPED`帧保留最近有效CNN画面直到`stale_after_ms`，并验证全部L5完成/丢弃/跳过/实际Hz/邮箱覆盖诊断。
- Gate开启但候选为空时，验证L3不调用prepare、L3/L5均`COMPLETED`、L5收到空batch，正式增强音频与Voice结果为空。

## 11. 迁移边界

代码迁移已完成：L1增加可切换的每麦IMCRA-Wiener预降噪；L2已删除Noise Estimation，使用L1对齐的两个20 ms概率执行Probability Gate，并在SRP候选后执行可选私有ID追踪与圆周卡尔曼；Runtime/UI预测方向旁路已经移除，本地试听ID/cache sidecar仅消费正式L3预览及其对齐的私有ID元数据，L3/L5/录音链路直接继承无ID的公共候选。试听sidecar已实现Center Mic原音参考、绝对时间轴连续拼接、2秒显示门槛、3秒等待、唯一ID换号续接、模式隔离、有界临时缓存和关闭清理。Runtime采用唯一`WindowKey`、冻结配置、L2/L3/L5逐层有界latest-wins、有界分区缓存和有序ResultJoiner，使稳态下L2(n)、L3(n-1)、L5(n-2)跨窗口并行；所有丢弃/拒绝均有DecisionRecord+watermark审计，停机以EOS drain或超时CANCELLED收束。L5已增加容量1的完成帧显示邮箱，不绕过正式Join/commit；零候选窗口已跳过L3重计算并以空L5成功终态收束。RecordingStore已使用原子result+watermark、逐chunk结果释放、有界event pre-roll、hotmap流写入及增强音频partial+journal恢复边界。预降噪、Runtime、配置、UI、时间轴和窗口接口的相关自动化门禁必须随本次流水迁移全部通过；静止、移动、交叉、混响和双声源实机数据上的参数标定仍属于部署验收，不得以模拟测试替代。

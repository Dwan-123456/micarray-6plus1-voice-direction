# 6+1 麦克风阵列二维人声方向识别系统
## Codex 项目执行规格 / Execution Specification

版本：v0.2 历史正文；当前主链采用v0.3迁移契约  
目标设备：U9 级 CPU + NVIDIA RTX 5060 GPU  
状态：本文件正文保留v0.2历史细节；当前v0.3主链迁移已经落地，权威跨层契约见[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)。自动化测试数量与结果以当前仓库完整`pytest`报告为准；方向平滑实机参数标定、硬件校准、目标域L5校准和最终production入口仍未完成。

当前执行优先级固定为：`ARCHITECTURE_V0.3_TARGET.md` > 本文件未被覆盖的正文 > `config/config.yaml` > 根目录`ENVIRONMENT.md` > 各层README > 当前新项目代码。`legacy_reference_only/`和已从主链删除但仍保留的旧模块文件不得作为完成度证据。

## v0.3当前架构速览

```text
Sipeed R6+1 + MA-USB8 native HostAudio [N,8]
    CH0..CH5=MIC0..MIC5，CH6=HardwareMix，CH7=Center
        ↓
Layer 1：解码、校准、逻辑重排、连续性guard
    【已完成】LogicalAudio [N,8] = MIC0..MIC5、Center、HardwareMix
        ├── PhysicalAudio [N,7]：几何/SRP/MVDR唯一阵列输入
        ├── HardwareMix [N]：预留接口、显示与录制，不进入阵列几何
        ├──【已完成】Cohen 2003 IMCRA：7物理麦每20 ms更新80～8000 Hz噪声PSD/SPP与特征
        │       └── 从500～4000 Hz聚合声源概率
        └──【已完成】可选IMCRA Wiener预降噪（默认关）：40 ms sqrt-Hann、20 ms WOLA、最低增益-18 dB
                └── 开启时只替换7路物理麦后送入WindowAssembler；HardwareMix与native_samples直通
    硬件通道、极性、固定延迟与真实角度校准仍未完成
        ↓
【已完成】IngestCoordinator + WindowAssembler：唯一session/epoch/sample时间轴与320 ms窗口
        ├── RecordingStore：异步保存8ch音频与IMCRA sidecar
        └── 每20 ms产生DecisionWindow [15360,8]，携带16个对齐IMCRA hop
                    ↓
    【已完成】WindowWorkItem：冻结窗口配置，唯一WindowKey=(session_id, stream_epoch, window_id, decision_sample)
                    ↓
        末尾40 ms的两个IMCRA概率取算术平均
                    ↓
Layer 2 1.1：
    【已完成】500～4000 Hz Probability Gate（mean_2x20ms_v1，默认0.60，可动态调整）
        ├──关闭：跳过SRP、候选为空
        └──开启：7物理麦在2000～4000 Hz执行SRP-PHAT 360°扫描 → Robust-Z+Sigmoid → 45°圆周NMS → 原始Top-3
                    ↓
    【L2 1.1】可选confidence_id_tracker_v2私有ID追踪（内部最多4轨，公开最多3角度；V1后端可回退）
        ├──逐候选私有元数据：ID、预测/观测、临时/正式、首次分配
                    ↓
    【L2 1.1】可选damped_circular_kalman_v2阻尼圆周滤波（默认关，依赖ID追踪；V1后端可回退）
        ├──Q/R倍率初始1.00，可在Test UI运行时调整并持久化
        ├──真实候选保留rank/时间/分数，仅theta_deg替换为后验平滑角
        ├──首个2秒累计自然Gate匹配≥5次且L5同窗人声≥1次后正式化；默认租约3秒，仅后续唯一匹配的L5人声可续命
        ├──正式ID存活时可在有效低概率窗口保持Gate开启；预热/缺失/无效概率仍安全阻断
        ├──L5只反馈流身份、时间、角度与人声结论，经容量256有界队列送回L2；不传公开ID
        └──真实候选优先，预测仅补足Top-3并继续满足任意两点45°圆周间距
    ID不进入公共CandidateDirection、L3/L5、录音或数据集；仅投影到本机Test UI诊断界面，不参与正式算法
    Raw SpatialResponse保持未平滑360°响应
    Gate关闭或当前没有Raw SpatialResponse时不预测
    静止/移动/交叉/混响双声源实机参数标定仍未完成
                    ↓
Layer 3：8ch音频 + 平滑候选角度 + 16个IMCRA hop
    【已完成】imcra_spatial_separation CPU实现、Runtime接线与自动测试
        ├──内部只用7物理麦：Dual LCMV / soft-null loaded MVDR / loaded MVDR / DAS回退
        ├──16个IMCRA hop组成BeamformerNoiseContext，生成逐频点7×7空间噪声协方差
        ├──双候选从全局spatial_separability只读表按频点、绝对朝向和有符号角差查询空间可分度p
        ├──Test UI可在运行前/运行中切换三档：优化算法、7麦DS基线、全频Loaded MVDR基线
        ├──Loaded MVDR基线读取IMCRA噪声协方差，无法安全实现的频点回退DAS
        ├──固定30°与五频段波束模式均已删除；正式默认保持优化算法
        ├──未实现L2方向不确定度输出或不确定度驱动的动态波束宽度
        ├──每个候选输出EnhancedAudio：48 kHz mono float32 [15360]
        └──公共STFT、SpectrogramFeature与[33,169] FeatureExtractor已从主链删除
    CUDA/OOM、实机音质和实时性能门禁仍未完成
                    ↓
Layer 5：48 kHz EnhancedAudio独立副本 + 16个对齐IMCRA概率 → 响度补偿 → 内部降采样16 kHz → NVIDIA MarbleNet → 每方向人声概率
    【已完成】基准artifact/hash、primary/shadow、CPU/CUDA一致性测试与Runtime/Test UI接线
    【已完成】imcra_probability_rms_v1：16×20 ms概率加权、目标-23 dBFS、-3 dBFS新增增益峰值保护
    目标R6+1数据、微调、窗口概率校准和锁定test指标仍未完成
                    ↓
    【已完成】ResultJoiner：同键合并各阶段终态，commit按全局window_id有序原子提交DecisionRecord+watermark
    【已完成】有界分阶段流水：稳态L2(n) || L3(n-1) || L5(n-2)，同窗仍严格L2→L3→L5
    【已完成】L2/L3/L5逐层latest-wins：只替换本层未开始旧任务，丢弃与pre-joiner拒绝均有序审计
    【已完成】L5完成帧显示邮箱：容量1、latest-only、完整同窗，仅降低Test UI显示等待，不改变正式提交顺序

配套：
    【已完成】ApplicationRuntime：唯一WindowKey与冻结配置；有界L2/L3/L5 latest-wins、completion/backlog/commit硬限和ResultJoiner有序提交
    Development Test UI：8路电平、IMCRA/Gate、Raw SRP与平滑候选；L3三档实时切换；显示各阶段队列/worker/缓存
        ├──【已完成】SRP候选身份显示：首次出现灰色小点；临时ID灰色；正式ID三色；观测大、预测小
        └──【已完成】试听sidecar：Center Mic原音参考 + L2私有ID优先关联 + 20 ms绝对时间轴缓存
                ├──Kalman-ready临时ID即开始缓存并在转正式后沿用；累计≥2秒后显示；消失后等待3秒
                ├──ID换号仅在20°内唯一匹配时续接；同时出现的近角双ID绝不合并
                └──模式切换清缓存；临时文件有界且关闭删除；试听归一化不修改正式音频
    【已完成】RecordingStore v0.3 sidecar：原子result+watermark、逐chunk释放、有界event pre-roll、hotmap流写入与enhanced partial+journal恢复
    【已完成】Audio Data Manager基础版与选定样本注入Development Test UI
    正式app.main/最终方向GUI与完整实机UI门禁仍未完成

测试与验收：
    当前自动化结果以仓库完整pytest报告为准，不在长期文档中固化易失真的用例数量
    【最终门禁】硬件校准、动态标定、CUDA/OOM、目标域指标、端到端时延和30分钟稳定性
```

图中“已完成”只表示其文字限定的软件实现、接线和自动化测试范围，不代表所属整层已经通过实机门禁。紧贴左侧的整层标题只有满足完整完成定义时才标记，其下分支不重复标记；因此Layer 1～4仍保留未标记标题。本文件后续正文仍包含v0.2历史细节，仅用于迁移比对；凡与v0.3目标文档冲突之处均由v0.3覆盖。麦克风坐标从麦克风面观察：MIC0为+x，MIC1～MIC5逆时针排列，背面灯面装配图必须镜像后验证。

全局空间可分度`p`表的公共访问接口、索引语义、适配校验和重新生成规则，以[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)第6.1节为准。该资源位于根级`spatial_separability`包，不属于L1～L5；各层统一导入`lookup_p`或`load_p_table`，不得直接读取表文件或维护私有副本。

---

# 0. 项目目标与边界

系统接收 Sipeed R6+1 麦克风阵列的7个物理麦克风信号，在二维平面内扫描候选声音方向，对每个候选方向生成方向增强信号，再由 CNN 判断该方向点是否包含人声。

每个判断点输出：

```python
voice_direction_count: int
detections: tuple[VoiceDetection, ...]
```

每个 `VoiceDetection` 包含：

```python
theta_deg: float
voice_probability: float
is_voice: bool
```

`voice_direction_count` 是 `is_voice=True` 的方向点数量，不是物理说话人数。同一个人的多径、旁瓣或重复峰可能形成多个方向点；当前版本不做人物去重。

当前版本不实现：3D DOA、俯仰角、距离、对外Speaker ID/Source ID、人物身份追踪、身份识别、ASR、物理人数估计和最终多音轨分离。L2内部临时ID只用于候选关联和角度平滑，不构成可消费身份。CNN只做逐方向点 Voice / Non-Voice 二分类。

---

# 1. 固定总体架构原则

主处理链路见文档开头“架构速览”；本节规定其不可违反的跨层原则。

固定原则：

- Layer 1沿用现有约定，不另建第二套输入层。
- 所有模块边界的多通道PCM shape统一为 `[N,7]`，时间轴在前。
- 所有算法使用同一套真实物理坐标和同一个 `theta_deg`。
- `IngestCoordinator`是stream epoch与绝对sample index的唯一分配者；算法与录音不得各自推导另一套时间轴。
- 所有下游对象携带相同 `session_id`、`stream_epoch`、`window_id` 和绝对sample边界。
- 层内算法可以替换；层间字段、shape、单位、时间语义和错误语义不可随算法改变。
- GitHub开源实现只能放在层内适配器后面，上层不得依赖第三方库对象。

---

# 2. Layer 1 固定输入契约

Layer 1唯一公开音频对象沿用现有 `DecodedAudio`：

```python
@dataclass(slots=True)
class DecodedAudio:
    samples: np.ndarray              # float32, C-contiguous, shape [N,7]
    sample_rate: int                 # 48000
    sequence_id: int                 # 单调递增
    timestamp: float                 # 第一帧的单调时钟秒数
    native_samples: np.ndarray | None
    hotmap: CdcHotmapFrame | None
    noise_spectrum: NoiseSpectrumRecord | None  # 只记录；当前L2不消费
```

`native_samples`在实时设备输入中必须为float32、C-contiguous `[N,8]`，顺序为Host CH0..CH7；离线7ch资产无原生8ch时允许为`None`。若存在，必须与`samples`具有相同N且`samples`严格由`native_samples[:, [0,1,2,3,4,5,7]]`再施加同一校准得到。`hotmap`是读取该音频块时最新的不可变CDC快照，不保证硬件同步。`noise_spectrum`是校准后音频的逐通道、逐频点动态噪声PSD只读记录，与音频块共享sequence/timestamp；当前版本不得作为Layer 2输入。所有数组和标量在进入Coordinator前必须finite，`N>0`，`sequence_id>=0`，`timestamp`为该块第一个sample的单调时钟秒数；时间戳不得使用Unix wall clock。

为保留现有对象且让丢块可检测，实时source另发布只读诊断侧信道：

```python
@dataclass(frozen=True, slots=True)
class InputHealthEvent:
    event_id: int
    timestamp: float
    kind: str                  # input_overflow | handoff_drop | device_restart | source_error
    last_sequence_id_before_gap: int | None
    first_sequence_id_after_gap: int | None
    lost_sample_count: int | None
    message: str
```

`DecodedAudio`的字段和语义不变。实时实现迁移时必须在任何有界handoff之前分配`sequence_id`；若handoff丢弃块，下一个可见`DecodedAudio`必须出现sequence gap并同时发布`InputHealthEvent`。PortAudio只报告overflow而无法给出精确丢失sample数时，`lost_sample_count=None`，也必须触发epoch重置。事件必须携带可知的gap两侧sequence ID；同一gap的sequence跳变与健康事件由Coordinator去重，只递增一次epoch。禁止使用无界队列掩盖持续积压。

固定设备默认值：

```yaml
device:
  sample_rate: 48000
  device_channels: 8
  pcm_format: s16-le
  layout: interleaved
  block_size_samples: 960
  physical_channel_map: [0, 1, 2, 3, 4, 5, 7]
```

逻辑通道顺序：

```text
0 Ring0, 1 Ring1, 2 Ring2, 3 Ring3, 4 Ring4, 5 Ring5, 6 Center
```

Host CH0～CH5对应Ring0～Ring5，CH6为设备内部波束输出并排除，CH7为Center。设备打开后必须验证48 kHz、8ch、S16_LE；不一致属于启动失败，不能由下游猜测或自动重排。

Layer 1负责设备读取、PCM解码、映射、校准，并保留现有原始8ch/物理7ch录制与离线WAV回放能力。Layer 1同时维护`NoiseSpectrumRecord [7,n_fft/2+1]`；它经Coordinator后共享`session_id/stream_epoch/sample`时间轴，随同一音频chunk保存，但当前只用于记录和访问，不接入Layer 2。正式项目中的切块、manifest、Catalog、保留策略和Runtime/Test资产生命周期由Audio Data Manager统一管理；Layer 1不得另建一套资产目录或索引。Layer 1不负责连续性编号、窗口组装、DOA、Beamforming、STFT或CNN。

“保留现有录制能力”只表示复用原始字节与WAV编码能力；正式Runtime/Test/scratch录制的控制权、路径、状态机和writer所有权分别只属于第12节`RecordingStore`、`CorpusStore`与第11节`scratch_recorder`，禁止Layer 1同时启动第二个正式录音writer。

固定延迟校准具有跨块history。Layer 1内部在公开`DecodedAudio`之前必须有一个只负责状态清理的`CalibrationContinuityGuard`：source开始、收到`InputHealthEvent`、sequence非连续、采样率改变或timestamp超出第4节容差时，先`ChannelCalibrator.reset()`再处理当前块。该guard不得分配session/epoch/sample index，也不得吞掉当前块；Coordinator仍对公开输出执行同一连续性验证并是下游时间轴唯一权威。guard与Coordinator对同一输入是否重置必须有一致性测试，防止断流后第一块混入旧校准history。

---

# 3. 真实物理坐标与角度

内部物理坐标固定为：

```text
+x 向右
+y 向上
原点 Center
单位 meter
```

```python
MIC_POSITIONS_M = np.asarray([
    ( 0.000000000,  0.040000000),  # Ring0
    ( 0.034641016,  0.020000000),  # Ring1
    ( 0.034641016, -0.020000000),  # Ring2
    ( 0.000000000, -0.040000000),  # Ring3
    (-0.034641016, -0.020000000),  # Ring4
    (-0.034641016,  0.020000000),  # Ring5
    ( 0.000000000,  0.000000000),  # Center
], dtype=np.float64)
```

所有算法层使用标准数学极角：

```text
theta_deg=0°   -> 物理+x
theta_deg=90°  -> 物理+y / Ring0
theta_deg=180° -> 物理-x
theta_deg=270° -> 物理-y / Ring3
逆时针增加，范围 [0,360)
```

Ring0～Ring5分别为 `90°, 30°, 330°, 270°, 210°, 150°`。几何只允许由 `common/geometry.py` 提供，角度规范只允许由 `common/angle.py` 提供。现有L2中旋转/镜像后的旧几何必须删除，不保留兼容模式。

UI可以通过唯一 `UiAngleMapper` 改变屏幕0°位置或顺逆时针显示，但不得把显示角写回算法对象。本版不冻结最终UI显示映射。

---

# 4. 统一时间模型

所有时间范围采用绝对sample index和半开区间 `[start,end)`。每次启动一个音频source时生成不可复用的UUID `session_id`；即使未开启录音，该ID也存在。流开始的第一帧sample index为0。

```yaml
timing:
  decision_hop_samples: 960       # 20 ms @ 48 kHz
  doa_window_samples: 1920        # 40 ms
  context_samples: 15360          # 320 ms
```

```python
@dataclass(frozen=True, slots=True)
class IngestedAudioBlock:
    session_id: str
    stream_epoch: int
    start_sample: int
    end_sample: int
    sample_rate: int
    sequence_id: int
    timestamp: float
    samples: np.ndarray                 # float32 [N,7], C-contiguous, read-only
    native_samples: np.ndarray | None   # float32 [N,8], C-contiguous, read-only
    hotmap: CdcHotmapFrame | None
    noise_spectrum: NoiseSpectrumRecord | None  # 与块对齐的只读L1记录

@dataclass(frozen=True, slots=True)
class DecisionWindow:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    context_start_sample: int
    context_end_sample: int
    sample_rate: int
    samples: np.ndarray                 # float32 [15360,7], read-only
    source_sequence_ids: tuple[int, ...]
```

必须满足：

```python
doa_end_sample == context_end_sample == decision_sample
doa_end_sample - doa_start_sample == 1920
context_end_sample - context_start_sample == 15360
samples.shape == (15360, 7)
```

`IngestCoordinator`先消费所有待处理`InputHealthEvent`并验证 `DecodedAudio`，再把数组所有权移交给只读 `IngestedAudioBlock`；若输入数组不满足float32、C-contiguous或独占所有权，则复制一次后再设为只读。它是连续性判断、`stream_epoch`和sample边界的唯一权威。连续块必须满足相同session、48 kHz、`sequence_id == previous + 1`、有限timestamp且 `abs(timestamp - expected_timestamp) <= 5 ms`，其中 `expected_timestamp = previous.timestamp + previous_sample_count / 48000`。离线source必须生成同样的sample时钟。任一健康事件或条件失败时，Coordinator递增epoch、在新epoch把sample index重置为0并发布discontinuity事件；当前有效块作为新epoch第一块，不静默丢弃。若健康事件发生后尚无有效块，只先发布状态并等待下一块建立新epoch。

`WindowAssembler`只消费 `IngestedAudioBlock`，不得重新判断或重编号连续性。它可以接收任意正长度块；在每个epoch内，`decision_sample`固定为`15360 + k*960 (k>=0)`，即第一个正式窗口覆盖`[0,15360)`且`decision_sample=15360`，此后每累计960个新sample产生一个窗口。跨越hop边界的输入块必须切片消费，不能因L1块大小改变endpoint。epoch改变时清空buffer并重新预热。开始后累计不足15360 samples时状态为 `warming_up`，不输出正式判断；source结束时不能组成下一个完整960-sample hop的尾部明确丢弃并记入diagnostics。不得用零填充伪造正式上下文。

`stream_epoch`从0开始，每次连续性重置加1；`window_id`由WindowAssembler分配，在进程生命周期内始终单调递增且不复用。绝对sample index在每个epoch内从0开始，因此任何跨文件、日志和Catalog引用必须使用 `(session_id, stream_epoch, sample_index)` 三元组，不能只用sample index。RecordingStore与WindowAssembler必须订阅同一份 `IngestedAudioBlock` 对象。

Layer 2使用 `samples[-1920:,:]`；Layer 3、STFT和CNN使用完整 `samples`。任何层不得重新读取设备获取“最新音频”。

---

# 5. 权威跨层数据类型

以下类型必须在 `common/data_types.py` 定义。核心层之间禁止传递无字段约束的普通dict。

```python
@dataclass(frozen=True, slots=True)
class SpatialResponse:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    theta_degrees: np.ndarray       # float32 [360], exactly 0..359
    raw_scores: np.ndarray          # float32 [360], finite
    normalized_scores: np.ndarray   # float32 [360], finite in [0,1]

@dataclass(frozen=True, slots=True)
class CandidateDirection:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    theta_deg: float                # [0,360)
    raw_score: float
    normalized_score: float         # [0,1], not a probability

@dataclass(frozen=True, slots=True)
class DirectionalSignal:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    theta_deg: float
    sample_rate: int
    beamformer_backend: str         # "frequency_hybrid" or "das"
    fallback_reason: str | None
    stft_complex: np.ndarray        # complex64 [513,33]

@dataclass(frozen=True, slots=True)
class SpectrogramFeature:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    theta_deg: float
    beamformer_backend: str
    preprocessing_version: str
    spectrogram: np.ndarray         # float32 [33,169]

@dataclass(frozen=True, slots=True)
class VoiceDetection:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    context_start_sample: int
    context_end_sample: int
    theta_deg: float
    beamformer_backend: str
    model_version: str
    voice_probability: float        # finite [0,1]
    is_voice: bool

@dataclass(frozen=True, slots=True)
class DecisionResult:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    doa_start_sample: int
    doa_end_sample: int
    context_start_sample: int
    context_end_sample: int
    status: str                     # "ok", "degraded", "error"
    spatial_response: SpatialResponse | None
    candidates: tuple[CandidateDirection, ...]
    detections: tuple[VoiceDetection, ...]
    voice_direction_count: int
    diagnostics: tuple[str, ...]
    processing_latency_ms: float

@dataclass(frozen=True, slots=True)
class PipelineStatus:
    state: str                      # "stopped", "warming_up", "running", "degraded", "error"
    session_id: str
    stream_epoch: int
    buffered_samples: int
    required_samples: int           # 15360
    message: str

@dataclass(frozen=True, slots=True)
class ResultWatermark:
    session_id: str
    stream_epoch: int
    through_decision_sample: int    # <=该点的窗口均已得到结果或明确记录为dropped
    dropped_window_ids: tuple[int, ...]
```

`ResultWatermark`调用按同一epoch的`through_decision_sample`严格递增；`dropped_window_ids`是自上一次watermark（本epoch第一次则从epoch起点）到本次through范围内新确认丢弃的精确增量，必须排序、无重复，且每个ID对应的窗口endpoint不大于through。每个窗口只能处于“发布一个DecisionResult”或“进入一个dropped增量”二者之一。epoch切换时，旧epoch所有已入队窗口必须完成、报错或记为dropped并发送最终watermark，随后才发送新epoch的第一个watermark；迟到的旧epoch结果不得进入新epoch。

上述NumPy数组均必须C-contiguous、只读且dtype/shape精确匹配注释；核心契约对象构造后不得原地修改。正式跨层DTO和持久化边界统一使用CPU NumPy，GPU worker内部允许使用PyTorch张量，但同一`DecisionWindow`只上传GPU一次，`DirectionalSignal`与`SpectrogramFeature`只在本窗口整批CNN推理完成后各下载一次供DTO、UI或持久化使用，禁止在Layer 3与Layer 5之间逐候选GPU→CPU→GPU往返。训练数据离线生成也必须经同一GPU/CPU适配边界。

同一个结果链的 `session_id`、`stream_epoch`、`window_id`、`decision_sample`和对应sample边界必须完全相等，否则立即抛出接口错误。候选、方向信号、特征和检测tuple均按Layer 2 rank顺序一一对应，且同一tuple内`theta_deg`不得重复。`DecisionResult`构造时强制 `voice_direction_count == sum(d.is_voice for d in detections)`；`status="ok"`或`"degraded"`时每个candidate必须恰有一个同角度、同顺序detection，`degraded`只表示结果通过记录过的回退仍可正式消费。`status="error"`时`detections=()`且`voice_direction_count=0`；可以保留`spatial_response/candidates`作诊断，但GUI不得将其计入正式方向数。`processing_latency_ms`是从窗口在Ingest时间轴闭合并进入processing queue，到`DecisionResult`构造完成的有限非负单调时钟差，包含queue等待与算法compute time，不含320 ms历史上下文本身、首次预热、UI绘制、试听和异步写盘；它就是11.3节定义的`Latency`。WindowAssembler尚未形成首个完整`DecisionWindow`的启动预热阶段只发布`PipelineStatus`，不构造`DecisionResult`；形成窗口后若未来L2 Gate处于自身预热状态，则按6.1的`BLOCKED`语义记录该窗口。

---

# 6. Layer 2 1.1：SRP-PHAT候选方向

## 6.1 Probability Gate架构

L2不再负责Noise Estimation，也不维护Noise PSD或NE后端。L1为每个20 ms hop发布与绝对sample区间绑定的阵列声源概率；每个`DecisionWindow`的末尾40 ms必须恰好对应两个连续、同session/epoch且状态有效的概率：

```text
SourceProbability20ms(previous) + SourceProbability20ms(current)
    → p40 = (p_previous + p_current) / 2
    → Probability Gate
    → SRP-PHAT → Peak Detection → Raw CandidateDirection
    → Internal DirectionSmoother → Smoothed CandidateDirection
```

Gate默认门限为`0.60`。通常当`p40 >= threshold`时为`OPEN`并对原始末尾40 ms物理7通道音频执行SRP-PHAT；当`p40 < threshold`时为`CLOSED`，跳过SRP并输出空候选。当ID追踪与卡尔曼同时开启且当前至少一个正式ID仍存活时，任何有效`p40`均强制为`OPEN`；最后一个正式ID失效后的下一窗口恢复门限判断。概率缺失、跨epoch、区间不连续、预热或非有限时仍安全阻断，不能把缺失概率当作0，也不能沿用上一窗口方向。

Gate只消费概率并做判定，不重采样、不降噪、不修改PCM。门限由Development Test UI右侧独立滑动条实时调整，在下一个完整窗口边界生效、更新`gate_config_revision`并持久保存，直到用户再次修改；它不得与SRP候选峰值门限混用。Gate阻断不是处理错误，result watermark仍推进。

`Layer2PipelineResult`只区分`BLOCKED`与`PROCESSED`：前者没有空间响应且候选为空，后者执行现有SRP扫描。L2已删除`noise_estimation.py`和`sound_energy_gate.py`，正式Gate契约由`probability_gate.py`提供。

## 6.2 固定扫描接口

```python
class DirectionScanner(Protocol):
    def scan(self, window: DecisionWindow, geometry: MicGeometry,
             config: DirectionScanConfig) -> tuple[SpatialResponse, tuple[CandidateDirection, ...]]: ...
```

扫描器输入为`window.samples[-1920:,:]`并输出0～3个原始候选。L2 Pipeline先进行可选私有ID分配，再进行可选的按ID圆周卡尔曼滤波。两个模块默认关闭；卡尔曼只能在ID追踪开启时运行，关闭ID会同步关闭卡尔曼。卡尔曼状态跟随私有ID而非rank，公共接口不产生或消费`track_id/source_id`。

当两个模块同时开启时，临时ID从首次建立起按48 kHz绝对sample观察2秒，并在这首个2秒内累计归并至少5次自然Gate窗口候选且至少1个同窗角度被L5识别为人声后正式化。临时阶段的人声反馈只满足确认条件，不提前续命；正式化时获得3秒语音租约。角度匹配、卡尔曼校正、预测和ID强制Gate均不续命。L5向L2发送人声与非人声的session/epoch/decision sample/角度/概率；L2按同窗历史和20°圆周门限自动匹配。非人声结果只降低内部语义可信度，绝不隐藏L2角度；任何匹配的人声结果清除该ID此前的负面语义证据。正式化后唯一匹配到仍存活正式ID的人声结果，才把截止sample滑动到该人声点之后3秒。无历史、歧义、错流、非人声或已过期均不续命，迟到反馈不得复活。低P强制窗口可更新已有正式ID位置且预测后首次重匹配使用2倍测量可信度，但不能创建或晋升新ID。租约到期后在Gate判断和关联前删除ID及其卡尔曼状态，最后一个ID删除后恢复按P判断。公共候选DTO保持不变，只输出角度及原有Raw/Norm。

卡尔曼Q、R通过两个无量纲运行时倍率调整：Q倍率缩放基础过程噪声矩阵，R倍率缩放基础测量噪声方差。配置文件必须显式给出初始1.00；允许范围0.02～10.00，调节步长为0.1，0.02作为最小端点。Test UI分别提供当前值、减、加、应用控件；应用后保存并从下一完整窗口生效，不得重建ID或重置卡尔曼状态。

### 6.2.1 内部DirectionSmoother固定语义

`SpatialResponse`保持原始360°响应。内部ID只存在于L2实例内；Gate阻断或当前没有`SpatialResponse`时公共候选为空。对真实候选，平滑器不得删除、重排候选或改写时间/rank/score；成熟ID在已有当前响应但没有可归并候选时，可以按前述规则补充预测候选，Raw/Norm从当前响应在预测角处读取。平滑后不得重跑threshold、prominence、NMS、Top-3或排序，最终仍须满足Top-3与45°间距。session/epoch切换立即重置；追踪异常时本窗口回退原始候选并清空状态。完整规则以[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)为准。

## 6.3 默认SRP-PHAT实现

默认后端为全21麦克风对的远场2D SRP-PHAT。下列片段是第14节唯一配置中`layer2`的摘录；`speed_of_sound_mps`只来自`hardware.speed_of_sound_mps`，不得在`layer2`重复定义：

```yaml
layer2:
  probability_gate:
    backend: mean_2x20ms_v1
    threshold: 0.60
  scanner_backend: srp_phat
  angle_step_deg: 1.0
  frequency_min_hz: 2000.0
  frequency_max_hz: 4000.0
  n_fft: 2048
  window: hann_periodic
  remove_channel_mean: true
  phat_epsilon: 1.0e-12
  gcc_interpolation: 16
  normalization_backend: robust_z_sigmoid
  normalization_alpha: 1.0
  normalization_beta: 2.0
  direction_threshold: 0.35
  peak_prominence: 0.05
  min_peak_distance_deg: 45.0
  max_candidates: 3
```

数学符号固定如下。对目标方向：

```python
u(theta) = [cos(theta), sin(theta)]
tau_m(theta) = -(position_m dot u(theta)) / c
predicted_pair_delay(i,j,theta) = tau_i(theta) - tau_j(theta)
G_ij(f) = X_i(f) * conj(X_j(f))
Gphat_ij(f) = G_ij(f) / max(abs(G_ij(f)), phat_epsilon)
```

统一远场合成模型为：

```python
X_m(f) = d_m(f,theta) * S(f)
d_m(f,theta) = exp(-j * 2*pi*f*tau_m(theta))
```

因此 `ifft(X_i*conj(X_j))` 的峰必须位于 `tau_i-tau_j`。GCC中必须在该预测lag取值。强制合成测试必须验证从0°到359°的平面波不会镜像、反向或固定旋转。

可直接实现的计算流程固定为：

1. 从 `DecisionWindow.samples[-1920:,:]` 取DOA子窗，逐通道减去该子窗均值。
2. 乘长度1920的periodic Hann窗，零填充到2048点并沿时间轴做one-sided RFFT，得到complex64 `[1025,7]`；CPU参考实现可用complex128计算，但发布`raw_scores`前必须转换并按float32 contract验证。
3. SRP-PHAT只保留FFT中心频率落在2000～4000 Hz（包含两端）的bins；其他bins置零。L1用于Probability Gate的概率聚合仍为500～4000 Hz。
4. 对21对通道计算 `Gphat`，以 `irfft(..., n=2048*16)` 得到16倍插值的圆周GCC。
5. 预测lag换算为 `lag_samples * 16`，对相邻两个GCC索引做线性插值；负lag通过模长度索引。
6. `raw_score(theta)` 为21个插值GCC实数值的算术平均。所有角度使用同一缩放，所以不再额外clip raw score。
7. raw score非有限时该窗口返回`error`；不得把NaN替换为0后继续。

CPU NumPy和GPU PyTorch实现都必须满足这一语义。默认先迁移现有NumPy实现；性能需要时可增加CUDA后端，但输出接口和合成测试不变。

## 6.4 归一化与候选筛选

```python
median = np.median(raw_scores)
mad = np.median(np.abs(raw_scores - median))
scale = max(1.4826 * mad, 1e-6)
z = (raw_scores - median) / scale
logit = np.clip(1.0 * (z - 2.0), -80.0, 80.0)
normalized = 1.0 / (1.0 + np.exp(-logit))
```

候选顺序固定为：圆周局部峰 → `normalized>=0.35` → circular prominence `>=0.05` → 45°圆周NMS → 分数降序 → Top-3。分数相同时按较小`theta_deg`排序。0°/359°必须相邻处理。45°专指同一窗口内任意两个声源点之间的最小圆周角差，不约束单个ID的位置稳定性或移动速度。圆周距离小于45°的较低分峰被抑制，恰好45°的峰允许共存。限制前有效峰数及是否触发上限必须写入搜索诊断。

圆周prominence必须用同一实现：对360点数组做 `np.tile(scores,3)`，在三倍数组上调用 `scipy.signal.find_peaks(..., prominence=0.05, plateau_size=(1,None))`，只保留中间副本索引 `[360,720)`并减360。平台峰使用SciPy返回的中心索引；偶数长度平台按SciPy规则向较小索引取整。之后自行执行圆周NMS，不使用线性`distance`参数。这样0°边界与普通位置具有相同prominence语义。

`normalized_score`只是空间响应分数，不是概率。上述数值是可直接运行的初始默认值；以后调参只能修改配置和评测报告，不能改变接口。

### 6.6 可切换的迭代多峰实验路径

`layer2.iterative_peak_search_enabled=false`是正式默认值。关闭时必须完整旁路迭代引擎，保持本节原单次SRP-PHAT的Raw、Norm、候选及排序不变。Development Test UI在右侧候选表上方提供开关，并在其下方以固定高度单行实时显示同一DecisionWindow的NE `noise_mean_db`与`noise_std_db`；Gate阻断SRP时NE读数仍须发布。设置持久保存；切换以完整DecisionWindow为边界生效，不重算已经显示的窗口，也不占用左侧极坐标空间。

开启时使用`iterative_rank1_projection_v1`：第0轮仍执行本节基础扫描并保留其`SpatialResponse`作为蓝色360°图；接受最强候选后，在同一40 ms、7通道RFFT观测上构造该方向的远场导向矢量，执行带阵列匹配置信度的rank-1软投影，重新生成21对互谱、PHAT、16倍GCC及完整360°残余扫描。最多执行两轮，证据不足时必须允许只有一个候选，禁止为了填满两个名额而制造候选。

后轮固定使用第0轮MAD尺度，残余互谱必须保留相对原始互谱幅度的有效度因子，禁止把微小残渣再次完全PHAT放大。每轮执行threshold、prominence、45°圆周NMS、残余/首峰比例、有效频点及麦克风对支持检查。候选正式Raw/Norm仍取第0轮基础图对应角度；残余搜索分数、轮次和支持量属于独立诊断。每个DecisionRecord必须记录实际模式、算法版本、配置revision、执行轮数及停止/回退原因。迭代异常时当前窗口回退单次搜索，但不得静默关闭或改写用户开关。

---

# 7. Layer 3：多频段方向增强与融合

## 7.1 固定接口

```python
class Beamformer(Protocol):
    def process_batch(self, window: DecisionWindow,
                      candidates: tuple[CandidateDirection, ...],
                      geometry: MicGeometry,
                      config: BeamformerConfig) -> tuple[DirectionalSignal, ...]: ...
```

输出数量和顺序必须与候选完全一致。`m=0`时返回空tuple，不运行STFT、CNN或伪造方向。

## 7.2 共享STFT

MVDR、DAS和FeatureExtractor共享一次多通道STFT，不允许各自采用不同参数：

```yaml
stft:
  n_fft: 1024
  win_length: 960
  hop_length: 480
  window: hann_periodic
  center: true
  pad_mode: reflect
  normalized: false
  onesided: true
  return_complex: true
```

输入临时转为 `[7,15360]`，`torch.stft`输出complex64 `[7,513,33]`。根据PyTorch定义，`center=true`时 `T=1+L//hop=33`。STFT只使用已经闭合的320 ms历史上下文；右侧reflect padding只反射上下文内已有sample，不读取未来真实sample，因此持续运行时可以在每个20 ms endpoint立即开始处理。320 ms是历史上下文和首次预热时间，不是每个判断点额外等待320 ms。

## 7.3 统一steering vector

```python
d_m(f,theta) = exp(-j * 2*pi*f*tau_m(theta))
```

Center位于原点，因此 `d_center=1` 并作为相位参考。若合成输入为 `X_m=d_m*S`，所有Beamformer必须在目标方向恢复相同相位参考。

## 7.4 80～500 Hz：DAS低频段

```python
w_das(f,theta) = d(f,theta) / 7
Y(f,t) = conj(w(f,theta))^T X(f,t)
```

DAS处理80～500 Hz，并是其他频段单频点故障回退和频带外安全输出，不得删除。

## 7.5 500～2000 Hz：WNG约束超指向MVDR

该频段使用扩散噪声空间协方差 `Gamma_mn(f)=sinc(2*f*distance_mn/c)` 的超指向MVDR。通过逐级diagonal loading搜索满足无失真约束且WNG不低于配置门限的权重；全部重试失败的频点回退DAS。主要算法路线参考第7.8节记录的6+1阵列开源实现，但运行代码必须独立适配本项目PyTorch接口。

## 7.6 2000～8000 Hz：数据自适应MVDR

使用无mask、角度导向、频域Capon/MVDR。下列片段是第14节唯一配置中`layer3`的摘录：

```yaml
layer3:
  main_backend: frequency_hybrid
  baseline_backend: das
  fallback_backend: das
  covariance_estimator: context_sample_covariance
  loading_retry_factors: [0.001, 0.01, 0.1]
  solve_dtype: complex64
  low_frequency_min_hz: 80.0
  low_frequency_max_hz: 500.0
  robust_frequency_max_hz: 2000.0
  high_frequency_max_hz: 8000.0
  crossover_width_hz: 100.0
  robust_noise_model: diffuse_sinc
  robust_wng_floor_db: -3.0
```

对每个频率：

```python
Rxx = (X @ conj(X).T) / T             # X shape [7,T]，等价于 X @ X.conj().T
Rloaded = Rxx + loading * max(real(trace(Rxx))/7, 1e-8) * I
v = solve(Rloaded, d)
den = conj(d).T @ v
require real(den) > 1e-8 and abs(imag(den)) <= 1e-4 * real(den)
w_mvdr = v / real(den)
Y = conj(w_mvdr).T @ X
```

禁止显式普通矩阵求逆。依次尝试三个loading factor；任一中高频频点仍求解失败、非有限、违反 `abs(wᴴd-1)<=1e-3`，或稳健频段未达到WNG门限时，该频点回退DAS并记录 `fallback_reason`。

## 7.7 频带融合与工程试听

三个频段必须使用同一候选角、Center相位参考和共享STFT。500 Hz与2000 Hz边界各使用100 Hz raised-cosine互补交叉带；禁止硬切、分别ISTFT后再相加。融合后每个候选只发布一个 `DirectionalSignal.stft_complex [513,33]`，后端标记为`frequency_hybrid`。Development Test UI从同一融合频谱ISTFT得到15360-sample单通道试听副本；试听处理不得改写正式频谱或CNN特征。

## 7.8 开源实现规则

三个频段必须基于同一份STFT、相同候选角度和相同上下文；运行时只把融合频谱送入FeatureExtractor。注意本文的`X @ conj(X).T`是通道协方差；对PyTorch `[F,7,T]`实现等价为`X @ X.mH / T`。

可以参考或适配：

- [Voice-Separation-and-Enhancement 6+1阵列多算法实现](https://github.com/KyleZhang1118/Voice-Separation-and-Enhancement)；审核提交与许可证状态记录在`third_party/NOTICE.md`，因上游未提供明确许可证而仅参考算法，不复制源码。

- [PyTorch Audio multichannel PSD/MVDR（固定到已审核tag/commit）](https://github.com/pytorch/audio/blob/v2.7.1/src/torchaudio/transforms/_multi_channel.py)
- [Pyroomacoustics beamforming](https://github.com/LCAV/pyroomacoustics)

引入任何GitHub代码前必须记录仓库URL、不可变commit hash、license、NOTICE要求和本地适配文件；禁止把`main`/`master`分支URL当成可复现依赖。第三方对象不得出现在上述公开接口。TorchAudio已进入维护阶段，因此只作为公式与测试参考，不作为运行时硬依赖；运行主路径使用项目封装的PyTorch张量运算。Pyroomacoustics主要用于仿真和交叉验证，不进入实时安装集。

依赖规则（Windows x64目标运行环境）：

```text
Python >=3.12,<3.13
NumPy >=2.4,<2.5
SciPy >=1.17,<1.18
PyTorch ==2.12.1+cu132（当前项目已验证官方CUDA wheel）
PySide6 >=6.10,<6.12
sounddevice >=0.5,<0.6
PyYAML >=6.0,<7
safetensors >=0.8,<0.9
```

`pyproject.toml`声明兼容范围，`requirements-vscode.txt`记录人工维护的顶层精确版本，`requirements-lock.in`记录lock输入，Windows x64/Python 3.12的发布、开发和实验环境统一通过`requirements.lock`记录全部传递依赖与分发包SHA-256。安装必须使用`pip install --require-hashes -r requirements.lock`，禁止把无hash的临时安装当成可复现环境。PyTorch固定为已在目标RTX 5060验证且包含`sm_120`的官方稳定`2.12.1+cu132`构建；升级Python、驱动、PyTorch或CUDA wheel时必须重新生成lock并重复第10节环境门禁。Pyroomacoustics作为可选`validation`依赖，不得成为实时运行硬依赖。

---

# 8. SpectrogramFeature固定工程旁路契约

`DirectionalSignal.stft_complex`直接用于工程特征，不执行ISTFT后再重复STFT。当前NVIDIA MarbleNet基准不消费该矩阵；L5稳定公共输入仍是第9节定义的48 kHz、320 ms增强波形。

```yaml
feature:
  preprocessing_version: voice_logmag_v1
  frequency_min_hz: 80.0
  frequency_max_hz: 8000.0
  first_bin: 2
  last_bin_inclusive: 170
  log_epsilon: 1.0e-6
  normalization: training_set_per_frequency_zscore
  expected_shape: [33, 169]
```

```python
magnitude = abs(Y[2:171, :])                    # [169,33]
log_magnitude = log(magnitude + 1e-6)
feature = transpose(log_magnitude, (1,0))       # [33,169]
# 仅当未来某个模型明确消费该特征且提供统计量时：
feature = (feature - model.freq_mean[None,:]) / model.freq_std[None,:]
```

当前运行时输出未标准化的finite、C-contiguous、只读float32 `[33,169]`，用于工程检查和未来模型实验。若未来模型选择该矩阵作为输入，`freq_mean/freq_std`必须由锁定train split上所有训练样本的未标准化`log_magnitude`按频率bin聚合计算，validation/test不得参与统计；artifact必须携带float32 `freq_mean[169]`、`freq_std[169]`并要求两者finite且`freq_std>=1e-6`。届时训练和推理必须共用同一个FeatureExtractor，禁止复制两套预处理代码。

未来消费该工程特征的模型artifact目录固定为：

```text
models/<model_version>/
  manifest.json
  model.safetensors
  freq_mean.npy
  freq_std.npy
  metrics.json
  sha256sums.json
```

生产加载器只接受无可执行pickle的`safetensors`权重；加载前验证所有SHA-256。训练checkpoint可以使用其他格式，但不得直接作为生产artifact分发。当前MarbleNet基准artifact采用第9.2节的独立manifest，不要求`freq_mean/freq_std`。

如以后改变采样率、上下文、STFT、频带、log或归一化，必须升级 `preprocessing_version` 并重新训练模型，旧模型必须拒绝加载不匹配特征。

调试听音通过共享complex spectrum执行 `torch.istft(..., length=15360)`按需生成；其未做播放器归一化的48 kHz波形同时是当前MarbleNet适配器的公共输入，`SpectrogramFeature`则保持独立工程旁路。

---

# 9. Layer 5：CNN逐点二分类

## 9.1 固定接口

```python
class VoiceClassifier(Protocol):
    model_id: str
    def predict(self, waveforms_48k: NDArray[np.float32]) -> ModelPrediction: ...
```

当前稳定公共输入由`TrackAudioStreamHub`生成：L3的40/80/160 ms重叠窗（当前40 ms）按精确`(session_id, stream_epoch, track_id)`去重，每窗追加一个与IMCRA概率严格对齐的20 ms hop并完成响度补偿，形成最长3200 ms的连续48 kHz轨。Test UI试听、按ID音频资产和CNN使用同一补偿后波形；重叠L3原始窗只作瞬时输入，不再重复保存。`m=0`仍返回`COMPLETED`空结果。

`imcra_probability_rms_v1`每20 ms计算RMS dBFS，以`-23.0 dBFS`为目标且只放大；`p<=0.30`为0增益、`p>=0.80`为完整增益，中间线性加权并受`-3 dBFS`新增增益峰值保护。Test UI开关默认开启且可实时切换；切换不清空轨道，从下一个20 ms平滑过渡到新的增益状态。

## 9.2 默认模型

第一版基准为NVIDIA `Frame_VAD_Multilingual_MarbleNet_v2.0`。适配器接收可变长度连续48 kHz轨，polyphase重采样为16 kHz后使用官方80维log-mel和预训练网络，一次得到连续20 ms二类概率。基线标量采用`latest_80ms_max_contiguous_3frame_mean_v2`：完整连续轨提供卷积上下文，但只在最新80 ms内寻找连续3帧均值峰值，防止旧语音粘住当前结论。

L5采用插件架构：同一不可变音频batch可同时送入一个primary和零到多个shadow模型。只有primary输出进入正式`VoiceDetection`、方向点数和`DecisionRecord`；所有模型的原始概率、版本、适配器和延迟分别保存在`ModelPrediction`，用于后续对比。新增模型只实现`VoiceModelPlugin`并添加配置，不得改动L2/L3公共接口。

模型manifest必须包含：

```yaml
model_id: nv_marblenet_baseline_v1
architecture_id: nvidia_frame_vad_marblenet_v2.0_native_v1
source_model: nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0
input: {public_sample_rate_hz: 48000, public_samples: 15360, model_sample_rate_hz: 16000}
aggregation: max_contiguous_3frame_mean_v1
weights_file: weights.safetensors
weights_sha256: string
voice_probability_limit: 0.70
```

manifest、权重哈希或接口不匹配时拒绝加载。运行时只加载`safetensors`；官方`.nemo`原包、模型卡和NVIDIA Open Model License快照保留用于来源追溯。开发阶段可使用显式Mock验证接口，但生产模式没有真实模型时必须返回 `error/model_unavailable`，不得把Mock结果显示为真实判断。

默认训练配置：AdamW、lr `1e-3`、weight decay `1e-4`、batch 64、`BCEWithLogitsLoss`、最大100 epochs、early stopping patience 10、min delta `1e-4`、seed 42。每个epoch固定用seed生成train shuffle；以validation loss选择唯一best checkpoint并执行early stopping。best checkpoint在validation logits上用正数temperature最小化NLL，artifact保存finite `temperature>0`；推理固定为`probability=sigmoid(logit/temperature)`，不能重复Sigmoid。数据集版本必须先锁定test，再只用train/validation训练、选checkpoint和校准temperature；模型、temperature和0.70阈值冻结后，test仅运行一次并生成不可改写报告。分组规则见12.7，禁止同一连续录音、房间或说话人跨train/validation/test泄漏。标签表示该方向增强后的实际音频在320 ms上下文中是否含人声，不以“候选角是否接近真实声源角”代替听觉/信号标签；只要旁瓣或反射使增强音频中仍有人声，就不能标成`non_voice`。音乐、电视、风扇、冲击声以及增强后确实不含人声的错误候选作为hard negatives收集。

验证集完成temperature scaling后导出模型；默认阈值：

```yaml
layer5:
  voice_probability_limit: 0.70
```

GUI改变阈值时只重新计算 `is_voice = probability >= threshold` 和方向点数量，不重跑前端或CNN。

---

# 10. GPU与运行时架构

目标设备为U9级CPU与NVIDIA RTX 5060。不假设固定显存容量；启动时读取实际GPU、驱动和可用显存并写入session metadata。

## 10.1 固定基础环境

当前目标机已验证环境固定为：Windows 11 x64、CPython 3.12.10 x64、NVIDIA RTX 5060 Laptop GPU（compute capability 12.0、实测7.96 GiB）、NVIDIA驱动610.88、PyTorch 2.12.1+cu132、PyTorch CUDA runtime 13.2。按NVIDIA CUDA 13.x兼容规则，驱动需要580或更高；610.88满足要求。`nvidia-smi`显示的CUDA版本只表示驱动可支持上限，不能作为本机CUDA Toolkit版本，真正运行时版本以`torch.version.cuda`为准。

正常运行不安装系统CUDA Toolkit：官方PyTorch wheel自带CUDA runtime，主机只需要兼容驱动。只有新增自定义CUDA/C++扩展时，才额外要求CUDA Toolkit 13.2、Visual Studio 2022 Build Tools C++工作负载和匹配Windows SDK；该扩展必须显式构建`sm_120`并单独验收，不得成为未记录的隐含依赖。具体创建、重建和VS Code操作以根目录`ENVIRONMENT.md`为准。

项目只允许根目录`.venv`作为运行解释器。`.vscode/settings.json`固定`${workspaceFolder}\\.venv\\Scripts\\python.exe`，并通过tasks/launch提供环境自检、全部测试、L1服务和Development Test UI入口。`.venv`不得提交、复制或与其他项目共享。环境安装只允许由`scripts/setup_vscode_env.ps1`执行；脚本按hash lock安装、运行`pip check`并强制执行`scripts/check_runtime_env.py --require-cuda`。

GPU环境启动门禁必须真实执行并全部通过：解释器路径与Python版本检查；`torch.cuda.is_available()`；设备名、显存、compute capability与`sm_120`支持；CUDA complex64 STFT `[7,513,33]`；批量complex64线性求解finite；实际MarbleNet批量波形 `[5,15360]` 经16 kHz适配与模型前向；依赖无冲突。只打印GPU名称或只运行通用Conv2D smoke不算L5门禁通过。当前`scripts/check_runtime_env.py --require-cuda`已执行实际MarbleNet `[5,15360]`波形前向并校验finite `[5]`概率；自动测试另比较固定输入的CPU/CUDA输出一致性。

笔记本正式性能测试必须连接电源、Windows设为最佳性能，并把VS Code和`.venv` Python指定为高性能NVIDIA GPU。驱动、Python、PyTorch或lock任一变化都触发完整环境门禁、全部自动测试和实机性能复测。

## 10.2 并发与加速架构

```text
CPU capture thread:
    USB -> Layer1 -> IngestCoordinator -> WindowAssembler
                               └───────> RecordingStore audio queue

Admission:
    DecisionWindow + frozen config -> WindowWorkItem
    WindowKey = (session_id, stream_epoch, window_id, decision_sample)

Bounded staged workers:
    L2 queue -> Gate / SRP-PHAT / private smoother -> L2StageResult
    L3 queue -> shared STFT / covariance / BF / ISTFT -> L3StageResult
    L5 queue -> loudness compensation / resample / batched CNN -> L5StageResult
             └-> COMPLETED full same-window DevUiFrame -> latest_l5_dev_ui (capacity 1, UI only)
    steady state: L2(n) || L3(n-1) || L5(n-2)
    same window:  L2(n) -> L3(n) -> L5(n)

Ordered commit:
    ResultJoiner(WindowKey) -> JoinedWindowResult in window order
       ├──> RecordingStore atomic(result + monotonic ResultWatermark)
       └──> immutable Development Test UI snapshot

UI thread:
    consumes immutable joined audit results, latest_l5_dev_ui, and public processing_status only
```

加速策略：

- PyTorch CUDA是目标主路径；安装与当前NVIDIA驱动兼容的官方稳定构建，不在项目代码硬编码CUDA小版本。
- `torch.cuda.is_available()`、目标device和显存信息必须在启动日志中记录。
- STFT、steering、协方差、`torch.linalg.solve`和CNN按窗口/候选批处理；禁止为每个候选重复上传同一窗口。
- 复数DSP使用complex64，不使用FP16解线性方程。CNN可以在验证无精度回退后使用CUDA autocast FP16/BF16。
- L2、L3、L5分别保持单worker以保护私有追踪、滚动STFT和噪声统计的时序；并行来自不同窗口占据不同stage，不是让同一窗口绕过依赖。每个stage的有界等待队列都使用latest-wins，只替换尚未被本层worker取走的最旧任务；已开始计算不取消。
- CPU ComputeCache按stage分区并受全局字节硬限制；L3 prepared CUDA context固定小容量且不进入CPU cache。频率轴、窗、mask、steering和p查询只允许有界缓存，GCC、BF solve、ISTFT和CNN batch按窗释放。
- 各stage队列和最大在途窗口均有硬上限。L2/L3/L5每层都按latest-wins显式终止本层队列中未开始的最旧任务：L2丢弃使三阶段均`DROPPED`，L3丢弃保留L2而使L3/L5 `DROPPED`，L5丢弃保留L2/L3而使L5 `DROPPED`。每个丢弃、跳过、失败、超时或取消窗口都必须进入可审计终态并有序提交DecisionRecord+watermark。
- 当Joiner在窗口数/字节容量上无法注册新窗口时，新窗口在pre-joiner边界被拒绝；不保留320 ms波形，仅使用有界范围记录身份/sample/原因，commit再按window ID展开为轻量`error` DecisionRecord和watermark。completion主队列、后备backlog和commit乱序表同样有硬上限，拥塞时拒绝新接纳而不无界堆积。
- L5 worker只在正式L5结果`COMPLETED`后向容量1的`latest_l5_dev_ui`发布完整同窗L2/L3/L5帧。满时覆盖旧显示帧并计数；失败、丢弃、跳过不进入该邮箱。此side channel不改变ResultJoiner、DecisionRecord、RecordingStore或watermark顺序。
- CUDA不可用时只有按配置成功转CPU并产生完整输出才标记`degraded/cpu_fallback`；CPU路径不承诺实时。回退本身也失败时必须是`error`。
- CUDA OOM时清空本次临时结果，重试一次较小候选batch；仍失败时只有完整CPU重算成功才返回`degraded`，CPU也失败则整窗`error`。不得崩溃、复用部分GPU结果或把失败标成降级成功。

目标性能（正式验收在用户U9+RTX 5060实机测量）：

```text
warm processing latency, m=5:
  p50 <= 10 ms
  p95 <= 20 ms
  p99 <= 40 ms
continuous 30 min:
  no unbounded queue growth
  no NaN/Inf
  no process crash
```

系统首次正式结果需要320 ms预热；持续运行后的上述processing latency从每个窗口闭合到`DecisionResult`产生。历史上下文、首次预热和逐帧processing latency不得混为一项。

## 10.3 唯一应用编排与生命周期

正式应用入口固定为`python -m app.main --config config/config.yaml`，Development Test UI入口固定为`python -m gui.dev_test_ui.app --config config/config.yaml`。两者都调用同一个`ApplicationRuntime`装配层，禁止UI、Layer 1 API或任一算法层自行创建第二条主链路。`ApplicationRuntime`只负责生命周期与适配，不定义新的DSP参数或DTO。

启动前的应用级顺序仍为：加载/校验配置与hash → 环境/GPU检查 → 加载并验证模型artifact（development允许明确Unavailable，production必须成功）→ 创建Catalog/RecordingStore和ApplicationRuntime。当前Runtime的实际启动顺序固定为：重置IngestCoordinator/WindowAssembler、缓存和有界队列 → 建立RecordingStore session并启用录音模式 → 按`commit→L5→L3→L2`启动stage worker → 打开CDC/UAC pipeline → 启动L1读取线程。UI只在Runtime已创建后订阅公开状态。任一步失败按相反顺序停止pipeline、唤醒并join已启动worker、封闭失败录音session，不得留下串口、音频stream、writer、stage线程或CUDA任务。

每个`DecisionWindow`先冻结配置，在有容量时注册唯一`WindowKey`，再进入有界L2队列。默认L2/L3/L5/completion容量分别为10000/10000/10000/8，最大Joiner在途窗口30003，覆盖三层等待队列及每层1个正在执行的窗口；L2、L3、L5各自使用latest-wins，仅终止本层尚未开始的最旧等待任务。Joiner注册前容量不足时新窗口用轻量有界审计拒绝，不保留音频。completion队列和后备backlog均不超过8，commit乱序表软限30003、硬限`2*30003+2*8=60022`；它们拥塞时拒绝新接纳，不延伸为无界队列。按50窗/s计算，单层满队列约对应200秒等待工作，端到端累计等待可能更长；大队列只提供有界过载缓冲，积压会增加CPU内存和排队延迟。三个stage可乱序完成，但ResultJoiner/commit只按window ID发布完整终态。epoch切换前必须封闭旧epoch全部已接纳窗口；迟到旧结果不得污染新epoch的UI、DecisionRecord或watermark。

正常关闭顺序固定为：停止UAC/CDC pipeline产生新块 → L1刷出已延迟预降噪hop并声明输入结束 → 以EOS依次drain L2、L3、L5和completion/commit → Joiner提交所有完成/跳过/失败/取消终态并通过原子result+watermark推进最终水位 → finalize scratch与RecordingStore → 停止播放/UI订阅 → 清空有界缓存并释放GPU和Catalog。drain受`graceful_shutdown_timeout_seconds`限制；超时时已注册未完成项明确标记`CANCELLED/error`并唤醒worker。只有全部worker真正退出后才关闭RecordingStore；否则保留资源并报错，不能假装停机完成。崩溃恢复只恢复第12节已写资产，不伪造未完成算法结果。

`runtime.mode`只允许`development|production`。production禁止Mock、禁止模型Unavailable、要求CUDA环境门禁通过且`hardware_calibration_status=verified`；任一不满足则启动失败。development允许CPU fallback、未校准硬件和显式Mock，但所有GUI/报告必须持续显示对应降级水印，不能产生production-ready报告。

---

# 11. 分层开发测试UI / Development Test UI

Test UI显示原始`SpatialResponse`与L2最终候选，并在右侧提供私有ID追踪与按ID圆周卡尔曼两个运行时开关。首次出现为灰色小点；临时ID观测为灰色大点、预测为灰色小点；正式ID使用三种稳定颜色，观测大、预测小。试听sidecar从Kalman-ready临时ID开始缓存已有L3音频，转正式后沿用同一`ID-nnn`缓存。这些状态只用于显示和连续播放，不得改变候选、租约、波束形成、分类或正式录音。

该UI是实现期间的独立诊断程序，用于每完成一层就观察真实输出并做实机验收；它不是最终用户GUI，也不是12.8的Audio Data Manager。程序必须支持L1-only、L1+SRP、L1+SRP+BF、完整CNN四种能力级别。未实现、未连接或模型不可用的象限明确显示`NOT IMPLEMENTED / UNAVAILABLE`及原因，不生成Mock视觉结果；若开发者显式启用Mock，整个对应象限必须持续显示醒目的`MOCK`水印，Mock结果不得写入正式测试报告。

## 11.1 全屏四象限布局

默认在当前主显示器全屏启动，`F11`切换全屏，`Esc`退出全屏但不终止采集。内容区用水平、垂直十字线按50%×50%严格均分为四部分；窗口缩放时仍保持等分，最低支持1920×1080与Windows 125%/150% DPI：

```text
┌──────────────────────────────┬──────────────────────────────┐
│ 左上：L1 输入与测试录音      │ 右上：SRP-PHAT 360°响应      │
│ 灯控、录制、7通道电平        │ 圆环、候选点、角度、Norm值    │
├──────────────────────────────┼──────────────────────────────┤
│ 左下：三频段融合音频与谱图    │ 右下：CNN 360°人声概率        │
│ 波形、试听、自动重播、谱图    │ 阈值滑动条、概率、Voice结果   │
├──────────────────────────────┴──────────────────────────────┤
│ Warmup | Sample rate | Compute time | Latency | Pre | Recall | F1    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

四象限内容区仍严格50%×50%等分；性能栏是内容区下方独立、固定56逻辑像素高的单行最末栏，不占用或破坏四象限等分。在最低支持分辨率1920×1080和100%～200%系统缩放下不得换行、遮挡或截断指标名；较长的模型、数据集和不可用原因放入悬停提示。四个象限顶部固定显示层名、连接状态、`session_id`短码、`stream_epoch`、最新`window_id`、sample endpoint、数据年龄和本层耗时。全局状态栏显示采集/预热/运行/降级/错误、输入drop、处理drop、CPU/GPU和队列深度；最末行性能栏专门显示下面定义的算法性能与效果指标。任何象限超过500 ms没有新数据即标记`STALE`，不得继续把旧图形伪装成实时输出。

测试UI使用算法内部真实数学角：0°在圆环正右、90°在正上、逆时针增加。它不得使用最终用户UI尚未确定的`UiAngleMapper`，因为此界面还承担角度和板卡朝向校验。

## 11.2 诊断快照接口与线程边界

测试UI只消费不可变、有界的诊断快照，不直接修改算法对象或等待GPU；灯光、录音和播放通过命令接口执行。新增类型位于`dev_test_ui/contracts.py`：

```python
@dataclass(frozen=True, slots=True)
class L1MeterSnapshot:
    session_id: str
    stream_epoch: int
    end_sample: int
    sequence_id: int
    rms_dbfs: np.ndarray             # float32 [7], finite or -120 floor
    peak_dbfs: np.ndarray            # float32 [7]
    clipped: np.ndarray              # bool [7]
    light_state: str                 # "on", "off", "unknown", "error"
    recording_state: str             # "idle", "recording", "paused", "finalizing", "complete", "error"

@dataclass(frozen=True, slots=True)
class BeamformPreview:
    session_id: str
    stream_epoch: int
    window_id: int
    decision_sample: int
    theta_deg: float
    mvdr_waveform: np.ndarray | None  # float32 [15360], read-only；未实现/失败时None
    das_waveform: np.ndarray          # float32 [15360], read-only
    spectrogram: np.ndarray | None    # float32 [33,169]；FeatureExtractor未接入时None
    runtime_backend: str              # "mvdr" or "das"
    fallback_reason: str | None

@dataclass(frozen=True, slots=True)
class AlgorithmPerformanceSnapshot:
    session_id: str
    stream_epoch: int
    window_id: int | None
    warmup_buffered_samples: int
    warmup_required_samples: int       # 15360
    configured_sample_rate_hz: int      # 48000
    observed_sample_rate_hz: float | None
    compute_time_ms_current: float | None
    compute_time_ms_p50: float | None
    compute_time_ms_p95: float | None
    latency_ms_current: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    metrics_source: str | None
    metrics_unavailable_reason: str | None
    model_version: str | None
    dataset_version: str | None
    evaluation_threshold: float | None

@dataclass(frozen=True, slots=True)
class DevUiFrame:
    l1: L1MeterSnapshot | None
    result: DecisionResult | None
    previews: tuple[BeamformPreview, ...]  # same candidate order, max 10
    pipeline_status: PipelineStatus
    performance: AlgorithmPerformanceSnapshot
```

`DevUiFrame.spatial_response`提供SRP原始/Norm 360点与候选，`l5_result.detections`提供CNN概率。除L1独立电平外，同一帧内SRP、BF音频和CNN必须具有完全相等的`WindowKey`。正式审计Frame由ResultJoiner有序提交的`JoinedWindowResult`构造；阶段跳过、失败或超时时发布同一window带明确终态的诊断帧。L5 worker还在真实`COMPLETED`后发布一个完整同窗Frame到`latest_l5_dev_ui`，只供右下象限即时显示，不是正式结果。禁止把新SRP与旧BF/CNN拼到同一Frame或让该邮箱改变有序提交。

实时UI使用两个容量1的latest-value mailbox：一个承载有序审计Frame，一个承载最新L5完成Frame。UI慢时只覆盖旧显示帧，不反压采集或算法。有序`DROPPED/SKIPPED`或缺失帧不得立即擦除上一份有效CNN结果；只有超过`dev_test_ui.stale_after_ms`仍没有新L5完成帧才显示`STALE`。`processing_status`必须公开`l5_actual_completed`、`l5_dropped`、`l5_skipped`、最近1秒`l5_actual_hz`以及L5显示邮箱深度、容量和覆盖数。L1电平最多刷新25 Hz，SRP/CNN图最多20 Hz，波形最多10 Hz；UI绘制耗时目标p95 `<8 ms`。复数STFT、整段PCM和360点数组不通过Qt signal逐项复制，使用只读快照引用并在GUI线程一次取走。

## 11.3 最末行：算法性能指标

性能栏固定为单行；以下换行只为文档可读性，实际UI不得换行：

```text
Warmup 15360/15360 (Ready) | Sample rate 48000 / 47999.8 Hz
Compute 6.2 ms (P50 5.8 / P95 8.9) | Latency 8.1 ms (P50 7.4 / P95 12.6)
Precision 0.91 | Recall 0.96 | F1 0.93 | test:<dataset_version> @ threshold 0.70
```

指标口径固定如下：

- `Warmup`显示当前epoch的`buffered_samples/15360`和百分比；不足时显示`Warming`及按`(15360-buffered)/48000`计算的理论剩余毫秒，首个正式窗口产生后显示`Ready`。epoch重置立即清零，不能继续显示旧epoch Ready。
- `Sample rate`同时显示配置值和实测值，格式为`configured / observed Hz`。实测值用Ingest连续sample增量除以对应单调时钟跨度，在最近5秒滑动窗计算；少于1秒数据、断流或timestamp异常时显示`N/A`，不得用配置值冒充实测值。
- `Compute time`是同一窗口L2、L3和L5各stage实际运行时长之和，包含SRP、共享STFT、MVDR/DAS、ISTFT、响度补偿和CNN，排除各stage队列等待、Joiner等待、UI绘制、试听与RecordingStore。各CUDA stage必须在结束计时前同步本窗口CUDA event，禁止只测异步kernel提交时间。
- `Latency`是窗口被Runtime接纳的单调时刻到ResultJoiner完成并进入有序commit的端到端算法延迟，包含L2/L3/L5队列等待、compute time和Joiner等待，排除UI绘制、试听和异步写盘。它与`DecisionRecord.processing_latency_ms`采用同一数值；必须满足`latency >= compute_time >= 0`。每个DecisionRecord另保存stage状态、stage compute时长和stage queue wait，便于区分算力不足与排队抖动。
- Compute与Latency均显示最新有效窗口的current值，以及当前epoch最近最多500个已完成`ok/degraded`窗口的P50/P95；warming、error和dropped窗口不进入分位数，但单独的drop/error计数仍在全局状态栏显示。epoch改变时清空滚动统计。性能栏刷新上限2 Hz，不能为刷新而同步额外GPU工作。
- `Precision/Recall/F1`是CNN模型artifact中`metrics.json`保存的锁定test split结果，必须与当前`model_version`、`dataset_version`、模型artifact hash和`evaluation_threshold`匹配；这里的`Precision`即用户所说的`pre`。三项均显示0～1三位小数，同时显示`test:<dataset_version> @ threshold`来源。
- 实时运行没有逐窗口ground truth，因此禁止用在线预测自身计算Precision/Recall/F1。模型不可用、Mock、metrics缺失、hash/threshold不匹配或尚未完成正式test时，三项统一显示`N/A`并注明原因；不得沿用上一个模型的数字。UI临时调整阈值后，artifact效果指标立即显示`N/A (threshold differs)`，除非存在该精确阈值的锁定test报告。

`metrics.json`至少增加并冻结以下字段：

```yaml
schema_version: voice_metrics_v1
model_version: voice_cnn_v1
model_artifact_hash: sha256
dataset_version: string
split: test
evaluation_threshold: 0.70
sample_count: int
true_positive: int
false_positive: int
false_negative: int
true_negative: int
precision: float
recall: float
f1: float
evaluated_at_utc: ISO-8601
```

加载器必须从混淆矩阵重新计算三项并与文件值比较，绝对误差超过`1e-6`即拒绝该metrics；分母为0时对应指标为null并在UI显示`N/A`，不能填0掩盖无可评测样本。

效果指标公式固定为：`Precision = TP / (TP + FP)`，`Recall = TP / (TP + FN)`，`F1 = 2 × Precision × Recall / (Precision + Recall)`；只有各自分母非零时才定义。

## 11.4 左上象限：L1、灯控与临时测试录音

控制区固定包含：

```text
[灯光开] [灯光关]   状态: On/Off/Unknown/Error
[录制] [暂停/继续] [结束]   状态与计时器   当前临时文件
```

灯控复用现有Layer 1官方`E/e`语义（`E`开、`e`关）及`POST /lights/on`、`POST /lights/off`适配器；同一时刻只能有一个`SerialDevice`所有者。MA-USB8协议没有独立的灯状态读回，因此“确认”固定定义为串口/API成功写入完整1字节且未抛出设备错误，只表示命令已被主机接口接受，不表示光学传感确认。按钮发出命令后先进入`Pending`，收到上述写入确认才显示`On (commanded)`或`Off (commanded)`；2秒无确认显示Error。UI不得假设命令写出等于物理灯光已被观察成功；L1实机门禁中的开/关10次由操作者目视确认并记录。UI不得为灯控另开同一COM口。

临时测试录音与Runtime Recording、Test Corpus严格分开，固定写入：

```text
data/dev_test_ui/scratch/current/
  scratch_manifest.json
  segments/
    segment_000_native_8ch.wav
    segment_000_physical_7ch.wav
    segment_000_physical_7ch_float.npy
    segment_001_native_8ch.wav
    ...
```

录音状态机与按钮规则固定如下：

| 当前状态 | 录制 | 暂停/继续 | 结束 |
|---|---|---|---|
| idle/complete/error | 清除`current`旧测试音频后从新文件开始 | 禁用 | 禁用 |
| recording | 清除旧测试音频并立即开始一次新的录制 | 进入paused | 进入finalizing并封闭文件 |
| paused | 清除旧测试音频并立即开始一次新的录制 | 继续同一次录制 | 封闭当前已录部分 |
| finalizing | 禁用 | 禁用 | 禁用 |

因此可以连续进行任意多次测试录制，但测试区只保留最近一次；每次点击`录制`都必须先停止并封闭当前writer，再清除此前`current`内容、重置计时和波形，不能追加到旧测试音频。若用户只是想暂停后继续同一次录制，点击`暂停/继续`，不能点击`录制`。清除仅作用于上述scratch目录，不进入Catalog/Trash，也不得删除Runtime或Test Corpus资产。清除失败时不得开始新录制，必须显示具体错误。

`暂停`期间算法采集与四象限显示继续运行，只停止scratch写入并封闭当前segment；继续时创建下一个segment。manifest以同一source时间轴记录每个segment的`session_id/epoch/start/end sample`，播放器按段播放，不把暂停缺口拼进单个WAV伪造成连续sample。`结束`等待全部writer flush、fsync、frame count和hash完成后显示Complete。UI关闭时先停止并封闭writer，再删除整个scratch临时目录；source断开或异常时仍执行finalize并允许当前UI会话内诊断。L3连续合成试听音频写入`data/dev_test_ui/l3_audio_cache/current`专用临时目录，使用内存映射文件避免整段音频常驻内存；关闭UI时必须停止播放、解除映射并删除该缓存目录。

scratch每个segment的native/physical/float文件范围必须完全一致；若该source没有`native_samples`，manifest明确`native_unavailable`且不创建伪8ch文件。清除`current`时先把旧目录原子改名为同一父目录下唯一`deleting-<uuid>`，再创建新的`current`并由后台删除改名目录；任一步失败都不开始录制。只有`scratch_recorder`可以执行此操作，删除目标必须解析后仍位于`data/dev_test_ui/scratch`内。

电平区固定显示7条独立竖向meter，顺序为`Ring0..Ring5, Center`，每条同时显示名称、当前RMS dBFS、Peak dBFS、峰值保持和clip红灯。计算使用Ingest时间轴上最新连续960 samples（20 ms）物理7ch滚动窗，不依赖L1块大小；dBFS下限-120 dB，显示范围-90～0 dB；attack 50 ms、release 300 ms、peak hold 1 s。禁止对7通道求平均后代替独立显示。

## 11.5 右上象限：SRP-PHAT 360°输出

中心为标准数学角360°圆环，显示0/90/180/270°刻度、每30°小刻度和阵列中心。每个有效`SpatialResponse`执行一次绘制：

- 淡色闭合极坐标曲线显示全部360个`normalized_scores`，半径映射固定为`0.55R + 0.40R*score`，Norm=0/1分别位于0.55R/0.95R；
- 通过阈值、prominence和NMS后的`CandidateDirection`在外圆上绘制圆点；圆点直径固定映射`8 + 20*normalized_score`像素，颜色由蓝→黄→红表示0→1；
- 每个候选点旁标注`θ°`与`N=0.xxx`，按候选排序显示编号1～2；标签碰撞时使用引线自动错开，不隐藏角度或数值；
- 侧表列出Rank、Angle、Raw、Norm和当前阈值，并显示`0～3 candidates`；无候选时明确显示`NO CANDIDATE`；
- 原始分数只在侧表/光标悬停显示，圆环半径和颜色只使用Norm值，避免不同窗口raw scale造成误导。

点击候选点会设置全局`selected_candidate(theta, window_id)`，联动左下融合音频/谱图与右下CNN高亮；不重跑SRP、BF或CNN。

## 11.6 左下象限：三频段融合音频与Spectrogram

最多显示2个L2正式候选方向并与SRP rank顺序一致；L3不得自行截断更多候选。每个候选包含角度、算法、回退状态及48 kHz增强波形。

顶部固定包含：

```text
选中方向: xxx°   [融合音频试听] [播放/暂停] [停止]
[自动重播 On/Off] [跟随最新窗口 On/Off]   循环间隔: 100 ms
```

点击某行或SRP/CNN圆点时，播放器冻结该`window_id/theta/backend`的320 ms float32快照；播放过程中实时处理继续，但音频内容不被新窗口替换。`跟随最新窗口=Off`为默认。On时仅在当前一轮播放结束后原子换入同方向的最新完整快照，不在波形中途切换。`自动重播`默认On，播放到结尾后静音100 ms再从头循环；暂停保持当前位置，继续从当前位置播放，停止回到起点。

为防突发声，试听先去DC，再只对试听副本做峰值归一化到-6 dBFS并施加5 ms淡入/淡出；不得改写`DirectionalSignal`或训练特征。输出设备、主音量和静音状态必须可见，默认音量25%。播放/ISTFT在独立preview worker完成，不阻塞实时处理或GUI线程。

FeatureExtractor接入显示后，所选候选行下方展示其运行时后端生成的`SpectrogramFeature [33,169]`热力图；未接入显示前明确标记待实现。横轴为320 ms、纵轴80～8000 Hz，色标显示log-magnitude数值范围；标题包含`preprocessing_version`。该矩阵是工程特征旁路，不从试听归一化后的waveform重新计算，当前NVIDIA MarbleNet基准也不消费它；未来以该特征为输入的模型接入时才可称为对应CNN的真实输入。候选切换时必须与L5概率保持同一window。

## 11.7 右下象限：CNN逐方向概率与最终识别结果

使用与SRP完全相同的0°位置、逆时针方向、尺寸和选中状态。圆环只绘制`VoiceDetection`对应候选：圆点固定落在候选角外圆，直径`8 + 20*voice_probability`像素，旁标`θ°`与`P=0.xxx`。`P>=voice_probability_limit`使用绿色并附`VOICE`，否则灰色并附`NON-VOICE`；选中点增加白色描边。圆心显示：

```text
Voice directions: <voice_direction_count>
Threshold: 0.70
Model: <model_version>
```

侧表按SRP候选rank列出Angle、Norm、BF backend、Probability、Decision；这样可以直接比较“空间响应强但非人声”和“空间响应较弱但CNN判为人声”。空候选显示`NO CANDIDATE`，候选存在但模型不可用显示`MODEL UNAVAILABLE`，错误窗口不显示正式voice count。点击CNN点与SRP点使用同一全局选择并联动左下试听及Spectrogram。

测试UI右下象限必须提供可操作的阈值滑动条，范围`0.00～1.00`、步长`0.01`，启动值取当前运行配置的`voice_probability_limit`（默认`0.70`），并在滑动条旁实时显示两位小数。拖动时必须立即用已缓存的`voice_probability`重新计算屏幕上的`is_voice = probability >= threshold`、颜色、`VOICE/NON-VOICE`标签和方向点数量，不重跑L3、FeatureExtractor或CNN，也不得阻塞实时处理。当前运行配置阈值和临时UI阈值必须并列显示；临时值只属于本次测试UI会话，不写回`config.yaml`，除非用户执行独立的“保存为实验配置”流程。

## 11.8 分层启用与每层完成门禁

每写完一层，必须先通过单元/contract test，再在该UI完成对应门禁，结果保存为带时间、配置hash、git commit、设备、截图和操作者备注的`reports/dev_ui/<stage>/<run_id>/report.json`；截图只保存界面，不自动保存音频，音频仍遵循scratch规则。

| 阶段 | 必须可用的象限 | UI实机门禁 |
|---|---|---|
| L1 | 左上 | 灯开/关各10次均收到写入确认并由操作者目视灯光状态正确；7个meter逐通道tap对应正确；录制→暂停→继续→结束可播放且sample区间正确；再次点击录制后旧scratch不存在 |
| SRP-PHAT | 左上+右上 | 0/90/180/270°已知声源的圆点方向正确；Norm、角度、候选排序、空候选和STALE状态可见 |
| L3融合 | 左上+右上+左下 | 每个候选的融合音频与谱图同window；可播放、暂停、自动重播；三频段算法与fallback状态可见 |
| Spectrogram工程特征旁路 | 左上+右上+完整左下 | 热力图来自L3 `FeatureExtractor`而非试听副本，shape严格为`[33,169]`，版本和window可见；当前MarbleNet基线不消费该特征 |
| Voice / MarbleNet | 四象限+性能栏 | 每个候选角一一对应概率；右下阈值滑动条覆盖`0.00～1.00`、步长`0.01`且默认取运行配置值，拖动只基于已有概率即时刷新颜色/标签/count并证明没有重跑L3或MarbleNet；点选联动相同window的音轨与谱图；模型错误不显示假结果；性能栏显示Warmup、配置/实测采样率、Compute与Latency current/P50/P95，匹配的锁定test报告显示Pre/Recall/F1，不匹配时显示N/A及原因 |

以上人工门禁不能替代17节自动测试与量化验收；反之自动测试通过也不能省略真实硬件、声源方向、灯光、录音和试听检查。

## 11.9 测试UI错误与性能要求

- 任一绘图或播放异常只使对应象限进入Error，不停止采集和算法；全局致命输入错误仍按13节处理。
- UI只保存最多最新一帧和当前冻结试听快照；候选/window变化不得造成内存持续增长。
- 关闭测试UI时先停止播放并解除L3缓存映射，再finalize scratch writer，删除整个scratch与L3音频缓存目录，随后清空UI快照引用并释放UI订阅；灯光保持当前状态，不隐式发送开/关命令。删除范围不得包含Runtime Recording、Test Corpus或UI设置。
- 在m=5、全屏四象限、自动重播开启时，UI不得使算法processing latency p95增加超过2 ms，连续30分钟GUI内存增长不得超过100 MB且无未界定队列增长。
- 自动UI测试覆盖四象限等分、56 px单行性能栏、DPI缩放、按钮状态机、每次新录制清空scratch、角度映射、候选联动、window一致性、STALE、Mock水印、播放循环和错误恢复；性能指标测试必须覆盖epoch重置、实测采样率断流N/A、current/P50/P95滚动窗、`Latency >= Compute`、metrics匹配显示和不匹配清空。

---

# 12. 历史音频、测试音频与数据管理子系统

该子系统负责音频资产的存储、索引、统计、分类、标注、质量控制、回放和可视化管理。它独立于实时算法主链路；存储或UI故障不得阻塞USB采集与实时判断。

## 12.1 两类资产严格分开

### A. Runtime Recording / 实际运行音频

来源是系统正常运行时的连续输入，用于问题复现、回看、硬件诊断和候选样本发现。它没有天然ground truth，不得直接作为正式test set或CNN训练数据。

记录策略：

```yaml
recording.runtime:
  mode: manual                # off | manual | continuous | event
  chunk_seconds: 60
  record_native_8ch: true
  record_physical_7ch: true
  record_results_jsonl: true
  retention_days: 30
  max_storage_gb: 200
```

`event`模式只在有效人声判断触发时保存事件，触发条件以12.3固定规则为准，并额外保留事件前2秒和后3秒上下文；事件片段仍需引用原始session/sample范围。默认使用`manual`，避免未经用户选择长期录音。

每次source启动时RecordingStore随IngestCoordinator的`session_id`建立会话，但只有模式规则允许时才缓存或落盘。`manual`由用户执行Record/Pause，在同一source session中形成多个`recorded_intervals`；`continuous`从source可用到结束持续记录；`event`在内存中维护pre-roll并只落盘合并后的事件范围；`off`既不缓存pre-roll也不落盘。模式改变不生成新时间轴，source重启才生成新session。若整次session没有任何落盘区间，关闭时不创建空资产目录，只保留不含PCM的普通运行日志。

默认三份音频同时开启时，未计文件头的音频理论写入量为 `48000*(8*2 + 7*2 + 7*4)=2,784,000 bytes/s`，约10.0 GB/小时；50个判断/s的两个360点float32空间谱未压缩上界约0.52 GB/小时，总计约10.54 GB/小时，200 GB约可容纳19.0小时。实际NPZ压缩率不作为容量保证。`retention_days=30`是最长时间上限，不保证容量足以保留30天，实际清理按“超过天数或超过容量任一先发生”执行。UI必须显示实测写入速率、当前占用和按当前模式估算的剩余可录时长。

### B. Test Corpus / 专门测试音频

来源是受控专门录制、从Runtime Recording复制提升并完成人工标注、或外部数据导入。用于几何验证、DOA/BF评测、CNN训练/验证/测试。每条记录必须有标签、质量状态和来源追踪；没有完成最小元数据的资产只能进入`quarantine`，不能参与正式评测。

Runtime与Test在Catalog中使用不同实体类型、目录和生命周期规则；它们可以由同一个SQLite catalog统一索引，但不得共用资产记录或保留策略。把Runtime片段提升为Test时执行“复制 + 新ID + lineage引用”，不得移动或改写原始session。

外部导入音频若不是48 kHz、8ch原生格式或7ch物理格式，不得静默作为标准资产使用。Importer必须保留原文件及hash，在隔离worker中显式重采样/重排为新的derived asset，manifest记录原格式、转换工具版本、参数和父asset ID；无法确认通道物理意义的多通道文件进入quarantine。导入时必须填写license/consent及允许用途；权利不明、已过期或不允许ML训练的资产不能进入对应dataset split。

## 12.2 固定目录与文件标准

```text
data/
  catalog.sqlite
  runtime_sessions/
    YYYY/MM/<session_id>/
      session_manifest.json
      native_8ch/
        epoch000_start000000000000_end000002880000.wav
      physical_7ch/
        epoch000_start000000000000_end000002880000.wav
      physical_7ch_float/
        epoch000_start000000000000_end000002880000.npy
      results/
        epoch000_start000000000000_end000002880000.jsonl
      spatial_response/
        epoch000_start000000000000_end000002880000.npz
      hotmaps.jsonl
      events.jsonl
      diagnostics.jsonl
  test_corpus/
    <dataset_id>/
      dataset_manifest.json
      recordings/<recording_id>/
        recording_manifest.json
        native_8ch.wav
        physical_7ch.wav
        physical_7ch_float.npy
        annotations/
          v0001.jsonl
        evaluation_references/
        qa_report.json
  quarantine/
  trash/
```

文件标准：

| 资产 | 格式 | 要求 |
|---|---|---|
| 原生设备音频 | WAV PCM S16_LE, 8ch, 48 kHz | 保留设备字节语义，不做归一化或校准 |
| 物理音频便携副本 | WAV PCM S16_LE, 7ch, 48 kHz | 固定 `[Ring0..Ring5,Center]` 顺序；仅用于便携播放 |
| 算法精确副本 | NPY float32, shape `[N,7]` | C-contiguous，保留校准后的精确float值 |
| 实时结果 | UTF-8 JSONL | 首行为chunk header，随后每行一个`DecisionResult`摘要或drop记录，不写大数组 |
| 历史空间谱 | NPZ | float32 `raw_scores[K,360]`、`normalized_scores[K,360]`及对应window/sample索引 |
| 元数据 | UTF-8 JSON | `schema_version`必填 |
| 标注 | UTF-8 JSONL | sample级半开区间和版本化标签 |

所有完成写入的音频和manifest计算SHA-256；manifest自身hash写入同目录`manifest.sha256`侧车文件，避免自引用。WAV与NPY必须记录sample count、channel count、sample rate、dtype、epoch、起止sample和校准配置hash。文件名包含epoch和零填充起止sample，但不使用时间作为唯一身份；所有session、dataset、recording、annotation使用UUID。

原生8ch WAV由`native_samples`执行固定 `np.rint(np.clip(x,-1,32767/32768)*32768).astype('<i2')`；对设备S16输入必须bit-exact往返。校准后7ch可能超出S16范围，便携WAV按同一规则饱和并记录每通道clip count，算法复现与训练只以float32 NPY为权威。`hotmaps.jsonl`按CDC `sequence_id`去重，记录CDC timestamp、received_at、关联音频 `(epoch,start_sample)` 和16×16 uint8值；没有硬件同步时只表示“该音频块离开L1时的最新快照”，不得当作精确同步真值。

## 12.3 写盘与一致性

```python
class RecordingStore(Protocol):
    def start_session(self, session_id: str, metadata: SessionMetadata) -> None: ...
    def set_recording_mode(self, mode: str) -> None: ...
    def append_audio(self, block: IngestedAudioBlock) -> None: ...
    def append_result(self, result: DecisionResult) -> None: ...
    def advance_result_watermark(self, watermark: ResultWatermark) -> None: ...
    def append_result_with_watermark(self, result: DecisionResult,
                                     watermark: ResultWatermark) -> bool: ...
    def trigger_event(self, result: DecisionResult, reason: str) -> None: ...
    def stop_session(self, reason: str) -> SessionManifest: ...

class CorpusStore(Protocol):
    def import_recording(self, source: ImportSource, metadata: RecordingMetadata) -> str: ...
    def promote_runtime_segment(self, session_id: str, stream_epoch: int,
                                start_sample: int, end_sample: int,
                                metadata: RecordingMetadata) -> str: ...
    def add_annotations(self, recording_id: str,
                        annotations: tuple[Annotation, ...]) -> None: ...
```

JSONL行schema固定为：

```yaml
schema_version: decision_record_v3
session_id: uuid
stream_epoch: int
window_id: int
decision_sample: int
doa_range: [start, end]
context_range: [start, end]
status: ok | degraded | error
candidates: [{theta_deg, raw_score, normalized_score}]
detections: [{theta_deg, beamformer_backend, model_version, voice_probability, is_voice}]
voice_direction_count: int
diagnostics: [string]
processing_latency_ms: float
stage_statuses: {l2: string, l3: string, l5: string}
stage_timings_ms: {l2: float, l3: float, l5: float}
stage_queue_wait_ms: {l2: float, l3: float, l5: float}
terminal_reason: string | null
```

文件首行固定为`{"record_type":"chunk_header","schema_version":"decision_chunk_v1",...}`并包含session、epoch、start/end sample、config hash；普通结果行增加`record_type: decision`，主动丢弃增加`record_type: dropped_window`及window/sample/reason。这样空结果chunk仍是合法JSONL，不使用注释行或另一种容器格式。

空间谱NPZ逐chunk封闭，键固定为`window_ids int64[K]`、`decision_samples int64[K]`、`raw_scores float32[K,360]`、`normalized_scores float32[K,360]`；`theta_degrees`不重复保存，固定由`common.angle`生成0..359。四个数组首维必须一致，window ID必须与JSONL中带`SpatialResponse`的行一一对应；error窗口没有空间谱时在JSONL保留错误记录而NPZ不增加行。

音频与判断结果是两个异步流，按 `(session_id, stream_epoch, sample)` 关联；不得假设一个L1块同步对应一个判断结果。当前ApplicationRuntime每有序完成、报错、超时、丢弃或取消一个窗口后，调用`append_result_with_watermark`把DecisionRecord和同窗ResultWatermark作为**一条结果队列命令**原子接纳。队列已满时两者均不入队，生产者watermark不前进，并记录`result_overflow`缺口。兼容的分开append接口不是当前Runtime正式提交路径。watermark不可倒退或跨epoch。写盘线程与实时采集线程隔离，音频队列和结果队列分别有界，默认音频队列容量为10秒、结果队列容量为256条，且结果队列schema硬上限同样为256条。写盘不得修改传入数组；音频队列满时记录`recording_overflow`和缺失sample区间，实时主链路继续运行。每60秒切块，切块边界必须落在960-sample判断hop上。普通WAV、NPY、noise NPZ与IMCRA NPZ整批在首次final改名前写入`chunk_asset_commit_<stem>.json`事务journal；只有manifest已按hash完整索引整批资产才视为提交，否则恢复时将整批partial、未完整索引的final和该journal转入quarantine。程序启动时也会校验崩溃遗留的open manifest，并将它原子改写为可审计的incomplete状态。

事件录音的pre-roll由RecordingStore自己的至少2秒原生8ch与物理7ch环形缓存提供，不得回读算法WindowAssembler或阻塞实时处理。event触发条件固定为：任一`DecisionResult.status in {ok,degraded}`且`voice_direction_count>0`时触发`voice_detection`；如果没有可用CNN模型，event模式不得退化为“有SRP候选即录音”，而是显示`event_unavailable/model_unavailable`。同一session/epoch中后一个事件的pre-roll起点不晚于当前事件post-roll终点时合并为同一连续事件资产并延长终点；跨epoch或不重叠触发新建事件段。manifest `event_triggers`每个合并事件段只保留一条有界审计：兼容字段`window_id/decision_sample/reason`代表首触发，并记录`first_window_id/last_window_id`、`first_decision_sample/last_decision_sample`、`start_sample/end_sample`和`trigger_count`，不保存随50 Hz增长的逐窗ID列表。容量扫描只在新事件段开始前执行，合并触发在锁内O(1)扩展；若触发在旧post-roll结束后但其pre-roll仍相交，必须从2秒ring补回中间音频，保证事件资产连续。

同一chunk的native、physical和NPY使用完全相同的 `(session_id, stream_epoch, start_sample, end_sample)` 与frame count。epoch改变时立刻封闭旧epoch当前chunk，新epoch从sample 0建立新chunk；文件名必须包含epoch与起止sample。结果按 `start_sample < decision_sample <= end_sample` 归入该chunk；结果可以为空，其JSONL header仍须记录同一chunk范围。chunk在算法watermark达到其`end_sample`或source结束后才封闭JSONL和空间谱NPZ；watermark列出的dropped窗口写为显式缺口记录。音频范围或frame count不一致标记`corrupt`；缺失结果只标记`result_incomplete`，不把完整音频误判为损坏。

已关闭音频chunk的writer watermark到达`end_sample`时，RecordingStore立即写出该chunk的JSONL/NPZ/sidecar并从RAM释放该范围结果，不累积到session结束。event模式的音频和结果pre-roll均按sample裁剪为最新2秒；水位停滞时的待保留结果仍受硬数量上限保护。Hotmap按sequence去重后流式写入`hotmaps.jsonl.partial`，不保留session级矩阵列表。普通chunk资产使用`chunk_asset_commit_<stem>.json`保护整批改名与manifest索引边界；增强WAV在对应音频区间可写时先落到`.partial`并从RAM释放，session封存另以`enhanced_asset_commit.json`事务journal保护改名与manifest的原子边界。恢复时，manifest已完整索引且hash匹配的终态视为已提交；其他journal所指partial或孤立终态全部进入quarantine。

开始录音前要求可用空间至少5 GB。达到`max_storage_gb`或剩余空间低于5 GB时，先按“已过retention、未pin、未提升、未被实验引用、最旧优先”清理Runtime资产到Trash；清理后仍不足则安全停止录音、完成当前可flush chunk并发出`recording_storage_full`，实时算法继续运行。Test Corpus、实验引用和pin资产永不自动清理。

## 12.4 Catalog与元数据模型

`data/catalog.sqlite`是查询索引，不是唯一真相；每个资产目录中的manifest是可移植事实来源。启动时可以从manifest重建catalog。

最少表：

```text
sessions
audio_assets
datasets
recordings
annotations
quality_checks
tags
recording_tags
experiments
experiment_items
asset_lineage
audit_log
schema_migrations
```

所有表包含UUID、created_at、updated_at和schema_version。Catalog使用WAL模式；数据库变更通过事务完成。不得把整段PCM存入SQLite BLOB。

`SessionManifest`最少字段：

```yaml
schema_version: audio_session_v1
session_id: uuid
started_at_utc: ISO-8601
ended_at_utc: ISO-8601 | null
stop_reason: user | normal | crash_recovered | error
device_format: {sample_rate, channels, pcm_format, layout}
physical_channel_map: [0,1,2,3,4,5,7]
calibration_hash: sha256
geometry_version: r6plus1_physical_v1
config_hash: sha256
git_commit: string | null
runtime: {cpu, gpu, driver, torch, cuda}
algorithm_versions: {srp_phat, beamformer, preprocessing, model, model_artifact_hash}
recording_mode_history: []
recorded_intervals: []
chunks: []
missing_intervals: []
result_gaps: []
```

`RecordingManifest`最少字段：

```yaml
schema_version: test_recording_v1
dataset_id: uuid
recording_id: uuid
capture_session_id: uuid
source_type: dedicated | promoted_runtime | imported
lineage: {session_id, stream_epoch, start_sample, end_sample} | {source_uri, source_hash}
capture_time_utc: ISO-8601
environment_id: string
room_id: string
array_pose_id: string
source_count: int
source_categories: [human_voice | music | tv | fan | impact | ambient | other]
known_theta_degrees: [float] | null
distance_m: [float] | null
speaker_ids_anonymous: [string]
language_tags: [string]
rights: {consent_status, license_id, allowed_uses, expires_at_utc}
snr_db: float | null
quality_status: pending | passed | failed | quarantine
split: train | validation | test | calibration | unset
assets: []
evaluation_references: []
annotation_version: string
```

speaker ID只能是数据集内匿名ID，不是系统输出身份。

Runtime状态机固定为 `open -> finalizing -> complete | incomplete | corrupt -> trash -> purged`；Test recording状态机固定为 `quarantine | draft -> annotated -> qa_passed | qa_failed -> versioned -> trash -> purged`。锁定dataset version中的recording不可原地修改或进入Trash，必须先创建不含它的新版本。所有状态改变先写资产目录内追加式`audit.jsonl`，再完成文件fsync和manifest原子替换，最后用SQLite事务更新Catalog；Catalog失败不回滚已完成资产，标记`catalog_reconcile_required`并从manifest/hash重建索引。不得先提交Catalog再留下旧manifest。

## 12.5 标注标准

```python
@dataclass(frozen=True, slots=True)
class Annotation:
    annotation_id: str
    recording_id: str
    start_sample: int
    end_sample: int
    label_type: str       # voice_activity | source_direction | noise_event | exclusion
    label: str
    theta_deg: float | None
    confidence: float     # annotator confidence [0,1]
    annotator: str
    annotation_version: str
```

所有标注使用对应recording自身的48 kHz、从0开始的局部sample index和半开区间；对于promoted Runtime资产，manifest中的lineage另外保存原`(session_id, stream_epoch, start_sample, end_sample)`，换算固定为`local_sample = source_sample - lineage.start_sample`。方向标签使用算法内部真实物理`theta_deg`，不得使用UI显示角。人工标注必须保留修改历史，不覆盖旧版本。

L5样本构造必须通过锁定版本的WindowAssembler、L2和L3多频段增强离线运行，保存与在线相同的48 kHz、320 ms未做播放器归一化的方向波形；MarbleNet适配器的16 kHz重采样和80维log-mel前处理必须与推理共用同一实现。`SpectrogramFeature [33,169]`可随样本保存用于工程分析，但不是当前MarbleNet输入。每个20 ms endpoint的方向集合为“L2实际候选 + 人工指定评测方向”，并保存候选角、BF后端、预处理版本及全部artifact hash。

CNN标签只判断该方向增强后的实际音频是否含人声，与该点离真实声源角多远无关。标注员对盲化后的增强音频做sample级`voice_activity`：320 ms上下文中人声有效sample占比 `>=0.50` 为`voice`，`<=0.10`为`non_voice`，中间为`ambiguous`并默认不进入训练。正式test gold label要求两名标注员独立标注；标签不一致或边界差超过20 ms时由第三人审核。若有时间对齐的独立voice/non-voice component reference，可用相同BF权重生成辅助标签，但正式test仍需人工抽检至少20%。旁瓣、反射和非目标方向泄漏只作为分析tag；只要增强后仍有人声，就不能作为hard negative。

> **待修改（暂不执行）**：L5在线推理已改为“320 ms内出现约60 ms可信连续人声片段即可形成高窗口概率”，而本段训练数据规则仍按人声有效sample占比`>=0.50`标记`voice`，两者语义尚未对齐。现阶段保持既有录音、存储、标注和split不变；在启动下一轮L5训练数据生成前，必须单独评审并版本化修改标签阈值、`ambiguous`边界、历史标注迁移策略及对应dataset version，禁止静默重标已有语料。

DOA测试记录必须具有受控真实`theta_deg`或可靠外部标定。没有角度真值的运行音频不能计入DOA误差统计。

## 12.6 专门录制流程

测试录制向导至少支持：

1. 选择数据集、房间、环境、阵列姿态和声源类别。
2. 记录参与者同意、数据允许用途、保留期限和匿名speaker ID；不同意或权利状态不允许时禁止开始正式录制。
3. 进行通道健康检查：静音、削波、DC、重复通道、极性和固定延迟。
4. 选择或输入真实`theta_deg`、距离、声源数量和匿名speaker ID。
5. 由操作者开始、暂停/继续和结束目标录音；不预设固定时长，也不强制采集固定的前后环境噪声段。
6. 自动保存7个物理麦克风的独立单通道WAV、录制标签、manifest与SHA-256；当前专用向导不生成伪native 8ch或float32副本。
7. 自动QA；失败进入quarantine，用户修正元数据后重新审核。
8. 人工检查波形、通道、播放和标注。
9. 按分组规则预览dataset split并在确认后锁定版本。

建议角度覆盖0～350°每10°至少一条；单声源、双声源和无人声分别建场景。距离至少覆盖0.5 m、1 m、2 m；环境覆盖安静、风扇/空调、音乐/电视和混响。以上是采集计划，不改变算法接口。

## 12.7 QA与统计

每个音频资产自动计算：

- 时长、sample数、缺失区间和文件hash；
- 每通道RMS、peak、DC offset、clipping比例、silence比例；
- 通道相关矩阵、疑似重复通道、极性异常和固定延迟；
- native/physical时间对齐和映射一致性；
- 标签时长、类别、角度、距离、环境、说话人和split分布；
- voice/non-voice/ambiguous时长；
- 训练/验证/测试的room、session、speaker泄漏检查；
- 数据集版本、manifest完整率和QA通过率。

默认QA阈值：clipping比例 `<=0.1%`、绝对DC `<=0.02`、非静音通道RMS `>-60 dBFS`、所有文件hash匹配、frame count一致。阈值失败不自动删除文件，只把状态设为`failed`或`quarantine`。

数据集split不是简单拼接三个字符串。先建立recording图：任意两条记录只要共享非空的`capture_session_id`、`room_id`或`anonymous speaker_id`之一就连边，再以连通分量为不可拆分group，按固定seed 42进行分层贪心分配，使总时长尽量接近70/15/15并保持voice/non-voice、角度和环境分布；没有speaker的无人声记录仍按capture session与room连接。分配器必须输出实际比例和偏差，任一split与目标时长比例相差超过5个百分点时失败并要求补录，不得拆开group凑比例。正式test split锁定后不可在UI中直接改写，只能创建新dataset version。

`dataset_manifest.json`包含递增语义版本和全部recording ID/hash。任何增删、标签修订、QA状态或split变化都创建新dataset version；已经被实验引用的版本不可变。统计和模型训练必须记录精确dataset version，不能只记录可变名称。

## 12.8 可视化管理UI

GUI增加独立“Audio Data Manager”，至少包含：

```text
Runtime Sessions
Test Corpus
Recording Wizard
Annotations
Quality Control
Statistics
Storage
Experiments
```

当前基础版已把上述职责按“操作首页、运行录音、测试语料库、测试录制向导、质量与标注、系统维护”六个中文任务页实现，并接入独立UAC采集主机、manual/continuous/event控制、导入导出、QA、版本化标注、split预览/锁定、Trash恢复、实验快照和选定7通道样本注入Development Test UI；相应自动测试已经存在。运行录音片段提升仍缺少波形范围选择器，完整多通道波形/频谱/相关矩阵和统计图可视化、真实硬件UI报告仍未完成，因此这里的完成标记只限定为架构图所写的“基础版与自动测试”，不代表本节全部实机门禁通过。

必需功能：

- 按日期、session、数据集、类别、角度、距离、房间、QA、split和tag筛选；
- 查看8ch/7ch波形、每通道电平、频谱、通道相关矩阵和360°结果时间线；
- 选择单通道、Center、MVDR或DAS播放，播放路径不改变源文件；
- sample级选择、打标签、批量tag、注释版本历史和审核；
- 专门测试录制向导及实时通道健康状态；
- Runtime片段复制提升为Test记录，并显示lineage；
- 数据集统计图、类别/角度覆盖图、split泄漏警告和QA报告；
- 存储占用、保留期限、孤立文件、损坏文件和`.partial`恢复状态；
- 当前写入速率、按已选资产格式估算的每小时容量和剩余可录时长；
- 创建实验快照：固定dataset version、配置hash、模型版本和recording IDs；
- 导入、导出manifest与选定音频资产。

录音状态必须始终在主窗口显示可见的 `REC / Event / Off` 指示、当前session、持续时间和磁盘余量；任何录音开始都必须来自用户显式选择或已显示并启用的event规则，不允许隐藏启动。默认仅本机访问，音频与匿名speaker元数据不得自动上传；若以后增加网络共享或云同步，必须作为独立功能重新定义鉴权、传输加密、访问日志和用户同意，本规格不默认授权。

删除是破坏性操作：默认先移入`data/trash/<operation_id>`并写审计记录，UI二次确认后才执行；Test Corpus资产和锁定test split需要再次输入数据集名称确认。自动清理只作用于超过保留期且未pin、未提升、未被实验引用的Runtime Session。Trash默认保留7天。

UI所有重扫描、hash、统计、波形降采样和导入任务在后台worker执行，不阻塞实时音频线程。UI只读取Catalog和不可变结果；写操作经DataManager service事务提交。

## 12.9 数据管理验收

- 正常结束、强制终止和磁盘空间不足后，已有完成chunk均可恢复和校验。
- 60秒chunk的native、physical、NPY和results sample范围完全一致。
- 从Runtime提升到Test后，源session hash不变且lineage可双向查询。
- Catalog删除后能从manifest完整重建。
- 1万条recording索引下常用筛选查询p95 `<=500 ms`；波形预览通过多分辨率缓存加载。
- split leakage检查对构造的session/room/speaker泄漏100%报警。
- UI删除进入Trash且可恢复；只有显式永久删除才不可恢复。
- 实时运行开启录音时，写盘不得使算法processing latency p95增加超过2 ms。

---

# 13. 状态、错误与降级

统一规则：

| 条件 | 行为 |
|---|---|
| 上下文不足 | 只发布`PipelineStatus(warming_up)`，不生成DecisionResult |
| 无L2候选 | `ok`，空detections，count=0 |
| 输入shape/rate/非有限错误 | 当前窗口`error`，清空candidates/detections/count，不得继续 |
| 任一stage失败/超时/丢弃/取消 | 整窗`error`，保留已完成上游作诊断，明确stage终态与`terminal_reason` |
| sequence/timestamp断裂 | 清空buffer并重新warming_up |
| MVDR失败 | 该候选DAS回退，结果`degraded` |
| 启动时CUDA不可用 | 环境自检失败；开发模式按配置允许CPU并显示`degraded/cpu_fallback`，生产验收失败 |
| 运行中CUDA OOM | 清空本窗口GPU临时状态，最多3个候选的正式batch失败时返回明确错误或按该层规格降级；不得静默截断候选 |
| CNN模型/manifest不匹配 | 生产模式`error`，保留候选作诊断但detections/count清空 |
| 单候选CNN输出非有限 | 整个窗口`error`，detections/count清空，不得阈值化NaN或输出半截正式结果 |

每个fallback必须写入 `diagnostics`。同一窗口不得混合GPU产生的部分候选与CPU重算的部分候选：若OOM重试仍失败，整窗口从共享输入重新计算，保证candidate顺序和结果原子性。**成功产生完整正式输出的回退**才为`degraded`；回退也失败时为`error`且detections/count清空。日志必须包含session id、window id、sample边界、后端、模型/预处理版本和耗时，禁止记录整段音频数组。

其中“上下文不足”通过 `PipelineStatus` 表达，不生成空的正式 `DecisionResult`。

---

# 14. 配置唯一来源

项目只允许一个运行时配置入口 `config/config.yaml`，由PyYAML `safe_load`读取后交给拒绝未知字段的类型化schema；schema版本不支持、缺字段、未知字段、类型/range错误均启动失败。录音`chunk_seconds/audio_queue_seconds/result_queue_capacity/retention_days/max_storage_gb`必须大于0，`min_free_storage_gb`必须非负且严格小于`max_storage_gb`，其中`result_queue_capacity`默认为256且不得大于256；不得以零容量队列或非法存储预算启动。第14节代码块就是必须创建的完整默认文件，不是示例。代码不得再维护第二套业务默认值；类型化schema只做解析与校验，测试和所有入口都加载该文件。`pyproject.toml`将`config`列入包发现并把`config.yaml`作为package data打入wheel，因此源码树与分发包携带同一配置资产，不复制第二份默认值。现有Layer 1环境变量只允许覆盖部署绑定字段`device.device_name`、`device.host_api`、`device.serial_port`、`device.serial_required`和`device.light_service_url`，覆盖后必须打印来源并写入session manifest；采样率、通道数、shape、几何、时间、算法参数及路径禁止被环境变量覆盖。未知环境变量不影响配置。禁止在模块中散落magic numbers。

```yaml
schema_version: project_config_v2

paths:
  data_root: data
  models_root: models

hardware:
  physical_mic_count: 7
  ring_radius_m: 0.04
  speed_of_sound_mps: 343.0
  geometry_version: r6plus1_physical_v1
  hardware_calibration_status: unverified
  hardware_calibration_report_hash: null

device:
  sample_rate: 48000
  device_channels: 8
  pcm_format: s16-le
  layout: interleaved
  block_size_samples: 960
  physical_channel_map: [0, 1, 2, 3, 4, 5, 7]
  device_name: MicArray
  host_api: Windows WDM-KS
  serial_enabled: true
  serial_port: COM5
  serial_baud: 115200
  serial_required: false
  light_service_url: null

calibration:
  gains: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  polarity: [1, 1, 1, 1, 1, 1, 1]
  delay_samples: [0, 0, 0, 0, 0, 0, 0]

timing:
  decision_hop_samples: 960
  doa_window_samples: 1920
  context_samples: 15360
  timestamp_tolerance_ms: 5

layer1_noise_recording:
  enabled: true
  estimator: mcra_record_v1
  n_fft: 2048
  smoothing: 0.90
  noise_smoothing: 0.80
  minimum_history_frames: 75
  presence_ratio_low: 1.5
  presence_ratio_high: 4.0
  floor: 1.0e-12

layer2:
  probability_gate:
    backend: mean_2x20ms_v1
    threshold: 0.60
  scanner_backend: srp_phat
  angle_step_deg: 1.0
  frequency_min_hz: 2000.0
  frequency_max_hz: 4000.0
  n_fft: 2048
  window: hann_periodic
  remove_channel_mean: true
  gcc_interpolation: 16
  phat_epsilon: 1.0e-12
  normalization_backend: robust_z_sigmoid
  normalization_alpha: 1.0
  normalization_beta: 2.0
  direction_threshold: 0.35
  peak_prominence: 0.05
  min_peak_distance_deg: 45.0
  max_candidates: 3
  iterative_peak_search_enabled: false
  iterative_max_sources: 2
  iterative_suppression_strength: 0.75
  iterative_phase_power: 2.0
  iterative_pair_phase_threshold: 0.60
  iterative_min_pair_support: 6
  iterative_min_frequency_support: 24
  iterative_min_remaining_weight_ratio: 0.10
  iterative_min_residual_peak_ratio: 0.20

stft:
  n_fft: 1024
  win_length: 960
  hop_length: 480
  window: hann_periodic
  center: true
  pad_mode: reflect
  normalized: false
  onesided: true
  return_complex: true

layer3:
  main_backend: frequency_hybrid
  baseline_backend: das
  fallback_backend: das
  covariance_estimator: context_sample_covariance
  loading_retry_factors: [0.001, 0.01, 0.1]
  solve_dtype: complex64
  low_frequency_min_hz: 80.0
  low_frequency_max_hz: 500.0
  robust_frequency_max_hz: 2000.0
  high_frequency_max_hz: 8000.0
  crossover_width_hz: 100.0
  robust_noise_model: diffuse_sinc
  robust_wng_floor_db: -3.0

feature:
  preprocessing_version: voice_logmag_v1
  frequency_min_hz: 80.0
  frequency_max_hz: 8000.0
  first_bin: 2
  last_bin_inclusive: 170
  log_epsilon: 1.0e-6
  normalization: training_set_per_frequency_zscore
  expected_shape: [33, 169]

layer5:
  primary_model_id: nv_marblenet_baseline_v1
  models:
    - model_id: nv_marblenet_baseline_v1
      backend: nvidia_marblenet_window_v1
      model_artifact: models/nv_marblenet_baseline_v1
      role: primary
      enabled: true
  allow_mock: false
  voice_probability_limit: 0.70

runtime:
  mode: development
  preferred_device: cuda
  allow_cpu_fallback: true
  max_candidate_batch: 3
  capture_handoff_blocks: 100
  processing_queue_windows: 1  # legacy launch-profile compatibility only
  l2_queue_windows: 10000
  l3_queue_windows: 10000
  l5_queue_windows: 10000
  completion_queue_windows: 8
  max_inflight_windows: 30003
  compute_cache_max_bytes: 67108864
  overflow_policy: drop_oldest
  graceful_shutdown_timeout_seconds: 10.0

dev_test_ui:
  start_fullscreen: true
  stale_after_ms: 500
  performance_bar_height_px: 56
  performance_refresh_hz: 2
  performance_window_count: 500
  sample_rate_window_seconds: 5
  l1_meter_refresh_hz: 25
  polar_refresh_hz: 20
  waveform_refresh_hz: 10
  snapshot_mailbox_capacity: 1
  scratch_root: data/dev_test_ui/scratch/current
  autoplay: true
  loop_gap_ms: 100
  follow_latest_window: false
  preview_volume: 0.25
  preview_peak_dbfs: -6.0
  preview_fade_ms: 5
  default_selected_backend: mvdr

recording:
  runtime:
    mode: manual
    chunk_seconds: 60
    audio_queue_seconds: 10
    result_queue_capacity: 256
    record_native_8ch: true
    record_physical_7ch: true
    record_physical_float32: true
    record_results_jsonl: true
    record_spatial_response: true
    record_hotmaps: true
    record_noise_spectrum: true
    retention_days: 30
    max_storage_gb: 200
    min_free_storage_gb: 5
  event:
    pre_roll_seconds: 2
    post_roll_seconds: 3
  trash_retention_days: 7

privacy:
  local_only: true
  automatic_upload: false
```

字段交叉约束至少包括：`physical_mic_count == len(physical_channel_map) == len(gains) == len(polarity) == len(delay_samples) == 7`且map索引唯一并位于0..7；gain为finite且`>0`，polarity只能为±1，delay为非负整数且`min(delay_samples)==0`；`context_samples >= doa_window_samples`；三种sample数均与48 kHz换算整数毫秒且`context_samples % decision_hop_samples == 0`；STFT满足`hop_length<=win_length<=n_fft`并由默认参数严格产生33帧；`max_candidate_batch>=max_candidates`；L3频率边界严格递增且`high_frequency_max_hz<=sample_rate/2`；feature bins与STFT频率范围、`expected_shape`逐值吻合；三个backend固定为`frequency_hybrid/das/das`；所有容量和刷新率为正数；路径规范化后必须位于授权根目录；`hardware_calibration_status=verified`时report hash必须是SHA-256，否则必须为null；production时`allow_mock=false`、CUDA与模型必需且校准状态为verified。任何交叉约束失败均启动失败。

配置hash固定为有效配置规范化JSON的SHA-256：递归按key字典序、UTF-8、无多余空白、保留JSON数值/布尔类型，并包含部署覆盖后的有效值。参数实验不覆盖`config/config.yaml`：实验配置固定存入`config/experiments/<experiment_id>.yaml`，必须是完整配置而非局部patch，并在报告中保存父config hash和自身hash；启动时只有显式`--config`才能选择实验配置，生产入口未显式指定时永远加载根默认配置。调参不阻塞初版实现，因为本节所有字段已有可运行默认值。

`hardware_calibration_status`启动默认必须为`unverified`且`hardware_calibration_report_hash=null`；完成17.2实机四轴向、通道、极性与固定延迟测试并保存校准报告SHA-256后，二者才允许一并改为`verified`和对应hash。未验证状态仍可运行合成/开发流程，但GUI必须显示“物理角未校准”，正式实机验收失败。相对路径全部以项目根目录解析，生产启动时创建并记录规范化绝对`data_root`，禁止把数据写到桌面或当前随机工作目录。

校准处理顺序固定为：设备S16解码为未校准float32 `native_samples [N,8]` → 按`physical_channel_map`选取7ch → 逐通道乘`gains*polarity` → 使用每通道非负`delay_samples`与跨块history对齐 → 输出`samples [N,7]`。delay语义为“给该通道额外延迟多少sample”；一组相对延迟必须整体减去最小值使`min=0`后写入配置。epoch/设备重启时必须清空校准history；正常连续块之间不得清空。默认全1/0仅表示未补偿，不表示硬件已验证。任何校准数组变化都改变config/calibration hash并要求重新执行几何与DSP实机门禁。

---

# 15. 项目结构与职责

当前新项目结构如下；`legacy_reference_only/`是不可导入的旧项目与厂商资料快照，不计入当前完成度：

```text
common/
  data_types.py
  geometry.py
  angle.py
  config.py
app/
  runtime.py                         # Development Test UI共用完整L1→L5链路
  # main.py尚未实现
ingest/
  coordinator.py
  fanout.py
windowing/
  assembler.py
layer1_input/                        # 当前采集、解码、校准、CDC与噪声记录实现
layer2_source_detection/
  interface.py
  pipeline.py
  probability_gate.py
  srp_phat.py
  candidates.py
  iterative.py
layer3_direction_signal/
  interface.py
  engine.py
  shared_stft.py
  steering.py
  das.py
  mvdr.py
  hybrid.py
  feature.py
layer5_voice_classifier/
  contracts.py
  engine.py
  marblenet.py
models/
  nv_marblenet_baseline_v1/         # manifest、safetensors与上游来源快照
gui/
  dev_test_ui/
    app.py
    contracts.py
    aggregator.py
    panels.py
    audio_id_tracker.py              # v0.2 UI方向追踪/预测试听；v0.3迁移时移除方向职责
    scratch_recorder.py
    preview_player.py
  production_ui/
    app.py                           # Audio Data Manager六页界面
    capture_host.py                  # 独立UAC录音主机；不运行L2→L5
data_management/
  service.py
  recording_store.py
  corpus_store.py
  catalog.py
  manifests.py
  annotations.py
  qa.py
  statistics.py
  retention.py
  export.py
  experiments.py
  dedicated_recording.py
data/
  external_sources/                 # L5公开bootstrap来源、许可、hash与清单
tests/
config/config.yaml
requirements.lock
requirements-lock.in
requirements-vscode.txt
ENVIRONMENT.md
scripts/
  setup_vscode_env.ps1
  check_runtime_env.py
  run_audio_data_manager.py
  acquire_l5_bootstrap_data.py
.vscode/
  settings.json
  project.env
  extensions.json
  tasks.json
  launch.json
third_party/NOTICE.md
```

每层README必须引用本规格并说明职责、输入、输出、默认后端、替换方法和错误行为；不得重复定义另一套shape、角度或时间规则。

---

# 16. 实施顺序

1. 以第14节逐值创建唯一`config/config.yaml`及拒绝未知字段/交叉约束的schema，完成`pyproject.toml`与hash lock；随后固化 `common` 数据类型、真实几何、IngestCoordinator和WindowAssembler，并把现有Layer 1部署变量接入同一配置加载器。
2. 实现Development Test UI外壳、四象限、快照聚合器、L1左上象限、灯控、scratch录音状态机；完成L1 UI门禁。
3. 在已完成的L2几何、40/20 ms、Robust归一化和Top-3之后增加可选私有ID追踪与圆周卡尔曼；公共候选不输出ID，成熟轨迹可按当前`SpatialResponse`合规续报。删除Test UI预测方向L3旁路，保留不改角度、不触发额外波束形成的纯试听ID/cache sidecar，完成新的SRP/平滑UI门禁。
4. 实现共享STFT、低频DAS、中频WNG约束超指向MVDR、高频自适应MVDR及平滑融合；接入左下融合音频试听。固定`[33,169]`工程FeatureExtractor已经接线，热力图显示仍待完成。
5. 完成L3 CPU与运行时自动测试；CUDA交叉验证、OOM降级和真实UI门禁仍需补齐。
6. 固化L5波形公共接口和插件架构，接入带hash的NVIDIA MarbleNet基准artifact、CPU推理、运行时与右下象限。
7. 完成目标R6+1数据采集、目标域微调、窗口校准、锁定test指标、实际MarbleNet CUDA一致性和production门禁；这些工作当前尚未完成。
8. 实现RecordingStore、CorpusStore、manifest、catalog、QA和统计；验证崩溃恢复与lineage。
9. Audio Data Manager、独立UAC录音主机、六页数据管理界面及自动测试已经实现；最终人声方向production GUI和`app.main`入口仍待实现，显示角度映射最后在 `UiAngleMapper` 决定。
10. 完成实机性能、真实房间、数据管理、测试UI和30分钟稳定性验收。

每一步必须先通过本层单元测试和上下游contract test，再进入下一步。

## 16.1 每一步统一完成定义

任一步只有同时满足下列条件才标记完成：

1. 该步目标代码、类型、配置和迁移删除项全部落地；不存在仍可被生产入口导入的旧公共Tracking、Test UI方向滤波或第二套配置；L2内部DirectionSmoother除外。
2. 本层单元测试、上下游contract test、CPU参考测试及该层适用的CUDA测试全部通过；测试必须从根目录专用`.venv`执行。
3. 第11节对应能力级别的真实UI门禁通过并生成报告；尚无真实模型时只能完成Mock接口阶段，不能宣称CNN生产完成。
4. 新增第三方代码已在`third_party/NOTICE.md`记录不可变commit、license与适配文件；新增模型/数据均有manifest和hash。
5. 相关根规格、唯一配置和本层README保持一致；不存在TODO/TBD或需要实现者自行选择的关键默认值。
6. 性能、错误、降级和资源释放路径已测试；失败不得留下半成品正式结果或无限队列。

步骤间允许为下一步预留接口或显示`NOT IMPLEMENTED`，但不得提前把预留入口计入完成范围。

---

# 17. 强制测试与量化验收

## 17.1 Contract tests

- 所有类型shape、dtype、finite、范围和只读约束。
- `session_id`、`stream_epoch`、`window_id`、decision/sample边界在全链路完全一致。
- 第一个窗口endpoint=15360，此后严格每960 samples一个；任意L1块切分产生完全相同窗口。
- 空候选、1候选、2候选、超过2个有效峰时的明确限额诊断、0°/359°、warming_up、断流和回退路径。
- L2真实候选平滑前后rank、时间字段和score逐项相同，仅`theta_deg`允许改变；成熟ID预测候选须从当前响应取score，最终仍为0～3个且满足45°间距；不存在公共ID。
- 圆周卡尔曼覆盖0°边界、静止降抖、移动跟随、双候选交叉、漏检/跳窗、Gate阻断、epoch重置和异常原始角回退。
- config schema与代码默认值一致，无未识别字段。
- 正式DTO为CPU NumPy而GPU worker不发生逐候选往返；候选rank在L2→BF→Feature→CNN→Result保持不变。
- `DevUiFrame`不混合window；四象限能力缺失、STALE、error与Mock状态不产生假正式结果。
- `latest_l5_dev_ui`固定容量1且只接收L5 `COMPLETED`的完整同窗Frame；覆盖只影响显示。有序DROPPED/SKIPPED帧保留最近有效CNN结果到`stale_after_ms`，L5实际完成/丢弃/跳过/Hz/邮箱覆盖诊断逐项验证。
- Gate开启但候选为空时L3不调用prepare，L3/L5均为`COMPLETED`且L5接收空batch；正式增强音频与Voice结果为空。
- `AlgorithmPerformanceSnapshot`与当前session/epoch一致；epoch切换清空Warmup和分位数，rolling window只接收最多500个`ok/degraded`窗口，Latency与`processing_latency_ms`逐值相同且不小于Compute。

## 17.2 几何与DSP tests

- 7个坐标与本规格逐值一致，误差 `<1e-9 m`。
- 最大阵元间距0.08 m，对应48 kHz最大物理TDOA约11.20 samples；预测不得超界。
- 实机逐通道tap、0°/90°/180°/270°已知声源验证通道映射、板卡物理朝向、极性和固定延迟；未通过前不得把真实世界`theta_deg`标记为已校准。
- 0～359°逐1°合成无噪平面波：SRP峰圆周误差 `<=1°`。
- 加20 dB白噪声、每10°测试：SRP误差P95 `<=5°`。
- DAS目标响应 `abs(wᴴd-1)<=1e-5`。
- MVDR目标响应 `abs(wᴴd-1)<=1e-3`，所有输出finite。
- 同一合成场景CPU/CUDA的SRP、DAS、MVDR最大数值差分别 `<=1e-4`、`<=1e-4`、`<=1e-3`。
- GPU OOM注入覆盖5→1重试与整窗口CPU重算，证明不混合部分后端结果。

## 17.3 Feature/CNN tests

- 15360 samples输入必须得到 `[7,513,33]`共享STFT和 `[33,169]`特征。
- bins必须严格为2..170，中心频率93.75..7968.75 Hz。
- 训练/推理对同一waveform产生的feature误差 `<=1e-6`。
- 模型manifest不匹配必须拒绝加载。
- `metrics.json`的模型hash、数据集版本、test split和阈值必须匹配；由TP/FP/FN重算Precision/Recall/F1误差不超过`1e-6`，分母为0、不匹配或Mock时UI三项均为`N/A`。
- train-only频率均值/标准差可复现，validation/test对统计量无贡献；logits只执行一次Sigmoid且batch顺序不变。
- 独立test set目标：Voice recall `>=0.95`、precision `>=0.90`、F1 `>=0.92`；若未达到，不得标记模型为production-ready，但Mock接口开发不受阻。

## 17.4 Beamformer评测（后续工作，不属于本次L3实现范围）

同一锁定benchmark同时跑DAS与MVDR。可计算SIR的case必须保存时间对齐的目标与干扰多通道component reference：仿真数据保存生成stem与RIR/seed，受控扬声器数据分别录制每个声源并固定阵列/声源姿态；仅有混合录音而没有component reference的真实数据只做稳健性和人工试听，不得伪造SIR。

基准集最低为180个case：36个方向（0～350°每10°）×5种条件（单人+白噪、双人、音乐/电视、风扇/空调、混响），至少包含3个room/environment ID、3档输入SIR（-5/0/+10 dB）且远场距离不小于0.5 m。对每个component分别施加相同波束权重，以80～8000 Hz频带计算：

```python
output_sir_db = 10 * log10(target_power / (interference_power + 1e-12))
sir_gain_db = output_sir_db - input_sir_at_center_db
delta_mvdr_vs_das_db = mvdr_output_sir_db - das_output_sir_db
```

MVDR通过门槛固定为：所有输出finite、fallback率`<=1%`、`delta_mvdr_vs_das_db`中位数`>=0 dB`、P10 `>=-1 dB`、至少70% case `>=0 dB`，且目标component相对DAS的中位RMS衰减不超过1 dB。任一门槛失败则版本状态为`mvdr_not_validated`并阻止生产验收；开发/诊断模式可显式选择DAS安全运行，但正式架构的主算法选择仍是MVDR，不能静默改名或宣称MVDR已通过。接口和数据格式保持不变。新MVDR参数或实现必须创建新benchmark report，不得覆盖旧报告。

## 17.5 数据管理测试

- Runtime/Test目录隔离、manifest schema、hash和SQLite重建。
- 唯一Ingest时间轴、录音chunk切分、绝对sample连续性、异步result watermark、`.partial`恢复、磁盘满和两类写盘队列溢出。
- promote复制、lineage、标注版本、QA、split锁定和泄漏检查。
- UI筛选、后台统计、Trash恢复与永久删除权限边界。
- Development Test UI scratch每次新录制先清空，在当前UI会话内仅保存最后一次，关闭UI后全部删除，不进入Runtime/Test Catalog；暂停分段保留真实sample区间。L3合成试听音频写入Test UI专用临时磁盘缓存并以内存映射方式播放，关闭UI后整个缓存目录删除。
- 录音保存原始SpatialResponse和平滑候选，但不保存内部ID；同一配置从epoch起点顺序重放必须复现平滑角。

## 17.6 实机验收

- 用户U9+RTX 5060，m=5，warm processing p95 `<=20 ms`。
- 30分钟连续运行无崩溃、无非有限结果、无无限队列增长。
- 全屏四象限和自动重播开启时算法processing latency p95增量`<=2 ms`，GUI内存增长`<=100 MB`。
- 原始8ch、物理7ch、空间谱、候选、MVDR/DAS输出、特征、CNN分数和最终判断均可按window id离线复现。

现有迁移前测试通过不代表新规格完成；上述新增contract/DSP/CUDA测试全部通过后，项目才进入功能验收。

---

# 18. 一句话定义

> 每20 ms以真实物理坐标执行SRP-PHAT并产生最多3个原始候选；随后可选分配私有ID，再可选按ID执行圆周卡尔曼。两个模块默认关闭，卡尔曼依赖ID追踪，内部ID不对外输出。

# 6+1 麦克风阵列项目 1.1.0：MUSIC 与公开方向 ID 目标架构

状态：**规划已确认；当前分支已完成L2 Rolling NormMUSIC、公共方向ID及配套跨层/记录接口迁移，但项目1.1.0尚未发布，真实阵列验收仍待完成。已发布版本仍为1.0.1。**

目标版本：项目 `1.1.0`，完成全部代码、测试与验收后才允许创建新标签 `v1.1.0`。不得移动、覆盖或重写已经发布的 `v1.0.1`。

适用范围：Layer 1～Layer 4、Windowing、Application Runtime、Development Test UI、Production UI、RecordingStore、数据管理、测试与资产。

覆盖规则：本文件是 **1.1.0目标架构** 的权威契约；各目录README必须明确区分“当前分支已实现”“尚待实机验收”和“已发布版本”，不得在完成全部验收前声称1.1.0已经发布。

## 1. 改造目标与非目标

1. 用宽带 MUSIC/NormMUSIC 替换 L2 的 SRP-PHAT 定位主链，并直接支持 0～3 个同时存在的方向峰。
2. 删除 iterative multiple peak 开关、配置、UI 和算法路径；多声源能力由 MUSIC 空间谱、声源数估计和圆周峰值筛选统一提供。
3. 将方向 ID 追踪设为 L2 永久在线能力，不再提供关闭按键；采用全局一对一线性分配，正确处理 `359° ↔ 0°`、候选排序变化、新 ID、短时漏检和超时后重新编号。
4. Kalman 只作为可选的方向平滑器；关闭 Kalman 不得关闭、重置或绕过 ID 追踪。
5. ID 从 L2 的私有 UI sidecar 元数据升级为 L2、L3、L4、Runtime、时间线、正式记录和逐 ID 试听共同使用的公共字段。
6. Test UI 根据 L2 的权威 ID 拼接 L3 音频；删除 UI 自己的二次角度关联、别名合并和贪心补救。
7. 录音管理和 Production UI 能按会话与 ID 查询方向时间线、L4 判断及增强音频，并提供逐 ID 试听。

这里的 `track_id` 是**阵列方向轨迹 ID**，不是人的生物身份或说话人身份。在两个声源处于同一方向、近距离交叉或空间证据不足时，系统不能承诺保持真实人物身份不交换。

## 2. 论文与开源实现依据

- MUSIC 的基础定义采用 R. O. Schmidt 的经典论文：[Multiple Emitter Location and Signal Parameter Estimation](https://codar.com/images/about/1986Schmidt_MUSIC.pdf)。
- 宽带实现优先参考 Pyroomacoustics 的公开源码：[MUSIC](https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/music.py)、[frequency-normalized MUSIC](https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/normmusic.py) 和 [DOA example](https://github.com/LCAV/pyroomacoustics/blob/master/examples/doa_algorithms.py)。本项目应提炼算法和测试方法，不直接引入不必要的完整运行时依赖。
- 声源数估计以 Wax/Kailath 的 MDL 方法为第一实现依据：[Detection of Signals by Information Theoretic Criteria](https://doi.org/10.1109/TASSP.1985.1164557)。
- 相干声源和强混响下若普通宽带 MUSIC 不稳定，CSSM 作为后续增强候选，而不是本轮第一实现：[Coherent signal-subspace processing](https://doi.org/10.1109/TASSP.1985.1164667)。
- Israel Cohen 的工作优先用于本项目的噪声统计、校准、鲁棒性和反馈思路。公开资料入口见 [Israel Cohen publications](https://israelcohen.com/publications/all-publications/) 和 [Source Localization with Feedback Beamforming](https://israelcohen.com/wp-content/uploads/2018/05/Source-Localization-with-FeedbackBeamforming-Thesis-Itay-Yehezkel-Karo.pdf.pdf)。检索阶段未发现可直接替换当前 L2 的 Cohen MUSIC 开源实现，因此不得虚构“Cohen MUSIC 代码”来源；实现以标准 MUSIC/NormMUSIC 为主，并复用现有 Cohen IMCRA 噪声估计结果。
- 全局关联使用 SciPy [`linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) 求解线性和分配问题。工程文档可称“匈牙利式全局一对一分配”，但代码注释应准确说明 SciPy 当前实现为改进 Jonker–Volgenant 算法，而不是声称调用了特定内部实现。

所有借鉴的代码必须核对许可证，并在实现文件和第三方声明中保留必要来源信息。

## 3. 目标主链

```text
48 kHz HostAudio [N,8]
    ↓
L1：解码、校准、逻辑重排、连续性检查、IMCRA、可选预降噪
    ↓
WindowAssembler：每20 ms发布一次、包含最近320 ms的DecisionWindow
    ↓
L2：Probability Gate
    → 7麦多帧STFT与频点协方差
    → 宽带frequency-normalized MUSIC 0～359°空间谱
    → MDL/跨频一致性估计0～3个声源
    → 圆周峰值与45° NMS
    → 永久在线全局分配方向ID
    → 可选按ID圆周Kalman
    → TrackedDirection + active_tracks
    ↓
L3：按同一WindowKey和track_id执行逐方向增强
    → EnhancedAudio(track_id, theta_deg, samples)
    ↓
L4：按track_id执行人声分类
    → VoiceDetection(track_id, theta_deg, probability)
    ↓
ResultJoiner：按WindowKey有序合并，逐ID精确对齐
    ├── DecisionRecord v4 / RecordingStore / ID时间线
    ├── Development Test UI / 按ID连续试听
    └── Production UI / 运行录音详情与逐ID回放
```

跨窗口并行仍为 `L2(n) || L3(n-1) || L4(n-2)`；同一窗口仍严格执行 `L2 → L3 → L4`。`track_id` 只增加对齐维度，不允许绕过 `WindowKey = (session_id, stream_epoch, window_id, decision_sample)`。

## 4. 公共方向与 ID 契约

### 4.1 `TrackedDirection`

L2 正式候选应改为不可变公共 DTO，至少包含：

```text
session_id
stream_epoch
window_id
decision_sample
track_id
rank
measured_theta_deg       # 当前观测；coasting时可为空
theta_deg                # 对外目标角；始终归一化到[0,360)
raw_score
normalized_score
track_state              # tentative / confirmed / coasting
is_observed
is_new_track
first_seen_sample
last_observed_sample
missed_samples
kalman_applied
```

字段名可在实现时按现有类型风格微调，但上述语义不可丢失。L2 公共结果同时区分：

- `directions`：本窗口真正交给 L3 的 0～3 个方向；每项必须有唯一 `track_id`。
- `active_tracks`：包含当前观测轨和仍在短时 coasting 的轨迹，用于 UI 与时间线；并非所有 active track 都必须触发 L3 波束形成。
- `spatial_response`：MUSIC 360 点伪谱及频率/归一化诊断。
- `model_order`：本窗口估计的声源数量和质量信息。

### 4.2 ID 作用域

- 正式关联键固定为 `(session_id, stream_epoch, track_id)`；跨层不得仅用角度匹配。
- 同一 session 内 `track_id` 单调分配且永不复用。epoch 切换清空运动状态和关联历史，但 ID 计数器在同一 session 内继续递增，避免覆盖旧时间线或试听文件。
- 新 session 可以从初始 ID 重新开始，因为完整关联键包含 `session_id`。
- L3、L4、Runtime、记录和 UI 只能继承 L2 ID，不得创建第二套“正式 ID”。

## 5. Layer 1 改动

L1 的 8 通道顺序、唯一采样时间轴、20 ms IMCRA 和可选 Wiener 预降噪保持。为 MUSIC 增加以下保证：

- L2 必须获得连续的48 kHz、7个物理麦校准音频，并可访问DecisionWindow内最多320 ms历史；MUSIC实际使用的滚动历史长度由独立配置和目标机基准确定。HardwareMix仍不得进入协方差、导向矢量或MUSIC伪谱。
- 校准元数据必须能区分 `verified / unverified`。Development Test UI 对未验证校准明确警告；Production 在完成实机标定后应支持要求 verified calibration 才启动正式定位。
- 当前增益、极性和整数 sample delay 校准继续兼容；MUSIC 实机误差若表明需要亚采样或频率相关补偿，应新增版本化的频域校准资产，不得静默改写旧 calibration hash。
- L1 不创建、不保存、不解释 `track_id`。

## 6. Windowing 改动

- `DecisionWindow [15360,8]` 和 20 ms 发布节拍保持不变；320 ms是L3/L4上下文和L2可用历史上限，不代表L2每次都重新计算整段音频。
- L2维护按session/epoch/sample连续的滚动STFT与协方差状态。每个新DecisionWindow原则上只加入最近20 ms产生的新帧并移出超出MUSIC历史长度的旧帧；禁止每20 ms从头重算320 ms STFT和全部协方差。
- `music.context_ms`首轮至少比较`160 / 240 / 320 ms`。最终默认值由目标设备实时性能、合成多源精度和真实移动声源测试共同决定，不把320 ms预先固化成不可调整要求。
- Gate 仍消费与窗口末端对齐的两个 20 ms IMCRA 概率；没有活动ID时，Gate关闭会跳过新的MUSIC观测。当前窗口开始时只要存在任意未删除ID，低于门限的正式概率判决即强制放行MUSIC；最后一个ID删除后立即恢复概率门限。ID继续按3秒绝对sample TTL推进到coasting/超时；预热、缺失和无效概率仍保持阻断，epoch变化不得继承旧ID的强制状态。
- 窗口不得预先生成 ID。所有 L2 配置必须冻结进 `WindowWorkItem`，保证同一窗口的 MUSIC、ID 和 Kalman 参数一致。

## 7. Layer 2：MUSIC 定位

### 7.1 初始参数基线

- 输入：48 kHz、7个物理麦的连续滚动音频；可用历史上限320 ms，初始候选历史长度为160/240/320 ms三档基准后择优。
- STFT：`n_fft=1024`、分析窗 `960 samples`、hop `480 samples`，形成多帧快照；实际窗函数沿用项目统一定义并写入算法版本。
- 定位频带：首版保持 `2000～4000 Hz`，后续只能依据真实诊室数据和空间可分度测试调整。
- 每个频点形成 `7×7` 复协方差矩阵，执行对角加载或收缩，拒绝 non-finite、秩异常和严重病态输入。
- 对每个频点执行 Hermitian `eigh`，构建噪声子空间；在 0～359°逐度扫描。
- 频点融合采用 NormMUSIC 风格的逐频归一化，避免少数高能频点支配宽带结果。

### 7.2 20 ms增量更新与实时预算

- MUSIC的320 ms历史与20 ms输出节拍不是同一概念：首轮只在滚动状态预热期间等待足够快照；预热完成后每20 ms继续发布新结果，不额外引入320 ms批处理等待。
- STFT hop为480 samples时，每个20 ms决策通常只新增2个STFT帧，并移出同等数量的过期帧。每频点协方差使用可逆滑窗累加量或等价的稳定递推更新，不能重复遍历整段历史。
- 阵列几何、2～4 kHz频点和0～359°导向张量在配置/校准revision变化时预计算；正常窗口复用。特征分解使用批量7×7 Hermitian `eigh`，MUSIC投影使用向量化批量运算。
- 360°伪谱和ID关联仍每20 ms更新。MDL可按有界较低频率刷新（初始每100 ms一次），在协方差质量或谱形显著变化时立即重算；结果必须记录年龄，超过100 ms不得继续沿用。
- 目标设备门禁要求L2单窗p95明显低于20 ms，初始工程预算设为15 ms并保留至少5 ms调度余量。若超时，按顺序优化向量化/缓存、确定性频点子采样和MUSIC历史长度；不得用静默丢窗、重复旧伪谱或降低20 ms发布时间戳来伪装达标。

### 7.3 声源数与候选

- 使用 MDL 估计 `0～3` 个信号子空间维度，并用跨频一致性、有效频点数、Gate 状态和数值质量约束结果。
- 首版最多输出 3 个候选，圆周 NMS 最小间隔继续为 45°。
- 峰值选择必须原生处理数组首尾相邻，`359°` 和 `0°` 属于相邻角度。
- 无足够有效频点、协方差退化或模型阶数不可信时，返回可诊断的 blocked/degraded/failed 状态，不得静默复用上一窗伪谱冒充新观测。
- 原 SRP-PHAT、iterative multiple peak 与相关回退不再进入正式 1.1.0 主链；删除配置、运行时 setter、UI 开关和专属测试。若保留历史实现用于回归，只能放在明确的非运行时归档边界，不能被新 pipeline 导入。

## 8. Layer 2：永久 ID 与可选 Kalman

### 8.1 全局分配

每窗先预测现有轨迹，再建立“现有轨迹 × 当前观测”的代价矩阵。代价至少包含：

- 圆周最短角残差 `((measurement - prediction + 180) % 360) - 180`；
- 与时间间隔相关的最大角速度/关联门限；
- MUSIC 峰值质量、轨迹不确定度和连续性惩罚；
- 明确的 miss 与 birth 代价。

矩阵必须加入 dummy 行/列，使“不匹配旧轨”“新建轨迹”和“轨迹漏测”参与同一个全局最优分配，再调用 `linear_sum_assignment`。禁止逐候选贪心、`itertools.product` 穷举组合或按 rank 绑定 ID。相同代价必须使用固定 tie-break，保证重放可复现。

### 8.2 生命周期

- 状态为 `tentative → confirmed → coasting → deleted`。
- 首次无匹配观测立即分配新 ID；满足连续观测条件后确认。ID 一经分配，在同一 session 内不得给其他轨迹复用。
- 短时漏检或 Gate 关闭进入 coasting；在 TTL 内重新落入关联门限应恢复原 ID。
- 超过 TTL 删除轨迹；之后出现的方向即使相近也必须获得新 ID。
- 所有确认、miss、coast 和 TTL 使用 48 kHz 绝对 sample 计算，不依赖“处理了多少窗”，从而正确应对 latest-wins 丢窗和 sample 跳跃。
- L4 不再拥有 ID 确认权、语音租约或生命周期；删除当前按角度把 L4 结果反向匹配给 L2 的反馈路径。将来若增加语义反馈，也只能携带完整 track key，并且不能取代 L2 的几何生命周期。

### 8.3 圆周与 Kalman

- 轨迹内部维护 unwrapped angle；`359° → 0°` 应表现为 `+1°`，反向为 `-1°`，公开时再 `% 360`。
- ID 关联永远开启，不存在 `enable_id_tracking` UI/配置开关。
- Kalman 可以开关，但只影响 `theta_deg` 平滑和短时预测，不影响 ID 的分配、确认、miss 或删除。
- 开关 Kalman 不得清空 tracker 或改变已有 ID。Kalman 状态按 `track_id` 建立；关闭时使用观测/几何预测，重新开启时安全重建滤波状态。
- 不确定度过大时 active track 可继续 coasting 展示，但不得发布虚假的 L3 目标。

## 9. Layer 3 改动

- 输入从无 ID 的 `CandidateDirection` 改为 `TrackedDirection`，以 `(WindowKey, track_id)` 为方向批次身份。
- `DirectionalSignal`、波束形成批次和 `EnhancedAudio` 都必须携带 `track_id`、`theta_deg` 与原候选顺序；输出不得重新分配、猜测或合并 ID。
- L3 在入口和出口校验：同一 WindowKey、ID 唯一、ID 集合/顺序、角度和音频数量完全对应；错误必须成为明确阶段终态。
- 默认仅处理本窗 `directions` 中可观测或满足受控短时预测条件的目标。仅用于时间线的长 coasting 轨不生成 L3 音频。
- optimized、ds_baseline、constant_beamwidth_baseline 三档仍保留；切换模式不改变权威 ID，只隔离各模式的试听缓存。

## 10. Layer 4 改动

- `Layer4AudioSegment`、`VoiceDetection` 和阶段结果均增加 `track_id`，并按 L3 的 `(WindowKey, track_id)` 原样返回。
- L4 入口/出口校验 ID 集合、顺序、角度与音频严格对齐；重新阈值判断只能改变 Voice/Non-Voice 结论，不能改变 ID。
- CNN、48→16 kHz 适配、响度补偿和 primary/shadow 边界保持不变。
- 删除 L4 通过角度向 L2 回送“正式化/续租”证据的路径。L4 是轨迹的语义标签消费者，不是方向 ID 的所有者。

## 11. Runtime、时间线与并行管理

- 保留 staged 单 worker、各层有界 latest-wins 队列、分区缓存和 ResultJoiner 有序提交。
- L2 worker持有滚动MUSIC状态，ComputeCache保存预计算导向张量和有界频点工作区；状态只能按worker实际取走且sample连续的窗口推进。发现sample跳跃时按缺口大小更新/重建滚动状态，并发布明确诊断，不能把不连续帧当作连续快照。
- 配置快照删除 iterative 和 ID enable 字段，增加 MUSIC、模型阶数、关联生命周期与 Kalman revision；旧配置加载必须显式迁移或拒绝未知冲突，不能悄悄保留旧开关语义。
- 每层 StageResult 都携带完整 ID 对齐信息；ResultJoiner 校验 L2 `directions`、L3 enhanced 与 L4 detections 的 `track_id` 一一对应。
- 丢弃、超时和跳窗按绝对 sample 更新轨迹；不得因某一层队列替换而重置整个 tracker。
- 移除 angle-only L4 feedback mailbox 和 Test UI 私有 ID 投影；Runtime 只传递正式公共 ID。
- `DecisionRecord` 升级到 v4。旧 v3 记录继续只读兼容，不原地改写。

## 12. Development Test UI 与逐 ID 试听

- 删除 “Iterative Multiple Peak” 开关、ID 追踪开关、相关持久化设置和运行时 setter。
- 保留 Kalman 开关及 Q/R 等调试参数；文案明确“仅平滑，不控制 ID 是否存在”。
- 右上面板从 SRP 改名为 DOA/MUSIC，绘制原始 360 点 MUSIC 伪谱、模型阶数和数值状态。
- 候选表显示 `track_id、measured_theta_deg、theta_deg、score、state、is_new_track、is_observed、L4 probability`；观测和预测样式可以不同，但颜色稳定绑定权威 ID。
- 左下试听继续保留 Center Mic 原音参考、20 ms 稳定 hop 拼接、可恢复真实音频补洞、过旧缺口补等时静音、交叉淡化、至少 2 秒显示、3 秒等待、有界分段和三档 L3 模式隔离。
- 方向音频只按 `(session_id, stream_epoch, track_id)` 拼接。删除 `_formal_aliases`、`_resolve_formal_track_id`、按角度贪心重关联和 ID 换号合并；UI 不再修补 L2 身份错误。
- coasting 期间保留轨道行并显示状态；只有 L2 删除轨迹或 session/mode 生命周期结束时封存对应试听轨。

## 13. RecordingStore、数据管理与 Production UI

### 13.1 DecisionRecord v4

v4 至少保存：

- L2 MUSIC 空间谱引用、model order、有效频点/协方差质量和算法版本；
- 每个候选的 `track_id`、观测角、输出角、状态、分数、是否观测/新建及生命周期 sample；
- L3 每个增强资产对应的 `track_id`；
- L4 每个检测对应的 `track_id`、概率和判断；
- `kalman_applied`、配置 revision、calibration version/hash；
- `active_tracks` 与窗口阶段终态。

增强文件名和 manifest 资产索引必须含 `track_id`，避免同窗多方向或角度跨 0° 时覆盖。Catalog/服务增加按 session + epoch + track ID 查询时间线、持续时间、角度轨迹、L4 概率和增强资产的能力。

### 13.2 界面

- Production UI 的 Runtime Session/运行录音详情增加方向 ID 列表、持续时间、首末 sample、角度变化、当前状态、L4 概率和逐 ID 增强音频试听，同时保留 Center 参考。
- 专用“测试录音向导”只采原始 L1 音频和热力图，不运行算法时明确显示“无算法方向 ID”，不得伪造 ID。
- 现有 native/logical/physical 通道试听、模拟测试、QA、标注、hash、恢复、Trash 和本地数据边界保持。
- `data/`、运行录音、Catalog、日志和临时缓存继续只保存在本地，不提交 GitHub。

## 14. 测试与验收门禁

### 14.1 MUSIC

- 1、2、3 个合成远场声源；0 个声源/纯噪声；方向包含 `359°/0°/1°` 和恰好 45°。
- 不同幅度、频谱、混响和部分相干输入；HardwareMix 注入不得改变结果。
- MDL 0～3 阶、跨频一致性、协方差秩不足、加载/收缩、non-finite 和低有效频点失败路径。
- 校准前后、错误极性/延迟、MIC 顺序和观察面镜像防错。
- 与独立 Pyroomacoustics/离线参考输出在约定容差内对照。

### 14.2 ID 与 Kalman

- `358→359→0→1` 和反向跨界不换 ID；公开角始终 `[0,360)`。
- 候选 rank 交换、两个/三个目标移动和会合前后使用全局一对一分配，不重复分配。
- 未匹配观测立即新建 ID；短时漏检恢复原 ID；超过 TTL 后同方向分配新 ID。
- Gate 关闭、latest-wins 丢窗、绝对 sample 大跳、epoch 切换、session 切换和确定性 tie-break。
- Kalman 开/关/运行时切换不改变 ID；不确定度过大不发布 L3 预测目标。

### 14.3 跨层、存储和 UI

- L2→L3→L4→DecisionRecord 的 WindowKey、ID 集合、顺序、角度和数量逐项一致。
- L3/L4 失败、超时、空候选、队列丢弃和停机 drain 仍生成唯一终态并推进正确 watermark。
- DecisionRecord v4、逐 ID 文件名、manifest、Catalog、恢复、旧 v3 读取和本地数据不入库。
- Test UI 不再存在 iterative/ID 开关或二次关联；按 ID 拼接、补洞、等待、封存、模式隔离和 Center 参考均有自动测试。
- Production UI 可查询并试听逐 ID 结果；L1-only 录音明确无 ID。

### 14.4 性能与实机

- 在目标设备上分别基准160/240/320 ms滚动历史，验证50 Hz持续流水、预热后每20 ms有新MUSIC结果、队列不持续积压；L2初始预算为p95不超过15 ms、硬门限小于20 ms。记录STFT、协方差、eigh、伪谱、MDL和关联的分项耗时。
- 增加“增量结果与从头离线重算一致”测试、长时间滚动数值漂移测试，以及配置/校准revision和sample跳跃触发安全重建测试。不得通过隐藏丢窗或复用过期伪谱宣称实时通过。
- 使用真实阵列完成静止、移动、`359° ↔ 0°`、新说话方向、短时静音、三声源、混响和长时间运行验收。
- 自动测试不能替代真实麦克风的校准和诊室验收。

## 15. 推荐迁移顺序与分支边界

1. **L1 + Windowing**：补 MUSIC 输入/校准契约和测试，不引入 ID。
2. **L2 MUSIC**：完成多帧 STFT、协方差、MDL、NormMUSIC、圆周峰值，并移除 iterative 正式路径。
3. **L2 Tracking**：完成公共 DTO、全局分配、生命周期、跨 0° 与可选 Kalman；不依赖 UI 修补。
4. **L3 + L4**：贯通公共 ID，删除 L4 angle-only lease feedback。
5. **Runtime**：更新配置快照、StageResult、Joiner、时间线和 DecisionRecord v4。
6. **Development Test UI**：删除旧开关，展示 MUSIC/ID，并按权威 ID 拼接试听。
7. **Recording/Data + Production UI**：保存、查询和试听逐 ID 资产，兼容 v3 只读。
8. **整合验收**：全量自动测试、性能、实机、CHANGELOG、语义版本与 `v1.1.0` 发布。

并行分支可以分别修改，但公共 DTO、字段命名、WindowKey/ID 对齐和 DecisionRecord v4 schema 必须先冻结；合并时以本文件为共同契约，禁止每个分支自行发明不同 ID 语义。

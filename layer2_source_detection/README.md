# Layer 2 1.1：Rolling NormMUSIC 与公共方向轨迹

本目录是项目`1.2.4`正式组成部分，L2公开版本为`1.1`，已实现Rolling NormMUSIC、公共方向ID、可选Kalman平滑、可选DPD与IMCRA白化。

IMCRA白化严格只读DecisionWindow携带的L1不可变快照，不拥有、更新或重置IMCRA状态。逐麦PSD形成的噪声模型为对角矩阵，收缩和diagonal loading后仍保持对角，因此实现使用逐麦逆平方根直接缩放协方差与steering，不执行逐频通用7×7 Cholesky/solve。16个hop的固定频率映射按批量向量化处理；缺少READY快照或有效对角项时明确退回未白化MUSIC。

## 正式主链

```text
DecisionWindow + 两个对齐的20 ms概率
    → Probability Gate（两概率取平均，运行时门限）
    → Rolling frequency-normalized MUSIC
         7路物理麦 / 2～4 kHz / 0～359°逐度
         20 ms增量加入/移除STFT协方差帧
         每频点7×7加载/收缩协方差 + Hermitian eigh
         MDL与跨频一致性估计0～6阶空间模态
         MDL仅作0～6阶诊断；实际阶数=手动上限1/2/3
         NormMUSIC逐频归一化后跨频融合
         UI阶数上限1/2/3轮贪心选峰 + 50°圆周NMS
    → 永久开启的全局方向轨迹分配
         birth/miss dummy行列 + linear_sum_assignment
         tentative / confirmed / coasting / deleted
         滚动200 ms内至少6次匹配才进入confirmed
         session内ID单调且不复用；寿命按48 kHz绝对sample
         Kalman关闭时，confirmed ID可按3秒圆周历史识别短时静止
    → 可选Kalman输出平滑（不拥有、不重置、不改变ID）
    → TrackedDirection[0..3] + active_tracks
```

`track_id`只表示空间方向轨迹，不是人物或声纹身份。只有已达到tracking `confirmed`且至少收到一次L4正向人声反馈、当前未被标记为噪声干扰的ID，才能在正式Gate概率低于门限时强制放行MUSIC。尚无人声证据的临时轨迹有当前观测时仍可进入L3/L4接受分类，但漏检后只在`active_tracks`中保留，不再作为公共coasting方向送入L3，也不能强制Gate；最后一个符合条件的人声ID消失后，Gate立即恢复按概率门限判断。ID仍使用3秒绝对sample TTL；预热、缺失或无效概率不会被伪造成有效概率。短漏检保留同一内部ID，超过TTL后再次观察会获得新ID。epoch会清除活动轨迹，但同一session的ID计数继续递增；新session建立新的ID命名空间。

L4按权威`track_id`异步回传该窗口的人声概率与判定，L2在下一处理窗口安全消费。某ID从建立或最后一次正向人声结果起连续3秒未再被L4判为人声时，内部标记为`is_noise_interference`。噪声轨仍可跟随原方向观测并按原规则更新寿命，但退出全局一对一关联的排他集合，因此普通ID接近时不会被错误归并到噪声ID。只有噪声ID的±45°内不存在其他普通ID，并且它在滚动3秒时间窗内累计收到5次人声判定时，才解除噪声标记；期间的非人声结果不增加次数，也不清空仍位于该3秒窗内的人声记录。L4反馈不直接修改MUSIC、Gate概率、几何TTL或Kalman参数。

内部活动ID硬上限为4，公共方向仍最多3个。新观测需要建立ID但内部已满时，优先淘汰未被本窗关联的噪声轨、无人声证据轨、tentative轨及最久未观测/低分轨；本窗已成功关联的轨迹受保护。该上限只控制ID内存与UI/试听扇出，不把Gate改成`WARMING_UP`。

## 滚动计算

比较历史固定包含160/240/320 ms，当前运行配置选择240 ms；`DecisionWindow`仍提供最多320 ms原始历史。连续窗口只计算新增的两个50%重叠STFT帧，并移出超出历史的两个旧帧；sample跳跃、epoch或session变化时从当前窗口重建。导向张量按阵列几何、频率轴和配置revision缓存。伪谱与轨迹每20 ms更新，MDL缓存最多100 ms。

`MusicDiagnostics`分别记录0～6的MDL诊断阶数、实际MUSIC阶数、手动上限导致的限制、高阶模型失配、有效频点、协方差质量、增量状态和协方差更新/eigh/谱融合/总耗时。Test UI可把手动上限设为1、2或3，下一窗口生效；该控制不覆盖MDL诊断值。MDL诊断大于3时标记模型失配并禁止该窗创建新ID，已有ID仍可关联或coasting。普通MDL路径不实现逐频真实峰支持和可靠性加权门禁；这些检查只属于默认关闭的DPD路径。目标机门禁为稳态p95不高于15 ms，单窗硬门限20 ms。

默认关闭的`DPD + rank-1 MUSIC`开启后不使用MDL决定候选数：可靠频点分别产生rank-1方向票，再进行跨359°/0°连续的圆周核聚类。每个方向簇必须至少有5个支持频点、覆盖4个等宽子带中的至少2个、获得至少0.25的可靠频点总权重且圆周集中度至少0.95，随后再通过峰门限和50°NMS。合格簇数量决定0～手动上限个候选；逐候选诊断记录支持频点、支持率、子带数、集中度、平均平面波拟合度和簇权重。开关仍可由Test UI持久化并在运行中切换，默认值保持OFF。

## 兼容和删除边界

- 原SRP-PHAT与iterative multiple peak正式实现、配置、setter、UI开关和专属测试已删除；包不再导出旧扫描器。
- ID追踪没有enable开关。Kalman仍可在运行中独立开启/关闭，切换不会清空或换号。Kalman关闭时，confirmed ID会检查最近3秒圆周角度历史：至少70%观测位于圆周均值±15°时进入短时静止，公开角度使用随可信观测滚动更新的圆周均值。静止期间，1秒内最多3个超出均值±20°的观测只刷新ID观测时间，不移动关联锚点或输出角；达到第4个时立即退出静止并跟随当前观测。跨359°/0°始终使用圆周距离。漏检/coasting时保持静止均值或最后一次可信观测，不使用角速度外推；Kalman开启时才允许滤波预测，且短时静止状态和历史不会介入Kalman输出。
- `CandidateDirection`仅作为尚未迁移消费者的同角度兼容投影；公共权威输出是`TrackedDirection`与`active_tracks`。
- 第8路HardwareMix只用于接口、显示和录音，不进入MUSIC协方差或导向计算。

## 算法来源

实现依据Schmidt MUSIC、Wax/Kailath MDL以及Pyroomacoustics MUSIC/NormMUSIC的公开算法表达；Pyroomacoustics为MIT许可。本项目没有复制或声称存在“Israel Cohen MUSIC开源实现”，Israel Cohen资料只用于项目另行记录的噪声估计与鲁棒性背景。

自动测试覆盖MDL 0～6阶、手动1/2/3阶上限、高阶窗禁止新ID、最多3个候选、跨0°、rank交换、新生/短漏检/TTL、Gate关闭、sample跳跃、epoch/session、确定性关联、Kalman切换、HardwareMix隔离、滚动增量等价性和实时性能。

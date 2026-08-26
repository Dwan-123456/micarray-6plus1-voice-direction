# Layer 2 1.1：Rolling NormMUSIC 与公共方向轨迹

本目录是项目`1.3.3`开发组成部分，L2公开版本为`1.1`。DOA固定使用Rolling NormMUSIC，随后进入Circular IMM-JPDA方向ID追踪器，并共用公共方向DTO和L3边界。Development Test UI只显示MUSIC方法，不提供后端切换。

IMCRA白化严格只读DecisionWindow携带的L1不可变快照，不拥有、更新或重置IMCRA状态。L1在低SPP频点持续维护完整7×7复数空间噪声协方差，L2对其做频率插值、收缩、loading及批量Cholesky白化，同时变换观测协方差和steering。缺少READY快照或空间噪声协方差不可用时，本窗明确退回未白化MUSIC。

## 默认MUSIC主链

```text
DecisionWindow + 两个对齐的20 ms概率
    → 始终维护Rolling MUSIC空间协方差（Gate关闭也每20 ms增量更新）
    → Probability Gate（两概率取平均，运行时门限）
         关闭：不做特征分解、伪谱融合和峰值搜索
         开启：立即用已预热的空间协方差计算MUSIC伪谱
    → Rolling frequency-normalized MUSIC
         7路物理麦 / 2～4 kHz / 0～359°逐度
         20 ms增量加入/移除STFT协方差帧
         每频点7×7加载/收缩协方差 + Hermitian eigh
         Test UI手动输入1/2/3阶信号子空间
         NormMUSIC逐频归一化后按4 cm阵列固定频率权重跨频融合
         UI阶数上限1/2/3轮贪心选峰 + 50°圆周NMS
    → 默认开启的全局方向轨迹分配
         Gate连续OPEN满200 ms（10个20 ms窗口）才允许MUSIC角进入ID追踪
         预热期间已有ID仍可匹配；confirmed ID可正常coasting
         birth/miss dummy行列 + linear_sum_assignment
         tentative / confirmed / coasting / deleted
         滚动500 ms内至少10次匹配才进入confirmed
         session内ID单调且不复用；寿命按48 kHz绝对sample
         Kalman关闭时，confirmed ID可按3秒圆周历史识别短时静止
    → 可选Kalman输出平滑（不拥有、不重置、不改变ID）
    → TrackedDirection[0..3] + active_tracks
```

`track_id`只表示空间方向轨迹，不是人物或声纹身份。当前项目的L5只在停机后的离线L4输出上运行，ApplicationRuntime不会调用L2的在线语义反馈接口，因此普通1.3.3开发线运行完全按Gate概率门限决定是否执行MUSIC。达到L2 `confirmed`的实测或coasting轨迹可在数量与50°角距限制内作为公共方向进入L3，不要求L5人声证据；Gate关闭时不再产生MUSIC观测，但正式ID仍可在最后真实观测后的2秒TTL内按保持/预测角继续进入L3。tentative需在500 ms内累计10次观测且存在概率达标才能确认，并可因低存在概率提前删除。confirmed漏检后固定保留2秒绝对sample TTL，存在概率按真实时间约每20 ms保留0.97地平滑衰减，不再因低于0.05而在TTL前提前死亡。在TTL内重新匹配会恢复原ID和`confirmed`状态；连续2秒无匹配才删除。关联角度使用固定50°硬上限和卡方门限20，不按漏检时长额外扩大。补救关联与新生保护同时比较轨迹的IMM预测角和最后真实观测角，候选距任一个不超过50°即恢复原ID并禁止重复birth。两轨后来进入50°以内时，同样依据滚动500 ms观测历史；只有近期观测至少两次交替且没有同窗双峰才归并。超过TTL后再次观测会获得新ID。epoch会清除活动轨迹，但同一session的ID计数继续递增；新session建立新的ID命名空间。

为兼容旧在线分类实验，`Layer2Pipeline.submit_voice_feedback()`与`GlobalDirectionTracker.apply_voice_feedback()`仍保留精确`track_id`接口和专项测试：外部若显式提供至少2次正向结果，可使符合条件的confirmed轨在低Gate概率下强制放行；长期无正向结果还可触发噪声干扰标记。该接口当前没有Runtime调用方，不属于1.3.3普通主链，也不得用离线L5结果回写已经结束的实时轨迹。

内部活动ID硬上限为4，公共方向仍最多3个。新观测需要建立ID但内部已满时，优先淘汰未被本窗关联的噪声轨、无人声证据轨、tentative轨及最久未观测/低分轨；本窗已成功关联的轨迹受保护。该上限只控制ID内存与UI/试听扇出，不把Gate改成`WARMING_UP`。

## 滚动计算

比较历史固定包含160/200/240/320 ms，当前运行配置选择200 ms；`DecisionWindow`仍提供最多320 ms原始历史。无论Probability Gate是否开启，连续窗口都只计算新增的两个50%重叠STFT帧并移出超出历史的两个旧帧，持续维护7路物理麦的MUSIC空间协方差；这不是IMCRA噪声协方差。Gate关闭或无效时立即停止Hermitian特征分解、伪谱融合和峰值搜索，并清零连续OPEN计数。Gate开启后的前9个20 ms窗口可形成仅供诊断的MUSIC谱，但候选角不会进入ID追踪或原始MUSIC兼容输出；连续OPEN达到第10窗、即完整200 ms后才放行候选角。中途任何一次关闭、warming、unavailable、sample跳跃、epoch或session变化都会重新计时；confirmed ID原有coasting生命周期保持独立。导向张量按阵列几何、频率轴和配置revision缓存，预热完成后伪谱与轨迹每20 ms更新。

`MusicDiagnostics`记录Test UI手动阶数、实际输出数、有效频点、协方差质量、增量状态、连续Gate OPEN窗口数、同步的10窗/200 ms候选放行门限和协方差更新/eigh/谱融合/总耗时。Test UI可把MUSIC阶数设为1、2或3，下一窗口生效；普通路径同时将该值作为信号子空间阶数和最多搜峰数。后台不再计算或缓存自动模型阶数；兼容DTO中的旧年龄字段恒为0。目标机门禁为稳态p95不高于15 ms，单窗硬门限20 ms。

跨频融合固定保留2～4 kHz：2.0～2.3/2.3～2.5/2.5～2.7/2.7～3.0/3.0～3.6 kHz权重依次为`0.35/0.55/0.75/0.90/1.00`；3.6～3.8 kHz由`1.00`线性降至`0.75`，3.8～4.0 kHz由`0.75`线性降至`0.45`。处理顺序为IMCRA空间噪声协方差白化、逐频MUSIC、逐频独立归一化、固定权重融合、峰值搜索；没有新增SNR、SPP、特征值间隙或时间稳定性动态权重。可选DPD保留原有门禁，只把同一固定几何权重乘入其既有频点票权。

默认关闭的`DPD + rank-1 MUSIC`开启后，可靠频点分别产生rank-1方向票，再进行跨359°/0°连续的圆周核聚类。每个方向簇必须至少有4个支持频点、覆盖4个等宽子带中的至少2个、获得至少0.20的可靠频点总权重且圆周集中度至少0.85。两个或多个归一化峰值均严格大于0.70、且组内任意两峰圆周距离不超过40°时，在50°NMS前按唯一支持频点并集的可靠性权重计算圆周平均角`theta_group`和`w_merge`；重复频点只计一次，禁止链式跨范围合并，融合证据重新通过原方向簇门禁且蓝色投票谱不重新归一化。合格簇数量决定0～手动上限个候选；逐候选诊断记录支持频点、支持率、子带数、集中度、平均平面波拟合度和簇权重。开关仍可由Test UI持久化并在运行中切换，默认值保持OFF。

## 兼容和删除边界

- 原SRP-PHAT与iterative multiple peak正式实现、配置、setter、UI开关和专属测试已删除；包不再导出旧扫描器。
- 正式主链默认启用ID追踪；Development Test UI可用持久化诊断开关临时关闭。开启后，JPDA把每个观测在已有轨迹、新生和伪报之间做联合概率分配；每条轨迹的IMM同时维护静止与慢速移动模型。滚动500 ms内10次观测确认，恢复匹配同时参考预测角与最后真实观测角，交替出现且无同窗双峰的50°内重复轨使用同一500 ms历史归并；confirmed漏检后最多预测2秒，内部最多4条轨迹、公共最多3条。关闭时L2只返回无ID原始峰值，L3/L5跳过；不再存在独立Kalman开关或Q/R运行时调节。
- `CandidateDirection`仅作为尚未迁移消费者的同角度兼容投影；公共权威输出是`TrackedDirection`与`active_tracks`。
- 第8路HardwareMix只用于接口、显示和录音，不进入MUSIC协方差或导向计算。

## 算法来源

实现依据Schmidt MUSIC以及Pyroomacoustics MUSIC/NormMUSIC的公开算法表达；Pyroomacoustics为MIT许可。本项目没有复制或声称存在“Israel Cohen MUSIC开源实现”，Israel Cohen资料只用于项目另行记录的噪声估计与鲁棒性背景。

自动测试覆盖手动1/2/3阶、最多3个候选、跨0°、rank交换、JPDA一对一、新生/短漏检/TTL、Gate关闭、sample跳跃、epoch/session、IMM预测与重基准、HardwareMix隔离、滚动增量等价性和实时性能。

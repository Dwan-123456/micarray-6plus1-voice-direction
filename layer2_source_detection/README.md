# Layer 2 1.1：Rolling NormMUSIC 与公共方向轨迹

本目录是项目`1.3.3`开发组成部分，L2公开版本为`1.1`。Development Test UI可在运行中选择Rolling NormMUSIC或GI-DOAEnet PM作为DOA；两者之后统一进入同一个Circular IMM-JPDA方向ID追踪器，并共用公共方向DTO和L3边界。

GI-DOAEnet链读取同一`DecisionWindow float32[7680,8]`，只取前7路物理麦，将48 kHz 160 ms上下文同相重采样为16 kHz，并补零高度坐标形成`[7,3]`阵列位置。网络最近时间帧形成360点方向概率；候选继续使用UI门限、prominence、50°圆周NMS及1～3输出上限。DOA后端切换会清空活动轨迹，但同一session内ID计数不复用。

上游GI-DOAEnet固定为提交`af865978c783f309fc929f0f2499769a1c5499d5`和PM权重SHA-256 `d465...9fe8`。因该提交没有LICENSE文件，源码和权重不进入本仓库；运行`scripts/install_gi_doaenet.py --acknowledge-upstream-terms`下载安装到Git忽略目录。模型懒加载，默认仍为MUSIC；本机CUDA稳态实测适配器约5.8～11.8 ms/窗，首次加载约2.7秒。

IMCRA白化严格只读DecisionWindow携带的L1不可变快照，不拥有、更新或重置IMCRA状态。逐麦PSD形成的噪声模型为对角矩阵，收缩和diagonal loading后仍保持对角，因此实现使用逐麦逆平方根直接缩放协方差与steering，不执行逐频通用7×7 Cholesky/solve。16个hop的固定频率映射按批量向量化处理；缺少READY快照或有效对角项时明确退回未白化MUSIC。

## 默认MUSIC主链

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
    → 默认开启的全局方向轨迹分配
         birth/miss dummy行列 + linear_sum_assignment
         tentative / confirmed / coasting / deleted
         滚动200 ms内至少3次匹配才进入confirmed
         session内ID单调且不复用；寿命按48 kHz绝对sample
         Kalman关闭时，confirmed ID可按3秒圆周历史识别短时静止
    → 可选Kalman输出平滑（不拥有、不重置、不改变ID）
    → TrackedDirection[0..3] + active_tracks
```

`track_id`只表示空间方向轨迹，不是人物或声纹身份。当前项目的L5只在停机后的离线L4输出上运行，ApplicationRuntime不会调用L2的在线语义反馈接口，因此普通1.3.2运行完全按Gate概率门限决定是否执行MUSIC。达到L2 `confirmed`的实测或coasting轨迹可在数量与50°角距限制内作为公共方向进入L3，不要求L5人声证据；tentative轨迹不进入L3。ID使用2秒绝对sample TTL；预热、缺失或无效概率不会被伪造成有效概率。tentative固定使用20°关联门限；confirmed从最后一次真实观测起按`min(50°, 20° + 15°/s × 漏检时长)`扩张。未匹配峰位于任一现存非噪声ID预测位置±20°内时禁止birth，避免单一峰分裂成两个ID；噪声干扰ID不参与该排他判断。短漏检保留同一内部ID，超过TTL后再次观察会获得新ID。epoch会清除活动轨迹，但同一session的ID计数继续递增；新session建立新的ID命名空间。

为兼容旧在线分类实验，`Layer2Pipeline.submit_voice_feedback()`与`GlobalDirectionTracker.apply_voice_feedback()`仍保留精确`track_id`接口和专项测试：外部若显式提供至少2次正向结果，可使符合条件的confirmed轨在低Gate概率下强制放行；长期无正向结果还可触发噪声干扰标记。该接口当前没有Runtime调用方，不属于1.3.2普通主链，也不得用离线L5结果回写已经结束的实时轨迹。

内部活动ID硬上限为4，公共方向仍最多3个。新观测需要建立ID但内部已满时，优先淘汰未被本窗关联的噪声轨、无人声证据轨、tentative轨及最久未观测/低分轨；本窗已成功关联的轨迹受保护。该上限只控制ID内存与UI/试听扇出，不把Gate改成`WARMING_UP`。

## 滚动计算

比较历史固定包含160/240/320 ms，当前运行配置选择240 ms；`DecisionWindow`仍提供最多320 ms原始历史。连续窗口只计算新增的两个50%重叠STFT帧，并移出超出历史的两个旧帧；sample跳跃、epoch或session变化时从当前窗口重建。导向张量按阵列几何、频率轴和配置revision缓存。伪谱与轨迹每20 ms更新，MDL缓存最多100 ms。

`MusicDiagnostics`分别记录0～6的MDL诊断阶数、实际MUSIC阶数、手动上限导致的限制、高阶模型失配、有效频点、协方差质量、增量状态和协方差更新/eigh/谱融合/总耗时。Test UI可把手动上限设为1、2或3，下一窗口生效；该控制不覆盖MDL诊断值。MDL诊断大于3时标记模型失配并禁止该窗创建新ID，已有ID仍可关联或coasting。普通MDL路径不实现逐频真实峰支持和可靠性加权门禁；这些检查只属于默认关闭的DPD路径。目标机门禁为稳态p95不高于15 ms，单窗硬门限20 ms。

默认关闭的`DPD + rank-1 MUSIC`开启后不使用MDL决定候选数：可靠频点分别产生rank-1方向票，再进行跨359°/0°连续的圆周核聚类。每个方向簇必须至少有4个支持频点、覆盖4个等宽子带中的至少2个、获得至少0.20的可靠频点总权重且圆周集中度至少0.85。两个或多个归一化峰值均严格大于0.70、且组内任意两峰圆周距离不超过40°时，在50°NMS前按唯一支持频点并集的可靠性权重计算圆周平均角`theta_group`和`w_merge`；重复频点只计一次，禁止链式跨范围合并，融合证据重新通过原方向簇门禁且蓝色投票谱不重新归一化。合格簇数量决定0～手动上限个候选；逐候选诊断记录支持频点、支持率、子带数、集中度、平均平面波拟合度和簇权重。开关仍可由Test UI持久化并在运行中切换，默认值保持OFF。

## 兼容和删除边界

- 原SRP-PHAT与iterative multiple peak正式实现、配置、setter、UI开关和专属测试已删除；包不再导出旧扫描器。
- 正式主链默认启用ID追踪；Development Test UI可用持久化诊断开关临时关闭。开启后，JPDA把每个观测在已有轨迹、新生和伪报之间做联合概率分配；每条轨迹的IMM同时维护静止与慢速移动模型。滚动200 ms内3次观测确认，confirmed漏检后最多预测2秒，内部最多4条轨迹、公共最多3条。关闭时L2只返回无ID原始峰值，L3/L5跳过；不再存在独立Kalman开关或Q/R运行时调节。
- `CandidateDirection`仅作为尚未迁移消费者的同角度兼容投影；公共权威输出是`TrackedDirection`与`active_tracks`。
- 第8路HardwareMix只用于接口、显示和录音，不进入MUSIC协方差或导向计算。

## 算法来源

实现依据Schmidt MUSIC、Wax/Kailath MDL以及Pyroomacoustics MUSIC/NormMUSIC的公开算法表达；Pyroomacoustics为MIT许可。本项目没有复制或声称存在“Israel Cohen MUSIC开源实现”，Israel Cohen资料只用于项目另行记录的噪声估计与鲁棒性背景。

自动测试覆盖MDL 0～6阶、手动1/2/3阶上限、高阶窗禁止新ID、最多3个候选、跨0°、rank交换、JPDA一对一、新生/短漏检/TTL、Gate关闭、sample跳跃、epoch/session、IMM预测与重基准、HardwareMix隔离、滚动增量等价性和实时性能。

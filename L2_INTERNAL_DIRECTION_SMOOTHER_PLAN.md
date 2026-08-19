# L2圆周卡尔曼与私有ID追踪拆分说明

> 当前实现已改为两个模块，固定顺序为私有ID分配后再执行按ID圆周卡尔曼滤波。两个模块均默认关闭并可在Test UI中持久化、运行时切换；卡尔曼依赖ID追踪，ID关闭时卡尔曼不可开启。下文原合并方案仅作历史背景。
>
> **历史计划提示：** 下文“不输出预测轨迹”“候选数量与原始候选完全相同”等内容是最初迁移约束，已被当前实现覆盖。现行规则以[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)为准：临时ID在首个3秒内累计匹配至少5次且至少1次被L4识别为人声后转正并获得3秒语音租约；角度观测不续命，只有后续L4同窗人声角度被L2唯一匹配后才滑动延长3秒；公共候选仍受Top-2与45°间距限制；Test UI可读取对齐的私有ID/预测/正式/首次分配元数据用于SRP诊断显示与试听cache，但不得把ID送入L3/L4或正式记录。

状态：**代码迁移已完成；自动化测试已通过，实机动态参数标定待执行**  
权威关系：本计划细化[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)的L2内部方向平滑契约；如与v0.2历史说明冲突，以本计划和v0.3目标架构为准。

## 1. 目标与不变量

在SRP-PHAT完成候选筛选后增加内部`DirectionSmoother`：使用私有临时ID进行圆周关联和角度/角速度卡尔曼滤波，但不把ID写入任何公共DTO。

```text
SRP-PHAT
  → raw CandidateDirection[0..2]
  → 内部ID关联 + 圆周卡尔曼预测/校正
  → public CandidateDirection[0..2]
```

公共输出保持以下不变量：

- 类型仍为`tuple[CandidateDirection, ...]`，不新增`track_id/source_id`字段；
- 候选数量、rank顺序、时间身份、`raw_score`和`normalized_score`与SRP原始候选完全相同；
- 只把`theta_deg`替换为对应内部轨迹的卡尔曼后验角度；
- `SpatialResponse`始终是未平滑的原始360°SRP响应；
- 不输出仅靠预测存在的轨迹，不为漏检轨迹伪造候选或分数；
- Gate关闭、SRP无候选或窗口无效时，公共候选仍为空。

因此接口shape不变，但`CandidateDirection.theta_deg`的语义从“本窗口SRP峰值角”升级为“以本窗口SRP峰为测量更新后的平滑角”。两个score仍表示该候选的原始SRP测量证据，不表示平滑角处重新采样的空间响应值。

## 2. 内部状态与处理规则

内部轨迹至少维护：私有ID、展开角度、角速度、2×2协方差、最后decision sample、命中数和漏检窗口数。ID只用于本进程内关联，可在session/epoch重置后从1重新编号，不进入日志主键、录音资产或下游接口。

每个正式DecisionWindow按以下顺序处理：

1. 根据绝对`decision_sample`预测所有旧轨迹；跳窗时按sample差计算真实`dt`，不能假定只过去一个hop。
2. 用圆周最短角距离和门限形成可关联矩阵。
3. 对最多2个原始候选执行确定性一对一全局关联；候选输入顺序不得改变。
4. 匹配轨迹使用候选原始角度做卡尔曼校正，并在原候选副本中写入后验角度。
5. 未匹配候选创建新内部轨迹，当前窗口仍输出其原始角度。
6. 未匹配旧轨迹只在内部coast/计数/过期，本窗口不输出。
7. Gate阻断或空候选窗口仍推进内部时间和漏检寿命，但公共输出为空。
8. session或stream epoch改变时，在处理新窗口前清空全部轨迹。

0°/360°边界必须采用角度展开和圆周innovation，不能在线性角度上直接平均。相邻40 ms DOA窗口每20 ms发布、共享一半音频，测量高度相关；测量噪声不得按完全独立样本调得过小。

## 3. 确定性与安全规则

- 关联先最大化有效一对一匹配数，再最小化圆周角代价；完全同分时按内部ID、候选rank确定。
- 平滑后不重新执行threshold、prominence、NMS、Top-2或排序。
- 若多个后验角恰好重复，较低rank候选本窗口回退其原始角，避免公共tuple出现完全相同角度；不得人为增加随机扰动。
- 非有限状态、协方差失效或追踪器异常时，本窗口原样输出SRP候选、记录诊断并重置内部追踪状态；不能让平滑故障导致整个L2丢失有效定位。
- `DirectionSmoother`不得读取音频、重新扫描空间谱、调用L3/L4或依赖GUI模块。

## 4. 配置规划

唯一`config/config.yaml`的`layer2`下已增加`direction_smoothing`配置，schema拒绝未知字段，包括：

```text
enabled
backend = circular_kalman_v1
association_gate_deg
process_angle_std_deg
process_velocity_std_dps
measurement_std_deg
max_missed_windows
```

现有Test UI追踪器的20°关联门限、1.5°角过程标准差、25°/s速度过程标准差和5°测量标准差只能作为迁移初始值，必须用静止、移动、交叉、混响和双声源数据重新标定。当前UI为试听保留的3秒轨迹寿命不应直接冻结为正式L2默认值。

## 5. 各部分调整范围

| 部分 | 调整级别 | 需要修改的内容 |
|---|---|---|
| L1 / Ingest / Window | 无算法改动 | 继续提供唯一session/epoch/sample时间；增加跨epoch和跳窗测试输入即可。 |
| Common配置 | 中 | 增加`direction_smoothing`唯一配置schema；`CandidateDirection`字段不变，只更新`theta_deg`语义文档。 |
| L2 | 大 | 新增纯`DirectionSmoother`；在SRP候选筛选之后调用；Gate阻断时推进空测量；实现重置、异常回退和诊断。 |
| ApplicationRuntime | 中 | 只消费L2平滑后的候选；移除Test UI追踪器注入、预测方向额外L3批处理及对GUI追踪协议的导入。 |
| L3 | 小 | 输入类型不变；把平滑角原样用于steering并输出；不得再次卡尔曼滤波或改变rank。 |
| L4 | 小 | 输入类型和CNN不变；方向标签继承L3平滑角；不新增ID聚合或跨窗口状态。 |
| Development Test UI | 大 | 删除方向卡尔曼/ID关联和预测方向波束旁路；显示原始SpatialResponse与平滑候选点，不再把UI ID描述为算法ID。 |
| Production UI | 小 | 显示平滑候选角；如显示原始圆环，应提示候选点可能不严格位于峰顶。 |
| RecordingStore | 中 | DTO格式不增加ID；记录平滑候选、原始SpatialResponse、平滑器版本与配置hash，以支持离线复现。 |
| Tests | 大 | 新增L2追踪单元/序列测试，调整Runtime、L3/L4、UI和录音契约测试，删除UI承担正式方向滤波的断言。 |

## 6. Test UI职责拆分

原`gui/dev_test_ui/audio_id_tracker.py`同时承担方向关联、预测角L3增强、连续音频拼接和磁盘缓存，不能整体移动到L2，现已删除。迁移结果：

- 圆周卡尔曼、全局关联、轨迹寿命迁入L2纯算法模块；
- Runtime不再请求“仅为UI试听”的预测方向L3输出；
- UI直接消费L2平滑候选和对应正式L3增强音频；
- SRP面板可消费逐候选私有元数据，仅用颜色和点大小表达临时/正式及观测/预测状态；连续播放缓存只消费正式ID、已有L3预览及其对齐元数据。两者都不得再次滤波角度、影响L2租约或把ID送入L3/L4和正式记录。

## 7. 录音与复现语义

`SpatialResponse`保存原始360°响应，`CandidateDirection.theta_deg`保存平滑后角度，score保存原始候选证据。session manifest必须记录平滑器后端、版本、完整配置和配置hash。由于内部ID不持久化，离线复现需要从同一epoch起点按decision sample顺序重放全部L2窗口，不能随机单窗重算出相同平滑角。

## 8. 强制测试

- 静止方向抖动方差明显下降，同时稳态偏差不恶化；
- 358°→359°→1°→2°连续通过0°且不跳到180°附近；
- 匀速转动、突然转向和停止时的跟随误差、滞后及收敛；
- 双候选顺序变化、靠近、交叉、分离和短时漏检时的确定性关联；
- Gate关闭、空候选、窗口drop和大sample间隔只推进内部寿命，不输出预测候选；
- session/epoch变化清空状态，旧轨迹不能关联新流；
- 输出tuple的数量、顺序、时间字段和分数逐项等于原始候选，仅角度允许变化；
- 后验角重复时的确定性原始角回退；
- 非有限/异常追踪状态回退原始候选且不破坏正式L2结果；
- L3、L4、UI和RecordingStore中不存在`track_id/source_id`公共字段；
- 移除UI预测方向L3旁路后，正式候选以外不发生额外波束形成；
- 固定输入序列多次运行得到bitwise一致的角度和候选顺序；
- CPU处理时间和端到端延迟增量满足实时门禁。

## 9. 推荐实施顺序

1. 先冻结`theta_deg`新语义、异常回退和内部状态重置规则。
2. 增加配置schema和纯L2单元测试，再实现`DirectionSmoother`。
3. 接入L2 Pipeline，验证公共候选除角度外完全不变。
4. Runtime改为直接把平滑候选送入L3，删除GUI追踪器和预测方向旁路。
5. 同步L3/L4、Test UI、RecordingStore和离线回放测试。
6. 自动化迁移已完成；下一步使用实机静止/移动/双声源数据标定参数。

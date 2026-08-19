# Layer 2：Probability Gate、SRP-PHAT与内部方向平滑

权威契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。Probability Gate、SRP-PHAT及内部方向平滑均已实现。

## 目标主链

```text
DecisionWindow + gate_probability_40ms
    → Probability Gate
    → SRP-PHAT 360°扫描
    → Robust-Z/圆周峰值/NMS
    → Raw SpatialResponse + Raw CandidateDirection[0..2]
    → 可选私有ID追踪（默认关）
    → 可选按ID圆周卡尔曼滤波（默认关；依赖ID追踪）
    → Raw SpatialResponse + Smoothed CandidateDirection[0..2]
```

L2不再估计噪声，不维护IMCRA/PSD，也不提供NE后端选择。它只消费Runtime已按sample对齐的40 ms阵列声源概率：L1先从500～4000 Hz聚合每个20 ms概率，再对末尾两个连续概率做算术平均。

Gate默认阈值为`0.60`，判断采用`probability >= threshold`。阈值可由Development Test UI右侧滑动条实时调整，在下一个完整窗口边界生效、更新`config_revision`并持久保存，直到用户再次修改。它与SRP候选峰值门限是两个独立设置。概率缺失、跨epoch或IMCRA尚未预热时为`WARMING_UP/UNAVAILABLE`，不得当作0。

Gate关闭时不执行SRP并输出空候选；Gate开启时，SRP只读取窗口末尾40 ms、7路物理麦的**2000～4000 Hz**频点。第8路HardwareMix没有物理坐标，不得进入麦对、GCC-PHAT或steering delay。L1的Gate概率仍由500～4000 Hz聚合，不能与SRP定位频带混为一项配置。

正式候选数量固定为0～2。峰值通过threshold、prominence和**45°圆周NMS**后，按normalized score降序、角度升序破同分，只构造前两个`CandidateDirection`。这里的45°专指同一窗口内两个声源点之间的最小角差，不是对单个ID移动速度或位置稳定性的限制。圆周距离采用`min(|a-b|, 360-|a-b|)`，例如359°与2°相距3°；方向平滑后再次检查两个声源点的45°间距，不满足时重置平滑器并回退本窗口原始SRP候选。这是L2搜索阶段的明确选择，不允许Runtime或L3收到更多候选后再静默截断；搜索诊断记录限制前有效峰数和上限是否生效。完整360°空间响应不截断。

## 坐标与输出

算法从麦克风面观察：MIC0为0°/+x，MIC1～MIC5沿逆时针为60°、120°、180°、240°、300°。所有360°响应、候选角度和UI显示均使用这一坐标，必须重做无镜像、无反向、无固定旋转测试。

SRP候选筛选后先由`DirectionIdTracker`按圆周距离进行一对一全局关联并分配私有ID，再由`CircularKalmanFilter`以该ID为状态键处理圆周最短角差并替换`theta_deg`。因此候选rank来回切换时，滤波状态仍跟随物理方向对应的ID。两个模块默认关闭；ID追踪可以单独开启，但卡尔曼只能在ID追踪开启时启用，关闭ID会同步关闭卡尔曼。运行时切换从下一个完整窗口生效。候选数量、rank、时间字段及分数始终不变，ID不进入公共`CandidateDirection`，`SpatialResponse`保持原始360°响应。L2内部结果可携带与候选对齐的ID、预测、正式和首次分配标志，唯一允许的外部消费者是本机Test UI诊断投影（SRP身份显示与试听sidecar）；L3、L4、录音和数据集均不得接收。

两个模块同时开启时，新建的临时ID从首次建立起按绝对sample观察3秒；在这首个3秒中必须累计归并至少5次自然Gate窗口的真实SRP候选，并至少有1个同窗角度被L4识别为人声，才转为正式ID。临时阶段的人声证据只满足转正确认条件，不提前赋予租约；正式化时获得3秒语音租约。角度匹配、卡尔曼校正、预测及ID强制Gate均不续命；只有后续L4人声角度被历史记录唯一匹配到仍存活正式ID时，才把截止sample滑动到该人声点之后3秒。L4只发送session/epoch/decision sample/角度，不读取或发送私有ID；L2以同窗历史和20°圆周门限自动匹配，匹配不到、存在歧义或ID已删除均拒绝续命。低P强制窗口可更新已有正式ID位置，但不能创建或晋升新ID；预测后的首次重匹配仍以2倍测量可信度回拉。租约到期后在Gate判断和关联前删除ID及其卡尔曼状态，最后一个ID删除后立即恢复按P判断。预测仅补足最多2个候选并继续满足两个声源点45°圆周间距。L2公共DTO不增加字段，只输出角度及原有Raw/Norm。

运行时调参不直接改写不同单位的Q矩阵元素，而使用`process_noise_scale`整体缩放基础过程噪声矩阵、`measurement_noise_scale`整体缩放基础测量噪声方差。二者初始1.00、范围0.02～10.00、调节步长0.1（0.02作为最小端点），可在Test UI分别暂调并点击应用；成功应用后原子持久化，从下一个窗口生效且不清空滤波状态。

Gate因概率预热、缺失、无效而阻断或当前没有`SpatialResponse`时公共候选为空；已有当前响应但没有可归并真实候选时，只有租约仍有效的正式ID才可预测续报，Raw/Norm必须从当前响应取样而非伪造。正式租约按48 kHz绝对decision sample推进，不按实际处理次数推进，因而丢窗不会改变3秒语义。L4反馈通过有界线程安全邮箱送回，仅由L2线程应用；迟到反馈不能复活已删除ID。session/epoch变化立即重置状态与反馈。追踪器异常或产生非有限状态时，本窗口回退原始候选并清空状态。平滑后不重新NMS、Top-2或排序。

该功能只提供跨窗口角度降抖，不提供人员身份、声纹或下游稳定ID。完整规则见根目录[`L2_INTERNAL_DIRECTION_SMOOTHER_PLAN.md`](../L2_INTERNAL_DIRECTION_SMOOTHER_PLAN.md)。

## 核心测试

- L2不实例化或调用Noise Estimation；
- 500～4000 Hz子带聚合及两个连续20 ms概率平均成40 ms概率；
- 默认0.60、包含等于门限、动态revision及Gate关闭不运行SRP；
- 单次、迭代、异常回退与UI重新筛选均最多输出2个正式候选；
- 0°圆周连续、静止降抖、运动跟随、双候选交叉、漏检/跳窗、Gate阻断及epoch重置；
- 平滑前后数量、rank、时间身份和score完全相同，仅角度可变；
- ID不进入公共候选、L3/L4或正式记录；仅Test UI的SRP诊断显示与试听sidecar可消费对齐的内部元数据；异常回退原始角且输出确定；
- 7麦21对、HardwareMix隔离、360°与边界峰；
- 麦克风面全角度定位无镜像，实机轴向与UI一致。

# IngestCoordinator：v0.3目标时间轴契约

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。**当前代码仍是v0.2，本文描述待迁移接口。**

IngestCoordinator消费L1逻辑8通道音频、InputHealthEvent及同sample区间的IMCRA结果，统一分配`session_id`、`stream_epoch`和半开绝对sample边界。它不改变通道内容，也不重新计算IMCRA。

输出`IngestedAudioBlock.samples`目标shape为`float32 [N,8]`，顺序为`MIC0..MIC5、Center、HardwareMix`；存在native副本时仍为Host原始8ch顺序。数组必须finite、C-contiguous、只读且`N>0`。

WindowAssembler与RecordingStore订阅同一份不可变block。sequence gap、timestamp/rate异常、设备重启、overflow或handoff丢块只在这里转换为epoch重置；不得补零或让不同消费者建立不同时间轴。同一次连续handoff溢出突发先在采集边界合并为一个带缺失范围的健康事件，因此只产生一次epoch重置；真正分离的多次缺口仍分别重置。IMCRA结果必须继承同一epoch和sample边界，旧epoch结果不得进入新窗口。

迁移测试覆盖8ch所有权、音频/IMCRA身份对齐、断流重置、overflow、有界handoff及录音/算法同时间轴。

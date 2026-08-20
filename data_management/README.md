# Audio Data Manager：DecisionRecord v4与公共方向ID存储

本目录已纳入项目`1.2.1`：新结果写`decision_record_v4`，MUSIC诊断、公共`track_id`、逐ID增强资产和时间线进入正式存储；旧`decision_record_v3`只读兼容，不原地迁移或补造ID。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。L1 IMCRA sidecar已按v0.3迁移；其余资产的状态以目标架构和当前代码为准。

RecordingStore旁路订阅IngestCoordinator的唯一时间轴，不参与实时算法控制，也不自行推导sample。Runtime Sessions与Test Corpus继续严格隔离，通过manifest、SHA-256、SQLite WAL Catalog和lineage索引。

## 新增/调整资产

- native Host 8ch：`CH0..CH5、HardwareMix、Center`原始顺序；
- logical 8ch：`MIC0..MIC5、Center、HardwareMix`算法顺序；
- 可选physical 7ch派生视图；
- 每20 ms IMCRA：算法版本`cohen_imcra_2003_l1_v2`、状态、342点频率轴，以及形状为`[record,7,342]`的噪声PSD、两轮平滑谱、两轮最小值、`q_hat`、SPP和先验/后验SNR；另存`noise_features[record,7,4]`、每麦概率、500～4000 Hz阵列聚合概率与来源sequence；
- 每40 ms Gate：两个来源hop、平均概率、阈值、revision和Gate结果；
- MUSIC/NormMUSIC 360点空间谱、model order、有效频点及协方差质量；公共方向的`track_id`、观测角、输出角、轨迹状态、`active_tracks`和Kalman应用状态；L3逐ID 48 kHz增强音频；L4逐ID概率与判断。

所有sidecar均以session、epoch及绝对sample区间关联，不能按写入时间猜测对应关系。manifest必须记录0～8000 Hz IMCRA频带、500～4000 Hz Gate证据带、方向算法版本/配置/hash、官方MIC/I²S关系、Host/logical映射和几何版本。HardwareMix没有物理坐标。离线复现有状态算法必须从同一epoch起点顺序重放，不能随机单窗推导内部状态。

IMCRA NPZ字段固定为`start_samples、end_samples、source_sequence_ids、algorithm_versions、states、frequencies_hz、noise_psd、smoothed_psd、conditional_smoothed_psd、minimum_psd、conditional_minimum_psd、spp、speech_absence_probability、posterior_snr、prior_snr、noise_features、noise_level_db、source_probability_per_mic、array_source_probability_20ms`。manifest中的`frequency_bin_count`必须从实际频率轴写入，当前版本为342，不能沿用完整RFFT的1025。

写盘使用独立有界音频/结果队列；故障、队列溢出或磁盘不足不得反压采集。Runtime对已有序合并的每个窗口调用`append_result_with_watermark`：结果与同窗水位以一条队列命令原子接纳，满队列时两者均不入队、生产者水位不前进，并在manifest记录`result_overflow`缺口。兼容的分开`append_result/advance_result_watermark`接口仍存在，但当前ApplicationRuntime不使用它们完成正式提交。

录音配置同样受唯一严格schema约束：`chunk_seconds`、`audio_queue_seconds`、`result_queue_capacity`、`retention_days`和`max_storage_gb`必须大于0，`min_free_storage_gb`必须非负且小于`max_storage_gb`。`result_queue_capacity`默认256条，schema和RecordingStore均以256为硬上限；不允许通过配置扩成更大的内存队列。无效队列容量或存储预算在创建RecordingStore前失败。根`config/config.yaml`由`config`包作为package data打入wheel，不另复制第二份录音默认值。

## 有界内存与崩溃恢复

- 音频仍按默认60秒chunk切分。结果只保留到对应已封闭chunk的writer watermark达到`end_sample`；满足后立即写出该chunk的JSONL/NPZ/sidecar并释放内存。若水位长时不到，待保留结果仍受硬数量上限保护，溢出记录`result_retention_overflow`。
- `event`模式的音频环和结果环都按sample只保留最新2秒pre-roll；旧的未触发结果会被裁剪。同一epoch内，后续触发的pre-roll起点不晚于当前事件post-roll终点时合并，manifest `event_triggers`每个合并事件段只写一条审计，兼容字段`window_id/decision_sample/reason`指首触发，并记录`first_window_id/last_window_id`、`first_decision_sample/last_decision_sample`、`start_sample/end_sample`和`trigger_count`。跨epoch或不重叠触发新建事件段；不保存逐窗触发列表。
- 存储容量扫描只在新事件段开始前执行，连续合并触发在锁内O(1)扩展审计和post-roll，不以50 Hz重复扫描磁盘。若新触发发生在旧post-roll之后、但2秒pre-roll仍与旧事件相交，则从有界音频环补回间隙再延长事件，避免manifest标记连续而实际音频有洞。`off`和非活动`manual`模式在复制大数组前就丢弃非录制结果。
- Hotmap按CDC sequence去重并直接流式写入`hotmaps.jsonl.partial`，session内不累积矩阵列表；停机时flush/fsync、原子改名并记录SHA-256与count。
- 普通chunk的WAV、NPY、noise NPZ和IMCRA NPZ使用同一个`chunk_asset_commit_<stem>.json`事务journal，且journal在首次final改名前持久化。恢复时，只有manifest已按hash完整索引全批资产才视为已提交；否则partial、未完整索引的final与journal整批进入quarantine。崩溃留下的open manifest会校验已索引资产，移除临时增强项后原子改写为incomplete恢复状态。
- 增强波形在其音频区间已写盘后立即落到WAV `.partial`，文件名和manifest索引都包含公共`track_id`，然后从内存结果中释放。session封存使用`enhanced_asset_commit_v2` journal保护“逐ID WAV改名→写manifest”窗口；恢复时，只有已被完整manifest按hash和track索引的终态文件视为已提交，其他partial或孤立终态文件全部进quarantine。

DecisionRecord v4中任一阶段`failed/timed_out/dropped/cancelled`都必须对应`status=error`；只有算法输出完整、但成功使用已声明回退路径时才是`degraded`。一旦一个方向批次携带ID，L2、L3、L4的ID集合与顺序必须一致，同窗ID不得重复。每个Runtime丢弃窗口在JSONL只保留一条正式终态DecisionRecord，watermark中的drop信息不再重复写第二条结果行。

Catalog新增`direction_observations`投影，正式键为`(session_id, stream_epoch, track_id)`。数据服务提供轨迹摘要、逐sample时间线、持续时间、首末角/角度变化、最新与平均L4概率、逐ID增强资产以及native/logical/physical资产查询。Catalog重建会从v4结果和manifest重新生成投影；v3行可读取但不会生成公共ID。

60秒切块、hash、Catalog重建、QA、标注、split、retention、Trash与导出的其他职责保持现有边界。

## UI调整

Production UI运行录音详情显示方向ID、epoch、首末sample、持续时间、首末角、角度变化、状态和L4概率；可以试听按20 ms决策增量去除320 ms重叠后的逐ID增强时间线、Center参考及native/logical/physical任意通道。专用L1录音向导和manifest明确显示“无算法方向ID”。

## 持续回归与实机验收

- native/logical/physical三种视图sample范围一致；
- IMCRA 20 ms、Gate 40 ms、SRP/L3/L4结果sample级对齐；
- 原始SpatialResponse、平滑候选和下游角度一致；资产及Catalog中不存在内部ID；
- 使用同一配置从epoch起点顺序重放可复现平滑角；
- 映射与麦克风面几何写入manifest并可重建；
- hash、恢复、overflow、Catalog、lineage、QA、Trash及不反压性能回归。
- 原子result+watermark队列溢出时两者均不接纳，水位不假前进；
- 连续多chunk运行后已封闭chunk结果被写出并从RAM释放，event未触发结果始终不超过pre-roll/硬数量上限；连续触发合并为一条有界审计、容量扫描每事件段最多一次，跨epoch不合并且post-roll间隙可由ring补回；
- hotmap流写入不保留session级矩阵列表，普通chunk整批资产及增强WAV在partial、journal、改名、manifest之间任一崩溃点均可恢复或进quarantine。
- 严格配置拒绝零/负chunk与队列容量及非法存储预算；wheel打包契约包含唯一`config/config.yaml`。

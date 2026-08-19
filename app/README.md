# ApplicationRuntime：唯一时间轴与分阶段流水

> 项目 1.1.0 待实现改动见[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#11-runtime时间线与并行管理)：保留 WindowKey、有界分层流水与有序 Joiner，新增跨层 ID 对齐、MUSIC 配置快照和 DecisionRecord v4，并删除 angle-only L4 feedback 与私有 ID 投影。下文描述当前 1.0.1 实现。

权威架构见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)，详细计划见[`L2_INTERNAL_DIRECTION_SMOOTHER_PLAN.md`](../L2_INTERNAL_DIRECTION_SMOOTHER_PLAN.md)。**本页描述已实施的Runtime边界。**

Runtime负责唯一L1→Ingest→Window装配，以及L2、L3、L4分阶段流水调度。每个`DecisionWindow`只生成一次不可变`WindowWorkItem`，正式身份固定为`WindowKey(session_id, stream_epoch, window_id, decision_sample)`；入队时冻结该窗口使用的Gate/SRP/方向平滑、L3模式、L4阈值、几何与config hash。所有StageResult必须继承完全相同的键。

L2、L3、L4各有独立单worker和有界latest-wins等待队列。同一窗口仍严格按L2→L3→L4传递；稳态时允许L2(n)、L3(n-1)、L4(n-2)并行。新任务遇到本层满队列时只替换尚未被worker取走的最旧任务，已开始计算不取消。L2队列丢弃使三阶段全部`DROPPED`；L3队列丢弃保留L2、标记L3/L4 `DROPPED`；L4队列丢弃保留L2/L3、标记L4 `DROPPED`。L2的私有方向状态和L3的滚动统计不会被多worker乱序更新。

`ResultJoiner`接收乱序完成的L2/L3/L4终态，只在同一WindowKey完整后生成一个`JoinedWindowResult`。commit阶段按全局window ID有序调用RecordingStore的原子`append_result_with_watermark`，之后再做Test UI投影；stage worker不得绕过Joiner直接发布正式结果。Gate阻断产生L3/L4 `SKIPPED`；任一阶段`FAILED/TIMED_OUT/DROPPED/CANCELLED`都产生`error` DecisionRecord；仅完整成功但使用声明回退的结果为`degraded`。

L4 worker另有一个严格限于Test UI的`latest_l4_dev_ui` side channel：只有正式L4计算`COMPLETED`后才发布包含完整同窗L2空间响应、L3预览和L4结果的`DevUiFrame`。邮箱固定`maxsize=1`，新完成帧覆盖旧显示帧并累计覆盖数；失败不发布。该路径绕过的只是有序UI等待，不绕过ResultJoiner的正式结果、录音或watermark顺序。UI收到后续有序`DROPPED/SKIPPED`帧时保留最近有效CNN画面，直至`dev_test_ui.stale_after_ms`超时。

ResultJoiner注册前若在途窗口/字节容量已满，Runtime不保留新窗口的320 ms音频，只在有界范围审计中压缩保留身份、sample边界与原因；commit遇到对应window ID时展开为轻量`error` DecisionRecord和watermark。这条pre-joiner拒绝路径不会把容量异常抛回L1采集循环。

Runtime的`ComputeCache`按L2/L3/L4分区并受窗口数、分区字节和全局字节硬限制。它只缓存CPU可复用artifact，不能接纳CUDA张量，也不是StageResult的正确性来源。L3自己的滚动STFT/噪声统计、静态steering/p查询及prepared GPU context另有小容量硬上限；完成提交后按窗口退休。

Join后也没有无界队列：completion主队列和后备backlog均受`completion_queue_windows`硬限制，两者都满时拒绝新接纳并由Joiner保留已注册有界结果待重试。commit乱序表的软限为`max_inflight_windows`，硬限为`2*max_inflight_windows + 2*completion_queue_windows`。

Runtime还负责L1预降噪选择：IMCRA始终先读取原始音频；`ImcraWienerPreDenoiser`持续维护40 ms/20 ms WOLA状态。开关关闭时发布原始hop，开启时等待一个hop固定延迟并发布sample边界相同、前7路已替换的降噪hop。开关从关闭切到开启时不得重复发布此前已经旁路的音频。

已经删除：

- 先规划UI轨迹、再为预测角构造临时`CandidateDirection`的逻辑；
- `process_with_tracking_previews`预测方向额外L3批次；
- 任何由Test UI反向驱动正式候选、角度或额外波束形成的生命周期。

当前保留两条仅限Test UI的L2私有元数据投影。右上SRP面板消费逐候选ID、预测、正式和首次分配标志，只用颜色与点大小显示身份/观测状态。可选`dev_audio_tracker`试听sidecar只在正式结果提交后消费已有候选、L3预览及其对齐元数据，并忽略临时ID和无ID候选；production入口不启用。它以正式私有ID为优先关联键，ID换号只在20°内唯一匹配时续接，同时出现的近角双ID保持分离；缺失方向等待3秒。音频按绝对decision sample逐20 ms拼接，可从当前320 ms预览恢复的跳窗补真实音频，过旧缺口补等时静音，连续边界交叉淡化。首行另缓存预降噪前LogicalAudio第7路Center Mic原音参考；方向轨累计2秒才显示。缓存按10秒分段、最多3段及8条已结束方向轨。Gate暂时不可用及同一session的epoch恢复不会删除缓存；旧epoch轨道归档为`ENDED`，重复的L2私有ID映射到新的Test UI试听ID。新session、L3模式切换和关闭UI仍按职责重置或删除缓存。播放器只对稳定快照作试听归一化，不修改正式音频。

Runtime不得把L2内部ID加入公共候选、L3/L4输入或正式记录；它只可将逐候选私有元数据原样转交Test UI的SRP身份显示和试听sidecar，不得据此再次平滑、重排或截断候选。L4只通过线程安全邮箱向L2回送最终判为人声的同窗时间身份与角度，ID匹配及3秒语音租约仍完全由L2负责；反馈不携带私有ID。Gate阻断时按L2空候选继续推进正式watermark；追踪器的重置、寿命和异常回退属于L2。

Gate开启且L2空间响应有效但候选为空时，L3直接产生`Layer3Output(())`，不执行prepare/STFT/协方差；L4仍调用空batch公共接口并以`COMPLETED`空结果收束。正式记录中的三阶段均为completed，增强音频和Voice方向为空。

启动顺序为：重置图和时间轴 → RecordingStore session → `commit,L4,L3,L2` worker → 设备pipeline → L1读取；启动失败按反向回滚并join所有已启动线程。正常停止不清空等待队列，而是先停设备/L1并刷出预降噪，再依次以EOS drain L2→L3→L4→completion/commit，完成最终Join与Recording水位后才关闭RecordingStore。超时的已注册窗口显式转为`CANCELLED/error`；仍有worker存活时拒绝假关闭。

Development Test UI只通过公开只读`processing_status`获取每阶段队列深度/容量、worker存活、在途窗口、缓存字节、完成数和错误数；其中L4诊断包括`l4_actual_completed/l4_dropped/l4_skipped/l4_actual_hz`及显示邮箱深度、容量、覆盖数。UI不得读取Runtime私有队列或据此修改调度。

回归测试证明：相同平滑候选一一进入L3/L4；正式候选以外没有额外波束形成；普通运行与Test UI运行使用同一正式L2/L3/L4链路。

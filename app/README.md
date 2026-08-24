# ApplicationRuntime：唯一时间轴与分阶段流水

> 本目录实现项目`1.3.2`的唯一时间轴、分层并行Runtime、有界缓存与有序结果提交；真实设备长时间运行仍按根架构门禁验收。

权威目标架构见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)。**本页描述当前分支已实施的Runtime边界。**

Runtime负责唯一L1→Ingest→Window装配，以及L2、L3与Hub长音频封存。每个`DecisionWindow`只生成一次不可变`WindowWorkItem`，正式身份固定为`WindowKey(session_id, stream_epoch, window_id, decision_sample)`。实时L5 worker只提交`offline_after_l4`跳过终态，不执行CNN；停机排空后由`offline_l4_sources`和`run_offline_l4()`把Hub封存包交给离线L4/L5。

Runtime保留L2、L3、L5三个有界阶段队列和有序审计结构；当前实时计算只执行L2与L3，L5 worker为每个成功L3窗口提交`offline_after_l4`的`SKIPPED`终态。L2 worker独占滚动MUSIC/方向轨迹状态和预计算导向缓存；连续窗口增量推进，sample跳跃或revision变化时安全重建并发布诊断。L4与真正的L5推理不占用实时队列，只消费停机排空后的Hub封存包。

`ResultJoiner`接收乱序完成的L2/L3/L5终态，只在同一WindowKey完整后生成一个`JoinedWindowResult`。L2与L3完成结果携带有序公共`(track_id, theta_deg)`并逐项校验；当前实时L5是没有检测输出的`offline_after_l4`跳过终态。commit阶段按全局window ID有序调用RecordingStore的原子`append_result_with_watermark`，之后再做Test UI投影；stage worker不得绕过Joiner直接发布正式结果。Gate阻断产生L3/L5 `SKIPPED`；任一阶段`FAILED/TIMED_OUT/DROPPED/CANCELLED`都产生唯一`error` DecisionRecord v5；仅完整成功但使用声明回退的结果为`degraded`。旧DecisionRecord v3仅支持只读，不原地改写。

`latest_l5_dev_ui` side channel为兼容性保留；全离线L5架构下实时链不会向其发布CNN结果。Development Test UI已经提供L3/L4/L5三栏和唯一的“发送到L4”入口；整批L4完成后由同一后台任务自动运行L5并直接更新离线面板。

ResultJoiner注册前若在途窗口/字节容量已满，Runtime不保留新窗口的160 ms音频，只在有界范围审计中压缩保留身份、sample边界与原因；commit遇到对应window ID时展开为轻量`error` DecisionRecord和watermark。这条pre-joiner拒绝路径不会把容量异常抛回L1采集循环。

Runtime的`ComputeCache`按L2/L3/L5分区并受窗口数、分区字节和全局字节硬限制。它只缓存CPU可复用artifact，不能接纳CUDA张量，也不是StageResult的正确性来源。L3自己的滚动STFT/噪声统计、静态steering/p查询及prepared GPU context另有小容量硬上限；完成提交后按窗口退休。

Join后也没有无界队列：completion主队列和后备backlog均受`completion_queue_windows`硬限制，两者都满时拒绝新接纳并由Joiner保留已注册有界结果待重试。commit乱序表的软限为`max_inflight_windows`，硬限为`2*max_inflight_windows + 2*completion_queue_windows`。

Runtime还负责L1预降噪选择：IMCRA始终先读取原始音频；`ImcraWienerPreDenoiser`持续维护40 ms/20 ms WOLA状态。开关关闭时发布原始hop，开启时等待一个hop固定延迟并发布sample边界相同、前7路已替换的降噪hop。开关从关闭切到开启时不得重复发布此前已经旁路的音频。

L2的`TrackedDirection`是唯一权威方向身份。Runtime把同一`track_id`、角度和原始顺序传给L3、DecisionRecord v5、Development Test UI与Production数据服务；停机封存后的L4/L5继续继承该ID。任何下游都不得按角度、rank或试听状态创建、合并、续租或修补ID。同一session切换epoch时L2清空运动状态，但ID计数器继续单调递增。Gate关闭、latest-wins丢窗和sample跳跃都按绝对sample推进coasting/TTL，不重置整个tracker。

Development Test UI的方向音频只按`(session_id, stream_epoch, track_id)`拼接，保留20 ms hop、真实音频补洞、过旧缺口等时静音、交叉淡化、Center Mic参考、有界分段和L3模式隔离。UI不再维护私有ID投影、角度贪心关联或别名合并。当前离线L5只继承公共ID并返回该ID的人声语义结果，不向L2反馈，也不拥有方向轨迹生命周期。

Development Test UI可实时关闭下游处理。关闭后，L2继续正常处理、追踪和显示；新L2结果直接生成`downstream_disabled_by_test_ui`的L3/L5 `SKIPPED`终态，已经排队但尚未开始的L3/L5工作也快速跳过，当前正在执行的单窗允许安全收尾。该状态不计为错误，不破坏ResultJoiner、DecisionRecord或watermark顺序；重新开启后从下一条L2结果恢复L3/L5。

Gate开启且L2空间响应有效但候选为空时，L3直接产生`Layer3Output(())`，不执行prepare/STFT/协方差；实时L5不调用空batch模型，仍以`offline_after_l4`的`SKIPPED`终态收束。增强音频和Voice方向为空。

启动顺序为：重置图和时间轴 → RecordingStore session → `commit,L5,L3,L2` worker → 设备pipeline → L1读取；启动失败按反向回滚并join所有已启动线程。正常停止不清空等待队列，而是先停设备/L1并刷出预降噪，再依次以EOS drain L2→L3→L5→completion/commit，完成最终Join与Recording水位后才关闭RecordingStore。超时的已注册窗口显式转为`CANCELLED/error`；仍有worker存活时拒绝假关闭。

Development Test UI只通过公开只读`processing_status`获取每阶段队列深度/容量、worker存活、在途窗口、缓存字节、完成数和错误数；其中`input_health`公开当前epoch、连续性中断次数/最后原因、input overflow、handoff drop及交接队列深度/容量/高水位，L5诊断包括`l5_actual_completed/l5_dropped/l5_skipped/l5_actual_hz`及显示邮箱深度、容量、覆盖数。Gate因新epoch重新预热时，UI错误文案附带`epoch_reset:<reason>`，可直接区分静音、处理丢窗与真实输入中断。UI不得读取Runtime私有队列或据此修改调度。

回归测试覆盖公共ID逐项进入L3/L5、跨层错序/角度/WindowKey拒绝、队列丢弃、sample跳跃、epoch变化、配置revision、停机drain、唯一终态/watermark和有序提交；普通运行与Test UI运行使用同一正式L2/L3/L5链路。

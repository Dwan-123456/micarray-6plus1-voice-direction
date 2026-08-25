# ApplicationRuntime：唯一时间轴与分阶段流水

> 本目录实现项目`1.3.5`开发线的唯一时间轴、分层并行Runtime、有界缓存、有序结果提交与L4-L6渐进旁路；真实设备长时间运行仍按根架构门禁验收。

权威目标架构见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)。**本页描述当前分支已实施的Runtime边界。**

Runtime负责唯一L1→Ingest→Window装配，以及L2、L3与Hub长音频封存。每个`DecisionWindow`只生成一次不可变`WindowWorkItem`，正式身份固定为`WindowKey(session_id, stream_epoch, window_id, decision_sample)`。实时L5审计worker只提交`offline_after_l4`跳过终态，不执行CNN。与它隔离的L4-L6旁路按配置化3～15秒ID块发布可替换preview；停机排空后仍由`offline_l4_sources`和`run_offline_l4()`把Hub完整封存包交给权威L4/L5/L6。

Runtime保留L2、L3、L5审计三个有界阶段队列和有序结构；L5审计worker为每个成功L3窗口提交`offline_after_l4`的`SKIPPED`终态。L2 worker独占滚动MUSIC/方向轨迹状态和预计算导向缓存；连续窗口增量推进，sample跳跃或revision变化时安全重建并发布诊断。真正的L4/L5/L6模型不占用这些正式队列：L3 host只发送轻量wakeup，专用chunk producer在后台从Hub claim连续块，sidecar接纳成功后才ack并推进ID游标。

Development Test UI的L2面板使用容量1的`latest_l2_dev_ui`旁路：L2 worker完成即发布不可变L2快照，不等待同窗L3/L5终态或有序Commit。正式`latest_dev_ui`仍只发布完整Join帧并服务L3显示、录音与审计；L2旁路只影响可视化，UI按`session/epoch/window/decision_sample`过滤旧流和倒序迟到快照。

`ResultJoiner`接收乱序完成的L2/L3/L5终态，只在同一WindowKey完整后生成一个`JoinedWindowResult`。L2与L3完成结果携带有序公共`(track_id, theta_deg)`并逐项校验；当前实时L5是没有检测输出的`offline_after_l4`跳过终态。commit阶段按全局window ID有序调用RecordingStore的原子`append_result_with_watermark`，之后再做Test UI投影；stage worker不得绕过Joiner直接发布正式结果。Gate阻断产生L3/L5 `SKIPPED`；任一阶段`FAILED/TIMED_OUT/DROPPED/CANCELLED`都产生唯一`error` DecisionRecord v5；仅完整成功但使用声明回退的结果为`degraded`。旧DecisionRecord v3仅支持只读，不原地改写。

`latest_l5_dev_ui` side channel为正式逐窗兼容性保留，不发布CNN结果。L4-L6旁路使用独立的latest-only revision邮箱；队列、模型或停机失败不会反压L1-L3，也不能把不完整前缀标成final。Development Test UI采集中显示渐进preview，封存后的完整canonical成功时才一次性替换。

ResultJoiner注册前若在途窗口/字节容量已满，Runtime不保留新窗口的160 ms音频，只在有界范围审计中压缩保留身份、sample边界与原因；commit遇到对应window ID时展开为轻量`error` DecisionRecord和watermark。这条pre-joiner拒绝路径不会把容量异常抛回L1采集循环。

Runtime的`ComputeCache`按L2/L3/L5分区并受窗口数、分区字节和全局字节硬限制。它只缓存CPU可复用artifact，不能接纳CUDA张量，也不是StageResult的正确性来源。L3自己的滚动STFT/噪声统计、静态steering/p查询及prepared GPU context另有小容量硬上限；完成提交后按窗口退休。

L3显式选择CUDA时，Runtime只把当前L3队列中已经存在的连续工作合并为最多`l3_cuda_microbatch_windows`个窗口，不等待未来输入；各窗依次维护滚动状态，STFT、IMCRA协方差、BF和ISTFT留在同一CUDA stream，短音频异步复制到pinned CPU buffer后只同步一次，再按原WindowKey顺序进入CPU TrackAudioStreamHub。默认配置仍选择实测更快的CPU路径。若L3与停机后L4都选择CUDA，创建离线L4前必须确认实时worker已排空，并释放L3 device cache。

渐进L4-L6默认把DS/L1-L3、Hub、MarbleNet、DNSMOS、CAMPPlus和L6留在CPU，只把MF2放到CUDA。配置禁止L3和渐进L4同时选择CUDA，避免两条实时链争用同一8 GB级显存。后台worker在首块累计期间预载模型，MF2/CAMPPlus实例在渐进与canonical间复用。L3每次完成只设置事件；Hub锁内只冻结区间，构造大块、补洞、拼接和SHA都由独立producer在锁外完成。Hub使用claim/resolve事务：下游队列未接纳时不推进cursor，可在下一轮精确重试。每轮claim数受空闲队列槽限制，不会一次物化全部backlog。

Join后也没有无界队列：completion主队列和后备backlog均受`completion_queue_windows`硬限制，两者都满时拒绝新接纳并由Joiner保留已注册有界结果待重试。commit乱序表的软限为`max_inflight_windows`，硬限为`2*max_inflight_windows + 2*completion_queue_windows`。

Runtime还负责L1预降噪选择：IMCRA始终先读取原始音频；`ImcraWienerPreDenoiser`持续维护40 ms/20 ms WOLA状态。开关关闭时发布原始hop，开启时等待一个hop固定延迟并发布sample边界相同、前7路已替换的降噪hop。开关从关闭切到开启时不得重复发布此前已经旁路的音频。

L2的`TrackedDirection`是唯一权威方向身份。Runtime把同一`track_id`、角度和原始顺序传给L3、DecisionRecord v5、Development Test UI与Production数据服务；停机封存后的L4/L5继续继承该ID。任何下游都不得按角度、rank或试听状态创建、合并、续租或修补ID。同一session切换epoch时L2清空运动状态，但ID计数器继续单调递增。Gate关闭、latest-wins丢窗和sample跳跃都按绝对sample推进coasting/TTL，不重置整个tracker。

Development Test UI的方向音频只按`(session_id, stream_epoch, track_id)`拼接。Runtime在每个L2完成窗口先登记confirmed/coasting权威20 ms绝对时间槽，L3只填入BF波形；首尾或中间没有BF结果的槽保留等时静音，因此轨长只由L2首尾sample决定。其余仍保留真实音频补洞、交叉淡化、Center Mic参考、有界分段和L3模式隔离。UI不再维护私有ID投影、角度贪心关联或别名合并。当前离线L5只继承公共ID并返回该ID的人声语义结果，不向L2反馈，也不拥有方向轨迹生命周期。

Development Test UI可实时关闭下游处理。真实麦克风模式每次启动默认进入临时预热阶段：只运行L1/IMCRA和L2 MUSIC/ID，缓存Center试听，禁止提前开启L3/4/5；点击“正式录音开始”只继承IMCRA噪声统计，在与L2 worker互斥的边界清空MUSIC和全部ID追踪状态，同时清空Center与下游音轨缓存后打开L3。关闭下游后，L2继续正常处理、追踪和显示；新L2结果直接生成`downstream_disabled_by_test_ui`的L3/L5 `SKIPPED`终态，已经排队但尚未开始的L3/L5工作也快速跳过，当前正在执行的单窗允许安全收尾。该状态不计为错误，不破坏ResultJoiner顺序。

Gate开启且L2空间响应有效但候选为空时，L3直接产生`Layer3Output(())`，不执行prepare/STFT/协方差；实时L5不调用空batch模型，仍以`offline_after_l4`的`SKIPPED`终态收束。增强音频和Voice方向为空。

普通Runtime启动顺序为：重置图和时间轴 → RecordingStore session → `commit,L5,L3,L2` worker → 设备pipeline → L1读取；真实麦克风Test UI临时模式省略RecordingStore session及其全部音频/结果写入。启动失败按反向回滚并join所有已启动线程。正常停止不清空等待队列，而是先停设备/L1并刷出预降噪，再依次以EOS drain L2→L3→L5→completion/commit；普通Runtime随后完成Recording水位并关闭RecordingStore，临时模式只封存可发送给离线L4的本轮UI音轨。完整模拟输入EOF不设有限排空期限，Test UI在此期间显示`FINALIZING`；实时/手动停止继续使用配置的安全期限。有限期限超时的已注册窗口显式转为`CANCELLED/error`；仍有worker存活时拒绝假关闭。操作者在模拟播放中点击“从头重播”属于独立的硬换代路径：立即暂停L1并清空所有尚未执行的L2/L3/L5/Commit工作，当前正在执行且不能安全抢占的单个kernel返回后也丢弃其结果，再用全新处理图从sample 0开始；该行为不改变正常EOF的完整排空和`stopped`计时封存。

Development Test UI只通过公开只读`processing_status`获取每阶段队列深度/容量、worker存活、在途窗口、缓存字节、完成数和错误数；其中`input_health`公开当前epoch、连续性中断次数/最后原因、input overflow、handoff drop及交接队列深度/容量/高水位，L5诊断包括`l5_actual_completed/l5_dropped/l5_skipped/l5_actual_hz`及显示邮箱深度、容量、覆盖数。Gate因新epoch重新预热时，UI错误文案附带`epoch_reset:<reason>`，可直接区分静音、处理丢窗与真实输入中断。UI不得读取Runtime私有队列或据此修改调度。

回归测试覆盖公共ID逐项进入L3/L5、跨层错序/角度/WindowKey拒绝、队列丢弃、sample跳跃、epoch变化、配置revision、停机drain、唯一终态/watermark和有序提交；普通运行与Test UI运行使用同一正式L2/L3/L5链路。

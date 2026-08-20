# Development Test UI：项目1.0.1 / Layer 2 1.1

> 项目 1.1.0 待实现改动见[`ARCHITECTURE_V1.1_TARGET.md`](../../ARCHITECTURE_V1.1_TARGET.md#12-development-test-ui-与逐-id-试听)：删除 iterative 和 ID 开关，保留独立 Kalman 开关，显示 MUSIC 伪谱/公共 ID，并只按 L2 权威 `(session_id, stream_epoch, track_id)` 拼接试听。下文描述当前 1.0.1 界面。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../../ARCHITECTURE_V0.3_TARGET.md)。**本README描述当前已迁移界面。**

当前回归状态：L1预降噪及界面相关自动化门禁已通过。L1区域已增加持久化的“IMCRA预降噪”开关；开启后Runtime等待对应20 ms降噪块完成，再将替换后的7路音频送入后续层。

本UI只消费同一个ApplicationRuntime快照，不得重开设备、重建时间轴或在界面线程运行算法。普通启动使用真实麦克风；数据管理系统发起完整模拟时，`RecordingReplaySource`把已登记的原始8ch音频和CDC热力图作为同一个虚拟阵列输入，两种方式共用完整L1→L4链路。

只有完整模拟输入模式会在L1显示操作者填写的音频名称以及“开始/继续、暂停、从头重播”控件。暂停不推进sample，也不在继续时追赶；播放结束保留最后结果并等待重播。重播立即清空上一轮L1～L4画面、试听缓存和旧结果邮箱，并通过新的stream epoch重新预热算法状态。普通真实设备模式不创建这些控件。

主界面使用10 ms精确定时器以100 Hz轮询两个容量1的latest-value邮箱。正式审计邮箱仍只接收ResultJoiner按`(session_id, stream_epoch, window_id, decision_sample)`合并并有序提交的快照；L4完成邮箱`latest_l4_dev_ui`只在L4真正`COMPLETED`后立即接收完整同窗L2/L3/L4 `DevUiFrame`，用于减少有序commit等待造成的CNN显示延迟。它不改变DecisionRecord、录音或watermark顺序，也不能混拼不同窗口。算法正式窗口仍为20 ms（50 Hz）；某阶段SKIPPED/FAILED时仍由有序审计快照表达真实终态。

顶部状态栏通过ApplicationRuntime公开只读`processing_status`显示L2/L3/L4/completion队列的“当前深度/容量”、worker RUN/STOP、各阶段完成/错误累计、在途窗口及计算缓存MiB。L4另外显示实际完成、DROPPED、SKIPPED、最近1秒实际完成Hz，以及完成帧邮箱的深度/容量/latest-only覆盖数。悬停可查看入口丢窗和最近阶段错误。该显示不得访问`_processing_windows`等私有字段，也不得反向改变队列或调度；Runtime缺少公开快照时只显示telemetry unavailable，不猜测内部状态。

## 四象限

- 左上L1：MIC0～MIC5、Center、HardwareMix共8路电平；显示IMCRA预热状态与7个物理麦的80～8000 Hz噪声dB摘要，并提供持久化“IMCRA预降噪”开关和当前平均频率增益；不在L1显示20/40 ms概率；保留灯控与scratch录音。
- 右上L2：显示500～4000 Hz Gate概率、状态和原始SRP 360°响应；候选点严格使用L2最终输出角度。首次出现为灰色小点，临时ID观测为灰色大点、预测为灰色小点；正式ID使用红/绿/琥珀三种稳定颜色，观测为大点、预测为小点。
- 左下L3：连续试听首行固定为预降噪前LogicalAudio第7路Center Mic原音参考。方向轨从L2声明临时ID已建立卡尔曼状态、能够持续预测时开始缓存，转为正式ID后沿用同一缓存；首次出现但尚未Kalman-ready的临时ID和无ID候选不写入。累计至少2秒后显示，候选消失等待3秒。模式切换清空旧模式试听缓存；跳窗按48 kHz绝对sample补真实音频或等时静音。所有处理只影响试听，不改变L2/L3/L4正式结果或录音。
- 右下L4：显示逐方向CNN概率与Voice结果；优先使用`latest_l4_dev_ui`的真正完成帧，保留独立的L4分类阈值滑动条。有序DROPPED/SKIPPED/缺失帧不立即清掉上一个有效CNN结果；超过`dev_test_ui.stale_after_ms`仍没有新完成帧才显示`STALE`。

L2 Gate滑条范围`0.00～1.00`、建议步长`0.01`。拖动后在下一完整DecisionWindow生效并显示新的`config_revision`，默认不写回`config.yaml`。L4阈值滑条只重判缓存的CNN概率。两个滑条必须用“L2声源Gate”和“L4人声判断”清晰区分。

所有算法信息按session、epoch、window和sample endpoint对齐。缺少任一20 ms IMCRA概率、跨epoch或尚未预热时，右上明确显示`WARMING_UP/UNAVAILABLE`，不能拼接旧数据或显示假SRP结果。

L2内部ID不进入公共`CandidateDirection`，但Runtime会向Test UI传递与候选对齐的私有ID、预测、正式、首次分配和Kalman-ready标志。右上SRP面板只据此绘制诊断样式；左下试听sidecar接收Kalman-ready临时ID与正式ID的已有L3音频，用于本地拼接、缓存和播放。ID换号仅在3秒等待期内、20°以内且唯一可续接时合并，同时出现的近角双ID不得合并。这些投影不能改变候选角度、L2租约、L3/L4结果或正式录音。

## 回归测试

- 8路meter及通道标签/顺序；
- IMCRA 20 ms与Gate 40 ms值显示一致；
- L1预降噪开关持久化、开启后等待替换且不重复/跳过sample区间；
- L2滑条动态revision、L4滑条只重判、二者互不影响；
- Gate关闭时SRP显示Blocked且候选为空；
- 原始SRP圆环和平滑候选点同窗显示，UI不执行二次滤波；私有ID状态只进入本机SRP诊断显示与试听sidecar，不进入公共DTO和正式记录；
- Test UI试听ID不产生正式候选之外的L3预测波束批次，普通Runtime不创建该旁路；Center Mic参考、2秒显示门槛、3秒等待、唯一换号续接、近角双ID隔离、跳窗等时补洞和关闭清理均有回归测试；
- L3只有音频视图，无内部`[17,169]`依赖；
- L3三档循环切换、Runtime模式透传、模式切换后试听缓存隔离；恒定波束档固定30° FNBW并在不安全频点回退DAS；
- L4完成邮箱固定容量1、只发布完整同窗COMPLETED帧、覆盖不改变正式结果；DROPPED/SKIPPED画面保留到stale超时；
- L4实际完成/丢弃/跳过/Hz/邮箱覆盖诊断与空候选L3免prepare、L4空batch成功路径；
- latest-value邮箱、不卡采集、scratch与正式录音隔离。

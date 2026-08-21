# Development Test UI：项目1.3.1

> 当前版本按[`ARCHITECTURE_V1.1_TARGET.md`](../../ARCHITECTURE_V1.1_TARGET.md#12-development-test-ui-与逐-id-试听)显示MUSIC伪谱/公共方向ID，并按L2权威`(session_id, stream_epoch, track_id)`拼接试听；ID追踪默认启用并可进入MUSIC-only诊断模式，Kalman保持可选。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../../ARCHITECTURE_V0.3_TARGET.md)。**本README描述当前已迁移界面。**

当前回归状态：L1预降噪及界面相关自动化门禁已通过。L1区域已增加持久化的“IMCRA预降噪”开关；开启后Runtime等待对应20 ms降噪块完成，再将替换后的7路音频送入后续层。

每次麦克风采集成功连接后，Test UI会尽力发送一次官方关灯命令`e`。麦克风未连接时不会访问CDC或发送灯控命令；启动默认关灯失败也不弹出灯控错误，手动“灯光开/灯光关”仍保留完整错误提示。

本UI只消费同一个ApplicationRuntime快照，不得重开设备、重建时间轴或在界面线程运行算法。普通启动使用真实麦克风并保留实时CDC热力图；数据管理系统发起模拟测试时，`RecordingReplaySource`只读取已登记的原始8ch音频。实时链到Hub封存为止；L4和L5由下半区两个“发送”按钮依次触发，并在后台工作线程执行。

主窗口默认以系统最大化窗口启动，保留标题栏、最小化和还原能力，不进入无边框全屏；只有配置显式启用`start_fullscreen`时才使用全屏模式。

L3单窗仍由`timing.downstream_audio_window_ms`控制（当前40 ms），但按ID长轨不再由UI自行拼接。Runtime中的`TrackAudioStreamHub`每20 ms生成一份已经去重、按IMCRA概率响度补偿的连续轨hop；Test UI试听缓存和L5 CNN逐样本读取同一份波形，播放端不再额外提高响度。L5面板提供默认ON的实时补偿开关，切换不清空ID或连续轨。

只有完整模拟输入模式会在L1显示操作者填写的音频名称以及“开始/继续、暂停、从头重播”控件。暂停不推进sample，也不在继续时追赶；播放结束保留最后结果并等待重播。重播立即清空上一轮L1～L5画面、试听缓存和旧结果邮箱，并通过新的stream epoch重新预热算法状态。普通真实设备模式不创建这些控件。

主界面使用10 ms精确定时器以100 Hz轮询两个容量1的latest-value邮箱。正式审计邮箱仍只接收ResultJoiner按`(session_id, stream_epoch, window_id, decision_sample)`合并并有序提交的快照；L5完成邮箱`latest_l5_dev_ui`只在L5真正`COMPLETED`后立即接收完整同窗L2/L3/L5 `DevUiFrame`，用于减少有序commit等待造成的CNN显示延迟。它不改变DecisionRecord、录音或watermark顺序，也不能混拼不同窗口。算法正式窗口仍为20 ms（50 Hz）；某阶段SKIPPED/FAILED时仍由有序审计快照表达真实终态。

按ID累计试听每个决策只追加`TrackAudioStreamHub`产生的同一稳定20 ms hop；GUI不再自行交叉淡化、拼接或做响度增强。声卡输出仅保留必要的衰减型峰值安全和首尾播放淡化，不会提高或改写缓存、L5输入或录音资产。

顶部状态栏通过ApplicationRuntime公开只读`processing_status`显示L2/L3/L5/completion队列的“当前深度/容量”、worker RUN/STOP、各阶段完成/错误累计、在途窗口及计算缓存MiB。L5另外显示实际完成、DROPPED、SKIPPED、最近1秒实际完成Hz，以及完成帧邮箱的深度/容量/latest-only覆盖数。悬停可查看入口丢窗和最近阶段错误。该显示不得访问`_processing_windows`等私有字段，也不得反向改变队列或调度；Runtime缺少公开快照时只显示telemetry unavailable，不猜测内部状态。

## 上二栏与下三栏

- 左上L1：MIC0～MIC5、Center、HardwareMix共8路电平；显示IMCRA预热状态与7个物理麦的0～10000 Hz噪声dB摘要，并提供持久化“IMCRA预降噪”开关和当前采集流的历史平均频率增益；不在L1显示20/40 ms概率；保留灯控与scratch录音。
- 右上L2：显示500～4000 Hz Gate概率、状态和原始MUSIC 360°伪谱；紧凑的`MUSIC阶数`下拉框右侧提供`ID Tracking`按钮。追踪开启时，候选点严格使用L2最终输出角度：首次出现为灰色小点，临时ID观测为灰色大点、预测为灰色小点，正式ID使用稳定颜色且观测为大点、预测为小点。追踪关闭时只显示原始MUSIC峰值对应的灰色小点，不显示彩色ID，L3/L5按正常跳过终态停止运行而不报错。
- 下左L3：连续试听首行固定为Center Mic参考，其余方向轨显示L2权威ID和角度。停止采集并等待L3排空后，“发送到L4”把Hub已经拼好的完整长音频整体交给L4。L3波形不再读取或绘制L5黄色人声区间。
- 下中L4：逐轨保存L4输出WAV并提供与L3一致的ID、角度、时长、波形和播放控件。全部轨处理完成后“发送到L5”才可用。L5一次读取完整长音频并为每个20 ms hop返回独立概率；达到当前UI阈值的区间只在本栏对应时间位置显示黄色背景。
- 下右L5：显示从L4人工发送后的逐方向CNN概率与Voice结果；阈值滑条即时重判已缓存概率并同步重绘L4黄色区域，不重新运行CNN。

L2 Gate滑条范围`0.00～1.00`、建议步长`0.01`。拖动后在下一完整DecisionWindow生效并显示新的`config_revision`，默认不写回`config.yaml`。L5阈值滑条只重判缓存的CNN概率。两个滑条必须用“L2声源Gate”和“L5人声判断”清晰区分。

L2面板的紧凑“MUSIC阶数”控件只能选择1、2、3。实际阶数始终为`min(MDL实际诊断阶数, 手动上限)`；右侧状态条同时显示`MDL`和`MUSIC`，便于直接比较。阶数和`ID Tracking`状态都写入Test UI本地设置；L2每次真正开始计算前读取最新revision，因此即使处理队列已有积压也会在下一次L2计算实时应用。阶数控件不修改MDL 0～6诊断结果，也不启用逐频支持或可靠性加权门禁。

所有算法信息按session、epoch、window和sample endpoint对齐。缺少任一20 ms IMCRA概率、跨epoch或尚未预热时，右上明确显示`WARMING_UP/UNAVAILABLE`，不能拼接旧数据或显示假SRP结果。

L2公共`TrackedDirection`直接携带权威`track_id`、观测/预测状态和Kalman应用状态。右上MUSIC面板只据此绘制；左下试听按同一公共ID拼接L3音频，不执行第二套角度关联或换号补救。这些显示逻辑不能改变L2轨迹、L3/L5结果或正式录音。

## 回归测试

- 8路meter及通道标签/顺序；
- IMCRA 20 ms与Gate 40 ms值显示一致；
- L1预降噪开关持久化、开启后等待替换且不重复/跳过sample区间；
- L2滑条动态revision、L5滑条只重判、二者互不影响；
- Gate关闭时MUSIC显示Blocked且公开方向为空；
- 原始MUSIC圆环和公共轨迹点同窗显示，UI不执行二次滤波或二次ID关联；
- Test UI试听ID不产生正式候选之外的L3预测波束批次，普通Runtime不创建该旁路；Center Mic参考、2秒显示门槛、3秒等待、唯一换号续接、近角双ID隔离、跳窗等时补洞和关闭清理均有回归测试；
- L3只有音频视图，无内部`[17,169]`依赖；
- L3四档循环切换、Runtime模式透传、模式切换后试听缓存隔离；包括优化算法、DS、
  全频Loaded MVDR和五频段鲁棒对照，后三种均保持独立试听分区；
- L5完成邮箱固定容量1、只发布完整同窗COMPLETED帧、覆盖不改变正式结果；DROPPED/SKIPPED画面保留到stale超时；
- L5实际完成/丢弃/跳过/Hz/邮箱覆盖诊断与空候选L3免prepare、L5空batch成功路径；
- latest-value邮箱、不卡采集、scratch与正式录音隔离。

# Development Test UI：项目1.3.2

> 当前版本按[`ARCHITECTURE_V1.1_TARGET.md`](../../ARCHITECTURE_V1.1_TARGET.md#12-development-test-ui-与逐-id-试听)显示DOA伪谱/公共方向ID，并按L2权威`(session_id, stream_epoch, track_id)`拼接试听；ID追踪默认启用并可进入DOA-only诊断模式，IMM属于追踪器内部且不可独立关闭。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../../ARCHITECTURE_V0.3_TARGET.md)。**本README描述当前已迁移界面。**

当前回归状态：L1预降噪及界面相关自动化门禁已通过。L1区域已增加持久化的“IMCRA预降噪”开关；开启后Runtime等待对应20 ms降噪块完成，再将替换后的7路音频送入后续层。L1区域另有持久化“CountNet人数估计”开关：只读校准后Center Mic，开启后先预热5秒、随后每100 ms显示0/1/2、三位P0/P1/P2、输入RMS、模型输入增益和标注end sample；关闭会清空其私有状态。该worker不在GUI或采集线程推理，第一阶段结果不进入Windowing和L2～L5。

每次麦克风采集成功连接后，Test UI会尽力发送一次官方关灯命令`e`。麦克风未连接时不会访问CDC或发送灯控命令；启动默认关灯失败也不弹出灯控错误，手动“灯光开/灯光关”仍保留完整错误提示。

本UI只消费同一个ApplicationRuntime快照，不得重开设备、重建时间轴或在界面线程运行算法。普通启动使用真实麦克风并保留实时CDC/DOA极坐标显示；数据管理系统发起模拟测试时，`RecordingReplaySource`只读取已登记的原始8ch音频，不读取录音时的CDC热力图；后台重新运行Gate、MUSIC、ID和L3，并把本次L2计算得到的DOA快照提交给极坐标控件。实时链到Hub封存为止；下半区只保留“发送到L4”按钮，L4整批完成后由同一后台工作自动且仅一次运行L5。

主窗口默认以系统最大化窗口启动，保留标题栏、最小化和还原能力，不进入无边框全屏；只有配置显式启用`start_fullscreen`时才使用全屏模式。

L3单窗仍由`timing.downstream_audio_window_ms`控制（当前40 ms），但按ID长轨不再由UI自行拼接。Runtime先根据每个L2完成窗口登记权威ID的绝对20 ms时间槽，`TrackAudioStreamHub`再把已经去重、按IMCRA概率响度补偿的L3 hop写入对应槽；没有BF结果的首尾或中间槽保留等时静音。Test UI的L3试听和停机封存读取同一份48 kHz波形，所以算法速度不改变轨长。L3栏提供默认ON的补偿开关，切换不清空ID或连续轨；L5读取的是L4最终原生16 kHz输出。

只有完整模拟输入模式会在L1显示操作者填写的音频名称以及“开始/继续、暂停、从头重播”控件。暂停不推进sample，也不在继续时追赶；播放到EOF后进入`FINALIZING`并等待L2/L3/L5/Commit数据完整排空，随后保留最后结果和各层总运行时长并进入`stopped`等待重播。播放途中点击重播会立即暂停输入并清空上一轮所有尚未执行的L2/L3/L5/Commit队列、L1～L5画面、试听缓存和旧结果邮箱；若一个BF kernel已经开始，只等待该调用安全返回并丢弃其迟到结果，之后从sample 0建立全新Runtime处理图并重新预热。普通真实设备模式不创建这些控件。

主界面按配置中L1仪表、极坐标和波形的最高刷新率启动精确定时器，正式配置为20 ms/50 Hz，不再以100 Hz重复刷新同一算法帧。L2面板独占容量1的`latest_l2_dev_ui`，在L2 worker完成时立即更新，不等待L3、L5或Commit；UI仅接受当前`session/epoch`且窗口身份单调前进的快照。正式`latest_dev_ui`仍只接收ResultJoiner按`(session_id, stream_epoch, window_id, decision_sample)`合并并有序提交的快照，用于L3显示、录音与审计。实时L5固定写入`offline_after_l4`跳过终态，因此兼容邮箱`latest_l5_dev_ui`不会发布CNN结果。离线L5结果由当前后台作业直接写入L4/L5面板，不进入DecisionRecord或watermark。算法正式窗口仍为20 ms（50 Hz）；某阶段SKIPPED/FAILED时仍由有序审计快照表达真实终态。

按ID累计试听每个决策只追加`TrackAudioStreamHub`产生的同一稳定20 ms hop；GUI不再自行交叉淡化、拼接或做响度增强。声卡输出仅保留必要的衰减型峰值安全和首尾播放淡化，不会提高或改写缓存、L5输入或录音资产。

顶部状态栏通过ApplicationRuntime公开只读`processing_status`显示L2、L3原始/准备/host、L5审计及completion队列的“当前深度/容量”、worker RUN/STOP、各阶段完成/错误累计、在途窗口及计算缓存MiB。每次UI刷新只读取一次该快照。实时L5诊断主要显示DROPPED与`offline_after_l4`的SKIPPED数量；离线CNN进度由L4/L5面板单独显示。悬停可查看入口丢窗和最近阶段错误。该显示不得访问私有队列，也不得反向改变队列或调度；Runtime缺少公开快照时只显示telemetry unavailable，不猜测内部状态。

## 上二栏与下三栏

- 左上L1：MIC0～MIC5、Center、HardwareMix共8路电平；显示IMCRA预热状态与7个物理麦的0～10000 Hz噪声dB摘要，并提供持久化“IMCRA预降噪”开关和当前采集流的历史平均频率增益；提供独立CountNet开关和100 ms人数读数，其中2表示2人或以上。CountNet行的`end`每100 ms推进，`input/gain`用于确认安静阵列输入已进入预训练模型的有效电平范围；不在L1显示20/40 ms IMCRA概率；保留灯控与scratch录音。
- 右上L2：显示500～4000 Hz Gate概率、状态和原始MUSIC 360°伪谱；紧凑的`MUSIC阶数`下拉框右侧提供`ID Tracking`按钮。追踪开启时，候选点严格使用L2最终输出角度：首次出现为灰色小点，临时ID观测为灰色大点、预测为灰色小点，正式ID使用稳定颜色且观测为大点、预测为小点。追踪关闭时只显示原始MUSIC峰值对应的灰色小点，不显示彩色ID，L3与实时L5审计按正常跳过终态停止运行而不报错。
- 下左L3：连续试听前两行依次为`Center Mic RAW`和`Center Mic IMCRA`；RAW保存校准后、预降噪前的Center，IMCRA行仅在预降噪开启且该20 ms hop确实采用降噪输出时追加。其余方向轨显示L2权威ID和角度。停止采集并等待L3排空后，“发送到L4”把Hub已经拼好的完整长音频整体交给L4。L3波形不再读取或绘制L5黄色人声区间。
- 下中L4/L5：两层共用一个音频面板，仅在面板标题显示`L4 / L5`，不再显示独立L5面板、说明行或L5状态文字。“合并”按钮位于MossFormer2左侧，尺寸与模型按钮一致，默认开启为绿色、关闭为灰色。开启时每条双人轨一拆二后按1～4 kHz匹配只显示高分候选；关闭时保留两条16 kHz候选，并按相同匹配度降序标记A/B（A高、B低），以原ID、A/B和三位小数分数显示供试听，不显示角度。最终L4波形运行DNSMOS，标签末尾追加按SIG/BAK/OVRL合成的0～1三位小数MOS，例如`1A · 匹配度 0.941 · MOS 0.823`；单条合并/旁路音频保留原ID和角度后追加MOS。两种模式的L4输出都会自动进入同一个L5，逐20 ms人声结果仍作为本栏波形黄色区间呈现。试听缓存是L4原生16 kHz单声道PCM16 WAV；播放端解析RIFF/WAV后以16 kHz直接建立声卡流，不做16→48 kHz重采样，也不得复用L3裸`.f32`缓存入口。播放只做去直流、必要的衰减型峰值保护和首尾淡化，不改写缓存或L5输入。
- 下中L4/L5：逐轨保存L4输出WAV并提供与L3一致的ID、角度、时长、波形和播放控件；不提供手动“发送到L5”按钮。L5自动一次读取完整长音频并为每个20 ms hop返回独立概率。
- 下右L6：占原L5极坐标区域。L4必须关闭“合并”以保留A/B双候选；两条候选均完成L5逐20 ms标注后才启用“运行L6”。L6只在手动点击后加载CPU模型；每行对应一个最终L6讲话人ID，所有ID统一到本批最早至最晚绝对sample时间轴，按时间拼接有效片段并在未讲话位置补等时静音，因此各行等长且可直接对齐。标签同时显示来源L2 ID、质量、时长和波形并支持试听。L5概率仍在L4波形上以黄色区间显示。

L2 Gate滑条范围`0.00～1.00`、建议步长`0.01`。拖动后在下一完整DecisionWindow生效并显示新的`config_revision`，默认不写回`config.yaml`。L5阈值滑条只重判缓存的CNN概率。两个滑条必须用“L2声源Gate”和“L5人声判断”清晰区分。

L2面板的紧凑“MUSIC阶数”控件只能选择1、2、3，并直接决定普通路径的信号子空间阶数与最多搜峰数；DPD路径将它作为最多候选数。右侧状态条显示当前手动阶数和实际输出候选数。阶数和`ID Tracking`状态都写入Test UI本地设置；L2每次真正开始计算前读取最新revision，因此即使处理队列已有积压也会在下一次L2计算实时应用。阶数控件不启用逐频支持或可靠性加权门禁。

所有算法信息按session、epoch、window和sample endpoint对齐。缺少任一20 ms IMCRA概率、跨epoch或尚未预热时，右上明确显示`WARMING_UP/UNAVAILABLE`，不能拼接旧数据或显示假SRP结果。

L2公共`TrackedDirection`直接携带权威`track_id`、观测/预测状态和IMM应用状态。右上DOA面板只据此绘制；左下试听按同一公共ID拼接L3音频，不执行第二套角度关联或换号补救。离线L4/L5继承同一ID；这些显示逻辑不能改变L2轨迹、音频结果或正式录音。右侧仅保留`ID Tracking`总开关，不再显示独立Kalman或Q/R控件。

## 回归测试

- 8路meter及通道标签/顺序；
- IMCRA 20 ms与Gate 40 ms值显示一致；
- L1预降噪开关持久化、开启后等待替换且不重复/跳过sample区间；
- CountNet开关持久化、5秒预热、100 ms更新、无平滑、gap重置和缺模型/推理异常显式INVALID；
- L2滑条动态revision、L5滑条只重判、二者互不影响；
- Gate关闭时MUSIC显示Blocked且公开方向为空；
- 原始MUSIC圆环和公共轨迹点同窗显示，UI不执行二次滤波或二次ID关联；
- Test UI试听ID不产生正式候选之外的L3预测波束批次，普通Runtime不创建该旁路；Center Mic参考、2秒显示门槛、3秒等待、唯一换号续接、近角双ID隔离、跳窗等时补洞和关闭清理均有回归测试；
- L3只有音频视图，无内部`[17,169]`依赖；
- L3四档循环切换、Runtime模式透传、模式切换后试听缓存隔离；包括优化算法、DS、
  全频Loaded MVDR三种模式，后两种均保持独立试听分区；五频段与固定30°波束模式已删除；
- 实时L5固定以`offline_after_l4`跳过，兼容完成邮箱不发布CNN结果；DROPPED/SKIPPED画面保留到stale超时；
- L5丢弃/跳过诊断、空候选L3免prepare，以及离线L4完成后自动L5的独立工作流；
- latest-value邮箱、不卡采集、scratch与正式录音隔离。

# 项目完整变更日志

本文件是6+1麦克风阵列项目的统一、持续维护记录，覆盖：

- Layer 1：采集、通道映射、校准、IMCRA与预降噪；
- Layer 2：Gate、SRP-PHAT、候选方向、内部ID与卡尔曼；
- Layer 3：方向波束形成、缓存及增强音频；
- Layer 4：采集结束后的可选双人语音分离与主讲话人选择；
- Layer 5：响度补偿、重采样、CNN与人声概率；
- Development Test UI；
- 独立 Pipeline Log UI；
- 正式音频录制、数据管理与Production UI；
- Application Runtime、唯一时间轴、跨层接口、缓存、测试和模型资产。

## 维护规则

1. 日志按时间倒序追加，已发布记录不得重写成与历史不符的内容。
2. 每次提交前必须记录本次实际变化；没有变化的模块明确写“无变化”，防止遗漏跨层影响。
3. 每条记录至少包含日期、版本/标签、变更类型、涉及文件、各模块具体变化、接口或兼容性影响、验证结果和Git LFS资产变化。
4. 功能尚未完成、未经实机验证或仅完成自动测试时必须明确标注，不能写成已经正式验收。
5. 本文件记录“发生了什么”；当前`1.3.3`开发架构以`ARCHITECTURE_V1.1_TARGET.md`为权威契约，最终发布基线为`v1.3.2`，实际参数以`config/config.yaml`和代码为准。
6. 更早的单次Test UI历史快照保留在`docs/DEV_TEST_UI_CHANGELOG_2026-08-14.md`，其过时算法描述不得覆盖当前实现。

---

## 2026-08-24 — L4加入按决定时间对齐的跨方向轨候选惩罚

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **L4选择逻辑**：保留`l3_bf_1_4khz_complex_coherence_v3`对每条L3参考的1～4 kHz初筛；封存整批新增`l3_bf_1_4khz_cross_track_penalty_v4`。初选候选与同次“一拆二”的同胞候选分别对所有同epoch其他ID的L3参考评分；若初选对某条其他轨的分数比同胞至少高`0.025`，判定初选更像其他方向轨并切换到同胞候选。
- **决定时间与不等长音频**：禁止按数组首端强行对齐。使用每条`Layer4LongAudioInput.start_sample/end_sample`的绝对48 kHz决定时间求交集，再将两侧偏移精确映射到16 kHz切片；跨epoch、无交集及真实重叠不足2秒的比较不触发切换，因此支持不同起点、不同结束时间和不同时长的L3长轨。
- **审计契约**：最终结果记录初选/最终候选索引、切换状态、最强冲突ID、角度、交叠决定sample范围、交叠长度、两候选对其他轨的1～4 kHz分数、惩罚差值和门限；输出hash、PCM16峰值保护及L5输入随最终候选同步更新。
- **真实缓存复核**：重建`0824-1639`场景第6轨（350.8°、22.94秒）后，原初筛以`0.510/0.791`选择候选1；按决定时间与ID 8对齐19.52秒后，两候选对ID 8参考得分为`0.151/0.329`，候选1多`0.178`，新规则会改选候选0。诊断WAV和JSON仅保留在被忽略的`data/dev_test_ui/diagnostics`，不纳入版本库。
- **验证**：修改文件Ruff通过；L4契约、离线L4/L5和Development Test UI直接消费链共`55 passed`，新增错位起点、不等时长、至少2秒重叠、触发切换及短重叠不切换回归。自动与当前缓存复核不替代更多房间、角度、语言和噪声场景验收。
- **未改变与资产**：L1、L2 MUSIC/ID/Kalman、L3 BF与Hub拼接、1～4 kHz初筛数学、48→16 kHz重采样、人数分类、MossFormer2/TIGER模型、L5 CNN、各UI和录音管理均无变化；无模型、音频或Git LFS对象变化。

---

## 2026-08-24 — 刷新1 m正上方实机增益校准并应用到L1

- **版本/标签**：项目仍为`1.3.3`开发线；不修改项目版本，不创建或移动发布标签。
- **L1实机校准**：使用手机在阵列中心正上方约1 m播放确定性48 kHz校准音频，原生8通道连续采集60秒且无设备状态错误或削波。20/20个200～10000 Hz chirp被稳定识别；7个物理麦去底噪信噪比中位数为`26.9972～29.8156 dB`，20次相对电平标准差为`0.0156～0.0922 dB`。新相对增益写入`config/config.yaml`，校准版本更新为`office_overhead_1m_20260824_v2`并形成新的配置/hash边界。
- **极性与时延证据**：本次所有通道极性均为正；1 m办公室反射下的整数延迟结果与既有高相关度近场报告不一致，因此没有用低置信结果覆盖时延。阵列、安装和通道映射未变化，继续继承2026-08-21报告中28～30次稳定chirp验证的极性与整数delay，并在新报告中记录父报告路径和SHA-256。状态保持`verified`只表示组合证据通过，不把本次反射测量冒充独立时延验证。
- **工具与报告**：新增确定性校准音频生成、原生8通道WDM-KS采集、周期chirp检测、去底噪增益及极性/整数lag分析工具；新增可审计报告`docs/L1_HARDWARE_CALIBRATION_2026-08-24.json`。三次原始录音和生成WAV保留在被忽略的`data/calibration/current`，不提交运行数据。
- **未改变**：物理/逻辑通道顺序、HardwareMix排除规则、MIC面坐标、IMCRA与预降噪、CountNet、Windowing、L2 MUSIC/ID/Kalman、L3～L5、各UI、正式录音schema和模型资产均无变化。
- **验证与资产**：增加校准刺激确定性、响度/防削波、文件hash和整数lag/极性测试，并执行配置、L1校准、Ingest/Windowing相关回归；无Git LFS对象变化。

---

## 2026-08-24 — 恢复模拟输入L2 DOA极坐标显示

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **Test UI修复**：撤销实时性能优化中误加的`live_microphone_mode`极坐标渲染限制。模拟WAV与数据库录音回放现在与真实麦克风一样，把独立L2 UI邮箱中的最新MUSIC/ID快照提交给DOA极坐标控件；Gate阻断时两种输入模式都清除旧快照，避免显示过期方向。模拟输入仍不读取录音时的CDC热力图，显示的是本次重新计算的L2 DOA。
- **回归与文档**：恢复模拟输入有效L2帧必须持有极坐标快照的Test UI回归，并更正根README及Test UI说明中对CDC录制热力图与L2重算DOA的区分。
- **未改变与资产**：L1、L2 Gate/MUSIC/ID算法、L2→L3契约、L3拼接、离线L4/L5、队列、STOPPED及分层性能监控均无变化；无配置、模型、音频或Git LFS对象变化。

---

## 2026-08-24 — L1/L2/L3/拼接/Test UI全链实时优化与实机验证

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **Runtime层间调度**：CPU L3拆为有界的候选无关准备（滚动STFT/IMCRA）、候选相关DS/Loaded MVDR/optimized BF+ISTFT、主机连续音频拼接/发布三段FIFO worker；阶段屏障只有在主机拼接排空后才冻结L3总时长。新增`runtime.torch_cpu_threads=1`，避免7通道小矩阵让每个PyTorch操作占满16个逻辑核并与L1/L2争抢；CUDA微批路径继续保留作显式诊断，但不作为默认最快路径。
- **L3无效计算删除**：optimized双声源只对实际命中的rho频点执行LCMV或soft-null分支；多个新角度的steering一次批量生成；IMCRA空间协方差只在L3有效频带计算后回填公开全频契约；滚动协方差仅在增删帧数确实小于全量计算时启用；Runtime只在最终host DTO边界执行完整有限值检查，保留条件数、约束响应、DAS fallback和绝对sample语义。
- **拼接与Test UI**：实时Runtime不再每20 ms重建并复制每个ID的完整3.2秒上下文；Test UI仅在ID首次确认时请求一次历史种子，之后只追加新的960-sample hop，Hub完整归档和停机L4输入不变。UI按正式50 Hz节拍轮询、控件/仪表只在值变化时重绘、每帧只读取一次Runtime状态。根据操作者最新要求，模拟WAV/数据库回放仍显示Gate、MUSIC、ID和文字诊断，但不再向极坐标控件提交热力图；真实麦克风继续显示实时极坐标图。
- **性能裁决**：同一28.84秒八通道双声源压力录音中，CPU单BF worker、PyTorch单线程为当前最快配置；L2约`28.83 s`、L3含拼接约`33.34 s`，处理丢窗和阶段错误均为0。相同整链的L3 CUDA约`42.33 s`，两个并发BF worker约`42 s`，均已判定更慢且不设为默认。真实MicArray 8通道48 kHz采集10秒并同时运行L1/L2/L3/拼接/Test UI：L2/L3均在`9.875 s`排空，L3各队列峰值0，输入溢出、交接丢块、处理丢窗、时间轴中断和阶段错误均为0。
- **未改变与资产**：L1 IMCRA论文输出字段及0–10 kHz范围、L2 Gate/MUSIC/ID算法、DS/MVDR/optimized数学定义、离线L4/L5模型、Production/Log UI、正式录音schema、数据集与发布标签均无变化；无模型、音频或Git LFS对象变化。10秒实机验证证明当前设备上的短时实时性，不替代长时间、三声源、设备断连和热降频压力验收。
- **验证**：完整Ruff通过；L3、Runtime、TrackAudioStream、Test UI、回放、配置等聚焦回归`235 passed`；项目完整pytest为`535 passed, 1 warning`，唯一警告是既有CountNet `torch.jit.load`弃用提示。

---

## 2026-08-24 — 修复模拟输入L2 DOA极坐标图不显示

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **根因与修复**：Test UI近期工作区改动把L2极坐标快照提交错误限制为仅`replay_source is None`，导致模拟WAV/完整录音回放虽持续收到Gate、MUSIC阶数、有效频点和LIVE标题，却从不向绘图区提交同一L2快照，中央固定显示`DOA UNAVAILABLE`。现已恢复真实采集与模拟输入共用同一独立L2快照渲染路径。
- **未改变**：L2 Gate/MUSIC/ID算法、独立L2 UI邮箱、session/epoch及单调窗口过滤、临时/正式ID样式、L2→L3方向、L3～L5、队列、STOPPED及性能计时均无变化。
- **验证与资产**：增加模拟输入模式直接渲染有效L2帧并持有极坐标快照的回归；无配置、模型、音频或Git LFS对象变化。自动测试不替代实际窗口视觉验收。

---

## 2026-08-24 — L4候选匹配频带扩展为1～4 kHz

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **L4匹配修复**：L3参考与MossFormer2/TIGER匿名候选的逐帧复频谱相干匹配频带由2～4 kHz扩展为1～4 kHz，算法标识更新为`l3_bf_1_4khz_complex_coherence_v3`；512点Hann STFT、160点hop、参考频带能量加权、2秒最低轨长、0.50最低分、0.025最低分差及低可信回退规则均保持不变。
- **真实缓存诊断**：在30°中文、210°英文的当前双声源缓存中，原2～4 kHz对ID 2的两候选评分为`0.823/0.973`并错误选择中文候选1；改用1～4 kHz后评分为`0.869/0.652`，以`0.217`分差正确选择英文候选0。该结论来自当前临时缓存，不把运行录音或诊断音频纳入版本库。
- **契约与文档**：同步L4选择结果的算法版本Literal、模块说明、根README、完整架构与使用文档；新增1.5 kHz目标候选回归，确保1～2 kHz内容实际参与匹配。
- **验证**：`tests/test_l4_speech_separation_contracts.py`与`tests/test_l4_offline_pipeline.py`共`16 passed`；修改后的正式匹配器读取清理前保存的同一ID 2诊断副本后，以约`0.865/0.653`、分差`0.213`选择英文候选0且不回退。完整测试为`519 passed, 15 failed`，失败集中在工作区既有、未纳入本次提交的Runtime/L3实验改动（CUDA默认值及L3 Stub新参数契约），与本次L4匹配文件无调用关系。
- **未改变与资产**：L1、L2 MUSIC的2～4 kHz定位频带、ID/Kalman、L3 BF与缓存、L4重采样/人数判定/两种分离模型、L5、各UI、录音和数据管理均无变化；无模型、音频或Git LFS对象变化。真实缓存验证不替代更多房间、角度、语言和信噪比场景验收。

---

## 2026-08-24 — 临时方向ID提前进入L3并保留语音开头

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **L2→L3契约**：L2当前窗口实际观测到的`tentative`临时ID现在可以进入L3，优先级位于正式观测`confirmed`之后、正式预测`coasting`之前，并继续受最多3路及50°最小间隔约束；没有当前观测的临时ID不作为预测方向发布。临时ID确认后沿用同一数字ID，因此L3可保留确认前的说话开头。
- **拼接与输出过滤**：TrackAudioStream从临时ID首次观测开始建立权威时间线并拼接L3音频；Test UI在临时阶段隐藏该音轨，待同一ID正式确认时从Hub完整连续波形回填开头。Hub同时记录ID是否曾进入`confirmed/coasting`，始终未确认的临时ID即使累计超过2秒也强制丢弃；捕获结束后仍按既有规则物理删除短于2秒、未进入最终精确ID允许列表或静音占比不合格的音频，离线L4/L5不会收到这些候选。
- **未改变**：正式ID编号和颜色、ID确认门槛/TTL、Gate、MUSIC、L3波束形成算法、2秒最终过滤阈值、离线L4/L5、录音schema、Runtime STOPPED及性能监控均无变化。
- **验证与资产**：增加临时ID进入L3、确认后完整回填开头以及现有短音频/allow-list过滤邻接回归；无配置、模型、音频或Git LFS对象变化。自动测试不替代真实语音开头和多人交叠场景实机验收。

---

## 2026-08-24 — L2极坐标图仅为正式ID分配颜色

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **Development Test UI**：L2极坐标图中的临时`tentative` ID统一改为直径10 px的中性灰色圆点，不再提前占用按ID分配的稳定颜色；正式观测`confirmed` ID继续显示为直径24 px的彩色圆点，正式预测`coasting` ID继续以相同ID颜色显示为直径10 px圆点。
- **语义与兼容性**：颜色现在只表达已经确认的正式ID身份，圆点大小区分临时/预测小点与正式观测大点；L2跟踪、确认门槛、ID编号、L2到L3仅发布`confirmed/coasting`、Gate、MUSIC、L3～L5、Runtime队列及STOPPED监控均无变化。
- **验证与资产**：增加临时、正式观测、正式预测三种marker样式回归；无配置、模型、音频或Git LFS对象变化。自动测试不替代实机视觉验收。

---

## 2026-08-24 — 真实麦克风Test UI两阶段临时采集

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **启动采集阶段**：真实麦克风Test UI每次启动默认仅运行L1/IMCRA和L2 MUSIC/ID，`L3/4/5`保持关闭且不能被手动提前开启；L3试听区在此阶段只缓存Center RAW及实际启用时的Center IMCRA参考音频。
- **开始正式录音阶段**：Test UI中的“正式录音开始”改为本轮临时BF阶段开关。切换时只保留已经预热的IMCRA实例及噪声统计；MUSIC滚动状态和全部L2 DOA/ID追踪生命周期在互斥边界内清零，Center试听及下游连续音轨缓存同步清空，随后自动开启L3并由新L2结果重新建立ID、执行BF和按ID拼接。暂停会再次切断L2到L3/L5。
- **存储边界**：真实麦克风Test UI不再创建RecordingStore session，不写入正式音频、DecisionRecord、watermark或事件触发资产；scratch试听录制和Test UI分段缓存仍为临时文件。关闭UI或重新启动采集继续清除本轮临时缓存。模拟WAV/完整录音回放、Production录音及普通ApplicationRuntime仍保留原正式RecordingStore语义。
- **未改变**：IMCRA、预降噪、MUSIC、ID追踪、L3 BF算法、离线L4、L5模型、Production/Log UI、正式录音schema和数据管理接口均无算法变化。
- **验证与资产**：增加真实麦克风临时模式的下游门禁、仅IMCRA连续、L2完整重置、无正式session及暂停回退测试。未进行真实麦克风长时间实机验收；无模型、音频或Git LFS资产变化。

---

## 2026-08-24 — L3 CUDA异步微批实验与实机性能裁决

- **版本/标签**：项目仍为`1.3.3`开发线；不修改版本，不创建或移动发布标签。
- **L3与Runtime**：新增设备驻留的prepare/BF/ISTFT结果、pinned host异步回传及仅合并现有积压的有界CUDA微批路径；批次不等待未来窗口，完成后仍按原WindowKey、track ID和绝对sample顺序进入CPU连续音频拼接与记录。逐kernel同步的finite检查和诊断格式化延迟到批次末统一完成，公开音频、fallback和diagnostics保持同步路径兼容。
- **设备生命周期**：显式配置L3与离线L4共用CUDA时，只有实时worker停止并排空后才能创建L4；Runtime先同步L3 stream、清除L3滚动/device cache并调用CUDA allocator清理，再加载离线L4。
- **性能裁决**：本机RTX 5060 Laptop GPU、连续双候选四窗隔离微批约`11.03 ms/窗`，同条件CPU约`6.21 ms/窗`。同一条28.8秒真实八通道录音完整回放，CPU排空约`29.07 s`，CUDA各版约`34.75/33.85/32.30 s`，均无错误、丢窗或残余队列。CUDA降低CPU占用但未提高端到端吞吐，因此`runtime.l3_device`正式默认保持`cpu`；CUDA路径仅供显式诊断，不作为更快的生产配置。
- **配置与兼容性**：新增`l3_cuda_microbatch_windows=4`及`l3_cuda_batch_wait_ms=0`；旧配置使用默认值，CPU路径行为不变。新增同步/延迟接口等价、并行排空与L3→L4 CUDA缓存交接测试。
- **未改变与资产**：L1/IMCRA输出、L2算法与ID、L4/L5模型算法、UI交互、正式录音schema及数据管理均无变化；无模型、音频或Git LFS资产变化。

---

## 2026-08-24 — Test UI增加Center Mic IMCRA试听

- **版本/标签**：项目仍为`1.3.3`开发线；不修改项目版本，不创建或提前发布`v1.3.3`标签。
- **Development Test UI**：L3试听区固定按`Center Mic RAW`、`Center Mic IMCRA`、方向ID音轨排序。RAW继续缓存校准后且预降噪前的逻辑Center；IMCRA只在开关开启且Runtime为该20 ms hop实际选择降噪块时追加，开关关闭期间的旁路原音不会混入IMCRA缓存。
- **缓存与播放契约**：两个Center条目使用独立的会话级临时分段和播放快照；RAW保留完整采集，IMCRA完整保留所有实际降噪hop。IMCRA保持48 kHz及原session/epoch/sample顺序，仍不进入L2方向ID空间，也不改变正式录音或L5输入。
- **未改变**：IMCRA/Cohen噪声估计与Wiener增益、预降噪音频本身、L1采集与校准、CountNet、Windowing、L2～L5算法、方向ID、Production/Log UI、正式录音和数据管理均无变化。
- **验证与资产**：增加RAW/IMCRA独立缓存内容、完整保留、播放路径、UI名称/排序和开关边界标记测试；无模型、音频或Git LFS资产变化。

---

## 2026-08-24 — Development Test UI四层独立性能计时

- **版本/标签**：项目仍为`1.3.3`开发线；不修改项目版本，不创建或提前发布`v1.3.3`标签。
- **问题与修复**：底部性能横栏原把实时Runtime中用于有序审计的L5占位阶段显示成L5处理时长，导致停止后L5总时长与L3几乎相同，同时完全缺少离线L4计时。横栏现明确区分实时与离线工作流：上一秒性能只显示L2/L3的20 ms窗口统计，L4/L5标记为离线；总处理时长固定显示L2、L3、L4、L5四项独立记录。
- **计时边界**：L2、L3继续使用既有首窗入队至各自排空的Runtime计时，保留STOPPED、暂停排除及后续运行时长监控语义，L3窗口处理包含TrackAudioStream长音频拼接；点击“发送到L4”后，L4整批处理和其后的L5整批推理分别使用独立单调时钟计时。再次发送或开始新一轮采集/重播会清除上一批离线计时，失败的L5仍保留本批已耗用的L4、L5时长供诊断。
- **兼容性与未改变**：不修改L1输入、L2算法、L3波束形成/拼接算法、离线L4分离算法、L5模型、Runtime队列/屏障/STOPPED状态、录音/数据管理、公开DTO、配置、模型及音频资产；实时Runtime的L5占位审计计时仍保留，只是不再被Test UI误标为离线L5性能。
- **验证与资产**：增加L4/L5分段计时、重复提交重置及四层横栏文案回归；覆盖Development Test UI、离线L4/L5和并发Runtime邻接测试。无Git LFS对象变化；自动测试不替代真实长音频与设备负载验收。

---

## 2026-08-24 — 修正CountNet Theano卷积移植并启用v2模型

- **版本/标签**：项目仍为`1.3.3`开发线；不修改项目版本，不创建或提前发布`v1.3.3`标签。
- **根因与L1修复**：对原Keras 1.2.2/Theano模型逐层复算发现，旧TorchScript转换器直接复制卷积权重，但Theano执行数学卷积、PyTorch `Conv2d`执行互相关，遗漏的空间翻转使输出从`conv1`起失真。转换器现对四层卷积核的高、宽轴各翻转一次，并通过内存序列化规避Windows含中文工作区路径下的TorchScript保存失败。
- **模型与配置**：新增并默认启用`countnet_crnn_16k_5s_v2`，SHA-256为`f655f168bbd9091efd18b950e63484825ba68052a911331cab1e845e27e505e4`；算法身份更新为`countnet_crnn_5s_100ms_v3`。旧v1资产保留以便历史追溯，但不再被运行配置引用。
- **数值验证边界**：使用固定上游revision的官方`examples/5_speakers.wav`在原Python 3.6/Keras 1.2.2/Theano环境和新TorchScript间逐层对照；修正后原生11类概率最大绝对误差为`4.77e-7`，而旧模型最大误差约`0.496`。该验证证明移植实现忠实于上游模型，不代表真实办公室阵列准确率已经验收。
- **现有录音复测**：截图对应的`0821-0928`两声源录音按100 ms连续推理，旧模型仅`6/277`（`2.2%`）窗口输出2+，修正后为`226/277`（`81.6%`），且三位小数概率共有129组，确认监视值会随音频更新。另对6条两声源录音按1秒抽样为`110/128`输出2+；但4条一声源录音仅`34/105`输出1。录音名称标的是整段最大声源数而非逐窗同时讲话真值，因此这些数字只作诊断；一人场景的域泛化仍不合格，不能宣称人数识别已完成实机验收。
- **未改变**：Center选择、48→16 kHz状态化重采样、5秒上下文、100 ms节拍、无平滑、电平适配、异步worker、IMCRA、预降噪、Windowing、L2～L5、录音/存储及UI交互均无变化。
- **测试与Git LFS**：增加Theano卷积核转换回归，并更新模型hash、配置和CPU推理门禁；新增v2 TorchScript Git LFS对象，无音频资产变化。

---

## 2026-08-24 — L3/L4/L5独立设备分配

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **默认设备拓扑**：Runtime由原先一个`preferred_device`统一控制L3/L4/L5，改为`L1 CPU → L2 CPU → L3 CPU → 离线L4 CUDA → L5 CPU`；L3实时BF和轻量L5不再占用L4分离所需GPU。
- **配置兼容**：新增`runtime.l3_device/l4_device/l5_device`；旧配置未提供独立字段时仍回退到`preferred_device`，任一CUDA层在`allow_cpu_fallback: true`且CUDA不可用时独立回退CPU。
- **Runtime与记录**：L3处理器/CUDA流、L4 MossFormer2或TIGER、L5 MarbleNet分别读取自己的设备；`processing_device`保留为L3兼容别名，运行状态及session metadata新增L1～L5完整设备映射。
- **未改变**：L1采集/IMCRA/预降噪/CountNet、L2 Gate/MUSIC/ID追踪、L3 BF算法与输出、离线L4分离/2～4 kHz匹配、L5概率算法、各UI交互、队列调度、正式录音和数据管理语义均无变化。
- **验证与资产**：增加独立设备、旧配置fallback、CUDA不可用回退及离线L4设备消费测试；无模型、音频或Git LFS对象变化。

---

## 2026-08-24 — L3有效频带求解与按需加载重试优化

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L3求解性能**：Loaded MVDR及正式optimized/adaptive波束形成只对最终会输出的80～8000 Hz频点构造加载协方差、执行Cholesky/条件校验和多右端求解；频带外继续保留DAS权重兼容内部API，并在设备侧增强频谱发布前清零，最终48 kHz音频频带语义不变。
- **按需retry**：原三档diagonal loading由全量并行计算改为保持顺序的惰性重试；每个频点和目标继续选择首个数值有效档位，仅仍未求解成功的频点进入下一档，DAS fallback计数、诊断字段和公开L3 DTO保持不变。
- **未改变**：L1采集/IMCRA/预降噪/CountNet、L2 Gate/MUSIC/ID追踪、离线L4、L5、Development Test UI、Production/Log UI、TrackAudioStreamHub、正式录音/数据管理、Runtime调度、配置、模型与音频资产均无变化。
- **验证与资产**：增加新旧固定retry数值等价、频带外不参与分解、仅失败频点继续重试及设备侧频谱清零回归；无Git LFS对象变化。自动测试不替代真实麦克风、低性能CPU或长时间GPU负载验收。

---

## 2026-08-24 — L2 Test UI刷新与L3/Join解耦

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **Runtime/L2 UI旁路**：新增容量1、latest-only的`latest_l2_dev_ui`。L2 worker完成每个窗口后立即发布包含Gate、DOA/MUSIC、方向ID、实际配置revision和精确窗口身份的不可变快照，不再等待同窗L3/L5终态或有序Commit；邮箱覆盖次数纳入只读Runtime诊断。
- **Development Test UI**：L2面板只消费独立L2快照，按当前Runtime `session/epoch`过滤旧流，并要求`window_id/decision_sample`单调前进；现有有序`latest_dev_ui`不再回写L2面板，只继续更新L1/L3、性能、录音和正式审计。因此L3积压时L2面板追随L2完成窗口，不再重放较旧的Join窗口。
- **兼容性与未改变**：L2算法、Gate/MUSIC/ID Tracking、L3、离线L4/L5、ResultJoiner、DecisionRecord、水位、队列容量、20 ms正式时间轴、模拟重播清队列、`stopped`运行时长和Production UI均无变化；无配置、模型、音频或Git LFS资产变化。
- **验证**：增加L3首窗阻塞且Commit为0时L2独立邮箱仍推进到最新完成窗口的回归，并覆盖旧session/epoch、同流倒序窗口过滤及Test UI/Runtime/契约测试；聚焦回归`96 passed`，完整pytest为`526 passed`，完整Ruff与差异检查通过。

---

## 2026-08-24 — 模拟播放途中重播立即丢弃旧处理队列

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **Runtime与队列**：新增仅供操作者主动重播使用的硬换代路径。点击“从头重播”后立即暂停模拟输入，标记旧处理图为丢弃并清空所有尚未执行的L2、L3、L5、completion及commit backlog；已经进入CPU/CUDA调用的单个窗口不做不安全抢占，调用返回后其Hub、UI和Commit迟到结果全部丢弃，旧Recording session以`replay_restarted_discarded`结束。
- **Development Test UI与回放源**：重播改为后台执行“丢弃旧图→从sample 0重开回放源→启动全新Runtime图”，命令期间按钮置忙但界面先立即清空；关闭过的回放源重播时同步递增generation并重置sample/sequence，保证同一按钮在运行中、EOF后及`stopped`状态均可重新开始。
- **监控与兼容性**：普通停止及模拟EOF仍完整排空并封存`stopped`下的L2/L3/L5总运行时长；只有显式点击重播才丢弃等待工作。L1～L5算法、L2权威音轨时间轴、离线L4/L5、公开DTO、配置、模型、音频和Git LFS资产均无变化。
- **验证**：增加L3首窗执行中且后续多窗排队时的强制重播回归，验证只执行已开始的一窗、其余队列和Commit状态归零、旧结果不落库并能立即启动新图；增加运行中Test UI重播确实更换Runtime session，以及关闭后回放源从sample 0重开回归。Runtime/Replay/Test UI聚焦测试`70 passed`；当时完整基线`523 passed`，新增界面断言在提交前再次定向验证。相关Ruff与差异检查通过。

---

## 2026-08-24 — 模拟输入长音频长度改由L2权威时间轴确定

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **类型与涉及文件**：修复Application Runtime模拟EOF排空、L2→TrackAudioStreamHub时间轴、Development Test UI试听封存及相邻测试；同步Runtime说明和本日志。
- **Runtime/监控**：完整模拟输入EOF使用数据完整排空，不再套用实时设备的10秒强制停机期限；排空期间Test UI显示`FINALIZING`，全部L2/L3/L5/Commit完成后才进入`stopped`，各层总运行时长继续封存并保持可读。实时或手动有限停机若取消窗口会明确写入Runtime错误并以`runtime_error`结束录音session，不再伪装为正常完成。
- **L2/L3/连续音轨**：每个L2完成窗口先按`(session_id, stream_epoch, track_id)`登记confirmed/coasting权威20 ms绝对时间槽；L3仅向对应槽写入BF波形。首个BF结果之前、处理中间缺口及最后一个BF结果之后的缺失槽均保留等时静音和未观测语义，Hub与Test UI最终长度由L2首尾sample决定，不再随BF算法吞吐速度改变。
- **离线L4/L5及兼容性**：离线L4继续读取同一Hub封存48 kHz长轨，缺失BF槽的方向数量沿L2时间轴保存；L4/L5算法、模型、阈值、公开音频DTO、录音格式、L1、L2定位/追踪算法及Production UI均无变化。
- **验证与资产**：增加不同BF处理覆盖率仍保持相同sample数、试听缓存尾部补齐、模拟回放越过实时停机期限仍完整排空、有限超时明确失败等回归；相关Runtime/Hub/Test UI定向回归`63 passed`，基于最新开发基线的完整pytest为`511 passed`，相关Ruff与差异检查通过。无模型、音频、配置资产或Git LFS对象变化；自动测试不替代真实GPU长回放试听验收。

---

## 2026-08-24 — 修复L1 CountNet低电平输入概率冻结

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **原因与L1修复**：实测Center约`-51 dBFS`，而官方LibriCount示例约`-16 dBFS`；原预处理在低电平输入下退化为近似恒定`P0=1`。`countnet_crnn_5s_100ms_v2`在模型前增加可配置的去直流和仅向上电平适配：高于`-70 dBFS`的5秒上下文向`-20 dBFS`提升，最多30 dB，超过0 dBFS时等比例限幅；低于门限的数字静音不放大。
- **L1监视**：`SpeakerCountAnnotation`增加适配前RMS和实际增益诊断；Development Test UI把概率显示提高到三位，并显示`input/gain/end_sample`，可直接确认100 ms标注是否推进，不再把两位小数造成的视觉不变误判为worker停滞。
- **验证边界**：增加约`-53.5 dBFS`阵列输入增益、数字静音不放大和配置回归；低电平官方5人示例离线探针从恒定0人恢复为2人以上。该探针确认电平退化已修复，不等于办公室真实人数准确率验收。
- **未改变**：CountNet权重及其Git LFS对象、48→16 kHz重采样、5秒上下文、100 ms发布、无时间平滑、L1非阻塞worker、IMCRA、预降噪、Windowing、L2～L5、录音和数据格式均无变化。

---

## 2026-08-24 — L2近距离候选续接旧ID并保持IMM平滑输出

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L2 ID追踪**：Circular IMM-JPDA正常关联后增加50°圆周范围内的一对一救援关联；统计门禁或关联概率短时失败时，候选优先续接存活旧ID，50°内的其余重复候选禁止创建新ID，只有距离所有存活轨迹超过50°且满足原新生概率门限的候选才可创建tentative ID。
- **角度输出**：救援候选仅作为IMM/Kalman观测进入状态更新，公开`theta_deg`继续输出滤波后验角度，不直接跳到原始候选位置；`measured_theta_deg`保留原始观测用于诊断，359°/0°继续使用圆周距离。
- **未改变**：MUSIC伪谱和候选提取、L1、L3、离线L4、L5、Development Test UI、录音/数据格式、模型与音频资产均无变化。

---

## 2026-08-24 — L2正式ID漏检恢复与存在概率衰减修正

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L2轨迹生命**：已确认ID漏检后保留到2秒绝对sample TTL，不再被逐窗Bayesian miss在数十毫秒内降至删除门限；存在概率改为按真实时间衰减，等价于每20 ms保留约0.97。在TTL内再次匹配时继续使用原ID、恢复`confirmed`并提高存在概率；连续2秒没有真实观测才删除。
- **L2关联与临时ID**：卡方关联门限从9调整为20，仍保留50°固定角差硬上限，不引入随漏检时长扩大的动态角度范围。tentative仍需在200 ms内累计3次观测才能确认，并保留低概率快速淘汰。
- **未改变**：L1采集、IMCRA、预降噪及其他L1开发中内容，Probability Gate，MUSIC/GI-DOAEnet扫描，DPD，白化，L3，离线L4，L5，Runtime/UI交互，录音格式和资产均无变化。
- **验证与Git LFS**：增加正式ID低概率仍存活到TTL、长漏检后恢复原ID以及配置回归；无模型、音频或Git LFS对象变化。

---

## 2026-08-24 — L1增加异步CountNet讲话人数诊断支路

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L1**：新增不可变`SpeakerCountAnnotation`和`AsyncSpeakerCounter`。支路只读取校准后Center Mic，
  以状态化抗混叠滤波完成48→16 kHz，每5个连续20 ms块产生一次100 ms标注；前5秒预热，不做
  时间平滑，epoch/sequence/sample/timestamp gap或队列溢出均清空5秒缓存并重新预热。
- **并发边界**：L1线程只用`put_nowait`投递到独立有界`l1-countnet-worker`；重采样、缓存、模型加载
  与推理均不占用采集回调。第一阶段不增加Windowing join或主链延迟，CountNet结果不写入
  `IngestedAudioBlock/DecisionWindow`，也不传播到L2～L5、长音频、正式存储或其他UI。
- **模型与依赖**：引入MIT许可的Stöter等人官方CountNet CRNN，经脚本从固定上游revision的Keras
  1.2.2模型与scaler转换为TorchScript；输入为16 kHz/5秒，原生0～10人输出折叠为P0、P1和
  P2+。模型和配置以SHA-256固定，许可证与第三方NOTICE已同步；`h5py`仅作为可复现转换脚本的
  开发依赖，实时推理仍只使用既有PyTorch。当前自动测试不代表真实办公室准确率验收。
- **Development Test UI**：L1区增加持久化手动开关和`warming_up/ready/invalid`只读状态，READY
  显示人数、P0/P1/P2及模型ID；关闭后清空私有状态，再开启重新预热。
- **未改变**：WindowAssembler、DecisionWindow、L2 Gate/MUSIC/MDL/ID、L3、离线L4、L5、
  Recording/Data Manager、Production UI、Log UI及现有录音格式均无变化。
- **验证与Git LFS**：覆盖DTO、5秒预热、精确降采样长度、5窗对齐、无平滑、连续性重置、非阻塞
  投递、模型hash/输出、稳定CPU推理门禁、Runtime/UI开关与旧配置读取；完整pytest和静态检查在提交前执行。
  新增TorchScript模型由Git LFS管理；不提交上游HDF5、测试音频或本地运行数据。

## 2026-08-24 — L2方向ID追踪重构为Circular IMM-JPDA

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L2**：以单一`circular_imm_jpda_v1`正式替换MUSIC/GI-DOAEnet后的旧Hungarian、简化LMB/JPDA、独立Kalman和静止锁定组合；采用带miss/new/false假设的有界JPDA、静止/慢速移动双模型IMM、Bernoulli存在概率及tentative/confirmed/coasting/deleted生命周期。确认规则为滚动200 ms内3次观测；confirmed漏检后预测2秒；内部最多4条轨迹，公共输出仍最多3条。
- **圆周与时间契约**：关联、滤波和输出正确跨越359°/0°；内部连续角状态定期按整圈重基准，避免多圈运动造成数值无限增长；确认、漏检、TTL、epoch/session全部使用48 kHz绝对sample。
- **L4反馈边界**：保留L4→L2反馈队列和接口以便后续启用；本版只校验并短期记录，不参与JPDA、确认、存在概率、轨迹寿命、Gate强制开启或IMM状态。
- **Runtime与Development Test UI**：两个DOA后端共用同一权威追踪器；删除独立Kalman按键和Q/R控件，只保留`ID Tracking`总开关。关闭时输出原始DOA候选且不建立持续ID；开启时运行完整IMM-JPDA及生命周期。
- **配置/接口**：删除`layer2.direction_kalman`配置段，将IMM、JPDA、概率和生命周期参数统一纳入`direction_id_tracking`；保留`TrackedDirection.kalman_applied`和旧Runtime快照字段作为下游兼容投影，其值不再代表可切换的独立滤波器。
- **未改变**：L1、Probability Gate、MUSIC/GI-DOAEnet扫描、DPD、IMCRA白化、L3波束形成、离线L4、L5模型、录音音频格式和模型/音频资产均无变化。
- **验证与Git LFS**：更新配置、L2、Runtime和Test UI回归；完整pytest及静态检查结果记录于本次提交；无Git LFS对象变化，不提交本地运行数据。

---

## 2026-08-24 — L2删除MDL并改用手动MUSIC阶数

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L2 MUSIC**：删除正式主链中的MDL阶数估计、刷新缓存、饱和诊断及对应配置。普通MUSIC每窗直接使用Test UI手动选择的`1/2/3`作为信号子空间阶数和最多搜峰数；DPD路径继续把该值作为最多候选数。
- **Runtime/Test UI/兼容性**：运行快照和右侧状态条改为显示手动MUSIC阶数与实际候选数，不再显示MDL。为避免破坏既有记录DTO，`ModelOrderEstimate`及旧`mdl_age_samples`字段暂时保留为兼容投影，值固定为`0`，没有自动阶数估计器在后台运行。
- **文档与测试**：同步更新L2权威架构、项目README、L2 README和Development Test UI说明；配置与回归测试改为验证手动阶数直接生效及MDL配置已删除。
- **未改变**：L1采集、CountNet、IMCRA、预降噪、Probability Gate、MUSIC频带/伪谱/DPD/白化、Circular IMM-JPDA、L3、离线L4、L5、录音格式和模型/音频资产均无变化。
- **Git LFS**：无模型、音频或其他Git LFS对象变化。

---

## 2026-08-24 — L2临时ID确认门限提高至5次

- **版本/标签**：项目仍为`1.3.3`开发线；不创建或提前发布`v1.3.3`标签。
- **L2 ID追踪**：tentative轨迹转为`confirmed`的观测门限由滚动200 ms内3次提高为5次；存在概率门限仍为0.70，观测不要求连续占满窗口。
- **未改变**：500 ms临时ID TTL、2秒正式ID coasting TTL、IMM-JPDA关联、50°关联/NMS、MUSIC/DPD/白化、L1、L3～L5、Runtime接口、录音格式及资产均无变化。
- **验证与Git LFS**：更新配置契约和L2文档并执行定向回归；无Git LFS对象变化。

---

## 2026-08-24 — 冻结v1.3.2并开始1.3.3开发线

- **版本/标签**：项目`1.3.2`最终版固定在发布提交及不可变标签`v1.3.2`，不移动、不覆盖；项目包和当前状态文档从本提交开始更新为`1.3.3`，尚未创建`v1.3.3`标签。
- **分支策略**：从`v1.3.2`建立`codex/develop-v1.3.3`；后续修改进入该开发线，`main`与`v1.3.2`继续保持1.3.2正式发布基线。
- **未改变**：L1～L5算法、Runtime、各UI、录音/数据管理、配置schema与参数、模型权重、测试和资产均无变化；本提交仅初始化下一开发版本。
- **验证与Git LFS**：版本、配置与文档契约定向测试`39 passed`，全仓Ruff及`git diff --check`通过；无新增或修改Git LFS对象，不提交本地运行数据。

---

## 2026-08-24 — 项目1.3.2整合发布

- **版本/标签与分支**：保持已发布`v1.3.1`固定在原提交，不移动、不覆盖；将其后的全部有效提交整合为`v1.3.2`，同步`codex/develop-v1.3.1`与`main`后创建新的不可变标签`v1.3.2`。
- **发布内容**：纳入7路物理麦实机校准、完整架构与使用手册、L3隐藏音轨最终清理、可切换GI-DOAEnet PM + LMB/JPDA完整L2后端，以及Development Test UI模拟播放结束后可从头重播；同时包含`v1.3.1`既有L1～L5、Runtime、各UI、录音/数据管理、测试、模型和精选资产。
- **版本文件与文档**：项目包、根README、权威架构、完整使用手册、Log UI契约、文件分类及各层README统一更新为`1.3.2`发布状态；L2公共接口版本继续保持`1.1`。
- **兼容与验收边界**：不重写历史DecisionRecord、录音或发布标签；自动测试与本机GI-DOAEnet推理记录不替代真实诊室双声源、长时间采集、GPU分离音质和目标域概率校准验收。
- **数据与Git LFS**：不提交`data/`、运行录音、Catalog、scratch、日志、缓存、partial、密钥、Token或本地代理设置；本次发布不新增模型权重，复用并核验既有Git LFS对象。
- **验证**：完整pytest共`513 passed`；全仓Ruff与`git diff --check`通过；同时核验暂存数据边界、Git LFS及远端引用。

---

## 2026-08-24 — Development Test UI模拟播放结束后允许从头重播

- **版本/标签**：项目仍为`1.3.1`；不创建或移动发布标签。
- **类型**：Development Test UI模拟输入控件状态修复与回归测试。
- **涉及文件**：`gui/dev_test_ui/app.py`、`tests/test_dev_ui.py`和本日志。
- **Development Test UI**：完整录音模拟输入到达EOF并完成Runtime停止、队列排空和音轨封存后，“从头重播”按钮恢复可用；保留`stopped`状态及其已封存的L2/L3/L5总运行时长，只有点击重播后才沿用既有流程重置计时、重新打开回放源、清空上一轮画面并启动新一轮处理。
- **测试与兼容性**：增加`stopped`回放状态下按钮可用、停机总运行时长保持冻结，以及点击后恢复`playing`的界面回归；模拟输入格式、Runtime计时与停机接口、L1～L5算法、离线L4/L5、录音与数据管理均无变化。
- **Git LFS/资产**：无二进制文件、模型、音频或Git LFS资产变化；自动测试不替代实机界面验收。

## 2026-08-21 — 增加可运行的GI-DOAEnet完整L2替代链

- **版本/标签**：项目仍为`1.3.1`；不创建或移动发布标签。
- **类型**：L2双后端架构、神经网络适配、概率数据关联、Runtime/Test UI与本地模型安装。
- **涉及文件**：`layer2_source_detection/gi_doaenet.py`、`global_tracker.py`、`pipeline.py`、配置/Runtime、Development Test UI、模型清单与安装器、定向测试、本README和本日志。

### L2与Runtime

- 保留默认`Probability Gate → Rolling NormMUSIC → Hungarian → 可选圆周Kalman`完整链。
- 新增可切换的`Probability Gate → GI-DOAEnet PM → 候选门控 → LMB/JPDA → 圆周Kalman`完整链；两套链读取同一DecisionWindow并输出相同360点SpatialResponse、最多3个方向、TrackedDirection及active_tracks。
- NN适配器只读取7路物理麦，48→16 kHz同相重采样，使用3维麦位和最后一层最近5帧平均概率；继续执行UI候选门限、prominence和50°圆周NMS。
- LMB/JPDA链在最多4个内部轨迹、3个观测的有界空间内枚举一对一联合假设，计算边缘关联概率和Bernoulli存在概率，再确定性提取关联；不使用rank绑定。
- Runtime配置快照新增完整L2后端，切换在下一窗口原子生效；目标链轨迹状态单独维护并在切换边界清理，录制诊断写入实际链与实际关联后端。

### Development Test UI与模型资产

- L2右侧顶部增加完整方案按钮，可在`MUSIC + Hungarian`与`GI-DOAEnet + LMB/JPDA`间实时切换并原子持久化；左侧360°谱与最终ID点绘制契约不变。
- NN方案下禁用MUSIC专属DPD/Whitening控件，状态栏显示实际后端、输出源数与推理状态。
- 固定上游提交和PM权重SHA-256；因上游无LICENSE，不提交其源码/权重，提供显式确认的本地安装器并将下载目录Git忽略。无Git LFS变化。

### 未改变

- L1/IMCRA、Probability Gate算法和门限、现有MUSIC数值路径、L3/L4/L5公共输入输出、录音音频内容及发布标签均未改变。

### 验证

- 新增NN 7麦/16 kHz适配、360点谱/双峰、双链独立关联器及UI设置持久化测试；既有配置、L2 MUSIC/ID/Kalman和Test UI回归同步执行。
- 全量回归测试`513 passed`，Ruff全仓检查与Git差异检查通过。
- 本机CUDA用官方PM权重完成真实推理：合成30°单源输出29°，稳态7窗均值约7.36 ms、p95约10.61 ms；首次懒加载约2.71秒。

## 2026-08-21 — 删除L3隐藏音轨并阻止其进入离线L4

- **版本/标签**：项目仍为`1.3.1`；不创建或移动发布标签。
- **类型**：Development Test UI试听缓存、TrackAudioStreamHub封存边界与Runtime离线L4输入修复。
- **涉及文件**：`gui/dev_test_ui/audio_id_tracker.py`、`track_audio_stream/service.py`、
  `app/runtime.py`、相关定向测试和本`CHANGELOG.md`。

### L3试听与L4提交

- 确认并修复两套数据源不一致：L3面板隐藏/过滤音轨后，Hub长音频归档此前仍可能保留同一ID并发送到L4。
- 采集与队列完全结束后，以L3最终保留的`(session_id, stream_epoch, track_id)`作为离线L4唯一白名单；
  不足2秒、声音占比不超过30%或因其他最终过滤而未显示的方向音轨，同时从Hub归档中物理移除。
- 白名单为空时清空全部方向归档；L3缓存最终过滤失败时采用失败关闭，不向L4泄漏不可见音轨。
- Center Mic仍仅作为试听对照，不进入方向音轨白名单，也不发送到L4。
- 切换L3处理模式时，立即删除已从界面消失的旧模式文件及对应Hub归档，避免相同正式ID混入隐藏模式音频。

### 未改变

- L1、L2的MUSIC/ID追踪、L3波束形成算法、L4分离模型、L5分类模型、正式录音、配置和模型资产均无变化。
- 未修改2秒显示门槛、30%声音占比门槛及其现有声音判定方式；没有Git LFS资产变化。

### 验证

- `pytest -q tests/test_track_audio_stream.py tests/test_dev_audio_id_tracker.py tests/test_runtime.py -k "track_audio or dev_audio or offline_l4 or stop"`：`35 passed, 20 deselected`。
- 覆盖最终白名单、空白名单、旧L3模式归档删除及相关停止/离线L4路径；`git diff --check`通过。

## 2026-08-21 — 新增完整架构图与首次使用手册

- **版本/标签**：项目仍为`1.3.1`，不创建或移动`v1.3.1`及任何发布标签。
- **类型**：文档与架构说明；不改变运行行为。
- **涉及文件**：`docs/COMPLETE_ARCHITECTURE_AND_USAGE.md`、根`README.md`、
  `ARCHITECTURE_V1.1_TARGET.md`和本`CHANGELOG.md`。

### 文档

- 按当前代码绘制实时主链、采集后L4/L5链、存储旁路和独立观察工具的完整Mermaid架构图。
- 为硬件输入、L1、Ingest、Windowing、Runtime封装、L2、L3、TrackAudioStreamHub、实时L5审计、
  ResultJoiner、RecordingStore、离线L4与L5逐项列出正式输入、内部处理单元、正式输出、数据形状、频段和节拍。
- 增加L1、L2、L3及Hub→L4→L5内部处理图，并明确`WindowKey`、绝对sample时间轴、公共`track_id`
  以及实时/离线结果的保存边界。
- 补充环境创建、自检、Development Test UI从采集到离线结果的完整操作顺序，以及L1 Spectrum UI、
  Audio Data Manager、命令行离线L4/L5和Pipeline Log UI入口说明。
- 补充界面区域数据来源、正式与临时资产位置、推荐测试场景、常见状态排查和当前物理/模型/性能限制。
- 根README与1.3.1架构契约增加手册入口；同时修正根README中测试WAV说明末尾的标点错误。

### 未改变

- L1、Ingest、Windowing、L2、L3、TrackAudioStreamHub、离线L4/L5、Runtime调度、ResultJoiner、
  RecordingStore和所有UI代码均未改变。
- `config/config.yaml`、模型、测试音频、数据目录、依赖和发布版本均未改变；没有Git LFS资产变化。

### 验证

- Markdown代码围栏、全部本地相对链接及Mermaid主图的`flowchart/subgraph/end`结构检查通过。
- 使用严格配置加载器核对48 kHz/8通道/20 ms、160 ms上下文、L2 2～4 kHz/240 ms/50°/最多3方向、
  L3 80～8000 Hz、L5阈值0.70、L4最短2秒和4 cm阵列半径，均与手册一致。
- Development Test UI与`python -m scripts.run_offline_l4`帮助入口验证通过；后者也用于修正手册中的离线命令写法。
- `pytest tests/test_config.py tests/test_parallel_config_and_docs.py tests/test_packaging_contract.py -q`：`39 passed`。
- `git diff --check`通过。

## 2026-08-21 — 完成7路物理麦实机增益、极性与整数延迟校准

- **版本/标签**：当前`v1.3.1`分支的实机校准更新；不修改项目版本，不创建或移动发布标签。
- **类型**：L1硬件校准配置、可审计校准报告及配置契约测试。
- **涉及文件**：`config/config.yaml`、`docs/L1_HARDWARE_CALIBRATION_2026-08-21.json`、
  `.gitattributes`、`tests/test_config.py`和本`CHANGELOG.md`。

### L1与硬件校准

- 在有声办公室内，以阵列正上方居中扬声器播放100～8000 Hz粉红噪声和30次200～10000 Hz扫频，
  直接采集48 kHz原生8通道；7路物理麦的有效校准信号比静音底噪高20.77～25.48 dB，且无削波。
- 仅按物理通道映射`[0,1,2,3,4,5,7]`写入相对增益、全正极性和整数补偿延迟；原生CH6
  `HardwareMix`未参与校准，也不进入MUSIC。
- 校准版本更新为`office_overhead_20260821_v1`，状态保持`verified`，并以校准报告文件的SHA-256
  固化报告边界；报告JSON固定使用LF，保证不同平台检出的文件哈希一致。原始实机录音遵守数据边界，
  不纳入Git。
- 30次扫频的极性判断全部一致；相对Center的整数到达偏移中，除MIC1有2次落在相邻sample外，
  其余通道30次完全一致。亚采样延迟和频率相关校准资产仍保持预留且未启用。

### 其他模块、兼容性与验证

- **Windowing、L2、L3、离线L4、L5、各UI、录音/数据管理、Runtime和模型资产**：无变化。
- 校准DTO和校正模型不变；配置哈希会随新的校准版本及参数变化，持续运行的旧epoch不得混用新旧校准。
- 验证：更新后执行配置契约、L1校准器及直接消费校准元数据的Ingest/Windowing测试。
- **Git LFS**：无新增或修改的LFS资产。

## 2026-08-21 — 按1.3.1代码重绘实时与离线总架构

- **版本/标签**：已发布`v1.3.1`的文档维护；不创建、移动或替换发布标签。
- **类型**：全项目只读盘点后的架构图、权威契约及分层README同步；不修改程序行为。
- **涉及文件**：根`README.md`、`ARCHITECTURE_V1.1_TARGET.md`、`app/README.md`、
  `layer2_source_detection/README.md`、`layer4_speech_separation/README.md`、
  `layer5_voice_classifier/README.md`、`gui/dev_test_ui/README.md`和本`CHANGELOG.md`。

### 架构同步

- 将主链明确拆成两部分：实时链只计算L2、L3与`TrackAudioStreamHub`，实时L5 worker仅提交
  `SKIPPED(reason=offline_after_l4)`供ResultJoiner逐窗审计；真正的L4/L5不进入实时ResultJoiner。
- 补全停机离线链：Hub按权威ID封存完整48 kHz长轨，Test UI选择MossFormer2或TIGER后提交L4；
  L4按`min(2, 整轨L2方向数最大值)`进行一人旁路或双人分离，以2～4 kHz复频谱相干匹配主候选，
  低可信时回退L3参考，并输出保留原ID/角度的原生16 kHz音频。
- 明确L4整批完成后由同一后台任务自动且仅一次运行离线MarbleNet L5；L5不再重采样，逐320样本
  输出20 ms概率并回写L4音频条，离线结果不伪装成实时DecisionRecord或自动回写RecordingStore。
- 将独立L1 Spectrum UI补入项目级架构，标明其自建L1-only链且不创建WindowAssembler、L2～L5、
  正式录音或数据管理服务。
- 按当前配置校准L2文档：普通MUSIC实际阶数直接取Test UI手动上限，MDL仅作诊断；DPD方向簇门禁为
  至少4个频点、支持率0.20、覆盖2/4子带、集中度0.85；确认门禁为200 ms内3次观测、TTL为2秒。
- 澄清离线L5不向L2反馈：代码保留精确ID在线反馈兼容接口，但当前ApplicationRuntime没有调用方；
  confirmed/coasting公共方向由L2自身状态投影，普通运行不会因离线语义强制开Gate或改变ID寿命。

### 未变化组件、验证与资产

- L1～L5代码、Windowing、Runtime实现、TrackAudioStreamHub、ResultJoiner、Recording/Data Management、
  Development/Production/Log/L1 Spectrum UI代码、配置schema与参数、测试、模型和音频资产均无变化。
- 文档代码块、Mermaid、本地链接、冲突标记、配置关键值与`git diff --check`静态检查通过；配置、L2、
  Runtime审计、TrackAudioStreamHub、离线L4/L5、Development Test UI与L1 Spectrum UI专项自动测试
  `190 passed`。
- 未修改Git LFS管理的模型、音频、空间表、论文或其他二进制资产，无新增LFS对象；未提交`data/`、
  录音、Catalog、临时L4 WAV、缓存、日志、密钥、Token或本地代理设置。
- 本次文档同步不代表真实7通道阵列、诊室双声源、中文目标域、两种L4模型音质或长时间负载已重新验收。

---

## 2026-08-21 — 项目1.3.1整合发布

- **版本/标签与分支**：将`codex/develop-v1.3.1`自`v1.2.4`以来的全部开发提交快进合入`main`，并创建新的不可变标签`v1.3.1`；既有发布标签、远端分支和历史保持原位，不移动、不覆盖、不删除。
- **分支收束**：发布前同时核对全部本地与GitHub分支；没有任何分支包含`codex/develop-v1.3.1`之外的独有提交，所有有效内容均已进入本次发布历史，不机械重放过时功能分支。
- **发布内容**：完整发布L1采集/IMCRA、L2 MUSIC与公共方向ID/Kalman、L3波束形成与按ID长音频封存、离线L4双人分离及2～4 kHz候选匹配、NVIDIA L5逐20 ms人声概率、Development/Production/Log UI、录音与数据管理、Runtime契约、配置、文档、测试、模型和精选资产。
- **L4/L5工作流**：停止采集并排空L3后，Test UI通过“发送到L4”处理整批封存音轨；L4完成后自动且仅一次把原生16 kHz结果送入L5。MossFormer2/TIGER模型选择、低可信安全回退、试听、逐20 ms概率和黄色区间均纳入发布。
- **发布门禁修复**：调整L1串口停止顺序，先通知并等待读取线程退出、再关闭端口，避免Windows pyserial重复释放事件句柄；清理L2 tracker一处不再使用的局部返回值，不改变追踪行为。
- **保持不变与验收边界**：不修改L1～L5算法参数、模型权重、音频资产和配置schema；除上述关闭竞态与静态清理外，本次只收束版本、发布状态与Git历史。自动测试不替代真实阵列、诊室双声源、长时间负载、GPU音质和目标域概率校准验收。
- **数据与Git LFS**：不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、partial、密钥、Token或本地代理设置；发布提交不新增Git LFS对象，版本所需既有LFS模型与精选资产随标签可获取。
- **验证**：完整pytest共`506 passed`；全仓Ruff与`git diff --check`通过；同时检查敏感数据暂存、Git LFS状态以及远端分支/标签指向。

---

## 2026-08-21 — L4完成后自动运行L5并回写音频条

- **版本/标签**：当前`1.3.1`开发线Development Test UI工作流调整；不创建或移动发布标签。
- **自动接线**：删除L4栏手动“发送到L5”按钮；每次L3重新提交并完成整批L4分离后，同一后台任务自动把本批L4原生16 kHz波形交给L5，避免漏点、重复点或旧批次结果串入。
- **结果显示**：L5逐20 ms概率和判定继续按原`track_id`回写L4试听快照，并直接在对应音频条显示黄色识别区；L5失败时保留已完成的L4试听音频并明确显示失败原因。
- **重复运行**：每次“发送到L4”仍先清空上一批L4/L5界面与缓存，再写入本批结果；成功后允许再次提交并完整替换。
- **验证**：更新Development Test UI布局与重复提交回归，覆盖手动按钮移除、每轮L4后恰好一次L5处理、识别结果回写及重复提交状态清理。
- **未改变**：L1、L2 DOA/ID/Kalman、L3 BF与长音频拼接、L4两种分离模型、L5模型/阈值算法、正式录音和模型资产均无变化。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — L4试听与离线L5统一使用原生16 kHz输出

- **版本/标签**：当前`1.3.1`开发线L3→L4→L5离线工作流修复；不创建或移动发布标签。
- **L4输出与试听**：`Layer4ProcessedAudio`终端波形改为完整16 kHz、每20 ms 320样本；Test UI写标准16 kHz单声道PCM16 WAV。播放器解析WAV容器后按16 kHz直接建立PortAudio输出流，L3裸float32试听仍保持48 kHz，明确禁止L4试听16→48 kHz重采样。
- **L5接线**：点击“发送到L5”时直接传递L4的同一份16 kHz波形；离线NVIDIA Frame-VAD新增原生16 kHz逐帧入口并拒绝48 kHz长音频，删除L4的16→48 kHz回升及L5再次48→16 kHz降采样。逐20 ms概率仍按原ID/角度返回，只用于L4预览条黄色标记。
- **实时边界**：Runtime继续只运行L2、L3和TrackAudioStreamHub；L3 worker不再创建未使用的实时L5音频输入，逐窗L5仅保留`offline_after_l4`跳过审计，不执行CNN。
- **持久化与兼容**：新离线L4结果WAV和manifest内嵌输出统一为16 kHz；这是当前开发线离线输出契约变更，不兼容依赖旧`waveform_48k/output_waveform_48k`字段的未发布调用方。
- **验证**：覆盖L4 WAV采样率/帧数、PCM16逐样本解码、16 kHz声卡流建立、L4同一波形直送L5、逐20 ms结果回写及官方MarbleNet真实语音帧检测；聚焦和完整测试结果记录于本次提交。
- **未改变**：L1、L2 MUSIC/ID/Kalman、L3 BF数值算法、L4人数分类/两种分离模型/2～4 kHz匹配、L5模型权重与UI布局均无变化。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — 修复L4试听将PCM16 WAV误读为裸float32

- **版本/标签**：当前`1.3.1`开发线Development Test UI试听修复；不创建或移动发布标签。
- **L4试听**：新增专用WAV读取入口，严格校验48 kHz、单声道、PCM16，解析RIFF容器并把PCM16样本正确转换为float32后再交给PortAudio；L4按钮不再调用L3裸`.f32`内存映射入口。
- **故障证据**：旧逻辑把18.76秒L4 WAV误读成9.38秒float32流，约38.4%样本成为NaN，其余可达约`3.4e38`，因此旧UI试听不代表实际L4输出。
- **未改变**：L4模型、候选选择、缓存WAV内容、48→16→48 kHz重采样、L5、L1～L3、正式录音和UI布局均无变化。
- **验证**：新增标准L4 WAV逐样本解码、完整时长和错误采样率拒绝回归；Development Test UI/L4聚焦测试及相关Ruff通过。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — Development Test UI短试听音轨改为结束后彻底删除

- **版本/标签**：当前`1.3.1`开发线Test UI试听缓存修复；不创建或移动发布标签。
- **Development Test UI / L3试听缓存**：继续使用配置中的2秒试听门槛；短音轨在ACTIVE/COASTING期间暂存，轨迹结束、采集停止或模拟输入处理完成后若总时长不足2秒，则删除对应分段文件和UI状态，不再出现“界面隐藏但缓存仍存在”的情况；恰好2秒的音轨保留。
- **L3→离线L4**：Test UI专用长音频封存应用相同门槛，少于2秒的权威ID归档会从内存归档中移除且不提交给L4，避免L4出现左侧已隐藏的0.7秒音频。
- **兼容与未改变**：Center Mic参考不受门槛影响；正式运行时未启用Test UI试听器的TrackAudioStreamHub保持原行为；L1、L2 MUSIC/ID/Kalman、L3波束形成、L4分离模型、L5、正式录音、配置值和资产均无变化。
- **验证**：新增短轨结束删除、恰好2秒保留以及短轨不进入离线L4的聚焦回归；自动测试结果见本次提交记录。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — Development Test UI支持重复提交L3到L4

- **版本/标签**：当前`1.3.1`开发线Test UI离线处理操作修复；不创建或移动发布标签。
- **Development Test UI / L3→L4**：L3封存完成后可多次点击“发送到L4”；每轮提交均先停止当前试听，清空上一轮L4界面结果、临时试听缓存、离线处理对象及待发送L5状态，再按当前选择的L4模型重新处理完整L3长音频。
- **操作状态**：新一轮处理期间禁用提交按钮；处理成功或失败后恢复“发送到L4”，避免首次完成后无法再次运行。
- **未改变**：L1、L2 MUSIC/ID/Kalman、L3波束形成与封存音频、L4模型算法、L5模型、正式录音、配置和资产均无变化。
- **验证**：新增连续两次替换式提交的界面回归，验证每轮均先清空再处理且按钮恢复可用；Development Test UI聚焦测试通过。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — 离线L5改为NVIDIA长音频逐20 ms概率输出

- **版本/标签**：当前`1.3.1`开发线L4→L5接口修复；不创建或移动发布标签。
- **L5 / NVIDIA Frame-VAD**：新增完整48 kHz长音频入口，统一降采样到16 kHz后直接读取NVIDIA MarbleNet原始frame softmax；输出按NVIDIA 20 ms帧索引裁齐为与输入每个960样本hop严格一一对应的概率序列，丢弃`center=true`产生的额外尾部边界帧。
- **L4离线结果与Test UI**：`Layer4OfflineResult`新增逐20 ms概率和布尔判断；L4波形黄色区域逐hop读取真实概率，不再把长音频末尾约80 ms的单一结论覆盖整轨。整轨概览单独使用完整序列的连续3帧最大均值；`offline_l4_job_v2`持久化完整帧序列。
- **兼容与未改变**：保留原整轨`l5_probability/l5_is_voice`作为概览字段；L1、L2、L3、L4分离/2～4 kHz匹配、方向ID和角度、模型权重、录音音频格式及UI布局不变。
- **验证**：新增NVIDIA边界帧裁齐、逐hop阈值、真实官方权重长音频前向和Test UI逐帧着色测试；聚焦回归`123 passed`，全量pytest`497 passed`，Ruff与`git diff --check`通过；真实官方权重样例以32个输入hop得到32个有限输出概率。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — 修正L4双模型候选选择与不可靠输出回退

- **版本/标签**：当前`1.3.1`开发线离线L4质量修复；不创建或移动发布标签。
- **L4候选匹配**：将只看2～4 kHz幅度的余弦分数升级为逐帧复频谱相干度，保留L3方向参考的相位和时序身份，并容忍全局极性翻转；修复MossFormer2和TIGER在相似语音频谱下选择错误匿名候选的问题。
- **安全回退**：不足2秒、最高相干度低于`0.50`或候选分差小于`0.025`时，保留原L3参考音频并记录原因，不再把短轨或身份不明确的分离伪影写入试听缓存和L5。
- **音频库实测**：使用“开始静音 · 0821-0928 · 2个声源”原始8通道录音完整回放L1→L3，并分别运行MossFormer2和TIGER；两模型各3条输出均由同一L5逐帧判为人声，长轨汇总概率为`0.9981～0.9994`，0.72秒短轨回退L3后为`0.8398`。试听WAV保存在本机临时验证目录，不提交音频库或运行数据。
- **未改变**：L1、L2、L3算法和缓存格式、人数判定、模型权重、48/16 kHz重采样、L5模型与手动发送顺序、UI布局、正式录音均无变化。
- **验证**：新增同幅度错相位候选、短轨回退和身份歧义回退覆盖；执行L4聚焦测试、完整测试套件、L4相关Ruff、真实双模型GPU推理、L5人声确认及差异检查。全仓Ruff另检出本次未修改的L2 tracker既有未使用局部变量，不纳入本次L4提交。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — 修复MossFormer2输出异常增益与L4缓存削波

- **版本/标签**：当前`1.3.1`开发线离线L4音频修复；不创建或移动发布标签。
- **离线L4 / MossFormer2**：补回官方推理流程要求的逐候选输入RMS恢复，修复低电平L3波束音频经模型后被异常放大约29–35 dB的问题；不足1秒的输入先补足模型推理窗口，输出再裁回原长度并按有效区间恢复RMS。
- **L4缓存安全**：在48 kHz输出进入UI PCM16缓存和L5之前增加仅衰减的峰值保护，保证超范围模型输出不会被硬削顶；实际衰减系数写入结果元数据供审计。
- **故障证据**：本次现场缓存中L3轨道RMS约`0.0018–0.0030`，旧L4缓存升至`0.051–0.166`，其中轨道1和3已到达`1.0`并发生真实削波；修复后的实际0.72秒MossFormer2样本不再触顶。
- **未改变**：L1、L2、L3波束形成及原始缓存、人数判定、2–4 kHz匹配、L5 CNN、UI布局、正式录音、其他模型资产均无变化；既有已损坏临时L4 WAV不会原地伪修复，需重新发送L3生成。
- **验证**：新增MossFormer2短音频补窗/RMS恢复与PCM16峰值保护回归；执行离线L4聚焦测试、Ruff、真实缓存轨道模型推理及差异检查。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — Development Test UI支持手动选择离线L4模型

- **版本/标签**：当前`1.3.1`开发线Test UI操作增强；不创建或移动发布标签。
- **Development Test UI / 离线L4**：L4区域新增互斥的`MossFormer2`与`TIGER`模型按钮，绿色表示当前选择、灰色表示未选；点击“发送到L4”时显式使用所选后端，处理提示同步显示模型名称。
- **本地设置**：模型选择写入`data/dev_test_ui/settings.json`并在下次启动时恢复；项目`config.yaml`的默认模型配置不被改写。
- **未改变**：L1、L2 MUSIC/ID/Kalman、L3音频及缓存、两种L4模型实现与资产、L5、正式录音和其他UI均无变化。
- **验证**：增加模型设置往返、非法值拒绝、按钮互斥及颜色状态覆盖，并执行Development Test UI聚焦测试、Ruff及差异检查。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — Hub按权威ID唯一封存L3长音频

- **版本/标签**：当前`1.3.1`开发线L3→离线L4衔接修复；不创建或移动发布标签。
- **TrackAudioStreamHub / L3缓存**：封存输出由“同一ID每个连续片段各生成一条输入”改为“每个`(session_id, stream_epoch, track_id)`只生成一条长音频”；同一权威ID的非连续片段按绝对采样时间合并，缺失区间补入严格等长静音并记录方向数0，保持真实时间轴且不伪造音频。
- **离线L4 / Development Test UI**：消除同一权威ID被拆成多个L4结果后触发`L4 UI output track IDs must be unique`的问题；不同ID仍依次逐条处理，原ID和末次角度保持不变。
- **未改变**：L1、L2 MUSIC/ID/Kalman、L3波束形成数值、L4模型/人数上限/匹配、L5、正式录音、其他UI、模型和资产均无变化。
- **验证**：新增同ID跨缺口封存为单条且精确补静音的时间轴回归覆盖；按跨层时间轴变更要求执行完整测试套件，并运行Ruff及差异检查。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — L4人数路由限制瞬时多方向为双人处理

- **版本/标签**：当前`1.3.1`开发线离线L4修复；不创建或移动发布标签。
- **离线L4**：讲话人数由原先直接采用“封存范围内L2方向输出数量最大值”改为`min(2, 最大值)`；历史20 ms窗口即使短暂记录三个方向，也按当前双人L4后端的上限处理，不再使全部L3方向音轨失败。多条封存L3音轨仍由现有离线任务依次逐条完成L4处理。
- **可审计信息**：保留原始最大方向数，并额外记录实际采用的人数及`min(2, maximum)`聚合规则。
- **未改变**：L1、L2 MUSIC/ID/Kalman、L3音频及缓存、L4模型/重采样/匹配、L5、其他UI、正式录音和资产均无变化。
- **验证**：新增瞬时三方向被限制为双人的回归覆盖；离线L4聚焦测试、Ruff及差异检查通过。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — 完整录音模拟回放结束后自动封存L3音轨

- **版本/标签**：当前`1.3.1`开发线Development Test UI修复；不创建或移动发布标签。
- **Development Test UI / Runtime衔接**：完整录音模拟输入到达末尾时，除通用输入耗尽标志外，同时识别录音回放源的`ended`状态，自动停止并排空Runtime、封存L3方向音轨，使已有缓存满足发送到L4的前置条件；修复界面显示“播放完成”但Runtime仍为`RUNNING`、发送按钮始终不可用的问题。
- **缓存与数据**：不删除或重写已有Center Mic及方向音轨缓存；录音回放仍可从头重播。
- **未改变**：L1～L5算法、MUSIC/ID追踪、波束形成、音频内容、正式录音、Production UI、模型和资产均无变化。
- **验证**：增加完整录音回放真实到达EOF后Runtime自动停止的回归覆盖；Development Test UI聚焦测试、Ruff及差异检查通过。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — 启动模拟测试后自动最小化数据管理窗口

- **版本/标签**：当前`1.3.1`开发线Production UI交互优化；不创建或移动发布标签。
- **Production UI**：在测试语料库选中音频并成功启动Development Test UI模拟测试后，当前“麦克风阵列录音与数据管理”窗口自动最小化，让新打开的Test UI直接成为主要操作窗口；样本校验失败或子进程启动失败时不最小化，继续保留错误提示上下文。
- **未改变**：模拟输入仍只读取原始8通道音频且不回放热力图；真实麦克风输入、录音存储、L1～L5算法、Runtime、标签、质量检查、模型和其他UI流程均无变化。
- **验证**：Production UI相关Ruff检查通过；可用性聚焦测试`19 passed in 4.67s`，覆盖成功启动模拟测试后只最小化一次。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — L3/L4操作按钮对齐L1尺寸

- **Development Test UI**：以当前L1“正式录音开始”按钮的Qt `sizeHint`为动态基准，将L3顶部四个操作按钮和L4“发送到L5”由固定`130×32 px`缩至与L1完全相同；字体或DPI变化时自动跟随L1，不再使用独立的大按钮规格。
- **保持不变**：按钮文案、颜色与行为、音频播放按钮、L3/L4/L5算法和发送顺序、其他UI、配置、模型及Git LFS资产无变化。截图中的L4重复ID提示不属于本次尺寸修改范围。
- **验证**：新增与L1按钮`sizeHint`直接相等的尺寸断言；定向UI测试、Ruff和`git diff --check`通过。

---

## 2026-08-21 — 精简L3顶部按钮文案与状态显示

- **Development Test UI**：BF模式按钮删除`BF：`前缀；连续轨按钮由“连续轨响度补偿”简化为“响度补偿”；下游按钮固定只显示`L3/4/5`，不再显示“运行中/已停止”，改由绿色/棕红色背景表达运行状态，并在悬停提示中保留状态说明。
- **保持不变**：按钮尺寸、点击行为、L3/L4/L5处理与发送时序、音频、其他UI、配置、模型和Git LFS资产无变化。
- **验证**：更新四档模式、响度补偿及下游开关的文字/颜色断言；Test UI测试、Ruff和`git diff --check`通过。

---

## 2026-08-21 — 缩小L3/L4顶部操作按钮

- **Development Test UI**：将BF模式、L3/L5运行、连续轨响度补偿、“发送到L4”和“发送到L5”五个按钮的统一尺寸由`160×42 px`进一步缩小为`130×32 px`，减少顶部控制区占用。
- **保持不变**：按钮文案与行为、音频行、L3/L4/L5算法和发送顺序、其他UI、配置、模型及Git LFS资产无变化。
- **验证**：增加明确的`130×32 px`尺寸断言；定向UI测试、Ruff和`git diff --check`通过。

---

## 2026-08-21 — 压缩L3音频行并取消横向滚动条

- **Development Test UI**：L3试听区禁用横向滚动条；每行播放键缩至40 px，ID/角度列缩至128 px并改为紧凑的`ID · 角度`，时长列缩至62 px，其余宽度全部交给可伸缩音频波形条。L4复用相同紧凑音频行，避免后续输出轨重复浪费横向空间。
- **保持不变**：音频内容、ID/角度数值、试听、L3/L4/L5发送与算法、其他UI、配置、模型和Git LFS资产无变化。
- **验证**：新增滚动条策略、紧凑列宽、ID/角度文本和波形最小宽度断言；定向UI测试、Ruff和`git diff --check`通过。

---

## 2026-08-21 — 统一L3/L4发送区按钮尺寸

- **Development Test UI**：将L3栏顶部的BF模式、L3/L5运行、连续轨响度补偿和“发送到L4”四个按钮统一缩为与L4栏“发送到L5”相同的`160×42 px`，并由同一尺寸常量约束，避免窗口拉伸或文案差异造成宽度不一致。
- **保持不变**：各按钮行为、L3/L4/L5算法与发送时序、音频、配置、其他UI、模型和Git LFS资产无变化。
- **验证**：新增五个按钮尺寸一致断言；定向UI测试、Ruff和`git diff --check`通过。

---

## 2026-08-21 — 修复模拟播放完成后无法发送L3到L4

- **Test UI自动排空**：修复模拟音频已播放完成后仍等待`processing_running=false`才调用`Runtime.stop()`的循环等待。阶段worker设计上只有收到stop/EOS后才会排空退出；现在输入到达EOF后立即提交正常stop，由Runtime完成L2/L3队列排空、Hub封存和录音收尾。
- **L4按钮**：Runtime完全停止后重新读取`offline_l4_sources`；存在封存长音频时启用“发送到L4”，因此无需操作者再额外点击停止按钮。
- **保持不变**：L1/L2/L3/L4/L5算法、人数路由、模型、匹配、音频格式、其他UI、配置与Git LFS资产无变化。
- **验证**：模拟WAV EOF自动停止定向测试通过，本次变更路径Ruff与`git diff --check`通过；自动测试不替代长录音GUI实机复核。

---

## 2026-08-21 — Test UI增加L3→L4→L5两步人工发送与三栏试听

- **界面布局**：Development Test UI下半区由L3/L5两栏改为等宽L3、L4、L5三栏；L3新增“发送到L4”，L4新增“发送到L5”。只有停止采集、L3排空且Hub封存完成后才允许第一次发送；只有全部L4轨完成后才允许第二次发送。耗时模型加载和推理继续在UI工作线程外执行。
- **L4试听与身份**：新增L4输出WAV试听缓存和波形栏，逐轨保留原`session/epoch/track_id/theta`，显示处理状态、时长并支持独立播放。L4后端拆出`process_l4_*`与`process_l5_*`公开接口，原一键离线/批处理接口继续兼容。
- **L5显示与颜色规则**：L4发送后L5才运行CNN并立即更新方向概率面板；Voice区间只在对应L4音频预览条显示黄色。取消旧的L5结果染黄L3试听条规则，L3始终只显示自身音频波形。
- **保持不变**：L1、L2 MUSIC/Gate/ID/Kalman、L3波束形成、L4人数判断/重采样/分离/2～4 kHz匹配、L5模型与阈值、RecordingStore格式、Production UI、Pipeline Log UI、模型和Git LFS资产均无变化；不创建或移动发布标签。
- **验证**：全量pytest `487 passed`；L4/Test UI/Runtime聚焦复测`64 passed`，本次变更路径Ruff与`git diff --check`通过。全仓Ruff仍报告上一提交`3324287`中L2 tracker的一个未使用局部变量，本次未越界修改该L2实现。自动测试不替代真实长音频GPU处理、播放设备和人工交互验收。

---

## 2026-08-21 — 抑制现存ID附近的重复方向轨新建

- **L2 ID birth**：Hungarian一对一关联后，任何仍未匹配且位于现存非噪声ID预测位置±20°内的MUSIC峰不再创建新ID，优先保留既有方向轨；角差使用圆周距离，覆盖359°/0°。
- **缺陷修复**：旧逻辑虽然识别到普通ID附近的未匹配峰，却没有把该候选从后续birth列表剔除，因而仍可能在同一声源附近生成重复ID。本次新增明确的birth抑制集合并接入容量/淘汰前的正式新建入口。
- **边界保持**：噪声干扰ID继续为非排他轨，不阻止附近潜在人声建立新ID。200 ms三次确认、2秒TTL、动态关联、Kalman、Gate、MUSIC/DPD/Whitening、L1、L3、L4/L5、UI、录音与模型资产均无其他变化。
- **验证**：配置与L2追踪定向测试`92 passed`；新增0°附近及359°/0°跨界重复峰用例。未完成真实麦克风声场验收，Git LFS资产无变化。

## 2026-08-21 — L2方向ID快速确认、动态重关联与阻尼Kalman预测

- **ID确认与寿命**：tentative轨迹改为滚动200 ms内累计3次一对一MUSIC匹配后进入`confirmed`；正式/临时轨迹的几何coasting TTL统一为最后真实观测后2秒，时间继续严格按48 kHz绝对sample计算。
- **动态关联**：tentative固定使用20°圆周关联范围；confirmed按距离该ID最后真实观测的漏检时长使用`min(50°, 20° + 15°/s × t)`，修复此前误用相邻处理窗口时间导致范围几乎不扩张的问题。Hungarian全局一对一分配、359°/0°处理、同session ID单调不复用均保持不变。
- **Kalman**：用带协方差的二维圆周Kalman替换内部Q/R alpha插值；状态为角度与角速度，最大角速度60°/s，无观测时以0.5秒半衰期衰减角速度并用阻尼积分预测。重新观测时随漏检时长把测量可信度从1倍平滑提高到最多2倍；预测不确定度冻结参数正式接入。关闭Kalman仍保持最后真实角，不公开速度预测。
- **配置与文档**：新增20°基础关联范围和15°/s扩张率配置；同步README、L2说明与1.3.1权威架构。Gate、MUSIC/DPD/Whitening、L1、L3、L4/L5、各UI、录音与模型资产均无算法或接口变化。
- **验证**：定向配置、L2追踪及Runtime测试共114项通过；完整测试`485 passed`。新增200 ms三次确认、按最后真实观测扩张关联范围、0.5秒速度半衰期测试；未完成真实麦克风声场验收。Git LFS资产无变化。

## 2026-08-21 — 完成Hub长音频驱动的离线L4双人分离与L5接线

- **链路改造**：实时链改为`L2→L3→TrackAudioStreamHub`，不再把拼接片段送入CNN；L3排空后Hub按`session/epoch/track_id`封存完整48 kHz单声道长音频、原ID/角度和逐窗L2方向输出数。Runtime预留`offline_l4_sources`与`run_offline_l4()`后端接口，UI页面和控件本次明确无变化。
- **人数路由与重采样**：讲话人数取封存范围内L2方向输出数量最大值；1人和2人均使用L4统一拥有的48→16 kHz多相重采样，L5删除私有重采样实现并复用该组件。1人绕过分离直接进L5；2人进入所选分离后端；大于2人明确拒绝。Hub音频已完成增益补偿，离线L5带禁用补偿诊断，避免二次放大。
- **分离模型与匹配**：加入官方ClearerVoice MossFormer2 SS 16K和TIGER speech的Apache-2.0推理源码快照、严格revision/SHA-256 manifest及权重；二者可配置切换。长音频按30秒、1秒重叠分块，并用重叠相似度修复匿名输出交换。完成512点Hann、160 hop、2～4 kHz参考能量加权幅度谱余弦匹配，含末尾补零、有界批处理和确定性平分规则，获胜音轨继承原ID与角度。
- **离线结果与恢复**：增加同步离线编排、完成session的哈希校验恢复入口、原子WAV/作业manifest输出和批处理脚本；正常Runtime优先直接读取Hub内存封存包，RecordingStore读取仅用于恢复/批处理。模型推理结果记录后端、revision、候选分数、输出哈希和L5判断。
- **测试与文档**：新增人数最大值、单人旁路、双人分离匹配、长块匿名排列稳定、Hub封存、模型契约和匹配边界测试，并把旧实时CNN测试更新为离线L5契约；同步根架构、模块说明、第三方NOTICE和配置。L1采集/预降噪、L2 MUSIC/Gate/ID/Kalman算法、L3波束形成数值算法、RecordingStore既有格式、Development/Production/Log UI及发布标签无变化。
- **Git LFS与验收边界**：新增MossFormer2 `model.pt`与TIGER `model.safetensors` LFS资产；严格加载及两模型短音频有限值冒烟已验证。自动测试不替代真实双人录音、长时GPU吞吐、分离听感和2～4 kHz匹配质量实机验收。

---

## 2026-08-21 — 现有CNN迁移为L5并冻结离线L4双人分离契约

- **命名与实时主链**：将原`layer4_voice_classifier`及其`Layer4*`公共类型、配置、Runtime阶段、队列、状态、Development Test UI、Log UI、脚本和测试统一迁移为`layer5_voice_classifier`及L5命名；当前实时链明确为`L2→L3→TrackAudioStreamHub→L5`。原MarbleNet权重、算法、阈值、响度补偿和推理行为不变。
- **离线L4框架**：新增`layer4_speech_separation`，规定L4只能在采集停止、L3处理排空、按ID长音频完成拼接并封存后由外部编排器调用。输入固定为携带原`session/epoch/track_id/theta`和SHA-256的48 kHz单声道完整20 ms hop长音频；一人决策必须绕过L4，两人决策才允许创建MossFormer2或TIGER请求。模型后端固定接收16 kHz音频并返回恰好两条匿名、等长、finite float32候选。
- **2～4 kHz选择标准**：新增`l3_bf_2_4khz_magnitude_cosine_v1`，对同一重采样后的原L3 BF参考和两个分离候选使用512点Hann STFT、160点hop、2～4 kHz逐帧幅度谱余弦相似度并按参考频带能量加权；整段得分较高者继承原ID和角度，另一候选不作为正式输出。分数相同固定选择候选0；首版记录两分数和差值但不设拒绝阈值，禁止每20 ms切换讲话人。
- **实现边界**：本次只搭框架、设标准并规定输入输出；未下载MossFormer2/TIGER，未实现讲话人数分类器、48→16 kHz公共重采样器、长音频分段/排列稳定、离线任务队列、结果存储或L4/L5离线编排，也未改变现有实时CNN运行方式。
- **文档、打包与测试**：更新根架构、环境、模块说明、脚本及打包发现范围；新增L4输入封存/两人准入、双候选严格性、2～4 kHz选择和ID/角度继承测试。L1、L2 MUSIC/Gate/ID/Kalman、L3 BF数值算法、麦克风采集、录音音频格式、模型/Git LFS资产及各UI交互行为无变化。
- **验证**：L4/L5/配置/数据管理/Log UI/Runtime/打包聚焦回归`143 passed`；全量pytest最终为`473 passed, 2 failed`，失败分别是未改动的L2 DPD 15 ms性能门限瞬时波动，以及Runtime并发屏障时序波动，两项随后同批隔离复跑均通过；Ruff与`git diff --check`通过。自动测试不替代真实双人录音、GPU分离质量和长音频验收。

---

## 2026-08-21 — 冻结v1.2.4并开始1.3.1开发线

- **版本/标签**：项目`1.2.4`最终版继续固定在提交`8bb2a7e`及不可变标签`v1.2.4`，不移动、不覆盖；项目包版本和当前状态文档从本提交开始更新为`1.3.1`，尚未创建`v1.3.1`标签。
- **分支策略**：从`v1.2.4`新建`codex/develop-v1.3.1`，后续修改进入`1.3.1`开发线；`main`和`v1.2.4`保持最终发布基线，不追加本次开发线初始化提交。
- **涉及文件**：更新项目版本元数据、根README、总架构、Log UI架构、项目文件分类及L1～L4、Runtime、Windowing、Development Test UI、Production UI、Log UI和数据管理模块README中的当前开发版本说明。
- **保持不变**：L1～L4算法、配置参数、Runtime行为、各UI功能、录音与数据格式、测试、模型和公开接口均无变化；不修改或新增Git LFS资产。
- **验证**：检查全部当前版本入口、Git差异、工作区状态、远端分支与标签指向；版本元数据与文档调整执行全量自动测试、Ruff和`git diff --check`。

---

## 2026-08-21 — 项目1.2.4整合发布

- **版本/标签与分支**：将`codex/develop-v1.2.4`自`v1.2.3`以来的17个开发提交整合为项目`1.2.4`发布，快进合入`main`并创建新的不可变标签`v1.2.4`；既有版本标签和远程分支保持原位，不移动、不覆盖、不删除。
- **发布内容**：纳入L2 DPD平面波门限、强近邻峰融合和方向簇门限调整；连续方向音轨新增逐20 ms L4人声语义；Development Test UI新增MUSIC-only/ID追踪、L3/L4旁路、热插拔麦克风识别、回放性能与布局交互改进；同步架构、模块文档、Runtime/录音数据契约及相关自动测试。
- **接口与兼容性**：项目包版本保持`1.2.4`，L2公开版本保持`1.1`；既有录音读取兼容路径、正式Runtime默认ID追踪、L1/L3核心接口及模型加载边界保持兼容。自动测试通过不替代真实阵列、GPU推理、诊室声场、声卡试听和长时间运行验收。
- **保持不变**：未修改或新增模型、音频及其他Git LFS资产；未纳入本地录音、运行数据、缓存、日志、密钥或代理设置；不发布尚未完成的实机验收结论。
- **涉及文件**：发布记录更新`CHANGELOG.md`，已发布基线说明更新`README.md`；具体代码、配置、测试和文档文件逐项记录在本条之后的1.2.4开发日志及对应提交中。
- **验证**：发布提交前完整pytest `470 passed`，全仓Ruff检查和`git diff --check`通过；发布文档修改后再次执行差异、仓库状态、敏感数据暂存边界和Git LFS状态检查。

---

## 2026-08-21 — L2 DPD方向簇门限放宽

- **L2 DPD参数**：将方向簇最小圆周集中度从`0.95`降至`0.85`、最小支持权重比例从`0.25`降至`0.20`、聚类角度容差从`10°`增至`15°`、最少支持频点从`5`降至`4`，提高移动声源和较弱第二声源通过逐频投票聚类的机会。
- **保持不变**：DPD特征值比`1.50`、平面波匹配门限`0.40`、至少2个子带、40°峰融合、50°圆周NMS、Gate、Whitening实现、MUSIC/ID/Kalman、L1、L3、L4、Runtime接口、Development Test UI交互和录音格式均无变化。
- **验证范围**：执行配置与L2 MUSIC定向自动测试，并使用同一份本地`−60°→60°`移动声源加`180°`静止声源录音进行DPD开启回放统计；不改动或提交录音与运行缓存。Git LFS资产无变化。

## 2026-08-21 — 连续方向音轨增加逐20 ms L4人声语义与黄色显示

- **L4/Runtime公共契约**：新增不可变`TrackVoiceAnnotation`，把每个成功L4检测按完整`(WindowKey, track_id)`严格绑定到其连续输入最新20 ms hop，保存绝对sample范围、概率、Voice/Non-Voice、模型和运行时阈值；ID、顺序、角度或音频区间不一致时拒绝关联。移除Runtime向L2提交语义正式化/续租反馈的调用，L4仅为已有方向轨增加语义。
- **连续音轨与录音**：Development Test UI的按ID分段缓存为每个20 ms音频位置保留对应L4结果或明确的无结果状态。DecisionRecord的连续hop元数据携带同一语义，RecordingStore合成长WAV时在manifest资产中按绝对sample顺序保存`voice_results_20ms`；失败、丢弃和缺失结果不伪造为Non-Voice。
- **Development Test UI**：方向波形在当前L4 UI阈值下把Voice区间底色显示为黄色；Non-Voice、无结果和失败区间保留原默认底色。阈值滑块只使用已缓存概率实时重绘，不重新运行L3或CNN。试听波形和CNN仍使用同一份连续响度补偿音频。
- **文档与测试**：同步总架构、根README、L4、Test UI和数据管理说明；增加精确ID/sample回填、错误ID拒绝、运行时最新hop映射、长WAV逐20 ms语义及动态阈值着色测试。
- **保持不变**：L1、L2 MUSIC/Gate/ID/Kalman几何生命周期、L3波束形成和连续音频样本、NVIDIA模型/权重/48→16 kHz推理、响度补偿、primary/shadow、正式分类阈值边界、Production UI布局和模型/Git LFS资产均无变化；不创建或移动发布标签。
- **验证**：Ruff全仓检查、Git差异检查及全量pytest `470 passed`；自动测试不替代真实阵列、GPU推理、声卡试听和长时间运行验收。

---

## 2026-08-21 — Development Test UI重复模拟播放后可发送到L4

- **Development Test UI**：修复模拟输入“从头重播”后沿用上一轮EOF停止标记的问题；每轮重播现在都会独立执行播放结束后的停止、处理队列排空和TrackAudioStreamHub封存。自动停止完成后立即进入与手动停止一致的UI状态，使存在封存L3长音频时“发送到L4”按钮正确启用。
- **保持不变**：L1、L2、L3波束形成、离线L4/L5算法、录音格式和数据管理契约均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：补充重播时EOF停止标记复位断言，并执行Development Test UI聚焦测试与静态检查；自动测试不替代实机音频验收。

---

## 2026-08-21 — Development Test UI L3标题按钮尺寸统一

- **Development Test UI**：L3标题行的“BF模式”“L3/L4运行/停止”和“连续轨响度补偿”三个按钮统一宽度与高度；尺寸根据当前字体和最长BF模式名称计算，避免不同显示缩放比例下文字截断或按钮大小不一致。
- **保持不变**：L1、L2、L3/L4算法、Runtime处理逻辑、录音和数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：补充三个L3标题按钮尺寸一致性断言并通过Ruff与Python语法检查；聚焦pytest受共享工作区中未完成的`app/runtime.py`语法错误阻断，自动检查不替代实机界面尺寸验收。

---

## 2026-08-21 — Development Test UI连续轨响度补偿控件移位

- **Development Test UI**：将“连续轨响度补偿”开关从L4面板顶部移至L3标题控制行，位于“L3/L4”运行/停止开关右侧；关闭状态使用灰色，开启状态使用绿色。原有持久化设置、Runtime响度补偿开关和下一完整20 ms生效语义保持不变。
- **保持不变**：L1、L2、L3/L4音频与推理算法、录音和数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：增加控件归属、L4旧位置移除、灰/绿状态色、Runtime联动及设置持久化的Development Test UI聚焦测试；自动测试不替代实机界面尺寸验收。

---

## 2026-08-21 — L2 DPD强近邻峰圆周融合

- **DPD候选处理**：在方向簇门禁之后、50°圆周NMS之前加入高峰融合。仅当峰组内每个峰的归一化值严格大于`0.70`且任意两峰圆周距离不超过`40°`时，才使用支持频点可靠性权重计算圆周平均`theta_group`；多峰采用组直径约束，禁止相邻峰链式跨越40°合并并正确处理359°/0°。
- **融合证据**：`w_merge`按成员方向簇支持频点的唯一并集求和，重复频点只计一次；融合后重新计算支持频点数、支持率、子带覆盖、圆周集中度和平均平面波拟合度，并重新通过原DPD门禁。360点蓝色投票谱不改变、不二次归一化，融合角的公开Raw/Norm继续取原谱对应1°网格值。
- **配置与兼容性**：新增`dpd_peak_fusion_distance_deg=40.0`与`dpd_peak_fusion_min_normalized_score=0.70`，算法版本更新为`frequency_normalized_music_dpd_peak_fusion_v7`。DPD关闭路径、UI开关、MUSIC阶数上限、ID/Kalman及L3/L4接口均保持不变。
- **保持不变**：L1、Gate、Whitening本身、各UI布局、录音与数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：增加唯一频点去重、严格`>0.70`、多峰防链式融合、359°/0°圆周融合及配置契约测试；L2 MUSIC/配置/Runtime聚焦回归`111 passed`，Development Test UI相邻契约`30 passed`，相关Python文件Ruff检查通过。自动测试不替代真实双声源角度融合标定。

---

## 2026-08-21 — L2 DPD平面波匹配门限调整

- **L2 DPD**：将可靠频点准入的最小平面波匹配度由`0.45`调整为`0.40`，提高较弱第二声源频点进入方向投票的机会；特征值比、频点支持比例、子带覆盖、集中度和50°圆周间距均保持不变。
- **保持不变**：L1、Probability Gate、MUSIC/Whitening计算、ID追踪、Kalman、L3、L4、各UI、录音与数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：项目配置加载测试`35 passed`；自动测试不替代双声源检出率和单声源额外候选率的实机标定。

---

## 2026-08-21 — Development Test UI默认最大化启动

- **Development Test UI**：默认启动由普通窗口改为系统最大化窗口，保留标题栏、最小化、还原及关闭操作，不使用无边框全屏；显式`start_fullscreen`配置和F11全屏切换仍保持有效。
- **验证**：增加窗口最大化状态契约检查，并运行Development Test UI定向测试与Ruff检查。
- **保持不变**：L1～L4算法、Runtime调度、队列、ID追踪、试听缓存、录音、配置schema、模型和音频资产均无变化；不创建或移动发布标签，无Git LFS资产变化。

---

## 2026-08-21 — Development Test UI新增MUSIC-only与ID追踪切换

- **版本/标签**：当前`1.2.4`开发线界面与L2诊断能力；不创建或移动发布标签。
- **Development Test UI**：将“MUSIC阶数上限”收紧为仅容纳标签和1/2/3数值的紧凑控件，并在右侧新增持久化`ID Tracking`按钮；绿色表示开启，灰色表示关闭。
- **L2与显示契约**：追踪开启时保持原有全局权威ID、稳定颜色和观测/预测角显示。追踪关闭时仍计算Gate、360点MUSIC伪谱及原始峰值，只在圆环对应角度绘制灰色小点，不生成或显示权威ID；切换边界在单一L2 worker内重置轨迹及旧L4反馈，重新开启后从新的ID状态开始。
- **下游隔离**：MUSIC-only模式不把无ID峰值送入L3/L4，每个窗口以`direction_id_tracking_disabled_by_test_ui`正常`SKIPPED`收束，不记录为处理错误；已有试听缓存仍可使用。
- **设置与兼容性**：Test UI设置schema升级并保存ID追踪开关，缺少该字段的旧设置默认迁移为开启；正式/默认Runtime仍开启ID追踪，项目配置没有新增enable字段。
- **验证**：增加MUSIC-only原始峰值、重启追踪ID、Runtime revision、设置持久化及紧凑布局测试；L2 MUSIC/ID、Runtime和Development Test UI聚焦回归`103 passed`，完整测试`463 passed`，Ruff与`git diff --check`通过。自动测试不替代真实阵列界面验收。
- **保持不变**：L1输入/IMCRA/录音、MUSIC伪谱算法和峰值门限、ID追踪开启时的关联/Kalman规则、L3波束形成、L4模型、试听缓存格式、正式录音、模型与音频资产均无变化；无Git LFS资产变化。

---

## 2026-08-21 — 存储音频模拟测试停用热力图回放

- **版本/标签**：当前`1.2.4`开发线性能优化；不创建或移动发布标签。
- **类型**：测试语料回放输入边界、Development Test UI、Production UI提示、测试与文档。
- **模拟输入**：从测试语料库启动模拟测试时，`RecordingReplaySource`只校验并读取`native_8ch`原始音频，不再打开、校验、解析或逐块注入录制的`cdc_hotmaps`；送入Runtime和Test UI的`hotmap`固定为空，以减少文件解析、矩阵构造和界面更新开销。
- **兼容性**：只有`native_8ch`资产的已登记录音现在也可模拟测试；录音时仍按既有规范保存CDC热力图，既有热力图资产不删除、不改写，可继续用于归档和其他离线用途。
- **未改变**：真实麦克风模式继续接收并显示实时CDC热力图；L1～L4音频算法、绝对sample时间轴、录音格式、质量检查、标签和模型均无变化。
- **验证**：相关Python文件Ruff检查通过；回放、Development Test UI和Production UI聚焦测试首轮`53 passed`，唯一失败为关闭界面时本机COM5串口退出竞态；该项与本次音频-only回放用例随后合并复测`7 passed`。
- **Git LFS资产**：无变化。

---

## 2026-08-21 — L1 Spectrum UI支持麦克风热插拔与无黑框最大化启动

- **版本/标签**：项目版本与发布标签无变化。
- **类型**：独立L1 Spectrum UI设备生命周期、启动体验、界面状态、测试与文档。
- **L1 Spectrum UI**：删除手动“连接麦克风/停止采集”控件，改为不可操作的连接状态按键；未连接为红色“未连接”，成功打开并采集后为绿色“已连接”。
- **自动发现**：程序启动后即使麦克风缺失也保持运行，每1秒重新创建UAC输入并扫描配置设备；启动失败不发送UI错误信号、不弹出独立报错窗口。采集中连续2秒没有音频或输入异常时释放旧pipeline、回到未连接状态并继续扫描。
- **启动显示**：主窗口默认最大化；PowerShell入口改为启动独立隐藏`pythonw`进程，桌面快捷方式直接指向`pythonw.exe`，避免GUI运行期间保留黑色控制台窗口。
- **灯控**：仍只在麦克风成功连接后尽力发送一次默认关灯命令；手动灯光命令及其明确错误弹窗保持不变。
- **L1算法与数据**：校准、IMCRA、预降噪、逻辑通道、连续性和频谱计算无变化；Windowing、L2、L3、L4、Runtime、录音、数据管理、Production UI及其他Development Test UI无变化。
- **测试与验收**：增加缺失设备后自动重试、无连接错误弹窗信号、红绿连接状态、隐藏启动入口和默认最大化契约测试；L1 Spectrum UI、L1 meter和输入链聚焦验证`47 passed`，相关Ruff、Python编译、PowerShell语法及`git diff --check`通过。尚未执行真实USB拔插实机验收。
- **资产**：无模型、音频及Git LFS资产变化；桌面快捷方式是本机启动入口，不进入Git。

---

## 2026-08-21 — 放大Development Test UI的DOA圆环并迁移状态

- **Development Test UI**：移除左侧极坐标图内的`DOA / MUSIC 360°`标题，把`MDL / MUSIC / valid / status`状态移到右侧Gate概率条正下方；圆环不再为标题和底部状态预留空间，在不裁切角度标记及方向点的前提下放大并居中。
- **验证**：Ruff检查和Development Test UI渲染/布局定向测试通过。
- **保持不变**：L1、L2 MUSIC/ID算法及状态内容、L3、L4、Runtime调度、音频缓存、录音、配置、模型资产和发布标签均无变化；无Git LFS资产变化。

## 2026-08-21 — 合并Development Test UI的处理开关布局

- **Development Test UI**：将右上区域的三个处理开关合并到同一行，等宽各占三分之一；按钮分别简化命名为`Kalman`、`DPD`、`Whitening`，仅通过绿色/灰色表示开启/关闭状态，切换待生效时沿用琥珀色。开关功能、提示和持久化逻辑保持不变。
- **验证**：运行Development Test UI布局定向测试与Ruff检查。
- **保持不变**：L1、L2 MUSIC/ID算法、L3、L4、Runtime调度、音频缓存、录音、配置语义、模型资产与发布标签均无变化。

## 2026-08-21 — 修正模拟输入与L3/L4旁路期间的总处理计时

- **Development Test UI**：手动暂停模拟音频输入时，L2/L3/L4总处理时长同步冻结；继续播放后从原累计值继续，不计入暂停等待时间。
- **下游旁路**：手动关闭`L3/L4`开关期间只累计L2总处理时长，L3/L4停在关闭前的累计值；重新开启后恢复累计，并正确处理“模拟暂停”和“下游关闭”同时存在的重叠暂停。
- **Runtime计时**：总处理计时器新增按阶段、按原因的排除区间；排空终点落在暂停区间内时仍能得到稳定终值，不改变正式窗口、队列、DOA/ID、L3/L4算法或录音数据。
- **验证**：Ruff检查通过；计时暂停/重叠暂停/恢复定向测试通过；完整测试`460 passed, 2 failed`，其中L3数值项单独复跑通过，另一项为模拟界面关闭时访问物理`COM5`的既有串口清理失败，与本次计时改动无关。
- **保持不变**：L1输入、L2 DOA/MUSIC与ID追踪、L3波束形成、L4推理、试听缓存和正式录音均无变化；无Git LFS资产或发布标签变化。

## 2026-08-21 — Development Test UI新增L3/L4下游隔离开关

- **界面**：删除L3顶部仅用于单窗的“播放/暂停”和“停止”按钮，在原位置新增`L3/L4：运行中/已停止`开关；按ID长音频试听按钮和已有缓存保持可用。
- **Runtime**：关闭开关后L1/L2继续运行，新L2结果不再进入L3队列；已排队但未开始的L3/L4窗口快速收束为`downstream_disabled_by_test_ui`的正常`SKIPPED`终态，正在计算的单窗安全完成。重新开启后从下一条L2结果恢复，不破坏ResultJoiner、DecisionRecord和watermark顺序，也不记录为处理错误。
- **诊断**：公开处理状态新增`downstream_processing_enabled`，顶部L3/L4状态在隔离期间显示`OFF`；L3和L4画面明确显示由Test UI停止，L2 DOA/MUSIC仍持续刷新。
- **验证**：Ruff检查通过；新增的UI开关、Runtime旁路和DecisionRecord跳过契约共`6 passed`；完整测试覆盖`456 passed`，其中一项本机L2性能阈值测试首次受调度抖动影响、单独复测`1 passed`，串口退出竞态项单独复测`1 passed`。
- **保持不变**：L1、L2 MUSIC/Gate/ID/Kalman算法、L3波束形成和L4模型实现、按ID试听缓存格式、正式录音及除上述跳过终态兼容外的数据管理流程均无变化；不创建或移动发布标签，无Git LFS资产变化。

## 2026-08-21 — 修复coasting方向进入L3时Development Test UI停止

- **根因**：L2已允许有效TTL内的正式`coasting` ID在没有当前MUSIC响应时按预测角继续进入L3 BF，但`DevUiFrame`仍强制要求所有L3预览必须附带MUSIC响应；首个prediction-only预览因此触发契约异常并停止处理线程。
- **Runtime与Test UI契约**：无MUSIC响应时，Runtime现在同步传递该窗的权威`directions/active_tracks`。Test UI以`(session_id, stream_epoch, window_id, decision_sample)`和`track_id`校验prediction-only L3/L4结果，只接受`confirmed/coasting`正式ID，拒绝跨窗、缺ID、换序或tentative音频。
- **保持不变**：L1、MUSIC计算、Gate判决、L2 ID/Kalman、L3 BF与拼接算法、L4模型、录音和数据管理、UI布局均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：Development Test UI、L2 MUSIC/ID与并行Runtime聚焦回归`104 passed`；全量pytest回归`457 passed`。自动测试不替代真实阵列长时间试听验收。

## 2026-08-21 — 归档DOA追踪文献与6+1阵列空间可分离度图

- **版本/标签**：当前`1.2.4`开发主线参考资料补充；不创建或移动发布标签。
- **类型**：重要研究文献、阵列研究图、Git LFS规则及参考资料索引更新。
- **新增资料**：在`docs/references/`新增两份DOA短时失联与ID连续性研究PDF，以及一张100～6000 Hz、0～180°的6+1阵列空间可分离度`rho`图；仓库采用简短稳定文件名，索引保留完整中文题名、原图文件名、用途和非规范性边界。
- **Git LFS**：两份PDF继续匹配既有`docs/references/*.pdf`规则；新增`docs/references/*.png`规则，使空间可分离度图也由Git LFS管理。桌面源文件仅复制、不移动、不删除，本地原件保持不变。
- **未改变**：L1、L2、L3、L4、Application Runtime、Development Test UI、Pipeline Log UI、Production UI、录音与数据管理、公共DTO、配置、模型、测试音频和算法实现均无变化。
- **验证**：核对三份源文件与仓库副本SHA-256完全一致；两份PDF可正常解析且分别为23页和18页；空间图可正常读取；检查Git LFS跟踪状态、Git差异和远端推送结果。

---

## 2026-08-21 — 冻结v1.2.3并开始1.2.4开发线

- **版本/标签**：当前最终整合提交`bf660a4`发布为新的不可变标签`v1.2.3`；既有`v1.2.2`继续固定在原提交，不移动、不覆盖。项目包版本和当前状态文档从本提交开始更新为`1.2.4`，尚未创建`v1.2.4`标签。
- **分支策略**：后续项目提交进入`1.2.4`开发线；`v1.2.3`仅作为最终只读基线，不再追加或改写。
- **未改变**：L1～L4算法、配置参数、Runtime、各UI、录音和数据管理、模型、测试资产及公开接口均无变化。
- **验证**：检查版本入口、当前状态文档、Git差异与远端标签指向；文档和版本元数据调整不运行pytest。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 修复Loaded MVDR模拟输入无方向预览音频

- **L3根因与修复**：合并后的`loaded_mvdr_baseline`仍调用已被性能优化移除的旧`_mvdr`
  内部求解器，每个有方向的窗口因`NameError`直接进入`L3 failed`，因而Test UI只有
  Center Mic对照而没有方向预览。该基线现与当前优化路径一致，批量对所有loading重试
  执行Cholesky分解/求解，选择首个数值有效权重，其余频点仍回退DAS。
- **模拟输入二次修复**：同一录音回放还暴露了Loaded MVDR诊断文字引用已移除的
  `CONTEXT_HOPS`固定160 ms常量，数值求解成功后仍因`NameError`丢弃整窗输出。现改为读取
  `prepared.stft.window_hops`，与40/80/160 ms可配置窗口一致。
- **验证边界**：新增双方向Loaded MVDR无失真约束、finite输出及批量求解回归；
  按用户要求本次未运行自动测试套件。使用报错的同一录音执5秒短回放，L3完成83窗、
  L4完成82窗，L2/L3/L4/commit错误计数均为0；修改文件静态格式和差异检查通过。
  `optimized`、`ds_baseline`、`subband_robust_baseline`、L1、L2、L4、Runtime时间线、录音、
  数据管理、UI交互和二进制资产均无变化。未创建或移动发布标签，无Git LFS变化。

## 2026-08-20 — 全部分支统一合入main

- **分支整合**：将`codex/integrate-all-branches-v1.2.1`、`main`发布历史及尚未合入的`codex/l3-loaded-mvdr-baseline`统一为同一提交历史；其余本地功能分支已是该整合历史的祖先。
- **合并结果**：保留当前L2正式coasting方向持续进入L3的契约，并纳入全频`loaded_mvdr_baseline`第四档L3对照模式及相应Runtime、Development Test UI和文档改动。
- **保持不变**：不移动或重写`v1.2.2`及既有标签，不删除分支，不修改模型、音频或其他Git LFS资产，不纳入本地运行数据、录音、缓存、日志或密钥。
- **验证**：按用户要求仅完成合并，本次合并后未运行自动测试；各原提交中的历史测试记录保持原样，不能视为本次整合后的重新验证。

## 2026-08-20 — L2正式coasting ID持续进入L3波束形成

- **L2→L3方向契约**：解除“必须先获得L4人声确认才能发布coasting方向”的门槛。所有仍在有效绝对sample TTL内的正式`coasting` ID，都按L2权威状态参与最多3路、50°最小角距的`directions`选择，并以原`track_id`和保持/预测角继续进入L3 BF，减少Development Test UI试听缓存中因漏检造成的等时静音段。
- **L4边界**：L4人声反馈只通过既有追踪反馈机制决定是否续租预测ID生命，不再作为coasting进入L3的准入条件，也不参与L3方向槽排序。实测confirmed方向仍优先，coasting再按漏检时长、score和ID稳定排序。
- **保持不变**：L1、MUSIC伪谱与候选生成、ID关联/Kalman、L3波束形成算法本身、L4模型、试听拼接与缓存文件格式、各UI布局、录音与数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：L2 MUSIC/ID追踪与Runtime跨层定向测试`76 passed`；全量pytest回归`454 passed`。自动测试不替代真实漏检声源的长时间试听验收。

## 2026-08-20 — L2 Kalman关闭时的短时静止方向稳定

- **L2 ID追踪**：仅在Kalman关闭时，对confirmed ID维护最近3秒圆周观测历史；至少70%观测位于圆周均值±10°时进入短时静止，公开角度和关联锚点改用持续更新的圆周均值，正确覆盖359°/0°边界。
- **异常观测退出**：短时静止期间，滚动1秒内第1～3个超出均值±20°的观测不会移动ID位置或公开角度；第4个外点立即解除静止并跟随当前观测。正常范围观测不会清空外点计数，超过1秒的旧外点自动过期。
- **兼容边界**：Kalman开启时清除并旁路短时静止私有状态，不改变ID、Gate、MUSIC、L3/L4接口或3秒TTL；配置新增历史长度、比例、角度范围、外点窗口和退出次数字段。
- **保持不变**：L1、Probability Gate、MUSIC候选、匈牙利分配、L3波束形成、L4分类、各UI、录音与数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：全量pytest为`454 passed`；相关配置、L2追踪、Runtime和Development Test UI定向测试为`137 passed`，相关Python文件Ruff检查通过。自动测试不替代真实静止声源长时间实机标定。

## 2026-08-20 — L2短时静止判定角度范围调整

- **L2 ID追踪**：Kalman关闭时，confirmed ID进入短时静止状态所需的3秒历史圆周角度范围由均值±10°调整为均值±15°，70%占比要求保持不变。
- **保持不变**：静止状态的±20°异常观测范围、滚动1秒内第4个外点退出规则、359°/0°圆周处理、Gate、MUSIC、Kalman开启路径、L1/L3/L4、各UI、录音与数据管理均无变化；不创建或移动发布标签，无Git LFS资产变化。
- **验证**：配置与L2静止追踪相关的聚焦测试`38 passed`；自动测试不替代真实静止声源实机标定。

## 2026-08-20 — 消除L3连续方向音频的20 ms周期拼接毛刺

- **TrackAudioStreamHub**：利用相邻L3窗口天然重叠的下一段20 ms估计，在每个新hop开头2 ms执行`cos²/sin²`等功率过渡；拼接点先延续上一窗口的同一BF解，再平滑切换到当前窗口，消除每960样本固定出现的阶跃和弱50 Hz电流/嗡声。统一后的波形仍同时供L4、Development Test UI试听和正式增强音频记录使用，不创建UI专用副本。
- **保持不变**：L1、L2、L3波束形成正式窗口与算法、40/80/160 ms统一配置、方向ID、L4模型与阈值、Runtime调度、录音结构和各UI布局均无变化；仅改变连续方向音频的跨窗接缝样本。
- **验证与资产**：增加带窗口偏置的确定性拼接回归，验证2 ms重叠过渡、其余hop逐样本不变和边界连续性；无模型、音频或Git LFS资产变化。当前自动验证不能替代用户声卡实际试听确认。

---

## 2026-08-20 — 模拟输入分层总处理时长计时

- **Development Test UI**：模拟WAV与完整录音回放模式在底部上一秒性能信息后新增“总处理时长”，分别显示L2、L3、L4从首个20 ms窗口开始入队到该层处理完最后一个输入并排空的累计时间，显示精度为0.01秒；处理中实时更新，完成后冻结最终值。真实麦克风采集界面不显示该组计时。
- **Runtime**：新增线程安全的单次运行分层总时长快照；普通模拟WAV随正常EOS停表，交互式完整录音回放在播放结束时向L2→L3→L4依次传递有序计时屏障，使各层处理完屏障前的全部内容后立即停表，而不要求可重播Runtime线程退出。强制取消或异常退出不伪报“已完成”。既有单窗阶段耗时、队列、丢窗统计和调度逻辑不变。
- **保持不变**：L1采集与IMCRA、L2 MUSIC/ID/Kalman算法、L3波束形成及试听拼接、L4推理、录音与数据管理均无变化；不创建或移动发布标签。
- **测试与资产**：补充分层计时正常排空和模拟UI显示覆盖；无模型、音频或其他Git LFS资产变化。

## 2026-08-20 — L3统一下游音频窗口调整为40 ms

- **统一配置与接口**：`timing.downstream_audio_window_ms`新增40 ms合法档并将当前全局值改为40 ms；统一派生为48 kHz `1920`样本、2个20 ms hop、5帧STFT及16 kHz `640`样本。原80/160 ms档继续保留为可选兼容配置。
- **L3与Runtime**：L3从160 ms `DecisionWindow`末尾读取40 ms音频和两个对齐IMCRA hop，每个方向输出`float32[1920]`；Runtime、滚动STFT、波束形成批次和Test UI单窗试听共同读取同一全局规格，不新增局部窗口常量。
- **L4与连续轨**：`TrackAudioStreamHub`仍从每个重叠L3窗口只追加一个20 ms hop；最长3200 ms连续轨和L4“最新80 ms连续3帧”分类聚合规则不变。
- **保持不变**：L1采集、160 ms `DecisionWindow`、L2 MUSIC/Gate/ID/Kalman、L3波束形成数学算法、L4模型、录音与数据管理均无变化；不创建或移动发布标签。
- **测试与资产**：补充40 ms配置、L3输出和Test UI派生规格覆盖；全量pytest为`450 passed`，相关Python文件Ruff检查通过；无模型、音频或其他Git LFS资产变化。

## 2026-08-20 — 项目1.2.2整合发布

- **版本/标签**：项目包版本更新为`1.2.2`，创建新的不可变标签`v1.2.2`；所有既有版本标签、远程分支和历史保持原位，不移动、不覆盖、不删除。
- **发布基线**：以全部本地和GitHub已提交分支合并后的`c6ba7a3`为功能基线，纳入该提交以前全部代码、配置、文档、测试、模型、Git LFS研究资料和精选资产。
- **L1**：IMCRA/预降噪频率轴扩展至10 kHz；新增独立L1频谱观察器、设备灯光控制及连接成功后自动关灯；采集、8通道映射、校准和唯一时间轴职责不变。
- **L2**：Rolling NormMUSIC按手动阶数搜索候选，圆周候选最小间隔调整为50°；DPD改为按频率投票聚类；确认门限调整为6次观测；Gate hold要求既有L4语音证据；公共ID、Kalman和最多3方向契约保持不变。
- **L3**：移除旧恒定波束宽度实验实现，加入稳健子带波束形成基线；160 ms输入输出、公共track ID、滚动STFT/噪声统计缓存、Loaded MVDR/DAS等正式接口保持一致。
- **L4与连续音频**：统一下游音频窗口配置；按公共track ID连续流式生成补偿音频，L2正式coasting方向可维持L3试听连续性；CNN模型与人声概率输出契约不变。
- **Runtime与UI**：各阶段有界队列扩展到2000窗口；Development Test UI显示轨迹最后角度并缩短试听交叉淡化；Production UI支持语料录音重命名并明确表格选择状态；Pipeline Log UI继续保持只读观察边界。
- **录音与数据管理**：RecordingStore、Catalog、恢复事务、正式录音资产和Production UI完整随版本发布；新增连续track音频资产与语料命名维护。运行录音、scratch、Catalog和本地data目录仍不进入Git。
- **文档/研究资产**：架构、模块README、文件分类和Log UI契约统一更新为项目`1.2.2`；L3双声源和实时波束形成研究PDF继续由Git LFS管理。
- **测试**：发布前干净`main`全量pytest为`446 passed`，核心源码与测试Ruff全部通过，全项目Python `compileall`通过。自动测试不替代真实阵列、诊室多声源和长时间运行验收。
- **未改变**：Layer 2公开版本仍为`1.1`；采样率48 kHz、8通道输入、20 ms决策节拍、160 ms L3/L4音频窗口、唯一WindowKey、DecisionRecord v4和旧数据只读兼容策略不变。
- **Git LFS与安全边界**：发布前检查LFS对象与工作树；不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、partial、密钥、Token或本地代理设置。

## 2026-08-20 — coasting BF双重L4确认与滚动续命

- **版本/标签**：当前开发分支行为收紧；不创建或移动发布标签。
- **L2/L3契约**：正式方向ID必须在两个不同的L4决策窗口中获得正向人声判定，之后MUSIC漏检进入coasting时，才可强制Gate开启并继续把保持/预测角送入L3做BF。同一`decision_sample`重复反馈只计一次。
- **coasting续命**：基础到期时间仍为最后一次MUSIC观测后3秒；coasting BF窗口再次获得L4正向人声判定时，到期时间滚动更新为该窗口后3秒，后续有效判定可继续滚动续命。
- **保持不变**：6次/200 ms tracking确认、匈牙利关联、Kalman、L3 BF算法、L4模型与反馈格式、L1、Runtime调度、各UI、录音和数据管理均无变化。
- **验证**：覆盖一次反馈、重复同窗反馈、两个不同窗口反馈、Gate hold和coasting续命；Git LFS资产无变化。

---

## 2026-08-20 — L1 IMCRA与预降噪频率范围扩展到10 kHz

- **版本/标签**：项目`1.2.1` L1公开频率轴调整；不创建或移动发布标签。
- **类型**：IMCRA公开输出、IMCRA Wiener预降噪、录音旁路资产及相关配置契约更新。
- **频率范围**：IMCRA公开`noise_psd`、`signal_psd`、`snr_db`和`speech_presence_probability`统一覆盖名义`0～10000 Hz`。在48 kHz采样率、2048点FFT下实际共有427个非负频点，最高频点为9984.375 Hz。
- **预降噪**：IMCRA Wiener增益应用范围同步扩展到`0～10000 Hz`，10 kHz以上频率继续原样透传；IMCRA及预降噪算法版本分别更新为`cohen_imcra_2003_l1_v3`和`imcra_wiener_wola_v3`。
- **录音与界面**：新录音中的IMCRA NPZ旁路资产、清单频率轴和L1频谱界面读取同步采用427频点；既有录音文件不迁移、不改写，继续保留其原版本和原始频率轴。
- **兼容性**：新的实时`ImcraHopSnapshot`严格要求v3/427频点契约；旧v2/342频点快照不能混入新实时流水线。L1 Gate仍只使用500～4000 Hz，L2 MUSIC仍使用2000～4000 Hz，L3/L4的80～8000 Hz处理范围均无变化。
- **未改变**：采样率、20 ms权威时间轴、通道映射、L2～L4算法本体、Runtime调度、Production UI交互、语料标签、数据集划分、模型及二进制资产无变化。
- **验证**：相关Python文件Ruff检查通过；完整自动测试`446 passed in 44.18s`。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — Test UI 与 L1 Spectrum UI 连接麦克风后默认关灯

- **版本/标签**：项目`1.2.1`L1设备启动行为调整；不创建或移动发布标签。
- **类型**：Development Test UI与独立L1 Spectrum UI的麦克风/CDC启动顺序统一。
- **启动行为**：两套界面每次成功连接UAC麦克风后才发送一次官方关灯命令`e`。Development Test UI的手动连接和模拟输入自动启动统一经过同一入口；L1 Spectrum UI每次启动或重新连接采集均执行相同行为。
- **静默边界**：麦克风连接失败时不访问CDC、不发送灯控命令，也不产生额外灯控错误；成功连接后的默认关灯是尽力执行，CDC缺失或写入失败时保持Unknown且不弹窗。用户手动点击“灯光开/灯光关”仍保留正式错误提示。
- **未改变**：麦克风连接失败本身仍由原界面状态报告；L1音频/IMCRA/预降噪算法、L2～L4、Runtime处理、录音/数据管理、频谱计算、模型和二进制资产无变化。
- **验证**：Development Test UI与L1 Spectrum UI聚焦测试`38 passed`，相关Ruff检查通过；自动测试未向真实硬件发送灯控命令。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 独立 L1 Spectrum UI 增加灯光控制

- **版本/标签**：项目`1.2.1`独立L1界面设备控制补充；不创建或移动发布标签。
- **类型**：左上角设备控制与串口生命周期完善。
- **L1 Spectrum UI**：左上第一行新增“灯光开”“灯光关”和命令状态；复用现有CDC串口配置及正式`led_command`协议，开/关分别发送`E`/`e`，检测并报告串口异常或不完整写入。串口命令在独立后台线程执行，不阻塞麦克风采集和20 ms频谱刷新；关闭界面时同步释放灯控串口。
- **运行边界**：UAC输入链仍只运行校准、Ingest、IMCRA、可选预降噪和L1显示；CDC仅在灯光命令首次发送时按需打开，不创建L2、L3、L4、录音或Hotmap消费者。
- **未改变**：L1 IMCRA/预降噪算法和公开DTO、Development Test UI、Production UI、Pipeline Log UI、L2～L4、Runtime调度、录音/数据管理、模型和二进制资产无变化。
- **验证**：L1 Spectrum UI、L1 meter及Runtime灯控相邻测试`45 passed`，相关Ruff和Python编译检查通过；未在自动测试中向真实硬件发送灯控命令。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 新增独立 L1 输入与 IMCRA 频谱观察界面

- **版本/标签**：项目`1.2.1`独立诊断界面新增；不创建或移动发布标签。
- **类型**：L1只读观察工具、实时频谱可视化与启动入口。
- **涉及文件**：新增`gui/l1_spectrum_ui/`、`scripts/launch_l1_spectrum_ui.ps1`和聚焦测试，并更新根`README.md`与本日志。
- **独立 L1 Spectrum UI**：启动后自动连接配置中的UAC麦克风，只创建校准、Ingest、IMCRA、可选IMCRA预降噪、L1电平和频谱分析，不创建WindowAssembler、L2、L3、L4、正式录音或数据管理服务。界面颜色和四象限布局沿用Development Test UI风格。
- **左上**：复用八路L1 20 ms RMS电平、IMCRA状态和预降噪开关；按项目真实6+1逻辑映射提供`MIC0`～`MIC5`、`Center`、`Mix`互斥选择，默认`Center`。这里没有重复创建一个虚假的`MIC6`：逻辑通道6本身就是Center。
- **右上/右下**：对所选通道每20 ms执行一次2048点FFT，以0～10 kHz柱状dBFS频谱刷新；“抓拍到右下”复制并冻结当前频谱及session/epoch/sample/sequence标识，后续实时帧不覆盖该抓拍。
- **左下**：直接显示正式`ImcraHopSnapshot.noise_psd`换算后的当前噪声频谱折线，并列出同一物理麦的noise、signal、SNR和SPP；硬件`Mix`不属于IMCRA七路物理麦估计，选择时明确显示不可用而不伪造数据。
- **未改变**：现有Development Test UI、Production UI、Pipeline Log UI、L1算法与公开DTO、L2～L4、Runtime调度、录音/数据管理、配置schema、模型和二进制资产均无变化。
- **验证**：新增UI、L1 meter、IMCRA、Ingest和输入链聚焦测试`63 passed`，相关新增文件Ruff与Python编译检查通过；完成1500×900离屏四象限渲染检查。尚未在本次自动流程中占用真实麦克风做实机验收。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — Runtime统一阶段队列扩容至2000窗

- **版本/标签**：项目`1.2.1` Runtime容量调整；不创建或移动发布标签。
- **类型**：L2/L3/L4流水线等待容量配置调整。
- **Runtime**：统一`runtime.stage_queue_windows`及schema默认值从1000改为2000，L2、L3、L4三个单worker等待队列同步扩容至2000窗；自动派生的`max_inflight_windows`由3003变为6003。按50窗/秒计算，每层最大等待跨度由约20秒增至约40秒。
- **权衡**：扩容可吸收更长的暂时性处理抖动并推迟latest-wins丢窗，但不会提高实际处理吞吐；持续过载时仍会积累更高延迟和内存占用，队列满后继续替换最旧未开始窗口并记录丢窗。
- **未改变**：L1采集与IMCRA、L2 MUSIC/ID/Kalman算法、L3/L4算法、UI交互、正式录音、数据schema和模型资产无变化。
- **验证**：配置/容量聚焦测试及完整自动测试；未进行长时间实机负载验收。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — Development Test UI试听音轨补充末次角度

- **版本/标签**：项目`1.2.1` Development Test UI显示调整；不创建或移动发布标签。
- **类型**：L3试听列表信息展示优化。
- **Development Test UI**：左下角方向试听音轨在权威ID序号后同步显示该ID最后一次输出的角度，格式为`ID  角度°`；沿用L2权威ID的稳定颜色。Center Mic对照、时长、波形、播放控制和缓存排序保持不变。
- **未改变**：L1、L2跟踪与MUSIC算法、L3音频生成和拼接、L4、Runtime调度、录音、缓存生命周期及模型资产无变化。
- **验证**：Development Test UI聚焦测试及静态检查通过；未进行实机音频验收。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — L4人声确认ID在coasting TTL内优先保持L3音频

- **版本/标签**：项目`1.2.1` L2→L3连续性修复；不创建或移动发布标签。
- **类型**：已人声确认方向ID的L3 BF槽位优先级与coasting连续性修复。
- **L2/L3契约**：L4已确认为人声的confirmed ID在MUSIC短时漏检后进入coasting时，只要仍处于最后真实观测起算的3秒几何TTL内，就优先占用最多3个L3方向槽位，并按保持/预测角每20 ms继续生成BF音频。普通临时MUSIC峰不得先占与该人声ID冲突的50°槽位，避免试听缓存因单窗漏检生成空hop。
- **未改变**：Gate概率、MUSIC谱与候选算法、ID关联与3秒删除TTL、Kalman、L4人声阈值、L3 BF算法、UI与录音格式均无变化。
- **验证**：L2 MUSIC/ID/Gate与Runtime v1.1聚焦测试`52 passed`，连续轨音频与Test UI音频ID缓存测试`21 passed`，相关Ruff检查通过。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 普通MUSIC改为手动阶数驱动的逐峰搜索

- **版本/标签**：项目`1.2.1` L2候选搜索修改；不创建或移动发布标签。
- **类型**：普通frequency-normalized MUSIC子空间阶数与多峰搜索语义调整。
- **L2**：Test UI手动MUSIC阶数上限1/2/3现在直接决定实际信号子空间阶数和候选搜索上限，MDL只作诊断。普通路径每轮选择符合当前Test UI候选门限和prominence的最强圆周局部峰，再屏蔽与已选峰距离小于50°的区域，直到达到手动上限或无达标峰；恰好50°仍允许共存。峰仅作为L2观测备选，ID与L4人声判断规则不变。
- **算法版本**：`frequency_normalized_music_greedy_peaks_v6`。
- **未改变**：Gate、候选门限滑动条及持久化、DPD路径、IMCRA白化、ID/Kalman、L3、L4、Runtime调度、录音和模型资产无变化。
- **验证**：L2 MUSIC/配置聚焦测试`78 passed`，L2 Gate/Runtime v1.1相邻契约测试`7 passed`。对30°/210°、20.2秒双声源缓存录音以0.20门限和2阶上限只读回放：1003窗中210°附近候选由旧逻辑的165窗增至635窗，30°与210°同时命中601窗。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — L2候选圆周最小间距调整为50°

- **版本/标签**：项目`1.2.1` L2参数调整；不创建或移动发布标签。
- **类型**：L2 MUSIC候选圆周NMS与公共方向间距参数调整。
- **涉及文件**：`config/config.yaml`、`common/config.py`、`layer2_source_detection/configuration.py`、`layer2_source_detection/pipeline.py`、L2相关文档与对应测试期望。
- **L2**：`min_peak_distance_deg`从45°调整为50°；普通NormMUSIC与可选DPD路径均通过该配置执行50°圆周NMS，公共方向及coasting补点同步执行两两至少50°。恰好50°允许共存，小于50°时抑制低优先级候选。
- **未改变**：L1、Gate概率、MUSIC 2～4 kHz与阶数选择、ID的45°关联门限与噪声语义邻域、Kalman、L3、L4、Runtime调度、UI交互和录音数据均无变化。
- **验证**：按用户明确要求未运行自动测试；仅检查最终差异和Git状态。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 测试语料选中样式不再遮挡名称

- **版本/标签**：项目`1.2.1`Production UI视觉修复；不创建或移动发布标签。
- **类型**：测试语料库表格选中态可读性修复。
- **涉及文件**：`gui/production_ui/app.py`、`tests/test_production_ui_usability.py`。
- 测试语料库选中音频改为浅蓝底和深色文字；移除文字区域内由系统绘制的白色焦点框，改用贴合名称单元格外缘的2 px蓝色边框，并为边框保留内边距，避免覆盖长名称。
- **未改变**：数据文件、标签、manifest、Catalog、录制流程、L1～L4算法、Runtime、Development Test UI、Pipeline Log UI和其他页面表格均无变化。
- **验证**：Production UI可用性聚焦测试`19 passed`，相关文件Ruff通过；离屏渲染确认浅蓝选中区和外缘蓝色边框生效、文字区域无白色焦点框。本次不修改本地语料。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 测试语料库支持手动修改所选音频名称

- **版本/标签**：项目`1.2.1`维护增强；不创建或移动发布标签，既有发布标签保持不变。
- **类型**：Production UI语料操作、标签一致性与审计功能。
- **涉及文件**：`data_management/{corpus_naming,service}.py`、`gui/production_ui/{app.py,README.md}`及对应测试。
- 测试语料库新增“修改所选名称”：对话框预填当前名称，保存后保持该行选中并立即显示新名称；取消不写入，空名称、控制字符和超过300字符的名称会被拒绝。
- 手动改名同步更新`labels.json`的`recording_name`、labels资产SHA-256、`recording_manifest.json`及sidecar、Catalog投影和文件/Catalog审计记录；Recording UUID、目录、PCM、热力图、绝对sample轴和其他结构化标签不变。锁定数据集或实验快照禁止原地改名。
- **未改变**：L1～L4算法、Windowing、Application Runtime、Development Test UI、Pipeline Log UI、录制流程、QA与数据集划分均无变化。
- **验证**：语料命名/改名与Production UI可用性聚焦测试`22 passed`，全量自动测试`433 passed`，相关文件Ruff通过；本次不修改现有本地语料名称。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 总架构图同步当前连续逐ID音频主链

- **版本/标签**：项目`1.2.1`文档维护；未创建或移动发布标签。
- **类型**：仅文档架构盘点与已实现状态校准。
- **涉及文件**：根`README.md`总架构图、算法说明和本`CHANGELOG.md`。

### 架构图与说明

- 按当前代码补充`TrackAudioStreamHub`：它在L3 worker内同步执行，不是独立Layer 3.5，也没有自己的等待队列；按
  `(session_id, stream_epoch, track_id)`从重叠L3窗口抽取不重复的20 ms hop，并以同一补偿后样本驱动
  Development Test UI试听、RecordingStore逐ID长WAV与L4 CNN。
- 将L4更新为连续逐ID音频输入：最长3200 ms的48 kHz上下文降采样到16 kHz，模型产生连续20 ms帧概率，
  当前窗口只聚合最新80 ms内连续3帧；响度补偿位置、`-23 dBFS`目标及`-3 dBFS`新增增益保护与代码一致。
- 将L3第三档更新为已接入的`subband_robust_baseline`五频段鲁棒对照，并明确旧
  `constant_beamwidth_baseline`已经移除和拒绝；保留其自由场steering仅为首版RTF代理、尚未完成在线RTF学习的限制。
- 按当前L2实现补充DPD逐频投票与圆周核聚类、滚动200 ms内至少6次匹配观测确认轨迹，以及只有已有L4人声证据且
  非噪声干扰的confirmed轨迹才可在低Gate概率时强制放行。
- 更新录音与运行关系：重叠L3窗不重复形成正式音频资产，20 ms hop按chunk/track合成长WAV；同窗顺序为
  `L2 → L3 → TrackAudioStreamHub → L4`，跨窗仍为有界单worker流水。

### 未变化组件、验证与资产

- L1、WindowAssembler、L2/L3/L4实现、Runtime调度、TrackAudioStreamHub实现、ResultJoiner、全部UI、
  Recording/Data Management、Production UI、配置、模型、测试、音频及空间表资产均无变化。
- 本次不声称真实7通道阵列、诊室声场、中文语音、五频段模式全链吞吐或长时间运行已经重新验收。
- README代码块、本地链接、冲突标记和`git diff --check`静态检查通过；L2/L3、连续音频枢纽、L4及
  Runtime文档契约专项自动测试`56 passed`。
- 未修改Git LFS管理的模型、音频、空间表或其他二进制资产，无Git LFS对象变化；未提交本地数据、录音、缓存、日志或密钥。

---
## 2026-08-20 — 新增全频Loaded MVDR可切换基线

- **版本/标签**：项目`1.1.0`并行迁移分支；未创建或移动发布标签。
- **类型**：L3实验基线、Test UI模式切换、文档与自动测试。
- **涉及文件**：`layer3_direction_signal/{adaptive_separation,hybrid,interface,prepared}.py`、
  `common/data_types.py`、Development Test UI模式显示/试听分区、根/L3 README、1.1架构文档及相关测试。

### L3与Test UI

- 在现有`optimized`、`ds_baseline`和`subband_robust_baseline`之外新增第四档
  `loaded_mvdr_baseline`。它对每个L2权威方向独立处理，在80～8000 Hz统一使用IMCRA噪声协方差、
  噪声置信度、混叠保护和重试loading求解diagonal-loaded MVDR；不查询空间`p`表，也不叠加
  IMCRA频点后滤波，从而保持纯Loaded MVDR对照含义。
- 单频求解病态或非有限时逐频回退DAS；同窗IMCRA不可用时整窗回退DAS。0～3方向、WindowKey、
  track_id、rank、角度、原顺序、160 ms/7680点输出及入口/出口严格对齐规则均不变。
- Test UI按钮与试听缓存增加独立Loaded MVDR分区，支持启动前和运行中四档循环切换；模式切换不改变
  L2权威ID。

### 未变化组件、验证与资产

- 原有三种L3算法的计算和参数无变化；L1、Windowing、L2、L4、Runtime调度/时间线、Recording、
  Data Management、Production UI、空间`p`表、模型与音频资产均无变化。
- 全量自动测试：`361 passed`；修改Python文件Ruff检查和`git diff --check`通过。
- CPU/CUDA双方向冒烟均输出finite；同窗热运行分别约`2.3 ms`和`3.2～4.4 ms`，仅用于本次
  实现检查，不作为正式跨窗口性能基线。无Git LFS资产变化。

## 2026-08-20 — 五频段鲁棒对照替换30°恒定波束宽度模式

- **版本/标签**：项目`1.2.1`集成；未创建或移动发布标签。
- **类型**：L3实验算法替换、Test UI模式切换、配置、文档与自动测试。
- **涉及文件**：`layer3_direction_signal/{subband_robust,hybrid,interface,configuration,prepared}.py`、
  `common/{config,data_types}.py`、`config/config.yaml`、Development Test UI模式显示/试听分区、
  L3/根README、1.1架构文档及相关测试；旧`constant_beamwidth.py`从Git工作树移除。

### L3

- 保留正式默认`optimized`与纯`ds_baseline`的算法、参数和输出不变；第三档由
  `constant_beamwidth_baseline`替换为`subband_robust_baseline`，旧模式字符串现在被明确拒绝。
- 新模式使用同一160 ms滚动STFT与同窗IMCRA噪声协方差，但不查询空间`p`表。80～500 Hz采用
  温和干扰感知loaded MVDR与声源专属Wiener增益；500～900 Hz、900 Hz～1.5 kHz、
  1.5～4 kHz采用逐步放宽WNG下限的LCMV/DAS连续混合；4～8 kHz采用防混叠加载MVDR。
- 第一版以当前自由场steering作为RTF代理，并从当前多通道混合协方差减去IMCRA噪声协方差后，
  对已知方向拟合非负rank-1声源SCM。所有数值不安全频点仍逐频回退DAS；IMCRA整窗不可用时
  整窗回退DAS。该限制写入运行诊断和文档，未冒充已经完成在线RTF学习。
- 0～3个公开方向、WindowKey、track_id、rank、角度、候选顺序、48 kHz 3840/7680点
  输出和L3入口/出口严格对齐校验均无变化。

### Development Test UI与其余模块

- Test UI第三个按钮和试听缓存分区改为“五频段鲁棒对照”；启动前及运行中三档循环切换规则不变，
  切换不会修改L2权威ID。
- L1、Windowing、L2、L4模型与输入、Runtime调度/时间线、Recording/Data Management、
  Production UI、空间`p`表、音频/模型/测试资产均无变化。

### 验证与资产

- L3、缓存、Runtime、Test UI、ID试听、配置、阶段契约与文档专项：`141 passed, 1 deselected`；
  被排除项是与BF无关且受latest-wins采样时序影响的既有UI预热断言，随后单独复跑`1 passed`。
- 全量自动测试最终复跑：`356 passed`。首次全量运行中的一个Recording异步落盘超时已单独复跑通过，
  随后的完整全量运行无失败。
- 修改Python文件Ruff检查、`git diff --check`通过；CPU/CUDA热运行的双方向五频段模式输出finite。
- 未修改Git LFS管理的音频、模型、空间表或其他二进制资产，无Git LFS对象变化。

---

## 2026-08-20 — L3/L4之间新增按ID连续补偿音频主链

- **版本/标签**：项目`1.2.1`连续Frame-VAD架构增强；不创建或移动发布标签。
- **类型**：Runtime公共音频轨、L4输入契约、NVIDIA连续序列推理、Test UI试听/开关、录音资产、架构图、文档和测试。
- **L1/L2/L3**：L1 IMCRA算法、L2 MUSIC/方向ID和L3波束形成数学算法无变化。L3仍输出当前80 ms重叠增强窗；新增`TrackAudioStreamHub`严格按`(session_id, stream_epoch, track_id)`从每窗抽取一个与IMCRA概率网格对齐的20 ms hop，避免重叠重复并维持绝对sample连续性。
- **连续轨与响度补偿**：拼接后立即执行`imcra_probability_rms_v1`，目标`-23 dBFS`、概率分段权重和`-3 dBFS`新增增益保护保持不变。Test UI开关默认ON且本地持久化，可在不中断ID、不清空上下文的情况下实时切换，增益从下一20 ms平滑过渡。试听、正式按ID轨和CNN逐样本使用同一补偿后音频。
- **L4/NVIDIA**：`Layer4AudioSegment`接受由完整20 ms hop组成的可变长度连续48 kHz轨，并记录有效sample范围及既有补偿诊断。NVIDIA Frame-VAD适配器对最长3200 ms连续轨执行48→16 kHz polyphase重采样并输出连续帧概率；窗口标量仅聚合最新80 ms内连续3帧，较早语音只作卷积上下文。primary/shadow仍读取同一不可变批次，阈值重判仍不重跑模型或改变ID。
- **Development Test UI**：正式长轨不再由GUI私有逻辑从L3窗二次形成；Runtime在L3完成后把公共补偿hop送入现有分段播放缓存，播放端取消额外响度归一化。L4面板新增“连续轨响度补偿”实时开关。旧`AudioIdTracker.update`仅保留兼容测试边界，正式Runtime使用`consume_stream_batch`。
- **录音/数据管理/Production UI**：重叠L3原始窗只作瞬时计算输入，不再作为正式音频资产重复保存；DecisionRecord接收每轨新增的补偿20 ms音频，RecordingStore按chunk和公共`track_id`合并为长WAV（时间缺口补等时静音），Production UI与数据接口继续按ID回放。Pipeline Log UI只读接口无控制逻辑变化。
- **Runtime/配置/架构图**：新增`layer4.continuous_context_ms=3200`和`nvidia_marblenet_continuous_v2`后端标识；总架构图增加`TrackAudioStreamHub`及Test UI/Recording/L4三路消费者。WindowKey、阶段队列、ResultJoiner、L2几何生命周期和L4精确ID反馈语义无变化。
- **验证**：新增按ID隔离、连续20 ms时间轴、缺口恢复/重置、实时开关不断轨、Test UI缓存与CNN逐样本一致、可变长度连续MarbleNet、按ID长WAV录音契约及项目模型库真实20 ms人声音频模拟；合并五频段L3分支后Ruff与全量`430 passed`。自动化验证不替代真实声卡播放、真实7通道阵列、房间声场和长时间GPU验收。
- **Git LFS与数据边界**：模型权重及其他Git LFS二进制无变化，仅更新文本manifest；不提交`.venv/`、`data/`、运行录音、临时播放缓存、日志、密钥或代理设置。

---

## 2026-08-20 — L3、L4与Development Test UI统一下游音频窗口为80 ms

- **版本/标签**：项目`1.2.1`跨层配置与契约修复；不创建或移动发布标签。
- **类型**：统一下游音频窗口配置、派生尺寸、Runtime注入、模型适配、UI显示、文档和自动化门禁。
- **统一配置与Windowing**：新增唯一参数`timing.downstream_audio_window_ms`，第一阶段只允许80/160 ms且当前设为80 ms；统一派生48 kHz样本数、20 ms hop数、STFT帧数和16 kHz模型样本数。`DecisionWindow [7680,8]`及20 ms发布节拍保持不变，继续作为160 ms上游容器。
- **L1/L2**：L1采集、IMCRA、预降噪、通道/校准契约均无算法变化；L2 Gate、240 ms Rolling MUSIC、DPD、方向ID、Kalman和20 ms调度均无变化。L3只读取DecisionWindow末尾对应的4/8个IMCRA hop。
- **L3**：从固定160 ms改为按统一规格截取末尾80/160 ms；STFT、滚动缓存、IMCRA上下文、波束形成批次、特征形状和ISTFT输出全部由规格派生。当前输出为48 kHz `float32[3840]`、9帧内部STFT，160 ms配置仍产生`float32[7680]`和17帧。
- **L4**：公开波形、响度补偿段数、批次宽度和MarbleNet适配器接受并严格校验统一规格；当前4个20 ms补偿段、48 kHz 3840样本降采样为16 kHz 1280样本，160 ms配置对应8段、7680和2560样本。模型manifest声明两档适配长度，权重和阈值不变。
- **Development Test UI、Runtime与数据系统**：Runtime向L3、L4和Test UI注入同一规格并只传递末尾4/8个概率；单窗试听波形、按钮文字和按ID恢复范围随规格变化。正式录音流程、Audio Data Manager、Production UI和Pipeline Log UI无功能变化；数据契约仅扩展为接受两档正式增强音频。
- **测试与文档**：新增80/160配置派生、L3末尾截取、两档STFT/缓存、L4批次和正式MarbleNet 80 ms前向、Test UI显示及跨层Runtime契约覆盖；同步根README、架构、Windowing、L3、L4和Test UI说明。自动化验证不替代真实7通道映射、方位、试听和长时间实机验收。
- **Git LFS与数据边界**：MarbleNet manifest文本更新，模型权重和其他Git LFS对象无变化；不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、密钥或代理设置。

---

## 2026-08-20 — Test UI累计试听交叉淡化缩短为2 ms

- **版本/标签**：项目`1.2.1`Development Test UI试听微调；不创建或移动发布标签。
- **类型**：仅调整Test UI按ID累计试听的窗口拼接淡化时长。
- **Development Test UI**：相邻且绝对时间对齐的L3波束形成hop，其`cos²/sin²`交叉淡化由10 ms（480 samples）缩短为2 ms（96 samples），减少相邻两窗不同BF估计被长时间混合的范围；轨道开始、结束和静音缺口边界的5 ms（240 samples）淡入淡出保持不变。
- **L1/L2/L3/L4与数据系统**：L1、L2、L3正式增强波形、L4输入与判断、Runtime调度、Production UI、Pipeline Log UI、录音/数据管理、模型、配置和资产均无变化。
- **验证**：运行Development Test UI音频ID跟踪定向测试，覆盖2 ms交叉淡化、5 ms边界淡化、20 ms绝对时间轴拼接、缺窗恢复、静音补洞、ID隔离和缓存生命周期；执行Git差异静态检查。
- **Git LFS与数据边界**：无Git LFS资产变化；不提交`.venv/`、`data/`、录音、scratch、Catalog、日志、缓存、密钥或代理设置。

---

## 2026-08-20 — 可选DPD路径升级为逐频投票与圆周聚类

- **版本/标签**：项目`1.2.1`L2实验算法增强；不创建或移动发布标签，开关默认值保持关闭。
- **类型**：L2 MUSIC候选生成、配置、诊断、Test UI说明和权威文档更新。
- **L2**：保留既有`DPD + rank-1 MUSIC`运行时开关及持久化语义；开启后，由通过主特征值比、平面波拟合及IMCRA SPP/先验SNR可靠性检查的频点分别产生rank-1 MUSIC方向票，再执行359°/0°连续的圆周核聚类。每个方向簇新增至少5个支持频点、4个等宽子带中至少2个覆盖、加权支持率至少0.25、圆周集中度至少0.95四项门禁，并继续执行方向门限、45°NMS和手动1/2/3候选上限；合格簇数量取代MDL成为DPD路径的0～3候选数，MDL仅保留诊断。普通MUSIC OFF路径逐值逻辑不变。
- **诊断/Runtime**：算法版本升级为`frequency_normalized_music_dpd_cluster_v5`；逐候选记录支持频点数、支持率、子带数、圆周集中度、平均平面波拟合度和簇权重，DecisionRecord诊断同步持久化。无可靠频点与无合格方向簇分别报告`dpd_no_reliable_bins`和`dpd_no_qualified_clusters`。
- **配置/UI**：新增DPD绝对频点数、子带数量/覆盖数和圆周集中度配置；Test UI按钮位置、默认OFF、原子持久化和运行时切换不变，仅更新提示文字以明确圆周聚类。
- **缓存回放**：对22.76秒单移动人声缓存以DPD开启、阶数上限1离线重放；113个IMCRA预热窗之外，917窗形成合格圆周簇、101窗因簇证据不足拒绝。该结果仅证明链路和真实缓存可运行，不构成参数已完成多房间/多声源验收。
- **ID/Gate/Kalman及其他层**：200 ms内6次ID确认、L4人声资格、Gate概率/门限、Kalman、L1 IMCRA、L3、L4、Runtime调度、各UI布局、录音/数据管理、模型和资产均无其他算法或接口变化。
- **验证**：配置、L2 MUSIC/跟踪、Runtime、Runtime v1.1契约、Development Test UI和并行配置/文档定向测试共`126 passed`；包括单源、双源、0°边界、窄带簇拒绝、DPD+白化20 ms性能门禁和开关持久化。
- **Git LFS与数据边界**：无Git LFS资产变化；不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、密钥或代理设置。

---

## 2026-08-20 — L2方向轨确认门限提高到200 ms内6次匹配

- **版本/标签**：项目`1.2.1`参数校准；不创建或移动发布标签。
- **类型**：L2 ID生命周期参数与权威文档更新。
- **L2**：将全局方向轨从tentative进入confirmed的正式门限由滚动200 ms内2次匹配提高到6次；窗口长度、20 ms更新周期、45°关联门限、匈牙利分配、3秒TTL、L4人声反馈资格、Gate/MUSIC、逐帧归一化和Kalman均不改变。匹配不要求连续占满全部窗口，未达到6次的轨迹保持tentative并在后续滚动窗口重试。
- **依据**：单移动人声缓存回放中，主要轨迹ID 1/2/4/6在首次200 ms内分别达到6/11/11/11次，ID 7在后续稳定窗达到11次；短暂错误ID 3/8最多2/3次，跳峰ID 5最多5次。持续风扇仍可能达到6次以上，继续由L4人声资格限制其Gate强制放行与公共coasting。
- **L1/L3/L4与其他系统**：L1、L3、L4算法与反馈接口、Runtime调度、各UI、录音/数据管理、模型和资产均无变化；L3/L4只会更晚收到达到tracking-confirmed的短轨迹。
- **验证**：`tests/test_config.py`、`tests/test_l2_music_tracking.py`与`tests/test_runtime_v11_contracts.py`定向测试共`74 passed`；不等同于真实阵列声场验收。
- **Git LFS与数据边界**：无Git LFS资产变化；不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、密钥或代理设置。

---

## 2026-08-20 — 收紧L2 Gate强制放行与coasting发布资格

- **版本/标签**：项目`1.2.1`缺陷修复；不创建或移动发布标签。
- **类型**：L2方向ID、概率Gate联动和L3方向发布规则修复。
- **L2**：只有tracking状态已为`confirmed`、至少收到一次L4正向人声反馈且当前未标记为噪声干扰的ID，才能在正式概率低于门限时强制Gate开启；未经L4人声确认的轨迹失去当前观测后仍可在内部3秒TTL中等待重关联，但不再作为公共coasting方向送入L3。其有当前观测时仍可按既有规则进入L3/L4接受分类。MUSIC、MDL、逐帧归一化、候选门限、Gate概率计算、ID关联、Kalman和3秒几何TTL均未改变。
- **L1/L3/L4与其他系统**：L1采集/IMCRA/预降噪、L3波束形成、L4分类器和反馈格式、Runtime调度、Development Test UI、Pipeline Log UI、Production UI、RecordingStore、Audio Data Manager、模型与资产均无算法或接口变化；L3只会少收到未经人声确认的漏检coasting目标。
- **文档与测试**：同步根README、L2 README和`ARCHITECTURE_V1.1_TARGET.md`；更新L2跟踪测试，覆盖“仅tracking-confirmed不能强制Gate”和“收到L4人声反馈后允许强制Gate及coasting发布”。
- **验证**：`tests/test_l2_music_tracking.py`与`tests/test_runtime_v11_contracts.py`定向测试共`46 passed`；不等同于真实阵列声场验收。
- **Git LFS与数据边界**：无Git LFS资产变化；不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、密钥或代理设置。

---

## 2026-08-20 — 收录L3双声源分离与Python实时优化研究报告

- **版本/标签**：项目`1.2.1`研究资料维护；不创建或移动发布标签，既有`v1.2.1`保持不变。
- **类型**：重要非规范性研究参考资料归档。
- **涉及文件**：`docs/references/README.md`、两份PDF研究报告、根目录`README.md`、`.gitattributes`和本日志。
- **研究资料**：收录“以Python为主的两声源波束形成分离与实时优化研究报告”和“4 cm间距6+1麦克风阵列双固定声源分离：针对L3波束形成的研究结论与优化方案”，覆盖Python批量数值优化、DOA-conditioned Mask-MVDR、track-specific RTF、speaker-specific SCM、WNG约束鲁棒BF、分频处理、低频后滤波、实验矩阵和验收指标。
- **权威边界**：两份报告是研究综述与实施建议，其中部分描述基于旧320 ms上下文；当前项目已经统一为160 ms L3/L4直接音频窗口。报告不得覆盖代码、`config/config.yaml`、`ARCHITECTURE_V1.1_TARGET.md`及发布文档的现行契约。
- **L1/L2/L3/L4与界面/数据系统**：算法代码、Runtime、Windowing、Development Test UI、Pipeline Log UI、Production UI、RecordingStore、Audio Data Manager、配置、模型、测试和运行数据均无变化。
- **验证**：复制前后两份PDF逐文件SHA-256一致；PDF可重新打开，页数分别为36页和26页；Git差异、LFS跟踪、链接与冲突标记进行静态检查。文档归档不构成报告方案已经实现或完成真实阵列/诊室验收。
- **Git LFS与数据边界**：新增`docs/references/*.pdf` LFS规则并将两份报告作为LFS资产提交；不提交`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、临时渲染、密钥或代理设置。

---

## 2026-08-20 — 改动前主线统一：合并全部本地与GitHub功能分支

- **版本/标签**：以项目`1.2.1`为合并基线，不创建新版本、不移动或覆盖既有`v1.2.1`标签；保留`codex/backup-before-major-v1.2.1`作为大改前回退点。
- **类型**：跨分支整合、160 ms公共音频契约迁移、Development Test UI历史合并与大改前云端封版。
- **合并范围**：将`feature/l2-music-tracking-v1.1`、`feature/l3-public-id-v1.1`和`feature/dev-test-ui-v1.1`的未合并提交完整纳入统一主线；其他远程功能分支此前已经是`main`祖先。本次不删除、不改名任何远程分支或标签。
- **L1**：采集、8通道映射、校准、IMCRA、预降噪和正式输入接口无算法变化；仅保留并验证Test UI按sample加权显示当前epoch历史预降噪增益。
- **L2**：合并Kalman关闭时的零阶角度保持，coasting阶段固定在最后观测角而不继续预测，ID和生命周期不变。单个DecisionWindow改为160 ms后，160/240/320 ms MUSIC上下文通过L2有界滚动帧状态跨窗口累计；Gate、MUSIC频带、最多3个公共方向和45°约束不变。
- **L3**：直接输入从320 ms缩短为160 ms，即`float32[7680,8]`与8个IMCRA hop；滚动STFT由33帧改为17帧，连续20 ms窗口复用13帧并重算4帧；每方向输出改为48 kHz `float32[7680]`。波束形成算法、频带、候选上限、缓存硬边界与回退策略不变。
- **L4**：每方向输入改为160 ms `float32[7680]`，对应8个20 ms补偿概率；内部仍按既有流程降采样进入CNN。模型权重、分类器、输出概率和公共track ID契约不变。
- **Windowing/Runtime/公共契约**：`DecisionWindow`固定为`[7680,8]`，每20 ms发布，epoch首个正式endpoint为7680；唯一WindowKey、流水线并发、ResultJoiner、队列策略、录音水位和停机协议不变。预Joiner拒绝窗口的内存说明同步为160 ms。
- **Development Test UI**：合并历史平均预降噪增益、停止状态保护、权威方向ID颜色/标签和一秒丢窗指标相关提交；保留当前新版的L4独立完成帧邮箱、跨epoch隔离和试听缓存行为。
- **录音与数据管理**：RecordingStore、Catalog、恢复和保留策略无功能变化；DecisionRecord/增强波形及试听重叠范围随公共上下文统一为7680 samples。正式录音系统仍随主项目纳入Git，运行录音数据不纳入版本控制。
- **测试与文档**：将跨层、录音v4、MUSIC滚动、L3/L4、Runtime和UI测试统一到160 ms契约；README、架构、Windowing、L3和数据管理说明同步，并明确L2滚动历史与L3/L4直接窗口的区别。
- **验证**：合并冲突解决后的定向测试共`120 passed`，160 ms核心链测试`86 passed`，录音v4与Runtime v1.1补充复测`8 passed`；最终全量pytest为`402 passed`，核心新项目路径Ruff全部通过，全项目Python `compileall`通过。
- **Git LFS与数据边界**：本次无Git LFS资产内容变化；模型、精选测试资产继续按现有LFS规则管理。.venv、data、运行录音、scratch、Catalog、日志、缓存、partial、密钥和本地代理设置不提交。
- **已知验证边界**：本次目标是保证已提交功能和时间契约在本地/GitHub一致，不等同于新160 ms配置已经完成真实阵列、诊室多声源或长时间实机性能验收。

---

## 2026-08-20 — 按1.2.1实际实现维护项目总架构图

- **版本/标签**：项目`1.2.1`文档维护；不创建或移动发布标签，既有`v1.2.1`保持不变。
- **类型**：README总架构图、相关算法流程与完成边界校正。
- **涉及文件**：`README.md`、`CHANGELOG.md`。
- **架构图**：按当前代码、配置、测试和各模块说明重新核对L1→Ingest→Window→L2→L3→L4→ResultJoiner主链；主标题使用`【已完成】`标识已经接通的代码模块，下级分支不重复标记。补齐公共`track_id`、DecisionRecord v4、Production UI和独立只读Pipeline Log UI，并明确Log UI不是Layer 5、独立进程未注入公开provider时显示`Unavailable`。
- **L2/Runtime契约**：移除旧SRP、迭代多峰、可关闭私有ID和L4转正/续租描述；改为240 ms Rolling NormMUSIC、MDL 0～6阶诊断、手动1/2/3阶上限、可选DPD/IMCRA白化、永久全局方向ID、可选Kalman、最多3个且两两至少45°的公共方向，以及当前统一`stage_queue_windows=1000`有界latest-wins队列。
- **L3/L4与结果链**：架构图更新为按`WindowKey + track_id`严格对齐，说明双候选`rho`分支、单/三候选Loaded MVDR、DAS回退、L4多语言MarbleNet以及ResultJoiner逐ID校验和有序提交；历史75.78%丢窗仅保留为旧v3证据，不再写成当前1.2.1性能结论。
- **界面与限制**：Development Test UI说明改为永久公共ID、MUSIC阶数上限及默认关闭的DPD/白化；完成清单纳入Production UI和Pipeline Log UI；限制部分改为当前最多3候选，但明确三候选能力不等于三人诊室分离已通过实机验收。
- **未改变**：L1、L2、L3、L4算法代码，Windowing、Application Runtime、Development Test UI、Pipeline Log UI、Production UI、RecordingStore、Audio Data Manager、配置、测试、模型、音频、阵列表和其他Git LFS资产均无变化。
- **验证**：README本地链接、代码围栏、关键配置/架构术语、Git空白与冲突标记静态检查通过；`tests/test_parallel_config_and_docs.py`与`tests/test_runtime_v11_contracts.py`共`7 passed`。文档核对不构成真实阵列、诊室声场、中文目标域或长时间负载验收。
- **Git LFS**：无变化；`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、密钥和代理设置不纳入提交。

---

## 2026-08-20 — 项目1.2.1整合发布

- **版本/标签**：项目`1.2.1`，创建新的不可变标签`v1.2.1`；`v1.0.0`、`v1.0.1`、`v1.1.1`、`v1.1.2`及全部历史分支保持原位，不移动、不覆盖、不删除。
- **发布范围**：整合`v1.1.2`之后的全部已提交功能和当前工作区修改，覆盖L2、Runtime、Development Test UI、Pipeline Log UI、Production UI、CorpusStore命名、配置、文档与测试；L1～L4、Windowing和完整录音/数据管理系统继续随项目发布。
- **L2**：优化IMCRA白化并保持丢窗UI状态；tentative轨迹可在滚动确认窗口内重新匹配并完成确认，减少短时漏检造成的重复ID。MUSIC、DPD、公共方向上限和L2公开版本`1.1`保持兼容。
- **Runtime**：L2/L3/L4阶段队列容量改为严格配置驱动；新增上一秒完整处理20 ms窗口与丢窗事件统计，按session/epoch隔离，丢窗率严格使用`丢窗/(完整处理+丢窗)`，不把启动以来累计值冒充一秒指标。
- **Development Test UI**：底部每秒显示L2/L3/L4平均耗时、L4后的统一输出刷新率、完整窗口数、丢窗数与丢窗率；保留低电平有效试听轨，空帧/错误投影不误删已有试听行。
- **数据管理与桌面入口**：测试语料录音采用标准化标签文件名；Production UI自动适配桌面可用区域；Pipeline Log UI增加桌面启动入口；录音回放、Catalog和旧记录兼容边界不变。
- **未改变**：L1采集/IMCRA核心算法、WindowAssembler时间轴、L3波束形成算法、L4 MarbleNet模型与概率语义、RecordingStore资产schema和Git LFS模型/测试音频均无新变化。
- **本地数据边界**：`.venv/`、`data/`、运行录音、scratch、Catalog、日志、缓存、临时报告、密钥和代理设置继续只保存在本机，不进入GitHub。
- **验证**：完整自动测试`399 passed`；核心源码与测试Ruff全部通过；全目录Python编译通过；项目元数据为`1.2.1`、L2公开版本为`1.1`；Git差异、冲突标记、敏感数据与LFS边界检查通过。
- **Git LFS**：现有模型、精选测试音频和大型数组继续按`.gitattributes`管理；当前工作区没有新增或修改LFS资产。

---

## 2026-08-20 — Development Test UI静音过滤与试听行同步修复

- **版本/标签**：项目`1.1.2`Development Test UI维护修复；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：L3试听缓存静音判定与UI缓存生命周期一致性修复。
- **涉及文件**：`gui/dev_test_ui/audio_id_tracker.py`、`gui/dev_test_ui/panels.py`及对应Development Test UI测试。
- L3试听音轨的绝对有声RMS门限暂由`-50 dBFS`调低为`-60 dBFS`，避免未经试听增益的低电平有效BF音频被过早计为静音；声音hop占比不超过30%的既有整轨过滤规则保持不变。
- 当完整权威试听快照确认某个方向ID的缓存已被过滤删除时，界面同步删除该ID的时长、波形和播放行；普通空帧/错误投影继续保留上次有效行，正常`confirmed/coasting/ended`音轨仍随缓存保留，不产生“波形仍在但暂无可播放缓存”的假行。
- **未改变**：L1、L2 MUSIC/Gate/ID/Kalman、L3波束形成算法和音频格式、L4模型、Runtime调度、Center Mic参考、正式录音/数据管理、Production UI、Pipeline Log UI、配置schema、模型和二进制资产均无变化。
- **验证**：执行L3试听追踪及Development Test UI聚焦测试；未进行耳机实机听感验收。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 阶段等待队列改为单变量配置并设为1000

- **版本/标签**：项目`1.1.2`Runtime配置调整；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：Runtime调度容量、严格配置schema、Joiner在途上限、文档与测试。
- 新增唯一常用变量`runtime.stage_queue_windows`，当前设为`1000`；L2、L3、L4三个单worker等待队列默认同步使用该值。以后调整容量只需修改这一处，不再同时维护三层队列和Joiner上限。
- `max_inflight_windows`在未显式覆盖时自动派生为三层实际队列容量之和再加3个正在执行的窗口；当前自动结果为`3003`。保留`l2_queue_windows/l3_queue_windows/l4_queue_windows`可选高级覆盖，供专项测试或诊断配置使用。
- 按50窗/秒计算，1000窗约为单层20秒等待容量。相较容量1可吸收短时过载并减少截图所示的激进丢窗，但会增加最坏端到端延迟与内存占用；满队列时仍按原latest-wins策略替换最旧等待窗并记录`DROPPED`。
- **未改变**：采集handoff容量、completion队列、L1、MUSIC/MDL及实验开关、匈牙利ID、Kalman、L3/L4算法、Development Test UI布局、录音/数据管理、Production/Log UI、模型与音频资产均无变化。
- **验证**：共享变量派生、严格schema、Joiner容量、Runtime调度和全量自动测试；未运行长时间负载。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — 测试语料名称包含完整录制标签

- **版本/标签**：项目`1.1.2`维护修复；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：测试语料显示命名、已有本地标签迁移与Production UI可读性修复。
- **涉及文件**：`data_management/corpus_naming.py`、`scripts/migrate_corpus_names.py`、`gui/production_ui/{app.py,README.md}`及对应测试。
- 新录音名称统一为“环境 · 月日-时分 · 声源数 · 各声源类型（移动方式） · 噪音来源”；名称只作为可读展示字段，Recording UUID、资产目录、音频和热力图文件名保持不变。
- 新增可重复执行的本地语料名称迁移工具；迁移同步更新`recording_manifest.json`、`labels.json`、labels资产SHA-256、manifest sidecar、Catalog投影和审计记录。旧式“环境-单人声固定声源-噪音背景噪音”名称可恢复为结构化标签后再命名。
- **本地数据**：当前`data/test_corpus`内8条已有标记语料已完成迁移；本地录音和Catalog继续受忽略规则保护，不纳入Git或Git LFS。
- **未改变**：L1～L4算法、Windowing、Application Runtime、Development Test UI、Pipeline Log UI、录音PCM/热力图资产内容、绝对sample轴、QA与数据集划分均无变化。
- **验证**：语料命名/迁移与Production UI聚焦测试`19 passed`，全量自动测试`397 passed`，Ruff通过；实际迁移后二次预览为0条待更新，并核对8条manifest、labels SHA-256、manifest sidecar与Catalog显示名一致。
- **Git LFS资产**：无变化。

---

## 2026-08-20 — L2 tentative轨迹滚动确认修复

- **版本/标签**：项目`1.1.2`维护修复；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：L2权威ID生命周期与L2→L3方向音频链路修复。
- **涉及文件**：`layer2_source_detection/global_tracker.py`、`tests/test_l2_music_tracking.py`。
- tentative轨迹不再把永久不变的`first_seen_sample`同时当作唯一确认截止时间；新增私有绝对sample滚动观测窗口，过期观测会被移出，后续有效观测可重新形成确认机会。`first_seen_sample`仍保持原始身份语义，不改ID、关联、角度、Kalman、coasting或TTL规则。
- 当任意最近200 ms窗口满足配置的观测次数后，原权威ID转为`confirmed`，随后按既有规则进入L2 `directions`，使L3能够按同一`(session_id, stream_epoch, track_id)`生成BF音频；未满足条件的tentative轨仍不会进入L3。
- **复现验证**：同一段20.56秒“会议室·2个声源”模拟录音修复前第二轨迹874个窗口始终tentative、0次进入L3；修复后为10个tentative、280个confirmed、583个coasting窗口，863次被选入L3、664次实际完成BF，阶段错误均为空。该回放仍有240个调度丢窗，不构成50 Hz性能验收。
- **未改变**：L1、MUSIC候选生成与Gate、L3算法/缓存格式、L4模型、Runtime队列策略、Development Test UI布局、正式录音/数据管理、Production UI、Pipeline Log UI、配置、模型和二进制资产均无变化。
- **验证**：L2 MUSIC/ID测试41项通过；Runtime v1.1契约、Development Test UI试听追踪与UI测试41项通过。Git LFS资产无变化。

---

## 2026-08-20 — 优化IMCRA对角白化并区分L2丢窗状态

- **版本/标签**：项目`1.1.2`维护修复；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：L2 MUSIC白化性能、Runtime到Development Test UI的丢窗诊断语义、文档与回归测试。
- **涉及文件**：`layer2_source_detection/music.py`、`app/runtime.py`、`gui/dev_test_ui/{aggregator,app}.py`、项目/L2 README、`ARCHITECTURE_V1.1_TARGET.md`及对应测试。
- L2继续只读DecisionWindow中的L1 IMCRA快照，不拥有、更新或重置IMCRA。逐麦PSD构成的对角噪声模型改用逆平方根逐通道缩放协方差与steering，数学上等价于原对角Cholesky白化；删除每20 ms逐频通用7×7 Cholesky和矩阵求解，并将16-hop频率插值改为批量向量化。DPD与白化同时开启时同窗复用一份IMCRA指标。
- `l2_admission_queue_overflow`等L2接纳丢窗不再伪装成Gate/IMCRA不可用：同一epoch保留最近一次成功MUSIC、Gate、方向和原始发布时间，标题显示`STALE | L2 DROPPED | last completed`；真正的Gate warming/unavailable仍按原契约清空当前空间结果。
- **未改变**：L1 IMCRA算法和状态机、概率Gate、MUSIC数学输出、ID/Kalman、L3、L4、Runtime latest-wins队列容量、录音/数据schema、Production UI、Log UI、配置、模型和音频资产均无变化。
- **验证**：L2、Runtime、并行调度和Development Test UI直接相关测试`109 passed`；L2白化聚焦测试`41 passed`；Ruff与`git diff --check`通过。60窗独立短基准中白化开启路径约为平均`7.26 ms`、p95 `9.10 ms`、最大`9.59 ms`，关闭路径约为平均`4.44 ms`、p95 `6.41 ms`；尚未完成真实阵列全链并发长时间验收。
- **Git LFS资产**：无变化；`data/`、录音、日志、Catalog和缓存不纳入提交。

---

## 2026-08-20 — Production UI默认窗口适配当前屏幕

- **版本/标签**：项目`1.1.2`维护修复；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：Production UI桌面启动窗口与响应式顶部布局修复。
- **涉及文件**：`gui/production_ui/app.py`、`tests/test_production_ui_usability.py`。
- 双击桌面入口后，录音与数据管理界面默认以最大化普通窗口打开，使用当前显示器可用工作区并保留标题栏、任务栏以及最小化/还原/关闭能力，不再按超出高DPI屏幕工作区的固定尺寸显示。
- 顶部会话状态与录音控制拆分为两行；六个主页面页签改为按可用宽度扩展并保留滚动按钮，降低控件内容把窗口最小宽度撑出屏幕的风险。还原窗口的初始参考尺寸由`1460×900`降为`1200×760`。
- **未改变**：L1～L4算法、Windowing、Application Runtime、Development Test UI、Pipeline Log UI、RecordingStore/Catalog、录音与测试语料schema、配置、模型、音频和精选测试资产均无变化。
- **验证**：Production UI可用性聚焦测试`17 passed`；相关文件Ruff与`git diff --check`通过。自动测试确认默认状态为最大化普通窗口而非无边框全屏；本次不构成多显示器及全部Windows缩放比例的人工验收。
- **Git LFS资产**：无变化；`data/`、Catalog、录音、日志、缓存和本地桌面快捷方式不纳入提交。

---

## 2026-08-20 — Pipeline Log UI桌面无控制台启动入口

- **版本/标签**：项目`1.1.2`维护改动；不创建或移动发布标签，既有`v1.1.2`保持不变。
- **类型**：Pipeline Log UI桌面启动入口、只读边界说明、自动测试与本机快捷方式。
- **涉及文件**：`gui/log_ui/{__main__,standalone}.py`、`gui/log_ui/README.md`、`tests/test_log_ui.py`；本机桌面新增快捷方式但不纳入Git。
- 新增`.venv/Scripts/pythonw.exe -m gui.log_ui`无控制台入口，使桌面快捷方式可直接打开独立五页Log UI窗口。
- 独立进程继续使用无能力provider并明确显示`Unavailable`；它不接受data root、不构造`DataManagerService`、不打开Catalog/SQLite/WAL，也不读取Runtime latest-only邮箱。完整封存session回看仍只在正式宿主注入公共只读查询provider时启用。
- **未改变**：L1～L4、Windowing、Application Runtime、Development Test UI、Production UI、RecordingStore/Catalog、数据schema、配置、模型、音频和精选测试资产均无变化。
- **验证**：Log UI及Recording v4公开查询边界聚焦测试`18 passed`；Ruff与`git diff --check`通过；桌面`.lnk`的target、arguments、工作目录和图标核验通过，并实际打开标题为`Pipeline Log UI — Read Only`的无控制台窗口。不构成真实封存session人工回放或诊室实机验收。
- Git LFS管理资产无变化；`data/`、Catalog、录音、日志、缓存、临时文件和本地设置不纳入提交。

---

## 2026-08-19 — 项目1.1.2整合发布

- **版本/标签**：项目`1.1.2`，创建新的不可变标签`v1.1.2`；`v1.0.0`、`v1.0.1`、`v1.1.1`及全部历史分支保持原位，不移动、不覆盖、不删除。
- **发布范围**：整合`v1.1.1`之后的已提交功能与当前工作区全部项目修改，覆盖L2、L3、Runtime、Development Test UI、录音数据管理、架构文档和自动测试。Layer 1、Windowing、Layer 4模型、Pipeline Log UI与Production UI继续作为完整项目组成部分打包上传。
- **L2**：MDL诊断范围扩展为0～6阶，公共方向仍最多3个；增加1/2/3可调实际MUSIC阶数上限、可选DPD rank-1 MUSIC与可选IMCRA噪声白化。confirmed方向漏检进入coasting后，在最多3路及至少45°分离约束内继续作为权威L3目标；tentative漏检轨不伪装成正式目标。
- **L3与Runtime**：优化多声源矩阵求解和跨跳滚动缓存；L2/L3/L4等待队列最终固定为容量1的低延迟latest-wins。停机在强制取消后按完整超时等待worker，全部退出后清空残留阶段队列，避免停止状态残留窗口和内存占用。
- **Development Test UI**：coasting权威ID继续接收并拼接真实L3波束形成音频；移除右上重复方向表，L3轨道直接显示稳定ID与对应颜色；预降噪增益改为本epoch历史平均；停止后不把残留latest帧重新显示为LIVE。已有可播放语音不会仅因末尾长静音被删除，整体有声占比不超过30%的轨道仍清理。
- **录音与数据管理**：纳入回收站操作同步更新Catalog的既有修复；RecordingStore schema、正式录音格式和CNN资产本次不变。运行录音、scratch、Catalog、日志和本机`data/`不上传。
- **未改变**：L1采集、通道映射、IMCRA核心算法、WindowAssembler时间轴、L4 MarbleNet模型与概率语义、Log UI只读边界、Production UI核心页面和历史记录兼容规则无新算法变化。
- **项目边界**：将报告渲染产生的`tmp/`纳入Git忽略；本机临时页面保留在本地、不删除也不上传。`.venv/`、缓存、密钥、代理设置和未精选本地数据继续排除。
- **验证**：完整自动测试`389 passed`；核心源码与测试Ruff全部通过；全目录Python编译通过；项目元数据为`1.1.2`、L2公开版本为`1.1`；Git差异、冲突标记、敏感数据与LFS边界检查通过。
- **Git LFS**：模型、精选测试音频和大型数组继续按`.gitattributes`管理；当前工作区没有新增或修改LFS资产。

---

## 2026-08-19 — coasting权威ID持续生成L3波束形成试听音频

- **版本/标签**：L2→L3权威ID试听链路修复；未创建或移动版本标签。
- **类型**：跨层数据契约与Development Test UI试听行为修复。
- L2的`directions`除已确认实测ID外，现会在最多3路和方向间隔至少45°的约束内纳入仍有效的`coasting`权威ID；优先选择等待时间短、得分高且ID稳定的轨迹，并沿用保持/预测输出角送入L3。
- 未确认轨迹失去观测后保持`tentative`，不伪装为`coasting`，也不会触发L3波束形成；正式`confirmed/coasting`元数据保持一致并继续使用原`track_id`。
- L3算法、三档模式和Development Test UI缓存格式无变化；但coasting窗口现在获得真实BF输出并写入同一`(session_id, stream_epoch, track_id)`试听轨，只有本窗确实没有该ID的L3输出时才按既有绝对时间轴补等时静音。
- L4算法无变化，但继续消费与L3相同的权威方向集合；L1、录音/数据管理、Production UI、Pipeline Log UI、模型和二进制资产均无变化，Git LFS资产无变化。
- **验证**：新增confirmed→coasting BF目标、tentative排除和同ID真实音频连续写入测试；完成相关跨层定向测试及全量测试（结果见本次提交验证记录）。

---

## 2026-08-20 — Development Test UI性能栏合并刷新率显示

- **版本/标签**：Development Test UI显示精简；未创建或移动版本标签。
- **类型**：纯UI文案与布局调整。
- 底部上一秒性能栏不再分别显示L2、L3刷新率，仅保留三层平均耗时，并在L4耗时后显示一个统一输出刷新率；20 ms完整窗口数、丢窗数和丢窗率继续显示且统计逻辑不变。
- Runtime调度、性能快照字段、L1/L2/L3/L4算法、录音与数据管理、Production UI、Pipeline Log UI、模型和二进制资产均无变化，Git LFS资产无变化。
- **验证**：更新底栏初始布局文本测试；当前桌面分支Development Test UI测试24项通过，正式提交分支25项通过。

---

## 2026-08-20 — Development Test UI增加上一秒窗口与丢窗性能指标

- **版本/标签**：Development Test UI性能监控增强；未创建或移动版本标签。
- **类型**：Runtime可观测性、UI显示和性能快照契约更新。
- Runtime在每次20 ms窗口被L2/L3/L4调度链丢弃时记录带session/epoch和单调时钟的丢窗事件；继续保留原累计`processing_drops`，不改变latest-wins、队列容量或算法调度行为。
- 性能快照新增上一秒完整处理窗口数、丢窗数和丢窗率；完整处理以L2/L3/L4均取得非失败终态的窗口计数，丢窗率按`丢窗/(完整处理+丢窗)`计算，session/epoch切换时清零。
- Development Test UI底栏继续每1秒刷新，在原L2/L3/L4平均耗时与刷新率后显示`20ms窗口、丢窗、丢窗率`；停止或尚无数据时稳定显示0，不使用历史累计值。
- L1采集、L2 MUSIC/ID、L3波束形成与试听、L4分类、正式录音/数据管理、Production UI、Pipeline Log UI、模型和二进制资产均无变化，Git LFS资产无变化。
- **验证**：增加一秒滑动计数、丢窗率、epoch重置和初始布局文本测试；Runtime/UI相关测试70项通过，全量自动测试322项通过。

---

## 2026-08-19 — 增加可选DPD rank-1 MUSIC与IMCRA噪声白化

- **版本/标签**：`v1.1.1`发布后的L2试验性鲁棒定位功能；不创建或移动版本标签。
- **类型**：L2 MUSIC候选生成与噪声白化、Runtime实时配置、Development Test UI控制、记录诊断、文档和回归测试。
- **涉及文件**：L2 `configuration.py`、`music.py`，项目配置，Runtime，Development Test UI的`app.py`、`panels.py`、`settings.py`，README、1.1.1架构说明及对应测试。
- 新增默认关闭的`DPD + rank-1 MUSIC`。开启后以逐频主特征值间隙和平面波拟合筛选可靠频点，以IMCRA `spp/prior_snr`加权rank-1 MUSIC方向票，并要求候选具备真实加权跨频支持；候选数仍受用户手动1/2/3上限约束，MDL在该路径保留为诊断而不直接规定候选数。
- 新增默认关闭的`IMCRA噪声白化`。白化严格只消费当前DecisionWindow中READY的公开IMCRA逐麦`noise_psd`，形成逐频对角噪声协方差并同时白化观测协方差和steering；当前接口没有跨麦互谱，因此没有虚构完整噪声CSM。缺少READY快照或数值分解失败时标记`unavailable`并安全退回未白化MUSIC。
- 两个开关均通过Test UI按钮实时修改revision并原子持久化；L2标题显示DPD选中频点数与白化状态。DecisionRecord/运行诊断增加开关、可靠频点、白化状态、IMCRA hop数量及每候选支持率/平面波拟合值。
- **未改变**：L1 IMCRA算法和预降噪、概率Gate、永久匈牙利ID及可选Kalman、L3、L4、Runtime队列策略、正式录音/数据管理、Production UI、独立Log UI、模型和音频资产均无变化。
- **验证**：配置、L2、Runtime、v1.1契约与Development Test UI重点回归`114 passed`；完整自动测试`386 passed`，相关文件Ruff与`git diff --check`通过。两个功能同时开启的30窗短基准为p50 `11.60 ms`、p95 `13.45 ms`、最大`13.79 ms`。按用户要求不运行10分钟负载，自动测试与短基准不构成真实阵列声场验收。
- **Git LFS资产**：无变化。

---

## 2026-08-19 — L2/L3/L4等待队列由10000改为1

- **版本/标签**：`v1.1.1`发布后的Runtime低延迟配置修正；不创建或移动版本标签。
- **类型**：分阶段流水队列容量、Joiner在途上限、配置、文档与回归测试。
- L2、L3、L4三个单worker阶段的等待队列默认值、根配置和schema上限均从`10000`改为`1`；每层最多保留一个尚未开始处理的窗口，满队列时继续使用既有latest-wins策略替换旧等待窗。
- `max_inflight_windows`从`30003`同步改为`6`，严格覆盖三个等待窗口和三个正在执行的窗口，避免Joiner在途容量与实际队列结构脱节。completion队列及后备backlog仍各为8，采集handoff仍为500块。
- 该调整以低延迟和实时控制为优先：持续算力不足会产生明确的`DROPPED`审计，而不会再积累最长约200秒的单层等待。它不承诺零丢窗，真实完成率仍取决于各层是否跟上20 ms输入节拍。
- **未改变**：L1采集/IMCRA/预降噪、MUSIC/MDL与手动阶数上限、匈牙利ID、Kalman、L3/L4算法、Test UI布局、录音/数据管理、Production/Log UI、模型和音频资产均无变化。
- **验证**：配置、Runtime latest-wins与容量相关快速测试通过；未运行长时间负载。
- **Git LFS资产**：无变化。

---

## 2026-08-19 — 增加Test UI可选MUSIC实际阶数上限

- **版本/标签**：`v1.1.1`发布后的L2诊断试验控制；不创建或移动版本标签。
- **类型**：L2 MUSIC诊断/执行阶数分离、Test UI运行时控制、ID出生保护、配置、文档和回归测试。
- **涉及文件**：`common/config.py`、`config/config.yaml`、`layer2_source_detection/configuration.py`、`music.py`、`global_tracker.py`、`pipeline.py`，Development Test UI的`app.py`、`panels.py`、`settings.py`、`srp_panel.py`及对应测试和架构说明。
- MDL继续完整估计并记录`0～6`阶空间模态；新增只允许`1/2/3`的`effective_order_limit`，实际MUSIC阶数严格为`min(MDL诊断阶数, 手动上限)`，默认上限3。设置通过Test UI下拉框持久化到本地设置；L2在每个窗口真正开始计算时读取最新值，即使队列已有积压也会在下一次L2计算实时生效，并把实际revision继续传给L3/L4、UI和录音，不覆盖MDL诊断值。
- 极图和L2标题分别显示`MDL`诊断阶数与实际`MUSIC`阶数。算法版本更新为`frequency_normalized_music_mdl_cap_v2`，配置revision随手动上限变化递增。
- MDL诊断阶数大于公共三候选上限时标记`mdl_saturated/model_mismatch`；该窗不创建新方向ID，但仍允许已有ID通过原匈牙利关联继续更新或进入coasting，避免高阶模型失配进一步制造新ID。
- **明确未加入**：没有实现逐频真实局部峰支持、SPP/SNR权重、特征值间隙权重或任何新的可靠性门禁；NormMUSIC仍为原有逐频最大值归一化后等权融合，candidate threshold、45° NMS、2次确认、匈牙利代价和Kalman均未改变。
- **实录短回放**：截图对应32.12秒单声源录音的第799窗在MDL=3时，上限1输出`86°`，上限2输出`87°/191°`，上限3复现`13°/89°/179°`；该结果用于证明手动阶数上限确实作用于MUSIC，并不构成真实角度精度验收。
- **其他模块**：L1/IMCRA与预降噪、L3、L4、Runtime队列策略、正式录音/数据管理、Production UI、独立Log UI、模型和音频资产均无变化。Development Test UI同文件中既有未提交试听/显示修改不属于本条算法范围。
- **验证**：L2/配置/Test UI/Runtime重点回归`111 passed`；截图对应实录只做单窗短回放，按用户要求未运行10分钟负载。完整测试和Ruff将在提交前继续执行。
- **Git LFS资产**：无变化；未修改或新增音频、模型和阵列表资产。

---

## 2026-08-19 — 修复L3多声源BF丢窗级联并扩容阶段队列

- **版本/标签**：`v1.1.1`发布后的L3性能修复；不创建或移动版本标签。
- **类型**：L3滚动缓存与矩阵求解性能、Runtime容量配置、1/2/3声源基准、文档和回归测试。
- **涉及文件**：`layer3_direction_signal/adaptive_separation.py`、`hybrid.py`、`noise_context.py`、`shared_stft.py`及L3说明，`common/config.py`、`config/config.yaml`、Runtime/项目架构说明、`scripts/benchmark_l3_l4.py`和对应测试。
- **跳窗滚动修复**：L3不再要求DecisionWindow严格相邻才复用。相同session/epoch且按960 sample对齐、仍有320 ms上下文重叠的`1～15` hop跳跃，按绝对sample复用`31-2N`个STFT内部帧，只计算`2+2N`个反射边界/新增帧；IMCRA只搬运新增N个hop，噪声协方差按对应过期/新增帧贡献滚动。达到16 hop无重叠、时间倒退、非hop对齐或身份/配置变化仍完整安全重建。
- **BF求解优化**：保持Dual LCMV、soft-null loaded MVDR、loaded MVDR、三档loading顺序和逐频DAS回退不变；用批量`cholesky_ex/cholesky_solve`复用同一加载协方差的LCMV/MVDR多右端，将两个soft-null目标合并求解，固定批量计算retry并统一选择首个有效结果。Hermitian正定矩阵的通用SVD条件数改为等价的特征值范围校验，核心retry循环移除逐档`bool/nonzero/item`主机同步，诊断计数合并为末尾一次传输。
- **Runtime容量**：按用户明确要求将L2/L3/L4等待队列默认值与schema上限均改为10000；为避免旧16窗门限遮蔽队列，`max_inflight_windows`改为30003，completion主队列和后备backlog仍各为8，单worker、L2→L3→L4依赖、latest-wins、ResultJoiner和有界缓存架构均不变。50窗/秒时每层最多约200秒等待；30003个窗口仅原始8通道float32音频的理论下限约13.7 GiB，另有IMCRA/StageResult开销，扩大队列不等于吞吐问题已解决。
- **基准与性能**：基准schema升级为`l3_l4_benchmark_v2`并真正生成三声源批次。本机RTX 5060 Laptop GPU、连续滚动窗口、每档120个样本的隔离L3端到端P95为1/2/3声源约`9.23/15.96/9.52 ms`，平均吞吐约`146.63/86.26/136.15窗/秒`；双声源P99约`17.33 ms`。gap=1/2/7/15时双声源滚动基准P95均低于15 ms。该结果不包含真实麦克风和L1/L2/L4/UI并发，仍需新的v4正式session验证真实丢窗率与端到端延迟。
- **契约与未改变项**：没有修改steering cache、角度key/量化、空间`p`表、候选排序、track ID、公开DTO、L1、L2算法、L4模型、Development Test UI、Production/Log UI、录音数据格式、模型或音频资产。基准JSON增加三声源字段且schema版本变化；Runtime配置默认容量及合法上限发生兼容性可见变化。
- **验证**：L3/cache/adaptive/benchmark/config/parallel Runtime重点回归`94 passed`；完整自动测试`375 passed`；全仓Ruff通过；`git diff --check`通过。新增1/2/3候选直接求解等价、首个有效retry、批处理形状、2/7/15 hop STFT与协方差等价、320 ms无重叠重建及CUDA多hop验证。
- **Git LFS资产**：无变化；未新增或修改模型、音频、阵列表或运行数据。

---

## 2026-08-19 — L2 MUSIC MDL试验范围扩展为0～6阶

- **版本/标签**：`v1.1.1`发布后的L2试验性修复；不创建或移动版本标签。
- **类型**：L2 MUSIC模型阶数、公共数据契约、文档与回归测试。
- **涉及文件**：`layer2_source_detection/music.py`、`common/data_types.py`、`tests/test_l2_music_tracking.py`、L2/项目README及`ARCHITECTURE_V1.1_TARGET.md`。
- MDL候选阶数由`0～3`扩展为7麦阵列可保留至少一维噪声子空间的`0～6`，跨频众数及一致性统计同步覆盖七种阶数；`ModelOrderEstimate.estimated_sources`允许记录`0～6`。
- 公共MUSIC候选和进入ID/L3的方向仍由`max_candidates=3`限制为最多3个；本次未增加并行声源输出上限，也未加入饱和拒绝、噪声白化或真实跨频峰支持。
- 测试扩展为合成特征值下MDL `0～6`全覆盖，并保留最多3候选约束。
- L1、Gate、ID匈牙利关联与Kalman、L3、L4、Development Test UI、Log UI、录音存储、Runtime调度、配置、模型和资产均无变化。
- **验证**：`tests/test_l2_music_tracking.py` 29项通过；`tests/test_runtime_v11_contracts.py`、`tests/test_dev_ui.py`、`tests/test_dev_ui_pipeline_status.py`合计29项通过，共58项相关回归通过。
- **Git LFS资产**：无变化。

---

## 2026-08-19 — 修复录音移到回收站后仍显示且无法再次删除

- **版本/标签**：`v1.1.1`发布后的分支修复；不创建或移动版本标签。
- **类型**：录音数据管理、Catalog迁移、可恢复回收站事务、Production UI与回归测试。
- **涉及文件**：`data_management/catalog.py`、`data_management/retention.py`、`data_management/service.py`、`gui/production_ui/app.py`及对应测试。
- Catalog schema迁移到本地revision 3，为运行录音和测试录音增加`trashed_at`软删除状态；默认查询、首页统计和Production UI列表不再返回已移到回收站的条目，恢复后重新显示，原有Catalog元数据不丢失。
- 启动数据服务时对旧版已经完成物理移动但仍残留在Catalog中的条目进行安全对账；只接受包含对应录音manifest的完整回收站数据包，兼容修复前已经执行的删除操作。
- 移动前确认源目录真实存在，防止重复点击为不存在的录音创建只有审计文件的无效数据包；恢复入口过滤不完整或已经恢复的历史操作。
- Production UI删除成功后立即刷新列表并在状态栏提示“已移到可恢复的回收站”；锁定数据集和实验快照保护保持不变。
- L1、L2、L3、L4算法、Development Test UI、实时Runtime处理、配置、校准、模型和音频资产格式均无变化；`data/`、实际录音、Catalog和回收站内容仍只保存在本机，不进入Git。
- 验证：回收站移动/隐藏/恢复、旧Catalog残留对账、锁定样本拒绝删除及Production UI选中删除测试通过；完整测试首轮`369 passed, 1 failed`，唯一失败为既有RecordingStore异步封存3秒时限波动，单独复跑通过，未修改该并发逻辑。
- **Git LFS资产**：无变化。

---

## 2026-08-19 — 项目1.1.1整合发布

- **版本/标签**：项目`1.1.1`，计划创建不可变标签`v1.1.1`；历史`v1.0.0`与`v1.0.1`标签保持原位，不移动、不覆盖、不重写。
- **发布范围**：合并并版本化Layer 1～Layer 4、Windowing、Application Runtime、Development Test UI、独立Pipeline Log UI、RecordingStore、Audio Data Manager、Production UI、配置、文档、自动测试、模型与精选测试资产。
- **Layer 1 / Windowing**：纳入校准后的7麦滚动输入、8通道记录边界、0～8 kHz IMCRA/预降噪、20 ms唯一时间步、320 ms历史窗口、采集回调减负、有界10秒handoff及连续性/epoch重置诊断。
- **Layer 2**：公开版本保持`1.1`；正式方向主链采用Rolling NormMUSIC与MDL 0～3源估计，永久公共`track_id`、内部最多4轨/公共最多3轨、可选圆周Kalman、活动ID Gate保持、噪声干扰标记与按ID的L4语义反馈均纳入整合版本。
- **Layer 3 / Layer 4**：L3按`(WindowKey, track_id)`消费权威方向并输出48 kHz增强音频；L4按同一身份完成补偿、重采样和MarbleNet推理，不按角度创建或修补ID。
- **Runtime与记录契约**：同窗严格L2→L3→L4，跨窗分层并行；有界latest-wins、ComputeCache、ResultJoiner、有序watermark、显式丢弃审计与停机排空进入发布。正式结果升级为`decision_record_v4`并贯通公共方向ID。
- **Development Test UI**：显示MUSIC伪谱与固定三行公共ID，按1秒窗口显示L4峰值，按L2权威ID拼接L3试听；增加Center参考、播放进度、低有效声音轨清理，并移除用户可关闭永久ID的旧控制。
- **Pipeline Log UI**：独立只读五页观察与回放界面已实现，使用版本化公共查询读取封存session；不进入、不控制、不消费或反压实时主链。
- **录音与数据管理**：RecordingStore、Catalog、崩溃恢复、流式chunk资产、逐ID增强音频、Production UI和专用测试录音流程均随项目上传；运行录音、scratch、Catalog、日志、缓存和本机`data/`仍只保存在本地，不进入Git。
- **兼容性与未改变项**：保留旧`decision_record_v3`只读兼容；GitHub仓库、历史分支和历史标签不删除。真实阵列、诊室声场、小时级长时间运行与目标域CNN质量仍需按实机门禁继续验证，不能由自动测试替代。
- **验证**：完整自动测试`349 passed`；核心源码与测试Ruff全部通过；全目录Python编译通过；项目元数据为`1.1.1`、L2公开版本为`1.1`；Git差异、冲突标记、敏感数据和LFS边界检查通过。根配置测试同步为当前已记录的`verified`硬件校准状态。
- **Git LFS**：现有模型、精选测试音频和大型数组继续按`.gitattributes`管理；本次版本整合不新增运行录音或本机环境资产。

---

## 2026-08-19 — 恢复内部4轨硬上限并隔离Gate预热故障

- **版本/标签**：项目`1.1.0`开发分支；仍未发布，不创建或移动版本标签。
- **L2 ID**：`GlobalDirectionTracker`恢复内部最多4个活动ID的硬约束，公共输出仍最多3个。容量不足时只淘汰未被本窗关联的低优先级轨迹，顺序优先噪声干扰、无人声证据、tentative、最久未观测及低分；本窗已成功关联的轨迹受保护。
- **正确性**：同一个噪声ID同窗最多关联一个观测；超额低分新生观测会被确定性舍弃，ID数量不会无界增长，也不会因为容量达到4而清空tracker或改变Gate状态。
- **故障归因**：`WARMING_UP`仍只来自IMCRA/新epoch，不由ID容量触发。配套的采集连续性修复已降低Gate长期强开时的输入溢出风险并公开epoch reset原因；本次ID上限进一步限制UI、记录和试听扇出。
- **未改变**：Probability Gate概率算法与强开规则、MUSIC/MDL、Kalman、L3、L4模型、音频格式和Git LFS资产均无变化。
- **验证**：新增两组三方向错开观测下内部轨迹始终不超过4的回归测试；执行L2/Runtime重点测试、Ruff和差异检查。尚未完成真实设备“多ID+长期强开Gate”复测。

---

## 2026-08-19 — 测试录音改为快速保存并修复后台保存崩溃

- **版本/标签**：项目`1.1.0`开发分支；仍未发布，不创建或移动版本标签。
- **类型**：Production UI录音保存流程、后台任务生命周期和回归测试。
- **涉及文件**：`data_management/corpus_store.py`、`data_management/dedicated_recording.py`、`gui/production_ui/app.py`及对应测试。
- 专用L1测试录音结束后只封存原始8通道音频、热力图、标签和manifest并登记Catalog，不再自动执行耗时质量检查；新录音保存为“待检查”，仍可在“质量与标注”页面按需手动检查。
- Production UI持续持有后台任务直至界面回调完成，避免Qt提前释放任务造成保存结束时原生崩溃，并恢复启动后异步加载录音列表的可靠性。
- 更新向导状态和说明文字，删除“保存时自动检查”的提示；无模型、音频、Git LFS资产变化。
- 验证：专用录音快速保存、待检查状态、无自动QA报告、后台任务回调生命周期和Production UI基础页面测试通过；尚未进行新的麦克风实机录制验收。

---

## 2026-08-19 — 加固长时间音频采集并公开IMCRA重置原因

- **版本/标签**：项目`1.1.0`开发分支；仍未发布，不创建或移动版本标签。
- **类型**：L1实时采集可靠性、连续性诊断、Test UI错误归因和回归测试。
- **涉及文件**：`layer1_input/capture.py`、`config/config.yaml`、`app/runtime.py`、`gui/dev_test_ui/aggregator.py`、L1/Ingest/Runtime说明及对应测试。

### L1、Ingest与Windowing

- 将RMS电平计算移出PortAudio实时回调，改为读取capture status时按需计算；回调只保留PCM复制、sequence/timestamp、健康事件和有界投递，降低完整L2～L4并发时因Python回调超时导致`input_overflow`的风险。
- 主链capture handoff由100个20 ms块（2秒）调整为500块（10秒），吸收Windows/GPU/UI短时调度停顿；队列仍有硬上限，持续过载不会无限增长。
- 新增专用`handoff_drop_count`及交接队列当前深度、容量、高水位；连续满队列丢块合并为一个带范围与lost sample数的健康事件，避免同一拥塞突发反复增加epoch和重复触发2.4秒IMCRA预热。
- 真实`input_overflow/handoff_drop/sequence_gap/timestamp_gap`仍增加epoch，WindowAssembler与IMCRA仍安全重建；不补零、不隐藏真实丢音。单纯静音或概率降低仍不会触发`warming_up`。

### Runtime与Development Test UI

- 公开`processing_status.input_health`，包含当前epoch、连续性中断计数、最后中断原因、input/handoff drop计数和交接队列水位。
- Gate因epoch变化等待L2或IMCRA重新预热时，诊断增加`epoch_reset:<reason>`，可直接识别`health_event:input_overflow`、`health_event:handoff_drop`、`sequence_gap`或`timestamp_gap`，不再只显示无来源的`WARMING_UP`。
- Development Test UI布局、控件、试听和用户当前未提交的L4概率显示改动均未由本任务修改。

### L2、L3、L4与录音数据

- MUSIC、模型阶数、全局ID关联、Kalman、L3波束形成、L4模型和处理队列策略均无变化；较高`processing_drops`仍是独立的算法吞吐问题，不会重置IMCRA。
- RecordingStore、Catalog、manifest、录音格式、Production UI和Pipeline Log UI均无变化；本次实机诊断不保存或上传音频。

### 测试、实机与资产

- 新增连续handoff overflow事件合并、回调不执行RMS、capture水位和Runtime输入中断公开原因测试，并锁定10秒handoff配置。
- 不落盘裸采集+IMCRA实机10秒：499块，epoch 0，input/handoff drop均为0，L1 p95约3.5 ms。
- 不落盘完整L1～L4实机120秒：6008块、约49.88 Hz，epoch 0，input/handoff drop均为0，handoff高水位3/500；算法处理丢窗685次，明确不属于输入丢音或IMCRA重置。
- 未进行小时级诊室录音、设备热插拔或强制CPU/GPU饱和故障注入；这些仍属于最终实机门禁。
- Git LFS模型、音频和阵列资产无变化；`data/`、录音、Catalog、日志和临时文件不纳入提交。

---

## 2026-08-19 — 恢复L4人声概率反馈并增加非排他噪声ID

- **版本/标签**：项目`1.1.0`开发分支；仍未发布，不创建或移动版本标签。
- **L4→L2接口**：Runtime在L4完成并通过方向身份校验后，按`session_id + stream_epoch + decision_sample + track_id`回传人声概率及`is_voice`。L2使用有界线程安全队列，在下一L2窗口统一消费；迟到旧epoch、已删除ID或非法概率不会改变当前轨迹。
- **L2语义**：ID从建立或最后一次正向人声反馈起满3秒仍无新的人声判定时标记`is_noise_interference`。噪声轨继续跟随自身方向观测并沿用3秒几何TTL，但不进入普通ID的Hungarian排他关联，防止慢速讲话人靠近时被错误并入噪声ID。
- **噪声恢复**：噪声ID仅在±45°内不存在其他普通ID，且滚动3秒内累计5次L4人声判定时解除标记；非人声结果不增加次数，也不清空仍在窗口内的正向记录。L4语义反馈不确认ID、不延长几何TTL，也不修改Gate概率、MUSIC或Kalman参数。
- **公共契约与文档**：`TrackedDirection`增加只读布尔字段`is_noise_interference`，录音/日志可审计噪声标记；同步更新L2说明和1.1目标架构。
- **未改变**：L1采集与IMCRA、MUSIC/MDL数值算法、Gate强制开启规则、L3波束形成、L4模型本身、音频格式、模型资产及Git LFS资产均无变化。
- **验证**：增加L4按权威track ID回传概率、3秒噪声标记、±45°普通ID防误归并、滚动3秒累计5次恢复且夹杂非人声不清零等自动测试；未完成真实风扇与讲话人靠近场景的实机验收。

---

## 2026-08-19 — 活动方向ID存在期间强制保持Probability Gate开启

- **版本/标签**：项目`1.1.0`开发分支；仍未发布，不创建或移动版本标签。
- **L2**：每个窗口在概率Gate判定前，先按绝对sample清理超过3秒TTL的轨迹。只要仍存在任意tentative、confirmed或coasting ID，低于门限的正式40 ms概率判决即改为强制OPEN并继续运行Rolling NormMUSIC；最后一个ID删除后立即恢复按概率门限判断。epoch/session变化不会继承旧轨迹的强制状态，预热、概率缺失及无效输入仍保持阻断。
- **接口与文档**：不增加公共DTO字段或运行时开关；Gate通过`reason=active_id_force_open`及诊断字段明确记录强制来源。同步更新L2说明和1.1目标架构。
- **未改变**：L1采集、IMCRA概率算法、MUSIC/MDL数值算法、全局ID关联、Kalman、L3、L4、Development Test UI布局、录音格式、模型和Git LFS资产均无变化。
- **验证**：增加“建立ID后低概率仍强制OPEN”和“最后ID超过3秒TTL后恢复CLOSED”的自动测试；未进行真实麦克风及长时间声场验收。

---

## 2026-08-19 — Kalman关闭时方向轨采用最后观测角保持

- **版本/标签**：`feature/l2-music-tracking-v1.1`开发分支；项目`1.1.0`仍未发布，不创建或移动标签。
- **L2**：ID追踪和绝对sample生命周期继续永久运行；当Kalman关闭且轨迹漏检/coasting时，公开角度改为严格保持该ID最后一次真实观测角，不再按内部角速度外推。Kalman开启时仍允许预测；运行时切换不重置、不删除、不更换ID。
- **其他模块**：Probability Gate、Rolling NormMUSIC/MDL、L1、L3、L4、录音与UI布局无算法变化；Test UI自动消费新的L2轨迹角语义。
- **验证**：增加有速度轨迹的OFF零阶保持、359°↔0°圆周保持及ON→OFF不换ID测试；运行数据和Git LFS资产无变化。

---

## 2026-08-19 — 非 Log UI 全仓静态可用性审查与边界加固

- **版本/标签**：项目 `1.1.0` 开发分支；未创建或移动发布标签，已发布的 `v1.0.1` 不变。
- **类型**：全仓静态审查、L1 串口控制失败语义、数据集锁定前校验和可执行入口补齐。
- **涉及文件**：`layer1_input/api.py`、`data_management/service.py`、`.vscode/launch.json`、`.vscode/tasks.json`、`docs/KNOWN_ISSUES.md`和根 `CHANGELOG.md`。
- L1 原始串口写入、指示灯、波束方向、热力图阈值和恢复默认命令统一校验完整写入；底层异常或短写均返回 503，不再误报成功。
- 数据集分组分割先在内存中完成泄漏检查，通过后才改写 Recording 清单和 Catalog；校验失败不再留下部分更新。
- VS Code 直接运行入口补齐 Audio Data Manager，并移除 Development Test UI 过时的“实现后”标记；已知问题文档明确区分软件阻断项与实机验收边界。
- **未改变**：L1 音频采集/通道映射/算法、Windowing、L2 MUSIC/轨迹 ID、L3、L4、Application Runtime、Development Test UI 功能、Production UI 功能、录音格式、公共 DTO/配置、模型、音频与精选测试资产均无变化；Pipeline Log UI 及其测试文件按用户要求排除并保持原样。
- **验证**：按用户要求未运行 pytest 或其他测试套件；对非 Log UI 代码执行 Ruff、全模块导入和 Python 语法编译检查，并执行 `git diff --check`。真实麦克风、CDC、声场、CUDA、长时录音及回放未实机验收。
- Git LFS 管理资产无变化；`data/`、录音、Catalog、日志、缓存、临时文件和本地设置未纳入提交。

---

## 2026-08-19 — 独立只读 Pipeline Log UI 完整实现

- **版本/标签**：项目 `1.1.0` 开发分支；未创建或移动发布标签，已发布的 `v1.0.1` 不变。
- **类型**：Pipeline Log UI 只读适配、标准模型、统计引擎、五页界面、按需回放与自动测试。
- **涉及文件**：`gui/log_ui/`、`tests/test_log_ui.py`；根 `CHANGELOG.md` 仅增加本条记录。
- 新增记录列表、会话总览、分页 Pipeline 时间线、单窗详情、ID 与异常五页；跨 epoch 显式断开，方向角支持仅用于显示的圆周连续展开，异常可分类筛选并跳转到对应单窗。
- 新增公开查询 capability 探测和 v3/v4 标准化：逐窗主键固定为 `WindowKey`，方向轨主键固定为 `(session_id, stream_epoch, track_id)`；未知 schema、坏记录、接口缺失和未封存数据 fail-closed，并分别显示 `N/A / 未记录 / 尚未封存 / 校验失败`，不推断成 0 或正常。
- 新增阶段终态、实际完成 Hz、compute/queue wait/end-to-end p50/p95/p99、样本数与缺失率统计；实际 Hz 只计 `COMPLETED`，分母按各 epoch 完整公开 sample 区间求和。
- Log UI 只能接受宿主注入的现有公开查询 provider，不接受 data root、不构造 `DataManagerService`、不打开 Catalog/SQLite/WAL、不消费 Runtime latest-only 邮箱，也不提供 Runtime/算法/录音/数据修改控件。音频仅在点击播放后调用公开校验资产接口按需读取，界面不展示绝对路径。
- 后台 session 加载支持取消，内存 session 使用有界 LRU，10万窗口级列表按页显示；关闭或加载失败不改变主 Runtime、Test UI、录音或数据管理状态。
- **未改变**：L1、Windowing、L2 MUSIC/ID、L3、L4、Application Runtime、Development Test UI、Production UI、RecordingStore/Catalog及公共数据契约的实现均无变化；模型、配置、音频和精选测试资产无变化。
- **验证**：Pipeline Log UI及其Recording v4公开查询边界聚焦测试 `17 passed`（其中Log UI专属测试13项）；Ruff 和 `git diff --check` 通过；完成 Qt offscreen 五页渲染检查。未进行真实封存 session 的人工回放、10万条真实磁盘记录性能或诊室实机验收。
- Git LFS 管理资产无变化；`data/`、Catalog、录音、日志、缓存、临时文件和本地设置未纳入提交。

---

## 2026-08-19 — L3试听波形显示实时播放进度

- **版本/标签**：Development Test UI试听界面调整；未创建或移动版本标签。
- **类型**：L3音频试听进度可视化。
- Center Mic参考与所有方向音轨的波形缩略图增加橙色竖向播放指示线，直接使用播放器已输出的真实采样位置映射到整段音频，而非按UI刷新次数估算。
- 播放时指示线实时移动，暂停后停留在当前位置；停止、播放结束、切换到320 ms正式预览或切换试听音轨时清除旧指示线并复位。
- L1、L2/MUSIC与权威ID、L3合成和拼接算法、L4、录音及缓存生命周期均无变化；新增播放器采样进度和L3行级进度绑定测试，Git LFS资产无变化。

---

## 2026-08-19 — 清理静音或低声音占比的L3候选试听轨

- **版本/标签**：Development Test UI试听缓存调整；未创建或移动版本标签。
- **类型**：L3候选试听质量过滤与本地缓存清理。
- 方向候选轨封存后，若任意连续静音达到3秒，或按20 ms RMS统计的有声片段占比小于等于30%，立即删除该轨的缓存分段并从Test UI列表移除；RMS有声门限为-50 dBFS。
- 过滤只作用于已由L2删除或因session/mode结束而封存的方向轨；活跃/coasting轨不提前删除，Center Mic参考不参与过滤。
- L2权威ID、MUSIC、L3合成算法、L4、录音及Production UI均无变化；新增30%边界、保留条件和连续静音测试，Git LFS资产无变化。

---

## 2026-08-19 — L2方向表统一行底色并显示1秒L4峰值

- **版本/标签**：Development Test UI界面调整；未创建或移动版本标签。
- **类型**：DOA/MUSIC方向表显示与L4概率聚合。
- 关闭三行表格的交替灰白底色，使固定三行使用相同背景；权威ID文字颜色和圆图颜色映射保持不变。
- `L4概率`按`(session_id, stream_epoch, track_id)`在每个1秒统计周期内累计最大值，只在周期结束时更新一次表格，显示刚结束那1秒的最大概率；切换session/epoch时清除旧统计。
- L1～L4算法、L2权威ID生命周期、录音和试听均无变化；Development Test UI定向测试通过，Git LFS资产无变化。

---

## 2026-08-19 — L2方向表固定三行并消除行重排闪烁

- **版本/标签**：Development Test UI界面调整；未创建或移动版本标签。
- **类型**：DOA/MUSIC方向表稳定显示。
- L2方向表固定为3行；0～3条权威轨迹只更新既有单元格，空余行保持空白，不再随每帧结果增删表格行。
- 单次快照更新期间暂停绘制，全部字段更新完成后统一刷新，减少运行时表格闪烁；MUSIC、权威ID、颜色、L4概率及L1/L3/L4逻辑均无变化。
- Development Test UI定向测试通过；Git LFS资产无变化，本地数据未纳入提交。

---

## 2026-08-19 — Development Test UI隐藏校准状态条

- **版本/标签**：开发分支界面调整；未创建或移动版本标签。
- **类型**：Development Test UI显示精简。
- 删除L1区域的校准状态彩色横条及其运行时文本/样式更新；verified和unverified状态均不再在Test UI占用一行空间。
- 校准状态、版本和哈希仍保留在L1/Runtime/录音数据契约中；L1～L4算法、灯控、录音、试听、Production UI和数据管理均无变化。
- Development Test UI定向测试通过；Git LFS资产无变化，本地数据未纳入提交。

---

## 2026-08-19 — 将当前硬件校准配置标记为verified

- **版本/标签**：开发分支配置更新；未创建或移动版本标签。
- **类型**：硬件校准状态配置。
- `config/config.yaml`中的`hardware_calibration_status`由`unverified`改为`verified`，Development Test UI不再显示“校准UNVERIFIED”警告。
- 本次仅按用户要求修改状态标记，没有重新测量或改变7路麦克风的增益、极性、整数延迟、校准版本及校准哈希；L1采集、L2 MUSIC/ID、L3、L4、录音、数据管理和全部UI逻辑均无变化。
- 配置加载与校准元数据测试通过；Git LFS资产无变化，本地录音、缓存和日志未纳入提交。

---

## 2026-08-19 — Runtime、时间线与公共方向ID跨层集成

- **版本/标签**：项目`1.1.0`迁移集成；未创建发布标签，已发布的`v1.0.1`不移动。
- **类型**：Runtime并行调度、公共DTO与跨层校验、滚动MUSIC状态、DecisionRecord v4、测试与界面契约。
- **涉及文件**：`app/`、`common/`、`config/`、`layer1_input/`、`layer2_source_detection/`、`layer3_direction_signal/`、`layer4_voice_classifier/`、`windowing/`、`gui/`、`data_management/`及对应测试。

### L1与Windowing

- 保留唯一48 kHz sample时间轴、8通道逻辑布局、20 ms发布节拍和`WindowKey(session_id, stream_epoch, window_id, decision_sample)`；DecisionWindow继续提供最多320 ms连续上下文。
- 校准元数据增加verified/unverified状态并随窗口传递；HardwareMix仍不进入L2物理麦定位输入。IMCRA、可选预降噪和既有通道顺序算法无变化。

### L2

- 正式定位主链改为滚动frequency-normalized MUSIC：`1024/960/480` STFT、2～4 kHz频带、逐频协方差、MDL 0～3阶、360点圆周谱和45° NMS；删除Runtime可达的SRP-PHAT与iterative multiple-peak路径。
- L2单worker按session/epoch/绝对sample维护滚动STFT/协方差及预计算导向缓存；连续窗口仅增删新旧帧，sample不连续、epoch/config/calibration变化时安全重建，并发布gap、复用帧、增删帧和导向缓存诊断。
- ID关联改为永久开启的全局一对一分配，公共`TrackedDirection`携带轨迹生命周期、观测/输出角和Kalman状态。同一session跨epoch保持单调ID计数；TTL、coasting和删除按绝对sample推进，Kalman revision只控制平滑而不控制ID存在。

### L3与L4

- L3方向信号、批次、频谱特征和增强音频继承L2的`track_id`、角度、顺序及WindowKey；三种波束模式和信号处理算法无变化。
- L4音频段、检测与重新阈值结果原样保留同一公共ID；CNN模型、48→16 kHz适配、响度补偿及primary/shadow边界无变化。
- 删除angle-only L4→L2反馈；L4不再确认、续租或创建方向ID。

### Runtime、跨层契约与时间线

- 保留L2/L3/L4各自单worker、有界latest-wins队列、分区ComputeCache、跨窗口并行和ResultJoiner按WindowKey有序提交。
- `ProcessingConfigSnapshot`删除iterative和ID-enable语义，冻结MUSIC历史、STFT/频点、MDL、关联生命周期以及独立Kalman/config revision。
- 每个`StageResult`导出有序公共ID/角度对齐信息；Joiner严格拒绝L2 directions、L3 enhanced和L4 detections之间的ID集合、顺序、角度或WindowKey不一致。
- 队列替换、超时、跳窗、epoch变化和停机drain继续为每个已接纳窗口生成唯一终态并推进watermark，不重置同一session的方向ID计数器。
- 正式记录装配升级为`DecisionRecord v4`，保存MUSIC/model-order/配置revision、公共方向生命周期、逐ID增强与L4结果；旧v3仅只读兼容，不原地迁移。

### Development Test UI、Production UI与数据管理

- Development Test UI删除iterative/ID开关和私有角度ID投影，只按L2权威`(session_id, stream_epoch, track_id)`维护试听缓存；Kalman文案明确仅平滑角度。
- Production UI和Catalog/服务增加逐ID时间线、持续时间、角度、L4概率及增强资产查询/试听；L1-only测试录音明确无算法方向ID。既有QA、标注、hash、恢复、Trash和本地数据边界无变化。

### 验证与资产

- 新增或更新跨层ID、WindowKey/顺序/角度拒绝、latest-wins丢弃、sample跳跃、MUSIC滚动重建、配置revision、epoch ID连续、停机drain、唯一终态/watermark、DecisionRecord v4及旧v3只读兼容测试。
- 自动测试：全量`310 passed`；Runtime/MUSIC/记录/UI重点回归`75 passed`；Ruff与`git diff --check`通过。未进行真实麦克风、目标设备p95、长时间录音或诊室多声源实机验收。
- `data/`、实际录音、Catalog、日志、缓存、临时文件和本地配置未纳入提交；Git LFS管理资产无变化。

---

## 2026-08-19 — 完成项目1.1.0分支的L2 Rolling NormMUSIC重构

- **版本/标签**：`feature/l2-music-tracking-v1.1`开发分支；项目`1.1.0`未发布、未创建`v1.1.0`标签，已发布`v1.0.1`不移动。
- **类型**：L2定位主链、公共方向轨迹、运行时/跨层DTO、Test UI诊断、DecisionRecord v4适配及回归测试。
- **涉及文件**：`layer2_source_detection/`、`common/config.py`、`common/data_types.py`、`config/config.yaml`、`app/`、`gui/dev_test_ui/`、L3/L4公共track_id透传、数据管理适配、README和相关测试。

### L1与Windowing

- L1采集、7物理麦音频质量、IMCRA概率/噪声算法及预降噪算法无变化；MUSIC仍只读原有DecisionWindow，不重采样、不修改PCM，第8路HardwareMix不参与定位。
- Windowing继续提供320 ms历史和20 ms决策步进；为Rolling MUSIC保留160/240/320 ms比较配置，当前正式运行候选为240 ms。

### L2

- 正式定位由SRP-PHAT替换为2～4 kHz宽带frequency-normalized MUSIC：多帧STFT、逐频7×7协方差、收缩/对角加载、Hermitian `eigh`、MDL 0～3源估计、跨频一致性及NormMUSIC式逐频归一化融合。
- 连续20 ms窗口只加入两个新增50%重叠STFT帧并移出两个过期帧；session/epoch/sample跳跃时从当前历史重建。导向张量按几何/频率/config revision缓存；伪谱和ID每20 ms更新，MDL最多复用100 ms。
- 0～359°逐度扫描，最多3个候选并执行45°圆周NMS；新增协方差更新、特征分解、谱融合和总耗时诊断。
- 删除iterative multiple peak算法、SRP正式扫描器、运行配置、setter、UI开关和旧专属测试路径；包不再公开旧实现。
- ID追踪永久开启，使用含birth/miss dummy行列的`scipy.optimize.linear_sum_assignment`做确定性全局一对一关联。内部使用unwrapped angle处理359°↔0°，按绝对sample维护tentative/confirmed/coasting/deleted；同一session ID单调且不复用，epoch清轨但不重置session计数。
- Kalman保持独立可选，只平滑同一ID的输出角；运行时切换不重置、创建、删除或改变ID。公共权威输出新增`TrackedDirection`与`active_tracks`，ID明确表示方向轨迹而非人物身份。

### L3、L4与跨层接口

- L3波束形成数学算法和L4 CNN分类算法无变化；输入/输出DTO改为继承L2公共track_id，禁止下游按rank猜测或重新分配ID。
- Runtime、Joiner、Development Test UI和DecisionRecord v4同步保存/校验MUSIC模型阶数、空间谱质量、轨迹状态与Kalman应用状态；L4不再向L2回送角度来改变ID生命周期。

### Development Test UI、录音与数据管理

- 删除iterative与ID enable控件，只保留独立Kalman控制；L2圆环显示MUSIC伪谱和公共方向轨迹，试听缓存按`session + epoch + track_id`拼接，不执行第二套角度关联。
- 正式记录、Catalog和Production UI适配公共轨迹与逐ID增强资产；录音事务、恢复、QA及原始音频格式无算法变化。

### 验证、性能与资产

- 自动测试覆盖0～3源、全角度/跨0°、45°NMS、HardwareMix隔离、滚动增量与全量重建等价、rank交换、birth/miss/短漏检/TTL、Gate关闭、丢窗/sample跳跃、epoch/session、确定性tie-break、Kalman运行切换、跨层ID和DecisionRecord v4。
- 完整自动测试：`310 passed`。本机Rolling MUSIC自动性能门禁满足稳态p95不高于15 ms且单窗低于20 ms；独立100窗CPU基准均值`2.131 ms`。基准输入同步升级为逻辑8通道并验证HardwareMix不参与算法。尚未完成真实麦克风、诊室混响、三声源和长时间目标机实机验收。
- Git LFS资产、CNN模型、精选音频和运行数据无变化；`data/`、录音、Catalog、日志和缓存未纳入提交。

---

## 2026-08-19 — Development Test UI迁移到DOA/MUSIC与权威方向ID

- **版本/标签**：项目`1.1.0`并行迁移分支；未创建发布标签，`v1.0.1`不移动。
- **类型**：Development Test UI、Runtime调试接口、MUSIC可视化与逐ID试听缓存。
- 删除Iterative Multiple Peak和ID追踪开关、持久化键及Runtime setter；旧设置加载时会被清除。保留Kalman开关和Q/R参数，并在界面中明确其只控制方向平滑、不决定ID创建、续存或删除。
- 右上区域改为`DOA / MUSIC`：绘制原始360点归一化MUSIC伪谱，展示model order、有效频点和数值状态；方向表按L2公开`track_id`展示观测角、输出角、score、tentative/confirmed/coasting、新建/观测标志及同ID的L4概率，颜色稳定绑定权威ID。
- 左下试听只按`(session_id, stream_epoch, track_id)`接收L2/L3结果；移除角度贪心关联、formal alias和换号合并。coasting由L2生命周期维护，默认等待3秒后由L2删除并封存；Kalman开关不改变该生命周期。
- 保留Center Mic全采集参考、内部稳定20 ms hop、可恢复真实音频补洞、过旧缺口等时静音、跨hop交叉淡化、至少2秒显示、有界10秒分段/3段保留、三档L3模式隔离，以及关闭窗口删除Test UI缓存。
- 新增/更新控件删除、MUSIC 360点与状态、权威ID字段/L4概率、精确ID拼接、跨0°不换轨、缺口回填、coasting等待/删除封存、模式隔离及Center参考测试。
- **未由本UI子变更调整**：L1采集/IMCRA/录音控制算法、MUSIC/MDL数值算法、L3波束形成算法、L4模型推理、Production UI和RecordingStore事务规则；这些1.1前置契约的并行变更另行记录。
- 验证：Development Test UI定向测试`31 passed`；配置/Runtime/UI重点回归`78 passed`；集成工作树全量测试`310 passed`，`git diff --check`通过。未进行真实阵列、声卡播放或诊室实机验收。
- `data/`、运行录音、试听缓存、日志和本地设置未纳入提交；Git LFS资产无变化。

---

## 2026-08-19 — Recording/Data/Production UI迁移到DecisionRecord v4

- **版本/标签**：项目`1.1.0`并行迁移分支；未创建发布标签，`v1.0.1`不移动。
- **类型**：录音schema、事务资产、Catalog/服务查询、Runtime记录适配与Production UI。
- **涉及文件**：`data_management/contracts.py`、`data_management/recording_store.py`、`data_management/timeline.py`、`data_management/catalog.py`、`data_management/service.py`、`data_management/corpus_store.py`、`app/runtime.py`、`gui/production_ui/*`、相关README与测试。

### L1

- 采集、通道映射、IMCRA和预降噪算法无变化。
- 专用L1测试录音manifest和向导明确显示“无算法方向ID”，不从角度、声源序号或模拟结果伪造ID。

### L2

- MUSIC、MDL、全局方向追踪与Kalman算法实现无变化；本分支只冻结并消费其v4持久化字段。
- DecisionRecord v4可保存MUSIC算法版本、model order、有效频点/协方差诊断、公共track_id、观测角/输出角、轨迹状态、active_tracks和Kalman应用状态。

### L3与L4

- 波束形成和CNN推理算法无变化。
- L3增强资产文件名、事务journal和manifest索引加入track_id；L4逐ID概率与判断进入v4结果和Catalog投影。同窗重复或跨层错序ID被拒绝。

### Runtime、录音与数据管理

- 新录音结果写`decision_record_v4`；配置与校准revision/version/hash随session和窗口保存。旧v3结果通过只读读取器兼容，不原地改写、不生成公共ID。
- Catalog新增按`session + epoch + track_id`索引的方向观测表，服务可查询轨迹摘要、持续时间、完整角度时间线、L4概率、逐ID增强资产和native/logical/physical资产。
- 增强音频事务升级为`enhanced_asset_commit_v2`，恢复继续把manifest未完整索引的partial、已改名final和journal送入quarantine，避免逐ID文件覆盖或半提交。

### Production UI

- 运行录音详情增加方向ID、epoch、首末sample、持续时间、首末角、角度变化、状态和最新L4概率。
- 增加逐ID连续增强试听、Center参考，以及native/logical/physical任意通道试听；逐ID播放器按决策sample只拼接新增hop，去除320 ms窗口重叠并对缺口补等时静音。
- QA、标注、hash、Catalog重建、恢复、Trash、模拟测试和后台任务边界保持。

### Development Test UI

- 无界面或算法行为变化。

### 验证与资产

- 新增DecisionRecord v4对齐、旧v3只读、逐ID文件防覆盖、Catalog/服务查询、增强事务恢复、页面展示/试听、Center参考、重叠去除和L1-only无ID测试。
- 本分支相关自动测试`83 passed`，Ruff与Git差异检查通过，并完成Production UI运行录音页离屏渲染检查。全仓库并行验证为`282 passed, 12 failed`；12项均属于尚未完成的L2/Test UI迁移测试（旧开关参数或缺少公共ID的测试桩），不属于本分支新增测试。未进行真实麦克风、长时间录音或诊室实机验收。
- `data/`、实际录音、Catalog、日志、缓存和临时文件未纳入提交；Git LFS资产无变化。

---

## 2026-08-19 — 规划与主链平行的独立只读 Pipeline Log UI

- **版本/标签**：当前项目仍为`1.0.1` / `v1.0.1`；本次把 Log UI 纳入下一目标版本`1.1.0`，未创建`v1.1.0`标签。
- **类型**：架构与界面规划；仅文档变化，无运行代码、配置或数据schema实现变化。
- **涉及文件**：新增`LOG_UI_ARCHITECTURE_V1.1_TARGET.md`；更新`ARCHITECTURE_V1.1_TARGET.md`、根`README.md`、`PROJECT_FILE_CLASSIFICATION.md`和`CHANGELOG.md`。

### L1、L2、L3与L4

- 明确 Log UI 与 L1～L4 平行，不是 Layer 5，不插入、控制、消费或反压 `L1 → L2 → L3 → L4` 实时处理链。
- L1采集/校准/IMCRA、L2 Gate/SRP/ID/Kalman现有实现、L3波束形成、L4分类、跨层DTO与现有测试均无变化。

### Development Test UI

- 明确 Log UI 是独立观察与回放子系统，不是 Development Test UI 的面板。
- Log UI 禁止消费`latest_dev_ui`、`latest_l4_dev_ui`等读取即移除的latest-only邮箱，避免抢走正式UI帧或改变被观察系统。
- Development Test UI的界面、试听、Runtime控制、设置和测试均无变化。

### Pipeline Log UI

- 新增1.1.0权威目标文档，定义“公开只读接口 → 标准化读模型 → 统计引擎 → UI”的独立结构。
- 规划五个页面：记录列表、会话总览、Pipeline时间线、单窗详情、ID与异常；以`WindowKey`对齐逐窗数据，以`(session_id, stream_epoch, track_id)`对齐方向轨、L3资产和L4结果。
- 统一阶段数量、实际完成Hz、p50/p95/p99、缺失率和方向ID统计口径；只有`COMPLETED`计入实际完成频率，`SKIPPED/DROPPED/TIMED_OUT/FAILED/CANCELLED`分开显示。
- 实际完成Hz以所选epoch/时间范围的完整权威观测区间为分母，包含首尾和非完成窗口；不能用首末completed sample简单相减，跨epoch按有效时长合并。
- 规划v3/v4 capability适配、未知schema fail-closed、十万窗有界加载、按需音频和严格只读验收；接口未提供的数据必须显示`N/A`，不能推断为零。
- 第一版定位为完成/封存session的离线回看，加可选同进程`processing_status`聚合概览；当前尚无公共跨进程逐窗事件流，不通过内部队列绕过限制。
- **本次实现状态**：未新增Log UI程序、目录、依赖、配置、启动入口或自动测试，不能将规划描述为已实现UI。

### 音频录制、数据管理与Production UI

- Log UI 只读取未来稳定的公开查询能力，不调用标注、导出、删除、恢复、Catalog重建或其他写接口。
- 记录当前1.0.1公共服务只能列出runtime sessions，尚不能完整公开回看单个session；`DataManagerService`构造Catalog会创建/初始化SQLite/WAL，因此不能作为严格零写入读取方式。目标实现只能使用显式只读端口，或由正式公共接口生成的版本化只读快照/流；Log UI不得自行复制、打开或解析Catalog文件。
- RecordingStore、Catalog、manifest、录音格式、恢复、Audio Data Manager和Production UI代码均无变化；本次未读取、复制或提交任何运行录音和本地`data/`内容。

### Runtime、接口、配置与兼容性

- 规划可选同进程Live只在外部宿主能够注入现有Runtime只读引用时轮询公开`processing_status`聚合状态；在不修改主项目的独立进程范围内该能力仍延期。Log UI不得启动/停止Runtime、修改参数或成为ResultJoiner/录音提交的依赖。
- 当前Runtime、队列容量、ResultJoiner、公开接口、DecisionRecord、配置schema和版本兼容代码均无变化。

### 测试与资产

- 新文档规定v3/v4、缺字段、坏记录、完整阶段终态、跨epoch、`359° ↔ 0°`、统计公式、资产校验、严格只读性和十万窗口性能门禁。
- 封存静态fixture验证文件hash、Catalog行数和schema前后完全一致；Live场景通过调用审计与对照运行证明Log UI不消费邮箱、不调用写接口且不引入额外状态变化，不要求自然变化的运行队列或WAL字节静止。
- 本次为纯Markdown规划，不运行pytest；以Markdown本地相对链接检查、`git diff --check`、最终Git差异和暂存文件范围检查验收。
- 自动测试源码、精选测试音频、模型、阵列资产及其他二进制文件均无变化。

### Git与Git LFS

- 仅提交上述五个Markdown文件，不提交并行功能分支、运行数据、Catalog、日志、缓存或临时文件。
- Git LFS资产无变化；不创建、移动或重写任何发布标签。

---

## 2026-08-19 — 扩展IMCRA统计与预降噪至0～8000 Hz

- **版本/标签**：项目仍处于`1.1.0`分支迁移阶段；未修改项目版本，未创建或移动发布标签。
- **类型**：L1 IMCRA输出契约、Wiener预降噪频带、录音sidecar契约、测试与文档。
- **涉及文件**：`common/config.py`、`common/data_types.py`、`config/config.yaml`、`layer1_input/imcra.py`消费的配置、`layer1_input/pre_denoise.py`、L1/数据管理/架构文档、基准脚本及相关测试。

### L1

- IMCRA仍按7个物理麦分别计算，但发布和宽频噪声统计范围由80～8000 Hz扩展为0～8000 Hz；2048点RFFT频率轴由338点变为342点，算法版本升级为`cohen_imcra_2003_l1_v2`。
- Gate使用的`mean_spp`证据带保持500～4000 Hz，因此直流和新纳入的低频bin不会直接改变L2 Gate聚合频带。
- Wiener预降噪改为对0～8000 Hz复数STFT系数乘每麦实数增益，再经IRFFT和40 ms/20 ms平方根Hann WOLA恢复时域音频；8000 Hz以上、HardwareMix和native音频保持直通。算法版本升级为`imcra_wiener_wola_v2`。

### Windowing、L2、L3与L4

- WindowAssembler、DecisionWindow大小/节拍和滚动历史契约无变化；L2 MUSIC、Gate阈值、方向ID与Kalman无变化。
- L3、L4算法和处理频带无变化；仅测试fixture适配新的IMCRA 342点输入轴。

### Development Test UI、录音与数据管理

- Development Test UI行为与布局无变化，继续显示由L1发布的噪声摘要。
- IMCRA录音sidecar的频谱数组从`[record,7,338]`变为`[record,7,342]`，manifest继续从实际轴写入`frequency_bin_count`；其他录音schema、Catalog、事务和恢复行为无变化。

### 测试、资产与验收状态

- 增加0 Hz起点、342点频率轴和预降噪0～8000 Hz掩码覆盖测试；同步配置、数据管理、L3 fixture与生产采集契约测试。
- 聚焦验证`104 passed`，新增DC频点回归单测`4 passed`，全量自动测试`358 passed`；本次改动文件Ruff检查和`git diff --check`通过。未进行真实硬件听感、低频噪声抑制或诊室验收。
- 无模型、音频或其他Git LFS资产变化，本地录音和数据目录不进入提交。

---

## 2026-08-19 — 完成1.1.0的L1与Windowing输入准备

- **版本/标签**：面向项目`1.1.0`的分支准备；未修改项目版本，未创建或移动`v1.1.0`及任何已发布标签。
- **类型**：L1校准公共契约、Windowing滚动输入契约、配置、Development Test UI、测试与文档。
- **涉及文件**：`common/config.py`、`common/data_types.py`、`layer1_input/`、`ingest/coordinator.py`、`windowing/assembler.py`、`config/config.yaml`、Runtime校准hash适配、Development Test UI L1状态、相关测试、README与`ARCHITECTURE_V1.1_TARGET.md`。

### L1

- 保留48 kHz、8通道逻辑顺序、20 ms发布节拍、按7个物理麦独立更新的Cohen 2003 IMCRA和可选预降噪；采集、IMCRA参数与音频处理算法无变化。
- 新增不可变`CalibrationMetadata`及未来资产身份，明确传播`verified/unverified、version、calibration_hash、correction_model、report_hash`，并为亚采样延迟和频率响应校准预留`uri/version/sha256`边界。
- 当前增益、极性和整数sample delay继续生效；尚未实现的未来资产配置会显式拒绝，避免被静默忽略。规范化校准配置hash变化触发新epoch，同一epoch内校准身份变化被拒绝。
- `IngestedAudioBlock`稳定向下游提供连续、校准后的7路物理麦，同时保留第8路HardwareMix用于显示/录制；L1未增加、创建、保存或解释方向ID。

### Windowing与L2输入边界

- `DecisionWindow [15360,8]`和每960 samples/20 ms发布一次保持不变，继续提供最多320 ms历史。
- 新增只含7路物理麦的`physical_samples`和`physical_history(160|240|320)`，HardwareMix不能通过该接口进入MUSIC；新增按session、epoch和decision sample定义的滚动状态键、最近20 ms更新起点及连续后继检查。
- 配置增加`layer2.music.context_ms`三档选择、固定`160/240/320 ms`比较集合和320 ms历史上限。现有SRP配置适配器显式忽略该准备字段，正式MUSIC/STFT/协方差算法和性能默认值仍待L2分支与目标机基准完成。
- WindowAssembler不创建STFT、MUSIC结果或方向ID；现有Probability Gate的两个20 ms IMCRA概率对齐语义保持不变。

### Development Test UI、Runtime与Production UI

- Development Test UI的L1状态增加校准状态、版本和hash摘要；`unverified`显示明确红色警告，`verified`显示绿色状态。
- Runtime与Production采集主机统一使用规范化校准配置hash，避免不同JSON序列化路径产生不同身份；调度、队列、正式定位启动策略及Production UI布局无变化。

### L2、L3、L4、录音与数据管理

- L2的SRP-PHAT、Gate、候选、私有追踪和Kalman算法无变化；仅增加未来MUSIC有效历史的配置/输入边界，不实现MUSIC或公共方向ID。
- L3波束形成、L4分类、跨层公共DTO、DecisionRecord版本、RecordingStore schema、Catalog、录音文件和数据恢复无变化。

### 测试、资产与验收状态

- 增加连续任意切块、epoch/校准hash重置、同epoch校准拒绝、verified/unverified传播、8通道映射、HardwareMix隔离、160/240/320 ms物理历史和滚动后继契约测试，并覆盖UI校准警告字段。
- 聚焦验证`108 passed`，全量自动测试`357 passed`；本次改动文件Ruff检查和`git diff --check`通过。全仓Ruff仍报告既有`layer2_source_detection/__init__.py`的12项E402，本分支未修改该文件。真实硬件校准、160/240/320 ms MUSIC性能比较和诊室实机门禁未在本分支执行，不能视为1.1.0正式验收。
- 未修改模型、精选音频或其他二进制资产；无Git LFS对象变化，本地录音和数据目录不进入提交。

---

## 2026-08-19 — 按改动范围选择测试验证级别

- **版本/标签**：`1.0.1`之后的工程工作流维护；未创建新发布标签，`v1.0.1`不移动。
- **类型**：工程规范与测试流程。
- **涉及文件**：`AGENTS.md`、`CHANGELOG.md`。

### L1、L2、L3与L4

- 算法、接口、模型、配置和现有测试均无变化。

### Development Test UI

- 界面、Runtime控制和测试无变化。

### 音频录制与数据管理

- 录音格式、Catalog、manifest、事务、恢复和Production UI均无变化。

### 跨层接口、配置与兼容性

- DTO、shape、dtype、WindowKey、配置schema和版本兼容性均无变化。

### 测试与验收

- 新增按影响范围选择验证级别的项目规则：单一功能改动只运行直接相关的单元、集成和契约测试；跨层架构、公共接口、多消费者配置、Runtime并发时间轴、录音事务恢复、构建发布或大范围重构才运行完整测试。
- 当影响边界不明确、相关测试暴露外溢或相邻模块失败时，逐级扩大到相邻测试及完整套件。
- 本次仅修改工程规则与变更日志，不运行pytest；以`git diff --check`和最终Git差异检查验收。

### Git与Git LFS

- 仅普通文本工程规则和本变更日志发生变化；Git LFS资产无变化，未创建或移动版本标签。

---

## 2026-08-19 — 第二轮精简并合并自动化测试

- **版本/标签**：`1.0.1`之后的测试维护；未创建新发布标签，`v1.0.1`不移动。
- **类型**：测试重构与维护。
- **涉及文件**：`layer1_input/tests/test_layer1.py`、`tests/test_benchmark_l3_l4.py`、`tests/test_l2.py`、`tests/test_dev_ui.py`、`tests/test_data_management.py`、`CHANGELOG.md`。

### L1

- 采集、解码、校准、IMCRA、预降噪和时间轴行为无变化。
- 删除只重复检查`DecodedAudio`简单属性的DTO冒烟测试；解码、Pipeline、校准和窗口测试继续覆盖正式8通道契约。

### L2

- Gate、SRP-PHAT、三候选、方向ID和圆周卡尔曼行为无变化。
- 非有限配置参数从每字段重复测试`inf/nan`精简为每字段一个代表值；两个字段的有限性校验仍分别覆盖。

### L3与L4

- 算法、缓存、模型、接口和测试无变化。

### Development Test UI

- 界面和Runtime控制行为无变化。
- 将6个独立设置持久化测试合并为完整往返测试与非法输入测试，继续覆盖L2阈值、迭代搜索、ID/卡尔曼开关、Q/R倍率、Gate阈值及L1预降噪，且验证字段互不覆盖。

### 音频录制与数据管理

- 录音格式、Catalog、manifest、队列、事务和恢复实现无变化。
- 删除把partial恢复、result overflow和10000行查询耗时混在一起的重复测试；journal/open-session恢复和多处overflow测试继续保留，并移除依赖机器速度的0.5秒断言。

### 跨层接口、配置与兼容性

- DTO、shape、dtype、WindowKey、配置schema、V1回退和Layer 2 1.1兼容性均无变化。
- 删除只验证平均值/P95算术的benchmark测试；自动设备与正式Runtime配置一致的契约测试继续保留。

### 测试与验收

- 测试节点由355项降至346项，共减少9项；相关测试112项通过，完整自动测试346项通过。
- Ruff和`git diff --check`通过。
- 未进行新的麦克风、灯控、三声源诊室或长时间实机验收。

### Git与Git LFS

- 仅普通文本测试和本变更日志发生变化；Git LFS资产无变化，未创建或移动版本标签。

---

## 2026-08-19 — 确认项目1.1.0 MUSIC与公共方向ID目标架构

- **版本/标签**：当前项目仍为`1.0.1` / `v1.0.1`；本次仅确认下一目标版本`1.1.0`，未创建`v1.1.0`标签。
- **类型**：架构研究、目标契约与迁移规划；无算法代码、运行配置或数据schema实现变更。
- **涉及文件**：新增`ARCHITECTURE_V1.1_TARGET.md`；更新根README、v0.3历史架构说明、L1/L2/L3/L4、Windowing、Runtime、Development Test UI、Production UI、数据管理、项目文件分类等README/索引。

### L1与Windowing

- 规划保留320 ms DecisionWindow作为可用历史上限，同时让MUSIC维护滚动STFT/协方差，每20 ms只加入新帧并移出过期帧；有效历史在目标机比较160/240/320 ms后确定，避免逐窗从头重算320 ms导致积压或丢窗。
- 规划预计算导向张量、批量7×7特征分解和向量化伪谱；360°伪谱与ID每20 ms更新，MDL最多沿用100 ms且在质量变化时提前刷新。L2初始p95预算15 ms、硬门限20 ms。
- 明确增加校准verified/unverified状态，并为未来亚采样或频率相关校准预留版本化资产边界；L1不创建方向ID。
- **本次实现状态**：L1采集、IMCRA、预降噪、校准代码、WindowAssembler和运行配置均无变化。

### L2

- 规划以多帧STFT、频点协方差、MDL声源数估计和frequency-normalized MUSIC替换SRP-PHAT正式主链，保持0～359°逐度扫描、最多3候选及45°圆周NMS。
- 规划删除iterative multiple peak算法路径、配置与界面开关；多源搜索统一由MUSIC完成。
- 规划将方向ID追踪改为永久开启的公共能力，使用带birth/miss dummy项的全局一对一线性分配，内部采用unwrapped angle正确处理`359° ↔ 0°`，并按绝对sample管理tentative/confirmed/coasting/deleted生命周期。
- 规划同一session内ID单调且不复用；短时漏检恢复原ID，超过TTL后的同方向观测分配新ID。Kalman保持独立可选，开关不得改变或重置ID。
- 明确ID表示方向轨而非人物身份；相同方向或轨迹交叉时不承诺真实说话人身份连续。
- 研究依据记录Schmidt MUSIC、Pyroomacoustics MUSIC/NormMUSIC、Wax/Kailath MDL、CSSM候选、Israel Cohen公开论文/反馈定位资料及SciPy线性分配接口；未发现可直接替换L2的Israel Cohen MUSIC开源代码，文档未虚构来源。
- **本次实现状态**：SRP-PHAT、iterative、现有可选私有ID、现有穷举关联和Kalman代码均无变化。

### L3

- 规划输入改为公共`TrackedDirection`，方向信号、波束批次和增强音频按`(WindowKey, track_id)`精确对齐；L3不得分配、猜测或合并ID。
- optimized、ds_baseline和constant_beamwidth_baseline三档保留。
- **本次实现状态**：L3公共类型、波束形成算法、模式和测试均无变化。

### L4

- 规划在音频段、VoiceDetection和阶段结果中贯通`track_id`，并严格继承L3顺序和角度。
- 规划删除按角度向L2回送ID正式化/语音租约的路径；L4只消费和标注方向轨，不拥有ID生命周期。
- **本次实现状态**：CNN、响度补偿、重采样、阈值、反馈与公共DTO代码均无变化。

### Runtime、时间线与并行管理

- 保留唯一WindowKey、L2/L3/L4分层单worker、有界latest-wins、跨窗并行和ResultJoiner有序提交。
- 规划删除iterative/ID enable配置快照，增加MUSIC、模型阶数、关联生命周期和Kalman revision，并在Joiner中校验跨层ID集合/顺序。
- 规划将正式记录升级为DecisionRecord v4，旧v3保持只读兼容；丢窗、跳窗、epoch和session边界均按绝对sample及完整track key处理。
- **本次实现状态**：Runtime、队列、缓存、Joiner、DecisionRecord v3和配置schema均无变化。

### Development Test UI

- 规划删除iterative和ID追踪开关，保留只控制平滑的Kalman开关；L2面板改为MUSIC伪谱、模型阶数和公共ID诊断。
- 规划试听只按L2权威`(session_id, stream_epoch, track_id)`拼接，删除UI角度贪心、别名和换号补救；保留Center参考、稳定hop、补洞、淡化、2秒显示、3秒等待、有界缓存和L3模式隔离。
- **本次实现状态**：现有Test UI控件、私有ID投影和试听sidecar代码均无变化。

### 音频录制、数据管理与Production UI

- 规划DecisionRecord v4保存MUSIC诊断、公共ID、active tracks、逐ID L3资产与L4结果；增强文件名和Catalog查询使用track ID。
- 规划运行录音详情增加逐ID时间线、持续时间、状态、概率和增强音频试听；L1-only测试录音明确不含算法ID。
- 本地`data/`、运行录音、Catalog、日志和缓存继续不上传GitHub。
- **本次实现状态**：RecordingStore、manifest、Catalog、恢复、Production UI、录音格式和本地数据均无变化。

### 测试、资产与兼容性

- 新文档规定MUSIC 0～3源、跨0°、45°边界、校准/秩异常、全局关联、新ID/短漏检/TTL、Kalman切换、跨层ID、DecisionRecord v4、UI试听、性能与真实阵列验收门禁。
- **本次实现状态**：自动测试源码、精选测试音频、CNN模型及其他二进制资产无本次文档任务所作变化；1.1.0功能尚未实现或验收。
- **验证结果**：本次为纯文档规划，按项目验证范围不运行pytest；本地Markdown相对链接检查通过，`git diff --check`通过。

### Git与Git LFS

- 本次只提交规划与README文档，不提交本任务范围外的工作区修改、运行数据或临时文件。
- Git LFS资产内容无变化；不创建或移动发布标签。

---

## 2026-08-19 — 发布项目1.0.1与Layer 2 1.1

- **版本/标签**：项目`1.0.1` / `v1.0.1`；Layer 2公开版本`1.1`。
- **类型**：功能、跨层接口、界面、文档与正式版本发布。
- **涉及文件**：`pyproject.toml`、`layer2_source_detection/`、`layer3_direction_signal/`、`app/runtime.py`、`common/config.py`、`config/config.yaml`、`gui/dev_test_ui/`、根README/规格及相关测试。

### L1

- 采集、8通道映射、IMCRA、预降噪和唯一采样时间轴无变化。

### L2 1.1

- 层的公开版本名称由开发阶段“V2”统一为“Layer 2 1.1”；内部`confidence_id_tracker_v2`、`damped_circular_kalman_v2`名称保留为配置兼容标识。
- 正式公开候选由最多2个扩展为最多3个，任意两点继续满足45°圆周最小间距；Runtime、DTO、配置和诊断同步执行0～3候选契约。
- 新的置信度ID追踪在公开候选筛选前维护最多4条内部轨迹，并按持续性、SRP分数和L4语义可信度排序。
- 临时ID确认观察期调整为2秒；首次出现立即分配临时ID，满足短时重复匹配后才进入卡尔曼持续跟踪。
- 阻尼圆周卡尔曼加入角速度半衰期、最大角速度和预测不确定度冻结参数，减少漏测期间的方向漂移。
- L4正、负分类结果均可回送作为内部语义证据；非人声证据不会隐藏L2角度，人声证据会清除既有负面语义并可用于正式化/租约续期。

### L3

- 输入候选上限同步扩展为3。
- 0～2候选保持原波束形成策略；3候选采用逐方向Loaded MVDR，单路失败时独立回退DAS，避免三约束病态求解。

### L4

- CNN模型、48→16 kHz内部适配、响度补偿和公共输出契约无变化。
- Runtime支持将最多3个同窗检测结果按原候选顺序送回L2，并严格校验数量与角度对齐。

### Development Test UI

- L2候选显示扩展为最多3个正式方向；正式ID增加第三种稳定颜色。
- 首次候选显示为灰色小点，Kalman-ready临时ID即可开始L3试听缓存，转正式后沿用同一缓存。
- SRP诊断补充实际候选上限，相关面板、状态DTO和文档同步0～3候选语义。

### 音频录制与数据管理

- RecordingStore、正式录音、Test Corpus、Production UI、manifest、Catalog、恢复和音频资产代码均继续纳入本次完整项目发布并上传GitHub。
- 本次未改变录音格式、存储路径或数据schema；本地运行录音和`data/`仍按安全边界不上传。

### 跨层接口、配置与兼容性

- 项目包版本更新为`1.0.1`；`layer2_source_detection.LAYER2_PUBLIC_VERSION`和`__version__`固定为`1.1`。
- `layer2.max_candidates`与`runtime.max_candidate_batch`更新为3；V1方向后处理后端继续保留为回退选项。
- L2/L3/L4仍使用同一`WindowKey`和有界流水线，内部私有ID不进入正式录音或公共候选DTO。

### 测试与验收

- 发布整理前完整自动测试为355项通过。
- 新增/更新三候选、L2 1.1方向后处理、三路L3增强、L4语义反馈、Test UI颜色与试听缓存测试。
- 未进行新的真实麦克风、三声源诊室或长时间实机验收。

### Git与Git LFS

- 当前完整代码、配置、文档、测试以及已跟踪的录音存储/数据管理系统均纳入普通Git发布。
- CNN权重、精选测试音频和大型数组继续使用Git LFS；本次LFS资产内容无变化。
- `.venv/`、`data/`、日常录音、Catalog、日志、缓存和本地代理继续排除。

---

## 2026-08-19 — 精简自动化测试套件

- **版本/标签**：`1.0.0`之后的测试维护；未创建新发布标签。
- **类型**：测试重构与维护。
- **涉及文件**：`tests/test_l2.py`、`tests/test_parallel_config_and_docs.py`、`tests/test_l1_v03.py`、`layer1_input/tests/test_layer1.py`、`tests/test_spatial_separability_table.py`、`tests/test_wizard_usability.py`。

### L1

- 无采集、通道映射、校准、IMCRA或预降噪行为变化。
- 删除未被正式路径调用的旧physical映射单测，以及与L2精确坐标契约重复的几何方向单测；正式logical 8通道映射与7麦精确坐标测试继续保留。

### L2

- 无Gate、SRP-PHAT、候选、ID追踪或卡尔曼算法变化。
- 删除11个已被0～359°逐度SRP测试完整覆盖的抽样角度参数用例；保留完整逐度、公开`scan()`契约、噪声精度及分支现有方向后处理测试。

### L3与L4

- 无算法、接口、模型或测试变化。

### Development Test UI

- 无界面或运行行为变化；删除仅锁定README固定措辞的测试，正式流水线状态和UI行为测试继续保留。

### 音频录制与数据管理

- 无录制格式、Catalog、manifest、恢复或Production UI行为变化。
- 删除只检查文档固定措辞的测试，以及未实际检查录音时长字段的重复向导合法输入测试；结构化向导校验与录音事务测试继续保留。

### 跨层接口、配置与兼容性

- DTO、shape、dtype、时间字段、配置schema和兼容后端均无变化；并行Runtime配置契约测试继续保留。

### 测试与验收

- 共精简20个重复或脆弱的测试节点，分支完整套件由370项降至350项。
- 独立干净分支快照的完整自动测试350项通过；Ruff与`git diff --check`通过。
- 未进行真实麦克风、灯控或诊室声学实机验收。

### Git与Git LFS

- 仅普通文本测试与本变更日志发生变化；Git LFS资产无变化，未创建或移动版本标签。

---

## 2026-08-19 — 录制向导改为结构化环境与逐声源信息

- **版本/标签**：`1.0.0`之后的录音元数据界面更新；未创建新发布标签。
- **类型**：Production UI、录音元数据契约与测试。
- **涉及文件**：`gui/production_ui/app.py`、`gui/production_ui/README.md`、`data_management/wizard.py`、`data_management/contracts.py`、`data_management/dedicated_recording.py`、`data_management/corpus_store.py`及相关测试。

### L1、L2、L3、L4与Development Test UI

- 无算法、实时处理或Test UI行为变化；不影响正在独立开发的L2三候选输出。

### 音频录制与数据管理

- 测试录制向导不再要求填写音频名称和自由备注，改为填写环境、数字声源数量、每个声源各自的类型与移动方式，以及噪音来源。
- 声源数量变化时动态生成或移除逐声源输入行；声源数量为0时可用于只录制环境噪音。
- 列表与模拟输入显示名称由环境、声源数量和录制时间自动生成。
- `labels.json`升级为`test_recording_labels_v3`，manifest同步保存环境、逐声源类型、逐声源移动方式和噪音来源。

### 验证

- 增加动态逐声源表单、字段映射、结构化labels/manifest及校验回归测试。
- 未进行真实麦克风实机录制验收；Git LFS资产无变化。

---

## 2026-08-19 — 建立统一变更日志与强制维护门禁

- **版本/标签**：`1.0.0`之后的仓库管理提交；未创建新发布标签。
- **类型**：文档与版本治理。
- **涉及文件**：`CHANGELOG.md`、`AGENTS.md`、`README.md`。

### L1

- 无算法或接口变化。

### L2

- 无算法或接口变化。

### L3

- 无算法或接口变化。

### L4

- 无算法或接口变化。

### Development Test UI

- 无界面或运行行为变化。

### 音频录制与数据管理

- 无录音格式、Catalog、manifest、恢复或管理界面变化。

### 工程与版本管理

- 新增本文件，统一记录L1～L4、Test UI和音频管理系统的逐次具体变化。
- 项目级Codex规则新增提交门禁：任何项目修改在验证、提交和上传GitHub前必须同步本文件。
- README增加权威变更日志入口。

### 验证

- 文档与打包契约测试通过后提交。
- Git LFS资产无变化。

---

## 2026-08-19 — 允许在启动采集前独立控制阵列灯光

- **版本/标签**：`1.0.0`之后的修复；未创建新发布标签。
- **提交**：`582238b6d2c7089012b14522cfd7861188156896`。
- **类型**：Test UI与硬件控制修复。
- **涉及文件**：`app/runtime.py`、`gui/dev_test_ui/app.py`、`tests/test_dev_ui.py`、`tests/test_runtime.py`。

### L1

- 音频采集、通道映射、IMCRA和预降噪无变化。
- CDC灯控端口允许在音频采集尚未启动时按需打开；Runtime关闭时会释放这一独立打开的控制端口。

### L2、L3、L4

- 无算法、接口或配置变化。

### Development Test UI

- “灯光开/灯光关”不再依赖采集运行状态，未启动采集时也可操作。
- 灯控命令增加Pending、commanded和Error状态反馈；短写或设备异常会明确显示失败。

### 音频录制与数据管理

- 无变化。

### 验证

- 增加采集前灯控、异常状态及Runtime关闭CDC端口的回归测试。
- Git LFS资产无变化。

---

## 2026-08-19 — GitHub持久化工作流

- **版本/标签**：`1.0.0`之后的仓库管理提交。
- **提交**：`32a627255a8813ccab9626c222d7c19576c3bf2f`。
- **类型**：仓库安全和自动提交规则。
- **涉及文件**：`AGENTS.md`。

### L1～L4、Test UI、音频录制与数据管理

- 无功能、算法或公共接口变化。

### 工程与版本管理

- 规定完成项目修改后必须检查差异、验证、提交并上传私有GitHub仓库。
- 规定模型、精选测试音频和大型数组继续由Git LFS管理。
- 规定`.venv/`、`data/`、日常录音、Catalog、日志、缓存、密钥和本地代理不得上传。
- 规定GitHub仓库不得删除，已发布历史与标签不得改写。
- 规定本地项目内容如需删除必须进入Windows回收站。

### 验证

- 提交已推送到`origin/main`，远端提交哈希核对一致。
- Git LFS资产无变化。

---

## 2026-08-19 — 版本1.0.0首次云端发布

- **版本/标签**：`1.0.0` / `v1.0.0`。
- **提交**：`c809c364421c6be40431f14a4bc16bfe2a534642`。
- **类型**：首个完整、可恢复的项目基线。

### Layer 1

- 固定48 kHz原生8通道输入，Host通道映射为`MIC0..MIC5、HardwareMix、Center`，逻辑顺序统一为`MIC0..MIC5、Center、HardwareMix`。
- 前7路作为具有物理坐标的麦克风阵列；HardwareMix只用于接口、显示、录制和实验，不参与几何、SRP或波束形成。
- 统一麦克风面坐标：Center为原点，MIC0为`+x/0°`，从麦克风面俯视逆时针增加。
- 建立采集、解码、校准、连续性检查及不可变8通道音频契约。
- 将IMCRA噪声估计移入L1，按20 ms更新7麦80～8000 Hz噪声PSD、SPP、SNR和诊断特征；从500～4000 Hz聚合阵列声源概率。
- 提供可切换IMCRA-Wiener预降噪：40 ms窗、20 ms步长、50% WOLA；开启后等待重建完成并替换下游7路物理音频，HardwareMix与native音频保持原样。
- epoch切换、断流和预热状态具有明确状态与清理规则。

### Layer 2

- 使用同一DecisionWindow末尾两个20 ms概率的平均值作为40 ms Probability Gate输入，阈值来自唯一配置并支持Test UI动态调整。
- Gate开启后执行二维远场SRP-PHAT 360°扫描；当前定位频带为2000～4000 Hz。
- 完成空间响应稳健归一化、圆周局部峰、阈值、prominence和圆周NMS。
- 公共输出最多2个候选方向，双候选最小圆周角距为45°；完整360点SpatialResponse仍保留诊断。
- 增加可选内部方向ID关联与圆周卡尔曼平滑；ID仅在L2内部管理，公共CandidateDirection保持原契约并输出平滑角度。
- 修复跨epoch、候选生命周期及方向平滑状态的隔离和重置问题。

### Layer 3

- 公共输入统一为同一320 ms、48 kHz、8通道DecisionWindow和L2候选角度；波束形成只读取前7个物理麦。
- 公共输出改为每候选一条48 kHz、15360点单声道EnhancedAudio，不再把`[33,169]`特征作为跨层输出。
- 提供optimized、DS baseline及constant-beamwidth实验入口；optimized按空间可分度在Dual LCMV、加载MVDR与DAS回退之间选择。
- 新增两阶段`prepare/process_prepared`接口，使滚动STFT、IMCRA统计和角度相关BF可以在保持同窗依赖的同时参与跨窗口流水。
- 相邻窗口复用29/33个STFT帧，仅重算新增帧；协方差、噪声统计、频率轴、窗、mask、steering和空间可分度查询均采用有界缓存。
- Prepared GPU上下文、steering及查询缓存均有硬容量，epoch、配置、几何或连续性变化时整体失效。

### Layer 4

- 输入改为L3输出的48 kHz增强音频及同窗IMCRA概率，不再接收旧`[33,169]`跨层特征。
- 对CNN输入副本执行概率控制的受限响度补偿和峰值保护，不修改正式L3增强音频。
- 内部降采样到16 kHz并使用NVIDIA Frame VAD Multilingual MarbleNet输出每方向Voice/Non-Voice概率。
- CNN权重、NeMo源模型和smoke音频纳入Git LFS；模型说明、配置和许可证纳入普通Git。
- 增加空候选快速路径、候选数量/顺序/角度契约校验及CPU/CUDA一致性测试。

### Application Runtime与跨层架构

- 建立唯一`WindowKey=(session_id, stream_epoch, window_id, decision_sample)`，贯穿L1窗口、L2、L3、L4、UI与RecordingStore。
- 将原L2→L3→L4单线程串行链改为跨窗口流水：稳态`L2(n) || L3(n-1) || L4(n-2)`，同一窗口仍严格`L2→L3→L4`。
- L2、L3、L4各自使用单worker和有界latest-wins队列，只替换尚未开始的旧任务；所有丢弃、失败、跳过和取消均生成明确终态。
- 新增不可变StageResult、ComputeCache、ResultJoiner和有序commit barrier；结果乱序完成但按唯一时间轴提交。
- ResultWatermark只在连续终态可越过时推进，RecordingStore只接收一次完整DecisionRecord。
- 强化启动回滚、EOS、graceful drain、卡死线程、跨epoch、队列饱和和存储故障隔离。
- CPU缓存、GPU prepared上下文、Joiner和所有队列均设置窗口数与字节硬上限。

### Development Test UI

- 建立L1/L2/L3/L4四区域开发界面，并复用正式ApplicationRuntime而非另建算法时间轴。
- L1显示8通道电平、IMCRA状态、噪声统计、预降噪开关、灯控及scratch/正式录音控制。
- L2显示360°空间响应、Gate概率、候选阈值、内部方向ID、圆周卡尔曼Q/R控制及候选角度。
- L3支持真实320 ms增强音频波形、算法模式切换、播放/停止、Center参考和按正式方向ID拼接的有界试听缓存。
- L4显示各方向CNN结果，并新增容量1的最新完整L4帧邮箱；有序丢弃帧不擦除刚完成结果，超时后才显示STALE。
- UI按WindowKey和epoch隔离更新，修复旧epoch L4迟到污染新epochSRP、Gate UNAVAILABLE时误清L3全部试听录音等问题。
- 顶部诊断显示各层队列深度、完成/丢弃/跳过计数、处理Hz、错误、inflight和缓存使用量。

### 音频录制与数据管理系统

- 实现RecordingStore的off/manual/continuous/event模式，录制native 8ch、logical/physical音频、float数组、IMCRA、噪声、空间响应、增强音频和L4结果。
- 采用60秒对齐切块、跨epoch封块、独立有界音频/结果队列和原子result+watermark接收。
- event模式支持有界pre-roll/post-roll和相邻事件段合并，避免逐窗口无限增长审计记录。
- Runtime Sessions与Test Corpus隔离；提供manifest、SHA-256、SQLite WAL Catalog、lineage、导入导出、标注、QA、统计、split、retention和Trash接口。
- 大型physical float、IMCRA和noise资产改为磁盘partial流式spool，避免60秒大数组在内存中累积。
- 增加普通chunk和增强音频prepared journal、open manifest checkpoint、崩溃恢复、quarantine及Catalog重建。
- 写盘失败、队列满、容量扫描和录音模式切换不得反压实时采集。
- Production UI提供Runtime Sessions、Test Corpus、录制向导、标注、质量、统计、存储和实验管理入口。

### 测试、环境与发布资产

- 固定Windows 11 x64、Python 3.12和PyTorch `2.12.1+cu132`运行路径，完整依赖与SHA-256写入`requirements.lock`。
- 提供环境创建、GPU自检、Test UI启动、数据管理启动和L3/L4性能基准脚本。
- 自动测试覆盖公共DTO、几何、时间轴、IMCRA、预降噪、SRP、方向平滑、L3缓存/BF、L4、Runtime并行、UI、RecordingStore和恢复流程。
- 发布前完整自动测试为362项通过；版本号打包契约测试通过。
- 首次上传包含189个文件；7个Git LFS资产约17 MB。远端重新克隆`v1.0.0`后，7个LFS文件大小与SHA-256均和源项目一致。

### 尚未完成或需实机继续验证

- 不同诊室、距离、混响、风扇噪声和多人同时讲话条件下仍需持续实机标定。
- 当前主要实时性能压力位于L3双候选波束形成；latest-wins会保实时性但产生可审计丢窗。
- 当前CNN为NVIDIA预训练模型，尚未使用目标诊室和R6+1专用语料完成正式微调与校准。

---

## 后续记录模板

复制以下模板到本文件最上方的最新记录位置：

```markdown
## YYYY-MM-DD — 变更标题

- **版本/标签**：未发布 / x.y.z / vx.y.z
- **提交**：提交后在交付报告中填写哈希
- **类型**：修复 / 功能 / 性能 / 重构 / 文档 / 模型或数据
- **涉及文件**：列出主要文件或目录

### L1
- 具体变化；无变化时写“无变化”。

### L2
- 具体变化；无变化时写“无变化”。

### L3
- 具体变化；无变化时写“无变化”。

### L4
- 具体变化；无变化时写“无变化”。

### Development Test UI
- 具体变化；无变化时写“无变化”。

### 音频录制与数据管理
- 具体变化；无变化时写“无变化”。

### 跨层接口、配置与兼容性
- DTO、shape、dtype、时间字段、配置和迁移影响。

### 测试与验收
- 自动测试、性能、实机验证和未完成项。

### Git与Git LFS
- 普通Git文件、LFS资产、分支、版本标签及上传状态。
```

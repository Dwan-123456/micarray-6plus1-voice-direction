# 项目完整变更日志

本文件是6+1麦克风阵列项目的统一、持续维护记录，覆盖：

- Layer 1：采集、通道映射、校准、IMCRA与预降噪；
- Layer 2：Gate、SRP-PHAT、候选方向、内部ID与卡尔曼；
- Layer 3：方向波束形成、缓存及增强音频；
- Layer 4：响度补偿、重采样、CNN与人声概率；
- Development Test UI；
- 正式音频录制、数据管理与Production UI；
- Application Runtime、唯一时间轴、跨层接口、缓存、测试和模型资产。

## 维护规则

1. 日志按时间倒序追加，已发布记录不得重写成与历史不符的内容。
2. 每次提交前必须记录本次实际变化；没有变化的模块明确写“无变化”，防止遗漏跨层影响。
3. 每条记录至少包含日期、版本/标签、变更类型、涉及文件、各模块具体变化、接口或兼容性影响、验证结果和Git LFS资产变化。
4. 功能尚未完成、未经实机验证或仅完成自动测试时必须明确标注，不能写成已经正式验收。
5. 本文件记录“发生了什么”；当前权威接口与参数仍以`ARCHITECTURE_V0.3_TARGET.md`、`config/config.yaml`和代码为准。
6. 更早的单次Test UI历史快照保留在`docs/DEV_TEST_UI_CHANGELOG_2026-08-14.md`，其过时算法描述不得覆盖当前实现。

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
- 完整自动测试：`310 passed`。本机Rolling MUSIC性能测试满足稳态p95不高于15 ms且单窗低于20 ms；尚未完成真实麦克风、诊室混响、三声源和长时间目标机实机验收。
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

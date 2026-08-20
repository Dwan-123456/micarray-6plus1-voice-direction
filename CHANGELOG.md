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
5. 本文件记录“发生了什么”；当前1.1目标接口与参数以`ARCHITECTURE_V1.1_TARGET.md`、`config/config.yaml`和代码为准；`ARCHITECTURE_V0.3_TARGET.md`保留为历史版本记录。
6. 更早的单次Test UI历史快照保留在`docs/DEV_TEST_UI_CHANGELOG_2026-08-14.md`，其过时算法描述不得覆盖当前实现。

---

## 2026-08-20 — 五频段鲁棒对照替换30°恒定波束宽度模式

- **版本/标签**：项目`1.1.0`并行迁移分支；未创建或移动发布标签。
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
- 0～3个公开`TrackedDirection`、WindowKey、track_id、rank、角度、候选顺序、48 kHz/7680点
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

## 2026-08-20 — 公共音频上下文由320 ms缩短为160 ms

- **版本/标签**：项目`1.1.0`并行迁移分支；未创建或移动发布标签。
- **类型**：跨层公共时间契约、L3/L4输入输出、Test UI试听拼接、Runtime、文档与自动测试。
- **涉及文件**：`common/{timing,config,data_types}.py`、`config/config.yaml`、`windowing/assembler.py`、`layer3_direction_signal/`、`layer4_voice_classifier/`、`app/runtime.py`、`gui/dev_test_ui/`、`data_management/`、MarbleNet manifest、环境/基准脚本、1.1架构文档、组件README及相关测试。

### L1、Windowing与L2

- 公共`DecisionWindow`从48 kHz `float32[15360,8]`改为`float32[7680,8]`，首个endpoint与预热需求从15360改为7680 samples；20 ms发布节拍和末尾40 ms DOA窗口保持不变。
- 每窗IMCRA上下文由16个连续20 ms hop改为8个；L1 IMCRA、预降噪算法、7+1通道顺序及L2 Gate、SRP/MUSIC、ID/Kalman算法均无变化。
- 新增共享锁定时间常量，避免L3、L4和UI再次各自硬编码不同窗长。

### L3

- 输入和每方向48 kHz单声道`EnhancedAudio`统一为160 ms/7680 samples；STFT时间维由33帧改为17帧，内部工程特征相应为`[17,169]`。
- 相邻20 ms窗口精确复用13/17个STFT帧，只重算0、14、15、16号4帧；IMCRA插值和协方差滚动边界同步改为8 hop/17帧，1000 ms/50 hop缓存硬上限不变。
- `optimized`、`ds_baseline`、`constant_beamwidth_baseline`三种BF策略、空间`p`表、公开ID顺序契约和逐频点DAS回退均无变化。

### L4

- 正式输入改为每方向`float32[7680]`，模型适配器由48 kHz 160 ms重采样为16 kHz 2560 samples；MarbleNet权重、三帧连续峰值聚合和Voice阈值不变。
- IMCRA响度补偿由16段改为8段20 ms概率，仍只作用于CNN副本；模型manifest的`public_samples`同步改为7680，权重文件未修改。

### Development Test UI、Runtime与数据管理

- 正式预览按钮和波形契约改为160 ms。ID试听仍按绝对decision sample逐20 ms拼接，稳定hop和交叉淡化规则不变；当前预览最多回填最近8个hop，超过160 ms的旧缺口保留等时静音，不压缩时间线。
- Runtime向L4严格提供8个同窗概率槽；并行阶段、WindowKey、latest-wins队列、有序Joiner和失败终态规则不变。
- RecordingStore增强波形契约改为7680 samples；录音schema、事务恢复、Catalog与Production UI无变化。

### 验证与资产

- 全量自动测试：`353 passed`。
- Ruff检查与`git diff --check`通过；CPU最小冒烟基准成功覆盖L3单/双方向、13帧滚动复用及L4单/双方向160 ms模型前向。该单次冒烟数值不作为正式性能基线。
- 未修改LFS管理的音频、模型权重或空间表资产；仅更新MarbleNet文本manifest，无Git LFS对象变化。

## 2026-08-19 — L3迁移到公开方向ID与严格批次对齐契约

- **版本/标签**：项目`1.1.0`并行迁移的L3分支；未创建发布标签，已发布`v1.0.1`不移动。
- **类型**：公共DTO、L3接口、Runtime阶段契约、文档与自动测试。
- **涉及文件**：`common/window_key.py`、`common/data_types.py`、`common/__init__.py`、`app/processing_contracts.py`、`app/runtime.py`、`layer2_source_detection/pipeline.py`、`layer3_direction_signal/{interface,prepared,engine,hybrid,feature}.py`、L3 README及相关测试。

### L1与Windowing

- 采集、8通道顺序、校准、IMCRA、预降噪、DecisionWindow尺寸和20 ms发布时间轴均无变化。

### L2

- MUSIC/SRP、模型阶数、跟踪算法、Kalman和生命周期实现均无变化；本分支不分配、猜测或修补方向ID。
- `Layer2PipelineResult`增加公开`directions`与`active_tracks`承接字段及WindowKey/ID唯一性校验，供L2 Tracking分支输出权威`TrackedDirection`；旧私有候选元数据不再被L3用于构造ID。

### L3

- 公共输入由`CandidateDirection`切换为0～3个`TrackedDirection`。新增共享`WindowKey`类型；`DirectionalSignal`、`BeamformedL3Batch`、`SpectrogramFeature`和`EnhancedAudio`携带`track_id`、原始`rank`、`theta_deg`及WindowKey身份。
- 入口严格校验同窗、类型、数量、ID唯一、rank唯一及处理许可；L3保持输入元组顺序，不按角度排序、分配、合并或修补ID。
- 观测目标可处理；coasting目标只有在L2明确设置`allow_l3_prediction=True`时才可作为短时预测目标处理。长coasting轨仅留在`active_tracks`时间线，误送入L3时明确失败，不生成增强音频。
- 波束形成批次和音频合成出口逐项校验WindowKey、ID集合与顺序、rank、角度和数量。错误转为Runtime L3 `FAILED`终态，L4按既有阶段规则跳过。
- `optimized`、`ds_baseline`和`constant_beamwidth_baseline`算法、数值参数、缓存、输出音频shape与fallback行为无变化；运行时模式切换不改变权威L2 ID。

### L4

- CNN、重采样、响度补偿、模型和阈值逻辑无变化。L4公开ID契约由独立L4迁移分支负责，本次未提前修改其DTO。

### Development Test UI、录音与数据管理

- Test UI和Production UI的正式界面行为、试听关联实现、录音schema、Catalog、manifest、恢复和本地数据均无变化；仅更新集成测试夹具以显式提供L2权威ID。
- 未修改或新增模型、音频、空间表格、录音或其他二进制资产。

### 跨层接口与兼容性

- `WindowKey`从Runtime内部定义提升为`common.window_key`公共类型，并由`app.processing_contracts`继续导出，保持原导入路径兼容。
- L3公共入口不兼容旧无ID`CandidateDirection`；缺失ID、重复ID、跨窗ID和出口顺序损坏均拒绝，而不是静默兼容。
- 本分支需要与L2 Tracking、L4、Runtime/Recording和UI的其余1.1.0并行分支整合后才能发布，不单独创建`v1.1.0`。

### 测试与验收

- 增加0～3方向、359.5°/0.5°跨0°、非排序输入、重复/缺失ID、跨窗、短预测许可、长coasting拒绝、出口ID篡改、三BF模式一致性、模式切换ID稳定及L3失败终态覆盖。
- 完整自动测试353项通过；全部改动文件的Ruff检查和`git diff --check`通过。未进行新的真实麦克风、诊室三声源或长时间实机验收。

### Git与Git LFS

- 仅Python、Markdown普通文本发生变化；Git LFS资产无变化，未创建或移动版本标签。

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

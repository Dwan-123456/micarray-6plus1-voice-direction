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

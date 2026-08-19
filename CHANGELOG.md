# 项目完整变更日志

本文件是6+1麦克风阵列项目的统一、持续维护记录，覆盖：

- Layer 1：采集、通道映射、校准、IMCRA与预降噪；
- Layer 2：Gate、SRP-PHAT、候选方向、内部ID与卡尔曼；
- Layer 3：方向波束形成、缓存及增强音频；
- Layer 4：响度补偿、重采样、CNN与人声概率；
- Development Test UI；
- 独立 Pipeline Log UI；
- 正式音频录制、数据管理与Production UI；
- Application Runtime、唯一时间轴、跨层接口、缓存、测试和模型资产。

## 维护规则

1. 日志按时间倒序追加，已发布记录不得重写成与历史不符的内容。
2. 每次提交前必须记录本次实际变化；没有变化的模块明确写“无变化”，防止遗漏跨层影响。
3. 每条记录至少包含日期、版本/标签、变更类型、涉及文件、各模块具体变化、接口或兼容性影响、验证结果和Git LFS资产变化。
4. 功能尚未完成、未经实机验证或仅完成自动测试时必须明确标注，不能写成已经正式验收。
5. 本文件记录“发生了什么”；当前1.1.1架构以`ARCHITECTURE_V1.1_TARGET.md`为权威契约，已发布1.0.1历史以`ARCHITECTURE_V0.3_TARGET.md`为基线，实际参数以`config/config.yaml`和代码为准。
6. 更早的单次Test UI历史快照保留在`docs/DEV_TEST_UI_CHANGELOG_2026-08-14.md`，其过时算法描述不得覆盖当前实现。

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

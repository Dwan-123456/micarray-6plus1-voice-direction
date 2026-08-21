# 6+1 麦克风阵列项目1.2.4：独立Pipeline Log UI架构

状态：**代码、公共只读查询、五个页面与自动化测试已纳入项目1.2.4；真实封存session人工回放和大规模实机数据验收仍需继续。**

开发版本：项目`1.2.4`，最终发布基线为`v1.2.3`。本文件定义Log UI的定位、只读边界、数据覆盖、页面与验收要求；未通过的实机项目必须继续明确标注。

上位架构：[`ARCHITECTURE_V1.1_TARGET.md`](ARCHITECTURE_V1.1_TARGET.md)。当前运行时与录音实现以[`app/README.md`](app/README.md)、[`data_management/README.md`](data_management/README.md)和代码为准；[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)仅保留历史迁移契约。

## 1. 定位

Pipeline Log UI 是项目级的**独立只读观察与回放子系统**，与下列部分平行：

- Layer 1、Layer 2、Layer 3、Layer 4；
- Development Test UI；
- RecordingStore、Audio Data Manager 与 Production UI。

Log UI **不是 Layer 5**，不插入 `L1 → L2 → L3 → L4` 实时处理链，也不是 Test UI 的一个面板。它只通过项目公开的只读接口读取已经公开或持久化的数据，统一展示一次运行记录中的性能、阶段终态、算法输出、方向 ID 时间线和可校验音频资产。

```text
实时处理平面
L1 → Windowing → L2 → L3 → L4 → ResultJoiner
                                      ├── Development Test UI
                                      ├── RecordingStore / 数据管理 / Production UI
                                      └── 公开记录与只读查询边界
                                                     ↓
观察平面                                     独立 Pipeline Log UI
                                             统计 / 时间线 / 单窗详情 / 回放
```

`ResultJoiner`、RecordingStore 和公开查询边界是数据生产者；Log UI 是被动消费者。Log UI 不得成为任何实时阶段完成、录音提交、界面刷新或停机排空的前置条件。

## 2. 目标与非目标

### 2.1 目标

1. 按 session 查看当前公开接口能够提供的全部记录覆盖，并明确显示未记录或接口未提供的内容。
2. 以权威 `WindowKey = (session_id, stream_epoch, window_id, decision_sample)` 对齐 L1/Gate/L2/L3/L4/commit 结果。
3. 统计每层完成、跳过、丢弃、超时、失败、取消、计算耗时、排队耗时、端到端延迟、实际完成频率和缺口。
4. 按 `(session_id, stream_epoch, track_id)` 回看 MUSIC 方向、ID 生命周期、L3 增强资产和 L4 判断。
5. 支持从会话总览逐级下钻到异常窗口和只读原始公开字段，便于定位丢窗、延迟、ID 和跨层对齐问题。
6. 对 v3/v4 等记录版本进行能力探测与兼容展示，不把缺失字段推断为零或正常。
7. 对封存静态记录证明读取前后项目文件、Catalog和录音资产不变；对运行中系统通过调用审计与对照测试证明不消费邮箱、不调用写接口，也不引入额外状态变化。

### 2.2 非目标

- 不启动、暂停、停止或重启 `ApplicationRuntime`；
- 不修改 Gate、MUSIC、ID、Kalman、L3、L4、录音或 UI 参数；
- 不进行标注、QA 修复、Catalog 重建、导出、Trash、恢复或数据迁移；
- 不直接读取各层私有对象、内部张量、以下划线开头的字段或实现细节；
- 不读取会与正式消费者竞争的单消费者队列；
- 不创建第二套采集、算法 Runtime、方向 ID 或结果时间线；
- 不承诺恢复项目从未公开或持久化的内部性能数据。

## 3. 不可破坏的只读边界

### 3.1 只使用公开接口

Log UI 只能依赖稳定的公开查询接口和公开 DTO。不得为了多显示一个字段而导入各层内部类、解析私有缓存、访问 Runtime 私有队列或绕过资产校验直接打开任意路径。

UI 所需字段尚未公开时，页面显示“接口未提供”，并在能力清单中记录缺口。缺口只能通过后续正式公共接口改造解决，不能由 Log UI 猜测、反射或复制主项目内部算法解决。

### 3.2 禁止消费实时邮箱

当前 Runtime 的 `latest_dev_ui`、`latest_l4_dev_ui`、`latest_l1` 和 `latest_windows` 属于容量有限、读取即移除的 latest-only 邮箱。Log UI 禁止调用这些队列的 `get()`；否则会抢走 Development Test UI 或正式消费者的数据，并改变被观察系统本身。

### 3.3 禁止隐式写 Catalog

通用`DataManagerService(data_root)`会构造`Catalog`；Catalog初始化会创建目录、打开SQLite、启用WAL并执行schema初始化。因此，Log UI不能直接针对正式数据根目录实例化该服务，而应使用专用公共只读查询端口。

目标实现只能使用项目提供的稳定、显式只读查询端口，或由正式进程通过公共接口生成并返回的版本化只读快照/流。Log UI 自身不得复制、打开或解析 SQLite 主文件、WAL/SHM及其他内部存储格式来绕过公共接口。

Log UI 不得在项目 `data/` 中创建数据库、WAL、SHM、索引、缓存或 UI 设置文件。公共接口若返回临时只读快照，其生命周期、完整性和清理由提供该接口的正式组件负责；Log UI 不得把快照变成新的数据真源。

## 4. 数据模式与当前能力限制

### 4.1 完成会话离线回看：第一优先级

首版以已完成、已封存的 runtime session 为权威数据源。通过公开接口读取 session manifest、DecisionRecord、方向轨时间线和经过校验的音频资产。统计结果只来自这些公开记录。

对 `open` 会话，只允许读取已经封存且接口明确承诺稳定的部分；页面必须标记“录制中/数据可能不完整”，不得把暂未出现的数据计为失败或零。

### 4.2 可选同进程实时概览

如果正式应用的外部宿主已经能够显式注入现有 `ApplicationRuntime` 的只读引用，Log UI 适配器可以轮询公开 `processing_status`，显示队列深度、容量、worker 状态、缓存、累计完成/错误/丢窗和 L4 实际频率等**聚合状态**。

该模式不得控制 Runtime，也不能代替离线权威记录。聚合快照不包含完整逐窗历史，不能在事后恢复为逐窗时间线。在“不修改主项目”的独立进程范围内无法注入该引用，因此此模式为可选延期能力，不是首版承诺。

### 4.3 独立进程实时逐窗观察

当前项目没有稳定的跨进程只读 IPC、HTTP、WebSocket 或发布订阅事件流。因此第一版独立 Log UI 的“实时逐窗”能力应明确显示为 `Unavailable`，不得通过读取 latest-only 邮箱伪造。

未来若项目新增公共只读事件流，只增加新的 `LogSource` 适配器；不得让 Log UI 反向成为 Runtime 的依赖。

### 4.4 历史记录与1.2.2能力

- 1.0.1历史记录可能缺少MUSIC、公共ID或完整逐ID资产，Log UI通过schema和能力探测降级展示。
- 1.2.2通过公共接口提供session decisions、track timeline、track audio assets和session audio assets等只读查询能力。
- 未探测到的能力显示`N/A`，不能绕过接口读取内部文件，也不能为旧记录补造不存在的数据。

## 5. 公开数据覆盖矩阵

| 数据域 | 可展示的公开内容 | 不可推断的内容 |
|---|---|---|
| Session/Recording | 状态、起止时间、schema、模式历史、配置与校准 hash、chunk、recorded/missing interval、result gap、水位 | 未封存的在途数据、未公开的进程内部状态 |
| L1/Ingest/Windowing | 公开的输入身份、sample/epoch 连续性、校准状态、Gate 输入摘要、记录缺口 | 未记录的单窗解码/IMCRA/装窗耗时、内部频谱和中间张量 |
| Gate/L2 MUSIC | Gate 状态、MUSIC/模型阶数、候选、空间谱引用、质量诊断、方向 ID、active tracks | 未公开的逐频协方差、特征向量和临时工作区 |
| L3 | 每个 `track_id` 的方向、阶段状态、增强资产元数据、公开后端/回退诊断 | 私有 BF 权重、未持久化的 GPU 张量 |
| L4 | 每个 `track_id` 的概率、阈值判断、阶段状态与公开模型诊断 | 模型内部激活与未公开特征 |
| Runtime/Joiner | 各阶段终态、compute/wait/端到端耗时、terminal reason、drop/gap；可选实时聚合状态 | 历史队列逐窗深度、CPU/GPU/内存曲线，除非未来正式记录 |
| 音频与资产 | 公开查询返回且通过 hash、范围和schema校验的 Center/L3 等资产 | 任意绝对路径、越界文件、校验失败内容 |

每个字段都必须带来源能力和缺失状态。`N/A`、`未记录`、`尚未封存`、`校验失败`和数值 `0` 是五种不同状态，不得混用。

## 6. 软件结构

```text
RecordingLogSource       RuntimeStatusSource（可选）
        \                       /
         PublicApiAdapter / Capability Probe
                         ↓
                 NormalizedReadModel
                         ↓
                  StatisticsEngine
                         ↓
                    Pipeline Log UI
```

### 6.1 `LogSource`

- `RecordingLogSource`：只读完成/封存会话，是首版权威来源；
- `RuntimeStatusSource`：可选同进程聚合状态，不提供逐窗权威历史；
- 未来公共事件流：作为新适配器加入，不改变 UI 和统计模型。

### 6.2 `PublicApiAdapter`

- 先探测 capability，再调用对应公开接口；不按版本号猜测字段一定存在；
- v3/v4 分别适配，未知 schema 默认拒绝参与统计并保留可读错误；
- 未识别的公开字段可以进入只读 JSON 视图，但不能在没有定义时参与指标计算。

### 6.3 `NormalizedReadModel`

- 逐窗主键固定为 `WindowKey`；
- 方向轨主键固定为 `(session_id, stream_epoch, track_id)`；
- 保留原始 sample 范围、阶段终态、缺失原因和来源 schema；
- 任何层都不得由角度重新生成、合并或修补正式 ID。

### 6.4 加载与资源边界

- 后台增量解析，长任务可取消；
- 按页、按时间范围和按需加载，不一次把全部波形放入内存；
- 使用有界内存 LRU，达到上限时淘汰可重建视图；
- 默认不加载音频波形，用户进入资产详情或播放时才通过公开接口请求；
- 10万窗口级记录仍应保持 UI 可操作，且不会在项目目录生成索引文件。

## 7. 页面设计

### 7.1 记录列表

显示 session ID、状态、起止时间、时长、项目/算法版本、schema、配置 hash、校准 hash、数据完整性和可用 capability。默认打开已完成或结果不完整但已封存的会话；`open` 会话带显著警告。

### 7.2 会话总览

显示：

- 总窗口数、适用窗口数、完成/跳过/丢弃/超时/失败/取消数量与比例；
- L1/Gate/L2/L3/L4/commit 的实际完成频率；
- 各层 compute、queue wait 和端到端 age 的 p50/p95/p99；
- sample gap、timeline gap、terminal reason 和资产校验异常；
- MUSIC 声源数、方向轨数量及 L4 Voice/Non-Voice 摘要。

每项指标旁同时显示样本数 `n` 和缺失率；缺少公开数据时显示 `N/A`。

### 7.3 Pipeline 时间线

横轴使用 authoritative sample/20 ms window，纵轴显示 L1/Gate/L2/L3/L4/commit。颜色只表示明确阶段终态：`COMPLETED / SKIPPED / DROPPED / TIMED_OUT / FAILED / CANCELLED`；`UNKNOWN`只表示终态字段缺失或schema不识别，不是正式终态。点击任意格进入单窗详情。

跨 epoch 处必须断开显示；不得把前后两段不连续音频拼成连续时间线。

### 7.4 单窗详情

显示完整 WindowKey、sample 范围、各阶段状态、计算/等待耗时、终态原因、公开 diagnostics、Gate、MUSIC 360点谱（若接口提供）、model order、候选与 active tracks、L3资产元数据、L4概率/判断，以及该窗的只读公开 JSON。

单窗详情用于回答“这个时刻在哪一层停止、为什么停止、ID和输出如何对应”，不提供任何参数编辑控件。

### 7.5 ID 与异常

- 按 `(session_id, stream_epoch, track_id)` 展示首末 sample、寿命、状态、观测/预测角、`359° ↔ 0°` 连续轨迹、L4概率和增强资产；
- 异常列表按 sample gap、drop、timeout、failed、ID新建/终止、跨层集合不一致、schema/hash/path校验失败分类；
- 筛选结果和时间线、单窗详情、资产详情双向跳转；
- `track_id` 明确标注为方向轨 ID，不表示人物身份。

## 8. 统计口径

### 8.1 阶段数量与比例

- 每层分母只计该层适用且有公开终态的窗口；
- `COMPLETED`、`SKIPPED`、`DROPPED`、`TIMED_OUT`、`FAILED`、`CANCELLED` 分开统计；
- 缺字段不进入任何状态数量，同时进入缺失计数；
- “处理成功率”必须在界面上写清分子与分母，不能把 Gate 正常跳过混为失败。

### 8.2 实际完成频率

```text
observed_duration_s = (range_end_sample - range_start_sample) / sample_rate
actual_completed_hz = 所选范围内COMPLETED窗口数 / observed_duration_s
```

`range_start_sample`和`range_end_sample`是所选 epoch/时间范围完整权威观测区间的含首、排他尾边界，必须包含范围内的丢弃、失败和首尾窗口覆盖，不能用首末 `COMPLETED` sample 代替。跨 epoch 分别计算，或以各 epoch 有效观测时长之和作为总分母；没有有效正时长时显示 `N/A`，避免单窗除零和帧率高估。

只有 `COMPLETED` 计入实际完成频率的分子。`SKIPPED`、`DROPPED`、`TIMED_OUT`、`FAILED`、`CANCELLED`、零毫秒占位终态和 UI 刷新次数均不得计入。启动/结束不足一个统计窗时明确显示边界影响。

### 8.3 延迟

- 分别计算 stage compute、queue wait 和端到端 age；
- 只对存在对应有效数值的样本计算 p50/p95/p99；
- 每个分位数显示 `n`；
- 离线记录没有可靠 wall-clock age 时，不用 sample 时间或 UI 加载时间伪造。

### 8.4 ID

- 所有角度差使用圆周距离；
- 时间轴展示支持 `359° ↔ 0°` 连续展开，但原始公开角仍保持 `[0°, 360°)`；
- ID 的 birth、coasting、恢复、deleted 和同方向新 ID 只按公开 L2 记录解释，不由 Log UI 重新关联。

## 9. 兼容、完整性与隐私

- v4：在公开字段可用时完整展示 MUSIC、公共 ID、逐 ID L3/L4 和时间线；
- v3：展示已有阶段、候选、L4、耗时与资产；缺少公共 ID/MUSIC 字段时显示 `N/A`；
- 未知 schema：会话可列出，但默认 fail-closed，不参与聚合统计；
- 坏 JSON、截断记录、hash 不匹配、路径越界和资产缺失分别形成只读异常，不自动修复；
- 音频只能经现有公开资产校验接口读取，不显示任意绝对文件路径，不跟随数据根之外的路径；
- Log UI 默认不持久化搜索历史、波形、音频或运行记录副本，不向 GitHub 或外部服务上传任何项目数据。

## 10. 实施阶段

1. **P0：观察矩阵冻结**。确定字段 → 公开接口 → 页面 → 缺失行为；明确历史记录与1.2.2 capability。
2. **P1：离线只读内核**。完成只读来源、v3/v4适配、标准模型、索引和统计公式。
3. **P2：五个页面**。完成记录列表、总览、Pipeline时间线、单窗详情、ID与异常联动。
4. **P3：可选同进程概览**。只显示公开 `processing_status` 聚合数据，不消费实时邮箱。
5. **P4：未来事件流适配**。只有项目正式提供公共只读流后才增加独立进程实时逐窗观察。

Log UI 可以在独立分支开发，但必须等待 Recording/Data 公共只读契约稳定后再冻结其适配器。任何为 Log UI 增加的主项目接口必须作为单独的公共契约改动进行评审，不能隐藏在 UI 实现中。

## 11. 测试与验收门禁

### 11.1 正确性

- v3/v4 golden fixtures、未知 schema、缺字段、坏 JSON、截断记录；
- WindowKey 排序、跨 epoch 断开、sample gap、阶段适用分母；
- `359° ↔ 0°`、ID时间线、同方向超时后新 ID、跨层 ID 对齐；
- completed Hz 只计 `COMPLETED`，p50/p95/p99 与缺失率公式有独立断言；
- open/partial/complete/result_incomplete 会话显示语义正确。

### 11.2 只读性

- 对封存静态 fixture，打开、筛选、回放和关闭前后项目记录文件 hash、Catalog行数与schema完全一致，且不新建SQLite/WAL/SHM文件；
- 对运行中系统，不要求自然变化的队列深度、累计计数或Catalog/WAL字节保持静止；通过mock/调用审计和有无Log UI的对照运行证明没有消费latest-only邮箱、调用写接口或造成额外队列/提交变化；
- 测试证明没有读取私有字段、直接打开内部Catalog文件或绕过公共资产接口；
- hash失败和路径越界时拒绝资产，不自动修复或重写。

### 11.3 性能与稳定性

- 10万窗口记录后台加载、筛选和取消加载；
- 内存上限和 LRU 淘汰可验证；
- 默认不加载波形，音频按需读取；
- 大记录、部分损坏记录和正在录制会话不会卡死主 UI；
- Log UI 关闭或崩溃不影响 Runtime、Test UI、录音提交和数据管理。

## 12. 完成定义

Log UI作为1.2.2已实现组件满足以下自动化与接口条件：

1. 公共只读数据契约已经冻结并有版本化测试；
2. 五个页面和统计口径完成；
3. v3/v4兼容、异常和大记录测试通过；
4. 只读性测试证明项目数据、Catalog和运行队列均未改变；
5. 文档、CHANGELOG、打包入口和发布验收同步完成。

实机与真实大规模封存数据验收状态以CHANGELOG和测试报告为准，不得仅凭版本号推断已经通过。

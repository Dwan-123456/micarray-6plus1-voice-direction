# 项目文件分类

> 当前开发版本为1.2.4，最终发布基线为`v1.2.3`；职责以[`ARCHITECTURE_V1.1_TARGET.md`](ARCHITECTURE_V1.1_TARGET.md)为准。L1～L4、Rolling NormMUSIC、永久公共方向ID、各UI和正式录音数据系统均纳入版本管理。

> 当前源码已迁移到v0.3主链，权威接口见[`ARCHITECTURE_V0.3_TARGET.md`](ARCHITECTURE_V0.3_TARGET.md)。ApplicationRuntime采用唯一WindowKey、L2/L3/L4逐层有界latest-wins、有界completion/backlog/commit以及有序ResultJoiner；同窗严格L2→L3→L4、稳态跨窗并行。实机标定和production入口的未完成项仍以权威架构文档为准。

本工作区分为“当前新项目”和“旧项目/外部参考资料”两类。运行、测试、打包和开发入口只能依赖当前新项目；`legacy_reference_only/`不得加入Python导入路径，也不得作为运行时数据源。

## 当前新项目

- `app/`：当前共用`ApplicationRuntime`、跨层契约、逐层latest-wins调度、pre-joiner轻量审计、ComputeCache与ResultJoiner，已由`pyproject.toml`正式打包；正式`app.main`入口尚未实现。
- `config/`：当前项目唯一业务配置包；`config.yaml`由严格schema加载，并通过`pyproject.toml`的package data进入wheel。录音chunk/队列容量和存储预算的非法组合在启动前拒绝。
- `gui/`：Development Test UI消费有序Join快照和容量1的L4完成帧显示邮箱；后者仅降低CNN显示等待，不改变正式结果、录音或watermark顺序。Production UI仍按其当前实现边界处理。
- `common/`：公共配置、数据类型、阵列几何和角度规则。
- `config/`：唯一业务配置。
- `ingest/`、`windowing/`：统一时间轴、分发和分析窗口。
- `layer1_input/`：当前新项目的Layer 1，与Layer 2、Layer 3、Layer 4在根目录并列。
- `layer2_source_detection/`：Probability Gate、滚动frequency-normalized MUSIC、永久全局方向ID及可选Kalman输出平滑。
  - `probability_gate.py`：消费L1对齐的两个20 ms声源概率，计算40 ms均值并按运行时门限决定是否运行MUSIC。
  - `music.py`：维护增量STFT协方差、MDL 0～3源估计、NormMUSIC 360°伪谱和圆周峰值。
  - `global_tracker.py`：使用带birth/miss dummy项的`linear_sum_assignment`维护公共方向轨迹。
  - `pipeline.py`：按Probability Gate → Rolling NormMUSIC → Global ID → optional Kalman编排，并区分`BLOCKED / PROCESSED`。
- `layer3_direction_signal/`：当前v0.2含共享STFT、DAS/MVDR和FeatureExtractor；v0.3公共输出将只保留逐方向48 kHz音频。
- `layer4_voice_classifier/`：已实现MarbleNet基准的公共契约、插件引擎、artifact校验、CPU推理和运行时接线；目标域校准及CUDA门禁未完成。
- `data_management/`：正式录音与数据管理；当前RecordingStore包含原子result+watermark、逐chunk结果释放、有界event pre-roll、hotmap流写入及增强WAV partial+journal恢复。
- `gui/`：Development Test UI，以及带独立UAC采集主机的正式Audio Data Manager；最终人声方向production GUI尚未实现。
- `scripts/`、`tests/`：当前项目脚本与测试。
- `models/`：当前MarbleNet基准artifact与来源追溯快照。
- `data/`：当前运行产生的数据、Catalog和scratch资产；`data/external_sources/`另含L4公开bootstrap数据、许可、hash和清单，不属于旧项目源码。
- 根目录执行规格、环境说明、依赖锁定文件、`pyproject.toml`及VS Code设置均属于当前新项目。

- `L1_NOISE_RECORDING_AND_L2_OPTIMIZATION.md`：L1动态噪声记录契约及未接入的L2后续优化路线。
- `LOG_UI_ARCHITECTURE_V1.1_TARGET.md`：独立 Pipeline Log UI 的1.1架构；可执行实现位于`gui/log_ui/`。

## 1.2.4平行子系统

- **Pipeline Log UI**：与`layer1_input/`、`layer2_source_detection/`、`layer3_direction_signal/`、`layer4_voice_classifier/`、Development Test UI和`data_management/`平行的项目级子系统。
- 它只通过公共只读接口统计、展示和回放 session 的性能、阶段终态、方向 ID、L3/L4输出与音频资产；不是Layer 5，不参与实时处理、控制、录音提交或数据修改。
- 实现目录为`gui/log_ui/`；通过公共只读查询契约读取封存session，不依赖空目录或占位实现。
- 权威范围、数据模式、五个页面、统计公式和只读验收见[`LOG_UI_ARCHITECTURE_V1.1_TARGET.md`](LOG_UI_ARCHITECTURE_V1.1_TARGET.md)。

## 旧项目与参考资料（仅供参考）

统一放在`legacy_reference_only/`：

- `usb-micarray-project/`：迁移前旧项目的完整快照，包含旧Layer 1～4、旧UI和旧测试。
- `sipeed-mic-array-6plus1/`：厂商/上游资料快照，包含硬件、示例和Wiki，不是当前程序源码。

该目录只允许人工查阅、公式交叉验证和溯源。禁止：

- 将其加入`PYTHONPATH`或`pyproject.toml`包搜索路径；
- 从当前源码导入其中模块；
- 从其中启动旧UI、旧服务或旧算法作为当前主链；
- 修改旧快照来代替修复当前新项目。

如确需采用其中代码，必须复制到当前新项目的对应层内适配器，记录来源、不可变版本和许可证，并按当前执行规格补齐测试。

## 移动与索引检查

整理时已确认当前源码、配置、测试、VS Code任务和打包设置均不引用原`.reference/`路径。Layer 1提升后，当前Python搜索路径统一为项目根目录；旧参考目录和已删除的`micarray_refactored/`均不需要加入兼容路径。

以后移动或重命名文件前，必须检查：Python imports、`pyproject.toml`、`.vscode/`、脚本、配置路径、测试发现路径和文档入口，并在移动后运行完整测试。

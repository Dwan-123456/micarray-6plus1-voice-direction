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
5. 本文件记录“发生了什么”；当前1.2.4开发架构以`ARCHITECTURE_V1.1_TARGET.md`为权威契约，已发布1.0.1历史以`ARCHITECTURE_V0.3_TARGET.md`为基线，实际参数以`config/config.yaml`和代码为准。
6. 更早的单次Test UI历史快照保留在`docs/DEV_TEST_UI_CHANGELOG_2026-08-14.md`，其过时算法描述不得覆盖当前实现。

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

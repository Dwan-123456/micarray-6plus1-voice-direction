# 6+1 麦克风阵列二维人声方向识别系统

> 当前开发版本：`1.3.5`；Layer 2公开版本：`1.1`。最近正式发布基线为不可变标签`v1.3.3`。

> **发布状态：项目 `1.3.5`开发线。** 本开发线从不可变正式标签`v1.3.3`建立，不继承`1.3.4`无DOA/ID/BF实验分支；自动化验收不替代真实双人录音、长时间运行和GPU质量门禁。

> 项目每次具体修改统一记录在[`CHANGELOG.md`](CHANGELOG.md)。任何L1～L6、Development Test UI、Pipeline Log UI、音频录制/数据管理、跨层接口、测试或模型资产变化都必须在提交前同步该日志。

> 面向首次接触项目的完整数据流、逐层输入/输出/内部处理单元和操作步骤，见[《完整架构与使用手册》](docs/COMPLETE_ARCHITECTURE_AND_USAGE.md)。

> `1.3.5` 的 L4-L6 可调分块伪实时设计、失败边界和性能门禁见[《L4-L6 可调分块伪实时链》](docs/REALTIME_L456.md)。

> L3双声源分离、4 cm 6+1阵列物理边界和Python实时优化的深度研究报告收录在[`docs/references/`](docs/references/README.md)。这些报告是重要研究参考，不替代当前代码、配置和架构契约。

## 独立 L1 频谱观察界面

运行 `scripts/launch_l1_spectrum_ui.ps1` 可无控制台启动只连接麦克风、校准、IMCRA和L1电平/频谱分析的独立界面；该界面每1秒自动扫描麦克风，允许先启动程序再插入设备，并默认最大化；
它不会启动L2、L3、L5或正式录音链。界面提供CDC灯光开关、八路互斥选择、当前20 ms输入频谱、
IMCRA当前噪声频谱及手动频谱抓拍。详细契约见 [`gui/l1_spectrum_ui/README.md`](gui/l1_spectrum_ui/README.md)。

## 项目要解决什么问题
本项目面向诊室环境下的医患对话记录，目标是根据声源的空间方向对音频进行分离和增强，并识别声源为人声的概率。诊室中通常同时存在医生、患者、陪同人员、设备噪声和墙面反射，单支麦克风虽然能够录音，却无法判断某段人声来自哪个方向，也难以将不同讲话人的声音分开。
系统使用一块由6个均匀分布在半径4 cm圆周上的外圈麦克风和1个中央麦克风组成的阵列。根据当前诊室使用目的和阵列物理限制，系统公开至多3个候选声源方向，而且任意两个方向的水平夹角必须大于或等于50°。系统持续完成以下工作：

1. 采集并校准多通道音频；
2. 判断当前窗口是否存在值得定位的声源；
3. 扫描阵列周围360°，找出最多三个、任意两点至少相隔50°的候选声源方向；
4. 对每个候选方向增强音频；
5. 判断该方向的增强音频是否为人声；
6. 将原始音频、方向、增强音频、模型结果和时间信息按同一时间轴保存。
7. 采集中渐进维护L4/L5与声纹预览；停止后用完整封存轨校正为最多三人，并按MOS择优拼接、压缩长静音。

当前系统输出的是“相对麦克风阵列的水平角方向”和“该方向为人声的概率”。方向轨迹ID只用于跨窗口对齐同一空间方向，不代表人物身份；系统不输出距离、俯仰角、声压级、说话内容或说话人身份。

## 基本模型

诊室内的主要使用目标是区分坐在阵列周围不同方位的人，且人与麦克风阵列的距离远大于麦克风之间的间距。因此，当前版本采用二维远场模型：把到达阵列的声音近似为平面波，只估计水平面内的方位角`theta_deg`。

- `0°`：从麦克风面观察时的正右方，也是MIC0方向；
- `90°`：正上方；
- 角度沿逆时针增加；
- 角度范围为`[0°, 360°)`。

这个模型用一个角度描述声源相对于阵列的位置，不估计距离，也不估计俯仰角。两个处在同一水平角但距离不同的人，在当前模型中不能被可靠区分。

## 重要物理限制：低频声源无法被准确分离

> **这是阵列尺寸决定的物理上限，不是简单调参或增加算力就能解决的问题。** 本阵列孔径较小。低频声音的波长远大于麦克风间距时，同一低频波到达各麦克风的时间差和相位差都很小，不同方向对应的导向矢量会高度相似。系统因此缺少足够的空间信息，无法把来自不同方向的低频内容准确分开。

项目的空间可分度图也反映了这一点：当前`80～1500 Hz`的方向分离效果较差，任意两个方向的夹角越小越难分离。系统只保留两两夹角大于或等于`50°`的候选，但即使满足这一条件，低频分离仍不可靠；随着频率升高或方向夹角增大，才逐渐出现更多可利用的空间差异。

![半径4 cm的6+1麦克风阵列空间可分度图：100 Hz～6 kHz、方向夹角0°～180°](docs/assets/rho_map_100Hz_100_6000Hz_angle1deg.png)

*图：半径4 cm的6+1阵列空间可分度。横轴为频率，纵轴为两个声源的方向夹角；`rho`越接近1，两个方向的阵列响应越相似，越难进行稳定分离。低频区域在很大的夹角范围内仍保持高相关，体现了小孔径阵列的物理限制。*

这带来几个必须明确的工程边界：

- L2选择2000～4000 Hz做宽带frequency-normalized MUSIC定位，是为了使用相对更有方向信息的频段；代价是对主要能量落在该频段之外、频谱异常或窄带的特殊声源识别与定位能力较差；
- L3虽然在最终音频中保留80～8000 Hz，但当前80～1500 Hz的方向分离效果较差；“保留低频声音”不等于“已经把低频声源分开”；
- 80～500 Hz主要通过DAS保留目标方向的声音和整体听感，不能宣称获得了可靠的双声源低频分离；500～1500 Hz即使使用更复杂BF，仍明显受小孔径限制；
- 1500～2000 Hz属于空间可分能力逐渐改善的过渡频段，实际效果仍取决于声源夹角、阵列朝向和房间混响；
- 当两个人同时讲话时，输出音频的低频部分仍可能包含另一方向的声音、风扇、空调或房间低频噪声；
- 若必须提升低频空间分离能力，需要增大阵列物理孔径、改变麦克风布局、使用多阵列或引入额外先验，而不能只依赖当前BF权重优化。

## 完整架构图

本节给出便于快速浏览的文本总图；包含逐层接口表、内部处理单元图、运行时边界和完整操作方法的版本见[《完整架构与使用手册》](docs/COMPLETE_ARCHITECTURE_AND_USAGE.md)。

```text
诊室中的医生、患者及环境声音
    ↓
Sipeed R6+1 + MA-USB8：48 kHz HostAudio [N,8]
    Host通道：CH0..CH5=MIC0..MIC5，CH6=HardwareMix，CH7=Center
    ↓
【已完成】Layer 1：输入、通道整理和噪声分析
    解码 → 校准 → 通道重排 → 连续性检查
    LogicalAudio [N,8]：MIC0..MIC5、Center、HardwareMix
    ├── PhysicalAudio [N,7]
    │     六个外圈麦 + Center，参与几何、MUSIC和波束形成
    ├── HardwareMix [N]
    │     只用于接口、显示、录制和实验，不进入阵列几何
    ├── 7麦IMCRA：每20 ms更新0～10000 Hz噪声PSD/SPP
    │     从500～4000 Hz聚合声源Gate概率p20 ∈ [0,1]
    └── 可选IMCRA-Wiener预降噪（默认关闭）
          40 ms窗 / 20 ms步长；只替换下游7路物理音频
    ↓
【已完成】IngestCoordinator：唯一session / epoch / 绝对sample时间轴
    连续性、校准身份与不可变IngestedAudioBlock
    ↓
【已完成】WindowAssembler
    每20 ms发布DecisionWindow [7680,8]
    每窗携带最近160 ms音频和8个对齐IMCRA结果
    ↓
WindowWorkItem
    WindowKey=(session_id, stream_epoch, window_id, decision_sample)
    冻结本窗Gate、MUSIC、ID、Kalman、L3和实时L5审计配置
    ↓ 有界L2 latest-wins队列（默认容量由runtime.stage_queue_windows=100统一派生）
【已完成】Layer 2 1.1：二维声源方向定位与公共方向轨迹
    Probability Gate
        末尾两个20 ms概率取平均得到40 ms Gate概率，默认阈值0.60
        正式逐窗L5不运行CNN且不回传L2，因此Runtime没有语义强制放行
        Gate关闭时跳过MUSIC；预热/缺失/无效概率同样保持阻断
    ↓ Gate放行
    2000～4000 Hz Rolling frequency-normalized MUSIC
        240 ms滚动历史；20 ms增量STFT/协方差；0～359°逐度扫描
        Test UI手动选择MUSIC阶数1/2/3
        普通路径信号子空间阶数与最多搜峰数=该手动值
        可选DPD逐频投票 + 圆周核聚类、L1完整空间噪声协方差白化（白化默认开启）
        圆周峰值 + 50° NMS → 最多3个方向
    永久在线Circular IMM-JPDA方向ID
        tentative → confirmed → coasting → deleted
        滚动200 ms内累计至少5次匹配观测后confirmed
        恢复匹配同时检查预测角与最后真实观测角，任一距离≤50°即沿用原ID
        两轨进入50°内且近期观测交替、无同窗双峰时归并，保留更正式/更早的ID
        session内track_id单调且不复用；内部最多4轨，公共输出最多3轨
        几何寿命按绝对sample计算，默认coasting TTL为2秒
        confirmed/coasting公共方向完全由L2状态投影；离线L5不改变ID或几何寿命
        JPDA联合计算track/new/false/miss概率
        静止/慢速移动双模型IMM自动融合并预测theta_deg
    ↓ TrackedDirection[0..3] + active_tracks → 有界L3 latest-wins队列
【已完成】Layer 3：按公共track_id增强方向音频（BF）
    输入：同一WindowKey、160 ms DecisionWindow末尾的配置化40/80/160 ms LogicalAudio和0～3个权威方向
    当前可用：
        ├── optimized：双候选按rho选择Dual LCMV / Soft-null MVDR / Loaded MVDR
        │     单候选和三候选使用Loaded MVDR；数值失败逐频DAS回退
        ├── ds_baseline：7麦Delay-and-Sum；当前只按单声源使用
        └── loaded_mvdr_baseline：全频段独立diagonal-loaded MVDR对照
    跳窗重叠STFT/IMCRA/协方差滚动复用；权重仍按当前窗口重新计算
    每个方向输出：EnhancedAudio(track_id, theta_deg, 48 kHz mono [1920/3840/7680])
    物理上限：低频波长远大于阵列孔径，80～1500 Hz方向分离效果差
    ↓
【已完成】TrackAudioStreamHub：公共逐ID连续补偿音频流
    实时拼接在L3 worker内同步执行；仅首次confirmed回填使用独立有界任务
    按(session_id, stream_epoch, track_id)从每个L3重叠窗只追加一个20 ms hop
    去除重叠重复；缺口按绝对sample审计；处理模式变化时安全重建该轨上下文
    新ID首次confirmed时，用确认时平滑角回溯最多1秒补做ID出生前缺失BF
    回填按绝对sample前插；已有实时BF槽位优先且不会被覆盖
    可选按每hop IMCRA概率执行imcra_probability_rms_v1（当前实验基线默认关闭）
    RMS目标-23 dBFS、只放大；新增增益受-3 dBFS峰值保护
    每个方向维护最长3200 ms连续48 kHz缓冲
    同一补偿后样本供L3试听、RecordingStore逐ID长WAV和完整轨停机封存
    ├── 独立chunk producer：轻量唤醒、后台claim；接纳成功后才推进每ID游标
    └── L4-L6伪实时旁路：默认4秒，可调3～15秒；失败不阻塞正式20 ms审计链
    ↓ 有界L5审计队列
【已完成】实时L5审计占位
    不执行CNN；每个成功L3窗口提交SKIPPED(reason=offline_after_l4)
    ↓
【已完成】ResultJoiner与实时有序提交
    按WindowKey和track_id校验L2/L3，并合并L5审计终态
    按全局window_id顺序提交DecisionRecord v5 + ResultWatermark
    失败、超时、丢弃和取消保留明确error终态，不静默消失
    ├── RecordingStore
    │     原始/逻辑/物理音频、IMCRA、MUSIC、连续补偿音频和实时阶段状态
    │     重叠L3窗不重复保存；20 ms hop按chunk/track_id合成长WAV
    │     60秒切块、异步写盘、journal恢复、保留和逐ID音频资产
    ├── Development Test UI
    │     实时有序审计帧 + L1/L2/L3显示 + 播放公共补偿后连续轨
    └── Audio Data Manager / Production UI
          Runtime/Test Corpus、标注、QA、回收站、导出和逐ID回放

采集中：每个完整配置块渐进发布可替换L4/L5/L6预览
    MF2使用GPU；L5/DNSMOS/CAMPPlus/L6使用CPU；2秒声纹余量可跨奇数秒块
    ↓
采集停止 + 实时队列完全排空 + 伪实时尾部有限冲刷
    ↓ TrackAudioStreamHub.seal()
【已完成】按权威ID封存离线输入
    每个(session_id, stream_epoch, track_id)只生成一条完整48 kHz长轨
    跨缺口补等时静音并记录L2方向数0；Test UI丢弃短于2秒的轨
    ↓ Test UI在封存完成后自动提交（封存前选择模型）
【已完成】Layer 4：采集后1/2人语音分离与主候选匹配
    48→16 kHz；人数=min(2, 整轨L2方向数最大值)
    ├── 1人：直接旁路为L4输出
    └── 2人：MossFormer2或TIGER → 两条匿名候选
          30秒分块/1秒重叠、块间排列修复与交叉淡化
          固定不合并：1～4 kHz复频谱相干度仅用于降序标记A/B（A高、B低）
          两条候选均显示原ID/A-B及分数，并分别进入L5
    ↓ 最终16 kHz音频运行DNSMOS，保存SIG/BAK/OVRL并显示0～1综合MOS分数
    ↓ 原生16 kHz PCM16 WAV写入L4临时试听缓存
【已完成】Layer 5：L4整批完成后自动运行人声判断
    同一L4原生16 kHz波形直接进入NVIDIA Frame-VAD MarbleNet，不再重采样
    每320样本输出一个20 ms概率和Voice/Non-Voice；默认阈值0.70
    整轨摘要=完整概率序列中连续3帧均值的最大值
    ↓ 按原track_id回写L4音频条；Voice区间标黄，失败保留L4试听音频
    ↓ 每批L4与L5成功完成后自动运行L6
【已完成】Layer 6：整次录音人物归类与重复音频择优
    每条A完整提取CAMPPlus 192维声纹
    B仅在A/B匹配度差≤0.20、B匹配度>0.50且B MOS>0.30时完整提取声纹
    入选整轨声纹两两计算相似度，经平均链接AHC聚为0～3人
    每个声纹关联一条或多条A/B音轨；按录音绝对时间线逐20 ms填入
    重叠部分保留MOS更高音频；无音频处补静音
    删除首尾静音，内部超过2秒的静音压缩为2秒，按声纹显示Speaker A/B/C

【已完成】独立L1 Spectrum UI（平行工具，不接入上述Runtime）
    自建L1-only采集链：设备自动扫描、校准、IMCRA、可选预降噪、电平与频谱抓拍
    不创建WindowAssembler、L2、L3、L4/L5、正式录音或数据管理服务

【已完成】独立Pipeline Log UI（观察平面，不是Layer 5）
    只通过Recording/Data公开只读查询接口读取封存session
    记录列表、总览、Pipeline时间线、单窗详情、ID与异常
    不消费Runtime latest-only邮箱，不打开私有Catalog，不控制或反压实时链
    独立进程尚无跨进程只读端口：未注入provider时明确显示Unavailable

运行约束：
    同窗实时审计严格L2(n) → L3(n) → TrackAudioStreamHub → L5(SKIPPED: offline_after_l4)
    跨窗正式审计稳态为L2(n) || L3(n-1)；L4-L6预览在独立旁路渐进运行
    停机封存后的完整L4/L5/L6仍是权威结果，并原子替换全部preview
    各阶段单worker、队列/Joiner/缓存均有界；满队列按latest-wins替换未开始旧窗
    L2先登记每个权威ID的绝对20 ms时间槽，L3只填BF波形，缺失槽保留等时静音
    既有optimized隔离L3基准已低于20 ms节拍；真实阵列全链并发仍待复测
```

上图描述当前1.3.5开发线代码实现；`【已完成】`表示模块和自动化契约已经接通，不代表真实阵列、诊室声场、中文目标域或长时间负载已经验收。独立Pipeline Log UI的详细只读边界见[`LOG_UI_ARCHITECTURE_V1.1_TARGET.md`](LOG_UI_ARCHITECTURE_V1.1_TARGET.md)。

### 渐进L4-L6与采集后权威校正

L4/L5/L6不进入正式20 ms `WindowKey`审计。采集中由独立旁路按默认4秒、可调3～15秒的连续ID块渐进运行：单人旁路，双人MF2使用1秒重叠修复换序；L5保留跨块上下文，只发布稳定帧；L6缓存跨块2秒CAMPPlus证据并节流更新。Preview可被后续revision替换，不写DecisionRecord或RecordingStore。停止采集并排空L3后，旁路在有限期限内冲刷尾部，`TrackAudioStreamHub.seal()`再封存完整ID长音频和L2方向数量。Test UI随后运行完整L4/L5/L6；该canonical批次成功后一次性替换preview，迟到preview不能回写。双人轨的两条原生16 kHz候选按累计1～4 kHz复频谱相干度标记A/B并分别进入L5；单人轨保留唯一旁路。完整批次执行精确DNSMOS、MarbleNet和CAMPPlus/L6，仍是最终权威结果。

## 算法流程说明

### 1. 输入和坐标统一

设备原生输出8通道。Layer 1先把它们整理成统一的逻辑顺序：`MIC0～MIC5、Center、HardwareMix`。前7路具有明确的物理位置，构成实际阵列；HardwareMix没有空间坐标，是阵列自己合成的。不参加MUSIC或波束形成。

整个项目只使用一套麦克风面坐标。几何、定位、波束形成、界面角度和录音manifest必须引用同一坐标，避免出现镜像、顺逆时针相反或固定角度偏移。

### 2. IMCRA噪声分析和可选预降噪

IMCRA是一种递归噪声估计算法。它持续估计每个麦克风、每个频率位置的噪声功率和语音存在概率。

项目各部分选取不同频率区间

- 0～10000 Hz：用于IMCRA噪声统计和可选预降噪；
- 80～8000 Hz：用于Layer 3增强；离线L4/L5读取16 kHz时域音频，不使用该实时频带裁剪；
- 500～4000 Hz：聚合成Layer 2的声源Gate概率；
- 2000～4000 Hz：用于Rolling NormMUSIC方向定位。

可选的IMCRA-Wiener预降噪默认关闭。开启后，每个物理麦克风使用自己的噪声估计计算Wiener增益，再用重叠相加方式连续重建音频。它只改变送往后续算法的7路物理音频，不修改HardwareMix和保存的原生音频。

### 3. 唯一时间轴和滑动窗口

系统为每次采集建立唯一`session_id`，用`stream_epoch`表示连续音频段，并用绝对sample编号描述时间。发生输入丢失或不连续时会切换epoch，防止把不连续音频误拼到同一个算法窗口。

WindowAssembler累计160 ms上下文，之后每20 ms产生一个新窗口。因此算法每秒最多形成50个判断点，每个`DecisionWindow`继续携带160 ms音频。L2在自己的有界滚动状态中累计当前配置的240 ms定位历史；L3从DecisionWindow末尾截取`timing.downstream_audio_window_ms`，当前为40 ms。L3之后的`TrackAudioStreamHub`按精确ID每窗只追加一个20 ms hop；响度补偿当前默认关闭。正式逐窗L5不读Hub片段，但隔离旁路会按3～15秒连续块运行带上下文的L5 preview；完整轨仍在停机排空后封存。

### 4. Probability Gate

Layer 2先计算窗口末尾两个连续20 ms声源概率的平均值，达到默认阈值`0.60`才运行MUSIC。L2内部仍保留按精确ID接收在线人声反馈后强制放行的兼容接口，但正式L5和渐进preview都不回传L2，因此普通1.3.5运行不会触发该强制放行。预热、缺失或无效概率同样保持阻断。

Gate阈值与MUSIC候选阈值是两个不同参数：前者决定“是否启动或继续定位”，后者决定“伪谱上的峰是否足够可信”。

### 5. Rolling NormMUSIC二维定位

MUSIC维护多帧STFT的逐频7×7协方差，只使用7个物理麦和2000～4000 Hz频率。每频点执行加载/收缩与Hermitian特征分解，普通路径直接使用Test UI手动选择的1/2/3阶信号子空间，NormMUSIC式逐频归一化后在0～359°逐度形成伪谱。该手动值同时决定最多搜索峰数。每轮取未屏蔽区域内符合当前Test UI候选门限与prominence的最强局部峰，屏蔽与已选峰圆周距离小于50°的区域后继续，恰好50°允许共存；无达标峰时提前停止，最终输出0～手动阶数个备选方向。

Test UI保留DPD与白化的独立持久化开关。`DPD + rank-1 MUSIC`按逐频主特征值间隙、平面波拟合度以及IMCRA的SPP/先验SNR筛选可靠频点，每个可靠频点按单源噪声子空间产生方向票，再执行跨359°/0°连续的圆周核聚类。当前每个合格簇至少需要4个支持频点、覆盖4个等宽子带中的2个、加权支持率不低于0.20、圆周集中度不低于0.85。归一化峰值均严格大于0.70且组内任意峰圆周距离不超过40°时，先按唯一支持频点权重融合为圆周平均角，再执行50°圆周NMS；蓝色投票谱不做二次归一化。合格簇数量决定0～手动上限个候选。`IMCRA噪声白化`直接读取DecisionWindow已有的READY L1 `noise_covariance[427,7,7]`，经频率插值、收缩和loading后，以批量Cholesky同时白化观测协方差和steering；不在L2重新估计噪声矩阵。没有READY数据或完整空间协方差时本窗明确标记`unavailable`并安全退回未白化计算。L2队列丢窗是独立的Runtime过载状态，不代表Gate或L1 IMCRA不可用，Test UI会保留最近一次成功结果并标记`STALE | L2 DROPPED`。

达到L2 `confirmed`的方向ID在最后一次MUSIC观测后的2秒几何TTL内继续保留。新MUSIC峰用于更新角度；短时漏检时，公共投影可按该ID的保持/预测角继续每20 ms生成BF音频。当前离线L5不参与实时槽位准入、排序或续租；tentative轨迹不会进入L3。

50°是同一窗口内两个候选之间的最小角距。完整360°空间响应会保留供诊断，公共候选只保留角度、分数和时间身份。

当前已知误检是笔记本电脑风扇：当麦克风阵列通过约50 cm连接线放在笔记本电脑附近、现场只有一个主要人声声源时，散热风扇仍可能形成一个稳定候选方向。Gate和MUSIC只能说明某方向存在符合频带和空间特征的声源。渐进L5可提前显示人声概率，但不回写或删除L2方向；停止后的canonical供最终检查。由于BF方向分离能力有限，另一方向的人声串入仍可能使L5误判。

### 6. 公共方向ID和圆周卡尔曼

方向ID追踪是Layer 2默认开启的正式能力（Development Test UI可为MUSIC诊断临时旁路）。它使用全局一对一分配，把不同窗口中的观测关联为`tentative / confirmed / coasting / deleted`轨迹；`track_id`在同一session内单调分配且不复用，并原样进入L3、Runtime、DecisionRecord v5以及停机后的L4/L5、Development Test UI和Production UI。它只表示空间方向轨迹，不表示人物身份。

新方向首次出现时立即分配tentative ID；只有在滚动200 ms窗口内累计至少5次匹配观测且存在概率达标，才进入tracking `confirmed`。候选关联使用50°圆周硬上限与卡方门限20。JPDA主关联不足时，补救匹配同时比较IMM预测角和该ID最后一次真实观测角；候选距任一个不超过50°即恢复旧ID并禁止重复birth。两条既有轨迹后来进入50°以内时，只有在滚动200 ms内两轨均有观测、观测至少两次交替且不存在同窗独立双峰时才执行同源归并；保留confirmed优先、随后更早、存在概率更高的ID，并吸收另一轨较新的IMM状态和观测证据。

达到tracking `confirmed`的轨迹可以在2秒TTL内以`coasting`状态继续作为公共L3方向输出；公共投影只依赖L2的权威观测、关联、角距和数量限制。代码保留在线语义反馈接口以兼容旧契约和专项测试，但当前离线L5不调用它，所以不会实时强制打开Gate、标记噪声干扰或改变轨迹寿命。

圆周IMM是ID Tracking的固有组成：追踪开启时自动在静止与慢速运动模型间调整概率，并按权威ID平滑角度、估计角速度和支持短时预测。Test UI不提供独立Kalman开关或Q/R控件；关闭ID Tracking会整体旁路公共轨迹，而不是只关闭平滑器。

### 7. 按方向增强音频

Layer 3对每个候选方向生成一条由`timing.downstream_audio_window_ms`统一控制的48 kHz单声道增强音频；当前为40 ms。`optimized`模式的0/1/2候选保持既有BF策略，3候选分别使用IMCRA噪声协方差Loaded MVDR，失败逐路回退DAS。`ds_baseline`只按单声源方法使用；`loaded_mvdr_baseline`提供全频独立对照。五频段`subband_robust_baseline`与固定30°`constant_beamwidth_baseline`均已从正式代码删除并由入口明确拒绝。

优化BF会按频率和空间可分度选择处理方式：两个方向导向矢量相关度较低时才适合施加较强的双约束分离；相关度较高时必须转为更保守的MVDR或DAS，避免病态求解和目标失真。尤其在低频段，阵列提供的方向差异不足，算法的目标是稳定保留音频而不是承诺把两个低频声源彻底拆开。

#### 逐频点选择算法

每个频率bin独立选择处理方法。下表中的`rho`是两个候选方向在当前频率上的空间相关度，不是Layer 2输出的声源概率：

| `rho`范围 | 当前算法 | 处理目的 |
|---|---|---|
| `rho < 0.3` | Dual LCMV | 保留目标，同时对另一人形成硬零陷 |
| `0.3 ≤ rho < 0.7` | Soft-null Loaded MVDR | 对另一人进行较柔和的抑制 |
| `rho ≥ 0.7` | Loaded MVDR | 不强行分离，只保持目标方向并抑制噪声 |
| 数值不稳定 | DAS | 单频点安全回退 |

历史实机链路的主要性能瓶颈位于Layer 3 BF。当前实现已经加入跳窗重叠复用，并将LCMV/MVDR改为批量Cholesky求解；本机隔离L3基准已低于20 ms节拍，但真实麦克风下与L1/L2/L5/UI并发的持续吞吐仍待重新验收，因此不能仅凭扩大队列或隔离基准宣称实机丢窗已经清零。

### 8. L5人声分类

Layer 5只判断L4输出中的逐20 ms人声状态。公共轨服务先把L3重叠窗变成按ID连续的48 kHz长音频；采集结束后，L4只执行一次48→16 kHz并把该16 kHz结果同时用于试听和L5。NVIDIA Frame-VAD直接读取每20 ms 320样本，一次输出逐帧人声概率，不再进行任何重采样。完整序列的连续3帧最大均值仅作为整轨概览，不能覆盖逐帧结果。

Test UI与CNN读取同一份Hub连续轨。响度补偿开关当前默认关闭，可实时切换且不清空轨道；这里只调整数字波形增益，不是dB SPL测量。

L5继承并标注L2已经分配的公共方向`track_id`，不向L2回送角度、不创建ID，也不改变方向轨迹生命周期。

### 9. 并行、过载和结果保存

正式实时窗口依次完成L2、L3，再由L5审计阶段写入`offline_after_l4`跳过终态；不同窗口的L2与L3可以形成分阶段流水。真正的L4-L6模型可在隔离旁路渐进执行，但只有Hub封存后的完整批次是canonical。

被丢弃、失败、超时或停机取消的窗口不会静默消失，而会按原时间轴写入明确的终态和ResultWatermark。这样即使实时负载过高，录音和离线分析仍能知道哪些时间点没有得到完整结果。

## Development Test UI使用方法

Development Test UI用于开发、算法观察和实机标定。它与正式处理共用同一个ApplicationRuntime。它目前是主要的交互入口，但不是最终产品界面。

### 1. 环境准备

已验证环境为Windows、Python 3.12、NVIDIA RTX 5060和项目根目录下的`.venv`。建议在VS Code中依次运行以下任务：

1. `环境：创建或更新 RTX 5060 专用环境`
2. `环境：GPU 完整自检`
3. `测试：当前规格全部自动测试`
4. `运行：Development Test UI`

也可以在项目根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
.\.venv\Scripts\python.exe .\scripts\check_runtime_env.py --require-cuda
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gui.dev_test_ui.app
```

使用本地测试WAV检查完整链路时，可执行：

```powershell
.\.venv\Scripts\python.exe -m gui.dev_test_ui.app --input-wav "D:\audio\test.wav" --auto-start
```

输入WAV必须是48 kHz、7通道或8通道。7通道离线素材没有原生HardwareMix时由输入适配器按离线契约处理。

### 2. 推荐操作顺序

1. 连接麦克风阵列和电源，确认Windows已识别MA-USB8设备。
2. 启动Test UI，先观察全局状态栏是否有设备、CUDA或模型错误。
3. 点击左上角“启动采集”。首次形成正式窗口前需要累计160 ms音频，IMCRA也会显示预热状态；L2若配置更长的滚动定位历史，会继续独立预热。
4. 检查8路电平是否都有响应，通道名称和实际敲击位置是否一致。
5. 真实麦克风模式启动后先只运行L1/IMCRA与L2；“正式录音开始”会保留IMCRA噪声统计，但清空L2 MUSIC/ID历史和Center预听，再开始本轮临时L3 BF/按ID拼接，不写入正式音频存储系统。只想留一小段临时素材时仍使用“录制/暂停/结束”scratch录音。
6. 在右上角观察Gate、360°空间响应和候选角度，再根据测试目的切换ID Tracking总开关或L3对照模式；IMM随ID Tracking一同运行，没有独立开关。
7. 在下左L3区域试听Center参考和已确认/保持的方向音频，并在L4区域预先选择MossFormer2或TIGER；结束采集后等待L3排空和Hub封存，Test UI会自动把完整长音频提交到L4。
8. L4整批完成后L5会自动运行；在下中L4音频条查看黄色Voice区间，在下右L5查看整轨概率。下一次采集或模拟重播封存后会自动清空并替换上一批离线结果。
9. 结束实验时先点击“正式录音暂停”，并结束仍在进行的scratch录音；再点击“停止采集”，等待流水线排空和需要的离线处理完成后关闭窗口。

### 3. 四象限分别表示什么

#### 左上：Layer 1输入与录音

- 显示`MIC0～MIC5、Center、HardwareMix`共8路电平和削波状态；
- 显示IMCRA预热、噪声摘要和预降噪平均增益；
- “IMCRA预降噪”可在采集中切换，修改从后续完整窗口生效；
- “灯光开/灯光关”用于检查板卡控制链；
- scratch录音位于`data/dev_test_ui/scratch/current`，用于临时测试；
- Production/普通Runtime的正式录音位于`data/runtime_sessions/YYYY/MM/<session_id>`，包含正式时间轴和算法sidecar；真实麦克风Test UI明确不创建该目录中的session。

#### 右上：Layer 2方向定位

- 圆环蓝线是当前窗口完整360°原始MUSIC伪谱；
- 候选点使用L2最终平滑角，因此可能与原始峰顶略有偏移；
- 临时候选显示为灰色；正式ID使用红/绿交替颜色；
- 大点表示当前窗口有真实观测，小点表示IMM预测；
- “L2声源Gate”阈值决定是否执行定位，默认0.60；
- MUSIC候选阈值决定伪谱峰是否进入候选；
- “MUSIC阶数”可设为1、2或3，直接设置普通路径的信号子空间阶数和最多搜峰数；
- DPD rank-1 MUSIC默认关闭；IMCRA空间噪声协方差白化默认开启，二者仍可独立切换；
- 公共方向ID追踪默认开启；Test UI只保留`ID Tracking`总开关，不再提供独立Kalman或Q/R控件；
- 开启ID Tracking即运行完整Circular IMM-JPDA，针对静止和慢速移动自动调整模型概率。

如果显示`WARMING_UP`或`UNAVAILABLE`，表示概率尚未准备好、输入不连续或必要数据缺失。

#### 左下：Layer 3方向增强与试听

- 前两行是`Center Mic RAW`与`Center Mic IMCRA`：前者为校准后、预降噪前原音，后者仅缓存预降噪开启期间实际采用的Center降噪输出；
- 方向轨严格按L2权威`track_id`缓存和显示，Test UI不再按角度创建第二套ID；
- confirmed方向短时漏检时可进入coasting并在2秒TTL内沿用同一ID；
- L3按键显示为`DS`、`MVDR`和`LCMV`，默认使用DS；内部仍分别对应`ds_baseline`、`loaded_mvdr_baseline`和`optimized`；
- 切换L3模式会清空旧模式的方向试听缓存；

#### 下中与下右：离线Layer 4 / Layer 5 / Layer 6

- L4只在采集停止、实时队列排空且Hub完成封存后可提交；可选择MossFormer2或TIGER；
- L4输出是原生16 kHz单声道WAV，保留原`track_id`和角度，可直接试听；
- L4整批完成后自动且仅一次运行L5，不再提供单独“发送到L5”按钮；
- L5为每个20 ms hop输出概率和Voice / Non-Voice，Voice区间在对应L4音频条标黄；
- L5阈值只重新判断缓存概率并重绘黄色区间，不改变L2 Gate，也不重跑模型；
- L4固定保留A/B双候选并分别进入L5；每批L4/L5完成后自动运行L6，结果按声纹显示Speaker A/B/C，标出关联音轨数、来源L2 ID和平均MOS。


### 4. 参数调整建议

初次实机测试应保持默认配置，只改变一个参数并记录前后结果：

1. 先保持当前实验基线：预降噪关、DPD关、IMCRA白化开、ID Tracking开，检查Gate、MUSIC伪谱和公共方向ID关联；
2. 再分别把MUSIC阶数设为1、2、3，记录候选数量和误峰变化；
3. 需要试验鲁棒定位时，每次只改变DPD或IMCRA白化中的一个，并与上述基线对照；IMM随ID Tracking固定运行，不在Test UI单独调参；
4. 使用同一段录音比较`LCMV`（内部`optimized`）、`DS`（内部`ds_baseline`）和`MVDR`（内部`loaded_mvdr_baseline`）；DS只用于单声源，双声源重点比较LCMV与MVDR；
5. Gate阈值、MUSIC候选阈值和L5分类阈值分别记录，不要把三者当成同一个“灵敏度”。

## 录音数据和配置

唯一业务配置是[`config/config.yaml`](config/config.yaml)。运行时队列容量、算法开关、模型、Test UI和录音保留策略都由严格schema检查，非法值会在启动前失败。

正式录音默认按60秒切块，可保存：

- 原生8通道、逻辑8通道和物理7通道音频；
- IMCRA噪声与概率sidecar；
- 360°空间响应和候选方向；
- `TrackAudioStreamHub`补偿后的逐ID连续增强音频（每个录音chunk合成长WAV）；
- 实时L2/L3结果、L5的`offline_after_l4`跳过状态、阶段耗时和错误原因；
- DecisionRecord、ResultWatermark、manifest和配置hash。

Test UI离线L4试听WAV保存在独立临时目录，L5逐20 ms概率保存在该次离线结果对象中；它们不会伪装成实时ResultJoiner结果。需要长期保存离线结果时使用`scripts/run_offline_l4.py`的显式输出目录。

配置中的`privacy.local_only=true`、`automatic_upload=false`表示项目默认只在本机保存且不自动上传。但这只是软件默认行为，不等同于完成医疗数据合规。诊室采集前仍需要取得授权、限制访问权限、制定保留和删除策略，并根据所在地区要求进行去标识化和审计。

## Pipeline Log UI（1.3.3已实现）

项目已提供独立只读的 Pipeline Log UI，用于查看单次运行记录中的阶段性能、终态、MUSIC/方向 ID、L3增强资产、已持久化时可用的L5结果、丢窗和时间线。当前Test UI离线L4/L5临时结果不会自动进入Log UI。其项目地位与 L1～L5、Development Test UI、RecordingStore/Audio Data Manager 平行，不属于算法流水线的下一层。

当前版本优先完整回看已完成、已封存的 session；若由同一正式进程显式提供 Runtime 引用，只额外展示公开的聚合运行状态。Log UI 不消费 Development Test UI 的 latest-only 邮箱，也不直接打开私有运行时数据绕过公共查询边界。

Log UI 只能统计、展示和回放，不得启动/停止 Runtime、修改算法参数、标注/导出/删除数据、重建 Catalog，或在项目数据目录写缓存。公开接口未提供的数据明确显示 `N/A`。页面、数据覆盖、统计口径、兼容性和只读验收详见[`Log UI 1.1架构`](LOG_UI_ARCHITECTURE_V1.1_TARGET.md)。

## 当前完成情况

以下主链已经接通并有自动化测试覆盖：

- L1多通道输入、IMCRA和可切换预降噪；
- 唯一时间轴与160 ms/20 ms窗口装配；
- L2 Probability Gate、Rolling NormMUSIC和Circular IMM-JPDA永久公共方向ID；
- L3优化BF、单声源DAS和全频loaded MVDR；五频段与固定30°波束模式均已移除；
- TrackAudioStreamHub逐ID去重拼接、响度补偿、连续试听/录音轨和停机封存；
- 采集后L4一/二人路由、MossFormer2/TIGER分离、复频谱相干匹配和原生16 kHz试听；
- L4完成后自动运行的离线MarbleNet L5逐20 ms人声结果与黄色区间；
- 有界实时L2/L3/L5审计流水、ResultJoiner和有序提交；
- Development Test UI、独立L1 Spectrum UI、正式录音、scratch录音、Audio Data Manager、Production UI和独立只读Pipeline Log UI。

“代码已接通”不代表已经完成诊室部署验收。下列实机与目标域门禁仍需完成。

历史v3/v4实测曾出现较高的处理丢窗率和偏低的有效输出帧率。这里指算法处理队列中的window/frame drop，不是已经确认的USB输入数据包丢失。当前1.3.3已经优化Layer 3滚动缓存和矩阵求解，但尚无新的v5真实阵列全链session可以证明持续输入下的最终丢窗率。

最近一份可计算的正式缓存来自2026-08-18 14:30，session为`ea30a66d-9d4b-44c5-bb52-e106c85e05ed`，记录了15.36秒、768个20 ms窗口。统计结果如下：

| 项目 | 数量/结果 | 占全部窗口 |
|---|---:|---:|
| 完整输出 | 186 | 24.22% |
| 任一处理阶段丢窗 | 582 | 75.78% |
| L3入队溢出 | 581 | 75.65% |
| L5入队溢出 | 1 | 0.13% |
| L2完成 | 768 | 100% |

这段缓存的完整结果有效输出率约为`12.11帧/秒`。187个实际执行完成的L3窗口中，BF计算耗时平均`81.3 ms`、P50为`79 ms`、P95为`125 ms`，明显大于每20 ms到达一个新窗口的输入速度。对应manifest中`missing_intervals=[]`，说明录制区间音频连续；75.78%主要是算法处理队列丢窗，而不是已确认的USB采集缺口。

需要注意，这份缓存使用`circular_id_tracker_v3`，早于当前v4实现。之后生成的v4 session目录目前只有manifest，没有正式DecisionRecord，因此尚不能从现有文件计算v4的真实丢窗率。README中的75.78%只能作为最近一次有完整证据的性能基线，不能冒充当前v4最终结果。

2026-08-19完成L3内部求解与跳窗滚动修复后，在本机RTX 5060 Laptop GPU上使用连续合成窗口、每档120个计时样本得到隔离L3端到端P95：1/2/3声源分别约`9.23/15.96/9.52 ms`，平均吞吐约`146.63/86.26/136.15窗/秒`；三档均低于20 ms输入节拍。该结果不包含真实麦克风、L1/L2/L5/UI并发，不能替代完整实机复测。当前`runtime.stage_queue_windows=100`会同步设置L2/L3/L5等待容量，单层等待硬上限约2秒，避免旧2000窗配置隐藏长期过载并占用过多内存；以后只需改这一变量，Joiner容量会自动派生。仍需用正式session持续核对队列水位、端到端延迟、内存与真实丢窗率。

2026-08-24短窗有效频带优化后的默认设备拓扑改为`L1 CPU → L2 CPU → L3 CPU → 离线L4 CUDA → L5 CPU`。L3、L4、L5分别由`runtime.l3_device/l4_device/l5_device`控制；旧`preferred_device`只用于兼容缺少独立字段的配置。上述2026-08-19 CUDA数据保留为历史基线，不代表当前默认L3设备。

同日进一步实现了L3 CUDA有界微批实验路径：STFT、IMCRA协方差、波束形成和ISTFT保持在GPU，只有批次完成后的短音频通过pinned host buffer回传，由CPU继续ID连续音频拼接和写盘；批次只合并已经积压的窗口，绝不等待未来窗口。RTX 5060 Laptop GPU实测双候选连续四窗约`11.03 ms/窗`，同条件CPU约`6.21 ms/窗`；28.8秒真实八通道回放的CPU完整排空约`29.07 s`，CUDA最佳约`32.30 s`，所以正式默认仍为CPU。显式设置`runtime.l3_device: cuda`可继续诊断，停机后创建L4前会同步L3 stream、释放L3缓存再清理CUDA allocator。

2026-08-24全链优化后，CPU L3进一步拆为“候选无关STFT/IMCRA准备→DS/MVDR/optimized BF+ISTFT→CPU ID拼接/发布”三个有界FIFO阶段，并将7通道小矩阵的PyTorch CPU intra-op固定为1线程。28.84秒双声源模拟压力录音中，L2约`28.83 s`、L3含拼接约`33.34 s`，无丢窗或阶段错误；10秒真实MicArray采集并同时运行Test UI时，L2/L3均在`9.875 s`排空且所有L3队列峰值为0。完整链实测CUDA约`42.33 s`、双BF worker约`42 s`，因此正式最快配置仍是CPU单BF worker与层间流水线。模拟输入不读取录音时的CDC热力图，但会把重新运行L2得到的DOA结果实时显示在Test UI极坐标控件中；按ID连续音频只在首次确认时复制一次历史，之后逐20 ms hop追加。

## 局限性和待解决问题

### 1. 低频空间分离受阵列孔径限制

- 低频波长远大于麦克风间距，不同方向在阵列上的相位和时延特征过于接近；
- 空间可分度`rho`在低频区域接近1，代表方向响应高度相关，双声源约束难以稳定建立；
- L3保留80～8000 Hz是为了保留人声频段内尽可能多的信息；当前80～1500 Hz无法分离；
- 低频串音、风扇和空调残留是当前架构可以预期的结果，不能通过提高卡尔曼精度或CNN阈值解决；
- 真正改善低频分离通常需要更大的阵列孔径或新的硬件布局，因此属于后续硬件方案问题。

### 2. 二维远场假设的限制

- 不估计距离、俯仰角或声源高度；
- 同一水平角上的近远两个声源无法区分；
- 近场说话、人在阵列正上方、明显高度差或快速移动可能破坏平面波假设；
- 墙面、桌面、玻璃和医疗设备造成的强反射可能产生错误峰或角度偏移。

### 3. 多人和角度分辨能力有限

- 当前每个窗口最多输出三个候选方向；
- 任意两个候选必须至少相距50°；
- 当前方向结果不包含物理响度、声压级或角度不确定度；
- 2000～4000 Hz内持续存在的非人声声源也可能形成稳定候选，例如靠近阵列的笔记本电脑散热风扇；
- 三候选是算法输出上限，不等于已经验证可可靠分离三人同时说话；方向过近、强混响或说话人交叉移动仍可能换轨或串音；
- 公共`track_id`只表示空间方向轨迹，不表示人物身份，不能据此判断谁是医生或患者。

### 4. 实机几何和动态参数尚未完全标定

- 硬件通道、极性、固定延迟和真实角度仍需校准；
- MUSIC候选阈值、Gate阈值、ID关联门限、IMM内部运动模型、2秒coasting TTL和3秒静止判定历史需要用真实诊室数据标定；
- 当前IMM内部运动模型偏保守，主要面向静止或低角速度声源；快速移动时可能产生明显跟踪滞后；这些内部参数不作为Test UI的独立Q/R控件暴露；
- 仍需覆盖静止、移动、交叉、混响、双声源和不同座位布局；
- 麦克风或设备型号变化后不能直接沿用原参数。

### 5. 人声模型仍需诊室目标域验证

- 当前MarbleNet是NVIDIA预训练模型的直接接入版本，没有进行项目数据微调；
- 模型同时使用中文和英文训练数据，但英文数据来源更多，预期英文人声检测表现可能优于中文；官方未公布中英文直接对比准确率，仍需用诊室中文对话实测；

### 6. 实时性和稳定性门禁尚未完成

- 现有CUDA逐阶段基准不等于端到端延迟；
- 历史v3/v4缓存的高丢窗主要发生在Layer 3；当前滚动STFT/IMCRA/协方差复用和批量求解已改善隔离性能，但没有新的v5全链实测可证明问题已经消失；
- 仍需测量当前有效输出帧率，并继续评估BF计算、GPU利用、默认100窗阶段队列的水位/丢弃以及结果投影开销；
- 仍需测试采集、GPU计算、队列等待、有序提交、录音写盘和UI共同运行时的实际延迟；
- CUDA OOM、CPU fallback、设备断连和持续过载需要实机故障注入；


### 7. 产品化和临床使用尚未完成

- Development Test UI面向开发者，控件和诊断信息不适合直接交给诊室工作人员；
- 正式`app.main`入口、最终用户方向界面和完整实机UI门禁仍未完成；


## 进一步阅读

- [完整架构、逐层输入输出与使用手册](docs/COMPLETE_ARCHITECTURE_AND_USAGE.md)
- [总执行规格](CODEX_PROJECT_SPEC_6plus1_2D_voice_direction_v0.2.md)
- [v0.3目标架构与迁移契约](ARCHITECTURE_V0.3_TARGET.md)
- [1.3.3 MUSIC、公共方向ID与平行子系统架构](ARCHITECTURE_V1.1_TARGET.md)
- [Pipeline Log UI 1.1架构](LOG_UI_ARCHITECTURE_V1.1_TARGET.md)
- [Windows与RTX 5060环境说明](ENVIRONMENT.md)
- [Layer 1说明](layer1_input/README.md)
- [Layer 2说明](layer2_source_detection/README.md)
- [Layer 3说明](layer3_direction_signal/README.md)
- [离线Layer 4双人分离契约](layer4_speech_separation/README.md)
- [Layer 5说明](layer5_voice_classifier/README.md)
- [Layer 6说明](layer6_speaker_consolidation/README.md)
- [ApplicationRuntime说明](app/README.md)
- [Development Test UI说明](gui/dev_test_ui/README.md)
- [Audio Data Manager说明](data_management/README.md)

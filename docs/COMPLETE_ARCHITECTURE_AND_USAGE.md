# 6+1麦克风阵列：完整架构与使用手册

> 适用版本：项目`1.3.2`，发布基线`v1.3.2`。本文根据当前代码、`config/config.yaml`和各层公开数据类型编写，供第一次接触项目的开发者和测试人员使用。

## 1. 系统目的与边界

系统用于诊室医患对话的本地采集、二维水平方向定位、按方向增强、采集后语音分离和人声概率判断。硬件是半径`4 cm`的6个圆周麦克风加1个中央麦克风，设备以`48 kHz / PCM16 / 8通道`输出。

系统估计的是相对于麦克风面的水平角`theta_deg ∈ [0°,360°)`：`MIC0=0°`，逆时针为正。系统不估计距离、俯仰角、说话内容、声压级或人物身份。`track_id`只代表一条空间方向轨迹。

当前实时链只运行到L3和按ID连续音频Hub。L4、L5必须在停止采集、实时队列排空并封存完整方向轨后运行。实时L5位置只写入`offline_after_l4`审计跳过状态，不运行CNN。

## 2. 完整系统架构图

```mermaid
flowchart TB
    SRC[诊室声场<br/>医生 / 患者 / 环境噪声] --> HW[Sipeed R6+1 + MA-USB8<br/>48 kHz PCM16 HostAudio N×8]

    subgraph REALTIME[实时采集与处理链：每20 ms推进一次]
      direction TB
      L1[Layer 1 输入处理<br/>解码·校准·重排·连续性检查<br/>IMCRA·可选Wiener预降噪]
      ING[IngestCoordinator<br/>session / epoch / 绝对sample时间轴]
      WIN[WindowAssembler<br/>160 ms DecisionWindow<br/>20 ms hop]
      WORK[WindowWorkItem<br/>WindowKey + 冻结配置]
      L2[Layer 2 方向定位<br/>Gate·Rolling NormMUSIC<br/>峰值/NMS·Circular IMM-JPDA]
      L3[Layer 3 按方向增强<br/>STFT/协方差·steering<br/>LCMV/MVDR/DAS·ISTFT]
      HUB[TrackAudioStreamHub<br/>每ID取唯一20 ms hop<br/>响度补偿·去重·补洞·连续缓存]
      AUDIT[实时L5审计占位<br/>SKIPPED: offline_after_l4]
      JOIN[ResultJoiner<br/>终态合并·顺序提交·watermark]

      L1 -->|DecodedAudio N×8| ING
      ING -->|IngestedAudioBlock N×8| WIN
      WIN -->|DecisionWindow 7680×8| WORK
      WORK -->|有界L2队列| L2
      L2 -->|TrackedDirection 0..3<br/>SpatialResponse 360点| L3
      L3 -->|EnhancedAudio 0..3<br/>48 kHz mono| HUB
      HUB --> AUDIT
      L2 --> JOIN
      L3 --> JOIN
      AUDIT --> JOIN
    end

    HW --> L1

    subgraph STORAGE[实时旁路与观察面]
      direction LR
      REC[RecordingStore<br/>60秒切块·事务写盘·恢复]
      DEV[Development Test UI<br/>L1/L2/L3观察与试听]
      PROD[Audio Data Manager / Production UI<br/>录音·标注·QA·回放·导出]
      LOG[Pipeline Log UI<br/>只读审计与时间线]
    end

    JOIN --> REC
    JOIN --> DEV
    REC --> PROD
    REC --> LOG
    HUB -->|同一补偿后20 ms音频| DEV
    HUB -->|逐ID连续WAV| REC

    subgraph OFFLINE[采集停止后的离线链]
      direction TB
      SEAL[Hub.seal<br/>完整48 kHz方向长轨<br/>Layer4LongAudioInput]
      ROUTE[Layer 4人数路由<br/>min(2, 整轨L2最大方向数)]
      SEP[双人分离<br/>MossFormer2或TIGER<br/>30 s块 / 1 s重叠]
      MATCH[合并开关<br/>开：1–4 kHz匹配并保留高分<br/>关：显示A/B双候选]
      L4OUT[L4输出<br/>16 kHz mono PCM16 WAV<br/>保留track_id与theta_deg]
      L5[Layer 5 NVIDIA MarbleNet<br/>每320样本输出20 ms概率<br/>阈值0.70]
      OFFOUT[Layer4OfflineResult<br/>逐帧Voice + 整轨摘要<br/>Test UI显示或显式持久化]

      SEAL --> ROUTE
      ROUTE -->|1人| L4OUT
      ROUTE -->|2人| SEP --> MATCH --> L4OUT
      L4OUT -->|整批完成后逐输出运行| L5 --> OFFOUT
    end

    HUB -->|停止采集并排空| SEAL

    L1ONLY[独立L1 Spectrum UI<br/>仅设备/L1/IMCRA/频谱<br/>不进入Runtime]
    HW -.独立入口.-> L1ONLY
```

图中实线是正式数据流，虚线是独立观察工具。L2有独立worker；CPU L3再分为候选无关STFT/IMCRA准备、候选相关BF+ISTFT、host连续音频拼接三个有界FIFO worker。同一窗口仍遵循`L2(n) → L3(n) → L5审计(n)`，不同窗口可形成跨层和L3内部流水；L3阶段只有host拼接排空后才算完成。L4/L5不与实时链并行运行。

## 3. 逐层输入、内部处理单元与输出

| 模块 | 正式输入 | 内部处理单元 | 正式输出 | 频率/节拍 |
|---|---|---|---|---|
| 硬件输入 | 诊室声场 | 6个半径4 cm圆周麦 + Center；MA-USB8通道汇聚 | Host原生PCM16 `[N,8]` | 48 kHz；默认每块960 sample=20 ms |
| Layer 1 | Host PCM16 `[N,8]`、校准参数、CDC状态 | PCM16解码；增益/极性/整数延迟校准；Host→Logical通道重排；连续性检查；7麦IMCRA；可选IMCRA-Wiener WOLA | `DecodedAudio`：Logical float32 `[N,8]`；Native `[N,8]`；IMCRA噪声PSD/SPP/20 ms声源概率；健康事件 | IMCRA 0–10 kHz；Gate证据500–4000 Hz |
| Ingest | `DecodedAudio`、sequence/timestamp、校准身份 | 建立`session_id`；检测缺口并切换`stream_epoch`；分配绝对sample；把IMCRA hop对齐到同一时间轴 | `IngestedAudioBlock`：48 kHz float32 `[N,8]`，含native、hotmap、IMCRA、校准元数据 | 输入块通常20 ms |
| Windowing | 连续同epoch的`IngestedAudioBlock` | 环形累计；检查校准身份与sample连续；组合来源sequence | `DecisionWindow [7680,8]`；末端40 ms DOA区间；最近160 ms上下文；8个20 ms IMCRA hop | 160 ms上下文，每20 ms发布 |
| Runtime封装 | `DecisionWindow`、当前UI/配置revision | 创建唯一`WindowKey=(session, epoch, window_id, decision_sample)`；冻结本窗Gate/DOA/IMM-JPDA/L3设置；有界latest-wins入队 | `WindowWorkItem` | 每个DecisionWindow一个 |
| Layer 2 | `DecisionWindow`、末尾两个20 ms声源概率、7麦几何、扫描配置 | 40 ms Probability Gate；Rolling NormMUSIC；圆周峰值与50° NMS；Circular IMM-JPDA方向ID；可选DPD/IMCRA白化 | `Layer2PipelineResult`：Gate状态；`SpatialResponse` 360点；0–3个`TrackedDirection`；active tracks；MUSIC诊断 | 每20 ms判断与更新 |
| Layer 3 | `DecisionWindow`末尾40/80/160 ms、0–3个公共方向、7麦几何、IMCRA噪声 | 共享STFT与协方差缓存；steering；按`rho`逐频选择Dual LCMV / Soft-null loaded MVDR / Loaded MVDR；或DAS/全频loaded MVDR；数值保护；批量ISTFT | `Layer3Output`，其中每方向一个`EnhancedAudio`：48 kHz mono `[1920/3840/7680]`，携带`track_id/theta/algorithm/fallback` | 当前默认40 ms音频；每20 ms产生新重叠窗 |
| TrackAudioStreamHub | L3的`EnhancedAudio`、本窗IMCRA概率、L2 active IDs/方向数 | 每ID只取末尾唯一20 ms；去除重叠；按绝对sample补洞；ID首次confirmed后用确认时平滑角回溯最多1秒补做出生前缺失BF并前插（实时槽优先）；2 ms模式切换淡化；IMCRA概率响度补偿；维护完整归档；仅在消费者明确请求时构造滚动上下文 | `TrackAudioBatch`：每窗`TrackAudioHop [960]`，ID首次确认可附最长3200 ms `ContinuousTrackAudio`；停机输出`Layer4LongAudioInput`完整48 kHz长轨 | 20 ms hop；目标均值-23 dBFS，峰值不超过-3 dBFS |
| 实时L5审计 | L3/Hub阶段终态 | 不运行模型，只形成可审计跳过原因 | `L5StageResult=SKIPPED(offline_after_l4)` | 每实时窗口一个终态 |
| ResultJoiner | 同一`WindowKey`的L2/L3/L5阶段终态 | 校验ID与角度对齐；等待完整终态；按全局window顺序提交；保留失败/丢弃/取消原因 | `JoinedWindowResult`、`DecisionRecord v5`、`ResultWatermark`、UI快照 | 有序逐窗提交 |
| RecordingStore | 原生/逻辑音频、IMCRA、Joined结果、Hub hop | 异步有界写盘；60秒切块；逐ID hop合并；SHA-256；journal事务；崩溃恢复；Catalog投影 | WAV/NPZ/JSONL/manifest/Catalog；逐ID连续48 kHz增强WAV | 不反压采集 |
| Layer 4 | Hub封存的`Layer4LongAudioInput`：完整48 kHz mono、ID、角度、L2方向数历史 | 48→16 kHz；人数路由；1人旁路；2人MossFormer2/TIGER；30秒分块/1秒重叠；排列修复；交叉淡化；1–4 kHz复相干匹配度仅用于A/B排序 | Test UI固定不合并：每双人父轨两条16 kHz候选，保留父`track_id/theta`及A/B标识；单人父轨一条旁路 | 离线整轨处理；每条L4输出分别进入L5 |
| Layer 5 | L4原生16 kHz完整波形 | NVIDIA MarbleNet Frame-VAD；每320 sample直接推理；阈值比较；连续3帧均值取整轨最大值 | `Layer5LongAudioResult`：每20 ms概率/布尔值、摘要概率/判断、模型与耗时；并入`Layer4OfflineResult` | 16 kHz；20 ms一帧；默认阈值0.70 |
| Layer 6 | L4 A/B候选或单人旁路、父L2 ID/角度、匹配度、MOS、绝对时间线及L5逐20 ms概率/bool | A整轨声纹；B按匹配差/匹配度/MOS门限准入；整轨声纹两两相似度与平均链接AHC聚为0～3人；重叠按MOS择优；首尾静音删除、内部静音最长2秒 | `Layer6Result`：按声纹显示的Speaker A/B/C压缩16 kHz音频、来源L2 ID、声纹到完整音轨的一对多审计 | 每批L4/L5后自动离线执行；不参与实时链 |

### 3.1 Layer 1内部图

```text
HostAudio PCM16 [N,8]
  → 解码为float32
  → Host通道 CH0..5, HardwareMix, Center
  → 校准：gain + polarity + integer delay
  → Logical通道 MIC0..5, Center, HardwareMix
  ├─→ Native/Logical音频旁路录制
  ├─→ 前7物理麦 → IMCRA → noise_psd / 7×7 noise_covariance / SPP / p20
  └─→ 可选Wiener WOLA → 替换下游7物理麦；HardwareMix不变
```

HardwareMix没有物理坐标，不进入MUSIC、steering或BF。

### 3.2 Layer 2内部图

```text
DecisionWindow + IMCRA p20
  → 末尾2个p20平均 → 40 ms Gate（默认0.60）
  ├─ Gate关闭 → 空方向结果/明确状态
  └─ Gate打开
       → 7麦2–4 kHz增量STFT
       → 240 ms滚动协方差
       → frequency-normalized MUSIC 0..359°
       → MDL/跨频一致性诊断
       → 局部峰 + prominence + 50°圆周NMS
       → 最多3个观测方向
       → Circular IMM-JPDA概率关联
       → 预测角/最后真实观测角双参考恢复 + 交替重复轨归并
       → tentative / confirmed / coasting / deleted
       → 静止/慢速移动双模型IMM融合与预测
       → 0..3个公共TrackedDirection
```

普通MUSIC路径的信号子空间阶数和最多搜峰数由Test UI手动选择的1/2/3直接控制。`confirmed`和`coasting`方向可进入L3；`tentative`不进入L3。

### 3.3 Layer 3内部图

```text
DecisionWindow末尾音频 + TrackedDirection[0..3]
  → 物理7麦切片
  → 共享STFT / IMCRA噪声 / 协方差准备
  → 每方向steering vector
  → BF后端
       optimized:
         rho < 0.3       → Dual LCMV
         0.3 ≤ rho < 0.7 → Soft-null Loaded MVDR
         rho ≥ 0.7       → Loaded MVDR
         数值不稳定       → DAS逐频回退
       ds_baseline             → 单声源Delay-and-Sum
       loaded_mvdr_baseline    → 全频loaded MVDR
  → 批量ISTFT
  → 每方向48 kHz mono EnhancedAudio
```

L3输出保留80–8000 Hz，但80–1500 Hz受4 cm阵列孔径限制，不能认为已可靠分离。DAS基线按单声源使用。

### 3.4 Hub、Layer 4与Layer 5内部图

```text
重叠EnhancedAudio窗口
  → 每ID只追加末尾20 ms
  → 缺口补等时静音 / 重复拒绝
  → IMCRA概率控制响度补偿
  → 实时3200 ms试听上下文 + 完整归档
  → stop + drain + seal
  → 每ID一条Layer4LongAudioInput
  → 48→16 kHz
  → 人数=min(2, 整轨L2最大方向数)
       1人 → 直接旁路
       2人 → MossFormer2/TIGER两候选
             → 分块排列修复与淡化
             → 合并开启：1–4 kHz复相干匹配父L3轨，低可信时回退父轨
             → 合并关闭：不匹配，保存并显示A/B两条16 kHz候选
  → L4 16 kHz WAV
  → 每条L4输出分别进入MarbleNet逐20 ms概率
  → Voice区间 + 整轨摘要
```

## 4. 数据身份与时间轴

所有实时结果必须由同一个`WindowKey`关联：

```text
(session_id, stream_epoch, window_id, decision_sample)
```

- `session_id`：一次采集会话；
- `stream_epoch`：同一会话中的连续音频段，发生真实缺口或重启时递增；
- `window_id`：该会话内的判断窗口编号；
- `decision_sample`：该窗口在48 kHz绝对sample轴上的右端点；
- `track_id`：同一session内单调分配的方向轨编号，不是人物编号。

不能用墙钟时间、UI刷新次序或相近角度代替这些正式键。离线L4/L5继承原`session/epoch/track_id/theta`，但不会回写并改变实时L2轨迹。

## 5. 使用前准备

### 5.1 硬件与系统

- Windows；已验证Python 3.12；
- Sipeed R6+1、MA-USB8，设备采样率48 kHz、8通道；
- 默认设备拓扑为`L1 CPU → L2 CPU → L3 CPU → 离线L4 CUDA → L5 CPU`；
- `runtime.torch_cpu_threads=1`是当前16逻辑核主机的全链最快配置；不要把隔离单层的多线程或CUDA结果直接当作整链配置；
- `runtime.l3_device/l4_device/l5_device`相互独立，旧配置缺少这些字段时才回退到`preferred_device`；
- L4配置CUDA但设备不可用时，在`allow_cpu_fallback: true`下自动回退CPU；离线分离会明显变慢；
- 麦克风面朝向与坐标定义保持一致，MIC0方向作为0°。

### 5.2 创建环境并自检

在项目根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vscode_env.ps1
.\.venv\Scripts\python.exe .\scripts\check_runtime_env.py --require-cuda
```

首次使用或依赖、模型发生变化时运行完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

唯一业务配置是`config/config.yaml`。修改前应先保存实验目的；初次运行建议保留默认值。

## 6. Development Test UI完整操作流程

### 6.1 启动

无控制台启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_dev_test_ui.ps1
```

需要查看错误输出时：

```powershell
.\.venv\Scripts\python.exe -m gui.dev_test_ui.app --config .\config\config.yaml
```

使用本地48 kHz、7/8通道WAV回放：

```powershell
.\.venv\Scripts\python.exe -m gui.dev_test_ui.app --input-wav "D:\audio\test.wav" --auto-start
```

### 6.2 从采集到离线结果

1. 连接阵列，确认Windows识别设备；启动UI并检查顶部设备、CUDA和模型状态。
2. 点击“启动采集”。等待160 ms窗口累计及IMCRA预热；L2的240 ms滚动定位历史会继续独立预热。
3. 检查左上8路电平。依次轻敲麦克风，确认MIC0–MIC5、Center及HardwareMix映射没有镜像或错位。
4. 在右上查看L2 Gate、360°MUSIC谱、手动MUSIC阶数、实际候选数和方向ID。初次测试保持DPD和IMCRA白化关闭；ID Tracking开启即使用完整IMM-JPDA。
5. 在左下查看Center参考和按`track_id`排列的L3方向轨，可切换三种BF方法进行同源比较。
6. 需要正式数据时使用“正式录音开始/暂停”；只做临时试听时使用scratch录音。两者不要混作同一资产。
7. 点击“停止采集”，等待L2/L3队列完全排空和Hub封存。未排空时不能提交L4。
8. 在L4区选择MossFormer2或TIGER，点击“发送到L4”。短于2秒的方向轨不会成为有效L4输入。
9. L4固定保留A/B候选并自动对每条运行L5，L4栏可试听16 kHz结果并查看黄色Voice区间。
10. L4/L5完成后L6会自动运行整轨声纹聚类、MOS择优时间线拼接和静音压缩；完成后按声纹试听Speaker A/B/C音频条。重复发送L4会重跑L5/L6并替换展示；L6不回写L2 ID或角度。

再次“发送到L4”会替换当前UI内的上一批离线结果。Test UI离线缓存不是长期归档。

### 6.3 界面区域与数据来源

| 区域 | 显示内容 | 数据来源 | 会不会控制算法 |
|---|---|---|---|
| 左上L1 | 8路电平、削波、IMCRA状态、预降噪、灯控、录音 | Layer 1与采集源 | 预降噪开关会从后续完整窗生效 |
| 右上L2 | Gate、360°谱、候选/ID、MDL、ID Tracking、DPD/白化设置 | `Layer2PipelineResult` | Gate、阶数和试验开关会更新后续L2配置 |
| 下左L3 | Center参考、各ID增强音频、BF模式、发送L4 | Hub连续音频与L3诊断 | BF模式影响后续L3；发送L4仅在封存后运行 |
| 下中L4 | 16 kHz结果、时长、波形、播放、黄色Voice区间 | `Layer4ProcessedAudio/OfflineResult` | 选择离线后端；不反向修改实时结果 |
| 下右L6 | 声纹Speaker A/B/C、关联音轨数、来源L2 ID、平均MOS、压缩后时长、波形、试听 | L4双候选及L5逐20 ms结果 | 仅手动执行；不反向修改实时结果 |
| 顶部性能栏 | 队列深度、worker、完成/错误/丢窗、缓存 | Runtime公开只读状态 | 只观察，不反压处理链 |

## 7. 其他入口

### 7.1 独立L1 Spectrum UI

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\launch_l1_spectrum_ui.ps1
```

用于设备连接、通道、电平、IMCRA噪声谱和频谱抓拍。它不启动Windowing、L2、L3、L4/L5或正式Runtime录音。

### 7.2 Audio Data Manager / Production UI

```powershell
.\.venv\Scripts\python.exe .\scripts\run_audio_data_manager.py --data-root data
```

用于专用原始录音、Runtime Session/Test Corpus管理、标注、QA、回放、回收站和导出。专用原始录音只采集L1输入与热力图，不自动生成方向ID；用该录音发起模拟测试后，方向结果来自重新运行的Test UI。

### 7.3 已封存session的命令行离线L4/L5

```powershell
.\.venv\Scripts\python.exe -m scripts.run_offline_l4 `
  "data\runtime_sessions\YYYY\MM\<session_id>" `
  --backend mossformer2_ss_16k `
  --device cuda
```

也可选择`--backend tiger_speech_16k`。该入口要求session已经封存并含逐ID连续L3资产；结果会显式持久化到session下，而不是只放在Test UI临时缓存。

### 7.4 Pipeline Log UI

Pipeline Log UI是只读观察面，用于查看已封存session的阶段终态、方向ID、时间线、丢窗和性能。它不启动/停止采集、不修改参数、不写标注，也不进入实时处理链。具体入口和数据边界见[`../LOG_UI_ARCHITECTURE_V1.1_TARGET.md`](../LOG_UI_ARCHITECTURE_V1.1_TARGET.md)。

## 8. 输出文件与保存边界

正式Runtime录音默认位于：

```text
data/runtime_sessions/YYYY/MM/<session_id>/
```

可包含原生8通道、逻辑8通道、物理7通道、IMCRA sidecar、MUSIC/方向结果、DecisionRecord、watermark、逐ID连续48 kHz增强WAV、manifest和hash。默认按60秒切块并异步写盘。

```text
正式Runtime资产 ── RecordingStore/Catalog长期管理
scratch录音      ── data/dev_test_ui/scratch/current，仅临时测试
Test UI L4/L5    ── 临时试听缓存，不自动进入RecordingStore
run_offline_l4   ── 显式写入封存session，可长期审计
```

项目默认`privacy.local_only=true`且`automatic_upload=false`，不会自动上传。但软件默认值不等于医疗数据合规；真实诊室采集仍需授权、访问控制、保留/删除策略和去标识化流程。

## 9. 推荐测试场景

1. 单人静止：验证通道方向、Gate、单峰、ID稳定和DAS基线。
2. 单人缓慢移动：检查IMM静止/移动模型切换、角度滞后和ID连续性。
3. 双人夹角≥50°轮流讲话：验证双峰、ID和优化BF串音。
4. 双人同时讲话：比较optimized与Loaded MVDR对照，并运行离线L4。
5. 风扇/笔记本噪声：记录2–4 kHz异常声源造成的误峰和L5结果。
6. 低频声源：确认80–1500 Hz不可可靠分离的物理边界，而不是用参数掩盖失败。
7. 长时间运行：记录队列高水位、丢窗率、端到端延迟、GPU/内存和写盘状态。

每次只改变一个参数，并保留配置hash、声源位置、距离、夹角、噪声条件和结果文件。

## 10. 常见状态与处理

| 状态/现象 | 含义 | 首先检查 |
|---|---|---|
| `WARMING_UP` | IMCRA、Window或MUSIC历史尚未积累完成 | 等待数百毫秒到IMCRA配置的预热时间 |
| `UNAVAILABLE` | 必需概率、校准或连续数据缺失 | 设备、通道、epoch切换和顶部错误 |
| Gate关闭、无方向 | 40 ms概率未达到门限 | L1 IMCRA状态、500–4000 Hz能量和Gate阈值 |
| 有稳定风扇方向 | L2判断空间声源，不判断是否人声 | 记录频谱；查看离线L5；不要把它当响度结果 |
| L3串音明显 | 小孔径、低频、混响或方向过近 | 声源夹角、80–1500 Hz限制、BF模式和`rho` |
| L4回退原L3 | 音频太短、匹配分数低或两候选分差太小 | 时长≥2秒、模型状态和匹配诊断 |
| 顶部显示DROPPED | 算法队列过载，非必然是USB丢包 | L2/L3队列深度、worker耗时和输入健康事件 |
| L5黄色区间不理想 | 模型目标域或阈值问题 | 中英文/诊室数据差异、阈值；必要时再训练或微调 |

## 11. 当前限制

- 4 cm阵列对80–1500 Hz方向分离能力差，这是物理孔径限制；
- L2使用2–4 kHz定位，对频谱特殊或主要能量在带外的声源能力较差；
- 最多输出3个实时方向，但离线L4只按1/2人处理；三方向不等于可靠三人分离；
- 候选间距至少50°，近角度、同一水平角、近场、高度差和强混响不可靠；
- 当前方向输出不带物理响度或角度不确定度；
- IMM针对静止/慢速移动调校，快速移动声源可能滞后；
- MarbleNet直接接入NVIDIA预训练模型，未做诊室中文目标域微调；
- 隔离性能基准不能代替真实阵列、UI、录音并发下的长时间门禁；
- Development Test UI不是临床产品界面。

## 12. 权威文档关系

- 当前操作入口与项目概览：[`../README.md`](../README.md)
- 1.3.2详细架构契约：[`../ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)
- 唯一业务配置：[`../config/config.yaml`](../config/config.yaml)
- 各层实现说明：[`../layer1_input/README.md`](../layer1_input/README.md)、[`../layer2_source_detection/README.md`](../layer2_source_detection/README.md)、[`../layer3_direction_signal/README.md`](../layer3_direction_signal/README.md)、[`../layer4_speech_separation/README.md`](../layer4_speech_separation/README.md)、[`../layer5_voice_classifier/README.md`](../layer5_voice_classifier/README.md)
- Test UI说明：[`../gui/dev_test_ui/README.md`](../gui/dev_test_ui/README.md)
- 变更记录：[`../CHANGELOG.md`](../CHANGELOG.md)

发生冲突时，当前代码和严格加载的`config/config.yaml`描述实际运行行为，`ARCHITECTURE_V1.1_TARGET.md`描述1.3.2契约；本文用于把两者组织成可操作的交接说明，不建立第二份配置来源。

# 6+1麦克风阵列：完整架构与使用手册

> 适用版本：项目`1.3.5`开发线，发布基线`v1.3.3`。本文根据当前代码、`config/config.yaml`和各层公开数据类型编写，供第一次接触项目的开发者和测试人员使用。L4-L6渐进旁路的详细契约见[`REALTIME_L456.md`](REALTIME_L456.md)。

## 1. 系统目的与边界

系统用于诊室医患对话的本地采集、二维水平方向定位、按方向增强、采集后语音分离和人声概率判断。硬件是半径`4 cm`的6个圆周麦克风加1个中央麦克风，设备以`48 kHz / PCM16 / 8通道`输出。

系统估计的是灯面朝上、从灯面正上方向下观察的水平角`theta_deg ∈ [0°,360°)`：Center→MIC0为`0°/+x`，逆时针为正，依次经过MIC5、MIC4、MIC3、MIC2、MIC1。系统不估计距离、俯仰角、说话内容、声压级或人物身份。`track_id`只代表一条空间方向轨迹。

正式20 ms审计链只运行到L3和按ID连续音频Hub；实时L5位置仍写入`offline_after_l4`跳过状态，不运行CNN。与审计链隔离的旁路按默认4秒、可调3～15秒的连续ID块渐进运行L4-L6并发布可替换preview；L6刷新周期与同一个块长变量同步。单个ID消失1秒且未恢复后会提前提交不足块长的尾段并逐轨冲刷。停止后用完整封存轨逐ID校验并转正实时final或安全逐轨检查点，只补算未完成或异常轨，再统一确认最终L6。

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
      HUB[TrackAudioStreamHub<br/>每ID取唯一20 ms hop<br/>可选响度补偿·去重·补洞·连续缓存]
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

    subgraph PROGRESSIVE[采集中的L4-L6渐进旁路]
      direction LR
      CLAIM[后台chunk producer<br/>claim / admission / ack]
      PL4[L4 GPU<br/>单人旁路 / 双人MF2<br/>1 s重叠换序修复]
      PL5[L5 CPU<br/>跨块上下文与稳定水位]
      PL6[L6 CPU<br/>2 s声纹缓存·随块长刷新]
      CLAIM --> PL4 --> PL5 --> PL6 --> DEV
    end

    HUB -->|默认4 s，可调3～15 s| CLAIM

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

图中正式逐窗审计仍遵循`L2(n) → L3(n) → L5审计(n)`；不同窗口形成跨层和L3内部流水。渐进L4-L6是独立旁路：L3只发轻量wakeup，chunk producer后台构块，接纳成功后才推进Hub游标。它不进入ResultJoiner、DecisionRecord或RecordingStore，也不反压L1-L3。

## 3. 逐层输入、内部处理单元与输出

| 模块 | 正式输入 | 内部处理单元 | 正式输出 | 频率/节拍 |
|---|---|---|---|---|
| 硬件输入 | 诊室声场 | 6个半径4 cm圆周麦 + Center；MA-USB8通道汇聚 | Host原生PCM16 `[N,8]` | 48 kHz；默认每块960 sample=20 ms |
| Layer 1 | Host PCM16 `[N,8]`、校准参数、CDC状态 | PCM16解码；增益/极性/整数延迟校准；Host→Logical通道重排；连续性检查；7麦IMCRA；可选IMCRA-Wiener WOLA | `DecodedAudio`：Logical float32 `[N,8]`；Native `[N,8]`；IMCRA噪声PSD/SPP/20 ms声源概率；健康事件 | IMCRA 0–10 kHz；Gate证据250–3400 Hz，三频带15%/35%/50%加权 |
| Ingest | `DecodedAudio`、sequence/timestamp、校准身份 | 建立`session_id`；检测缺口并切换`stream_epoch`；分配绝对sample；把IMCRA hop对齐到同一时间轴 | `IngestedAudioBlock`：48 kHz float32 `[N,8]`，含native、hotmap、IMCRA、校准元数据 | 输入块通常20 ms |
| Windowing | 连续同epoch的`IngestedAudioBlock` | 环形累计；检查校准身份与sample连续；组合来源sequence | `DecisionWindow [7680,8]`；末端40 ms DOA区间；最近160 ms上下文；8个20 ms IMCRA hop | 160 ms上下文，每20 ms发布 |
| Runtime封装 | `DecisionWindow`、当前UI/配置revision | 创建唯一`WindowKey=(session, epoch, window_id, decision_sample)`；冻结本窗Gate/DOA/IMM-JPDA/L3设置；有界latest-wins入队 | `WindowWorkItem` | 每个DecisionWindow一个 |
| Layer 2 | `DecisionWindow`、末尾当前20 ms声源概率、7麦几何、扫描配置 | 当前20 ms Probability Gate；Rolling NormMUSIC；圆周峰值与50° NMS；Circular IMM-JPDA方向ID；可选DPD/IMCRA白化 | `Layer2PipelineResult`：Gate状态；`SpatialResponse` 360点；0–3个`TrackedDirection`；active tracks；MUSIC诊断 | 每20 ms判断与更新 |
| Layer 3 | `DecisionWindow`末尾40/80/160 ms、0–3个公共方向、7麦几何、IMCRA噪声 | 共享STFT与协方差缓存；steering；按`rho`逐频选择Dual LCMV / Soft-null loaded MVDR / Loaded MVDR；或DAS/全频loaded MVDR；数值保护；批量ISTFT | `Layer3Output`，其中每方向一个`EnhancedAudio`：48 kHz mono `[1920/3840/7680]`，携带`track_id/theta/algorithm/fallback` | 当前默认40 ms音频；每20 ms产生新重叠窗 |
| TrackAudioStreamHub | L3的`EnhancedAudio`、本窗IMCRA概率、L2 active IDs/方向数 | 每ID只取末尾唯一20 ms；去除重叠；按绝对sample补洞；首次confirmed时按首次出生角回补`first_seen_sample`之前完整1秒BF；2 ms模式切换淡化；可选IMCRA概率响度补偿；维护完整归档；渐进块使用claim/resolve事务 | `TrackAudioBatch`；最长3200 ms试听上下文；3～15秒渐进`Layer4LongAudioInput`；停机完整长轨 | 20 ms hop；当前响度补偿默认关 |
| 实时L5审计 | L3/Hub阶段终态 | 不运行模型，只形成可审计跳过原因 | `L5StageResult=SKIPPED(offline_after_l4)` | 每实时窗口一个终态 |
| ResultJoiner | 同一`WindowKey`的L2/L3/L5阶段终态 | 校验ID与角度对齐；等待完整终态；按全局window顺序提交；保留失败/丢弃/取消原因 | `JoinedWindowResult`、`DecisionRecord v5`、`ResultWatermark`、UI快照 | 有序逐窗提交 |
| RecordingStore | 原生/逻辑音频、IMCRA、Joined结果、Hub hop | 异步有界写盘；60秒切块；逐ID hop合并；SHA-256；journal事务；崩溃恢复；Catalog投影 | WAV/NPZ/JSONL/manifest/Catalog；逐ID连续48 kHz增强WAV | 不反压采集 |
| 渐进Layer 4 | Hub已确认连续块 | 1人旁路；2人MF2；上一块1秒输入尾+新块；输出重叠换序与淡化；stable branch与累计A/B rank分离 | 可替换L4 revision与稳定水位 | 默认4秒，可调3～15秒；MF2 CUDA |
| 渐进Layer 5/6 | 稳定L4片段 | MarbleNet跨块上下文并延迟右端；DNSMOS降频；2秒CAMPPlus余量跨块；L6随L4块长刷新 | 可替换L5帧与provisional L6 speaker revision | CPU；latest-only；不入正式审计 |
| 权威Layer 4 | Hub封存的完整`Layer4LongAudioInput`与实时final/逐轨完成检查点 | 逐ID核验身份/范围/SHA/后端/分支/水位及逐轨final；一致则转正，仅未完成或异常轨运行完整L4 | 双人父轨两条16 kHz A/B候选；单人一条旁路 | 停止后选择性校正 |
| 权威Layer 5 | L4原生16 kHz完整波形 | NVIDIA MarbleNet Frame-VAD；每320 sample推理；阈值比较；连续3帧均值摘要 | `Layer5LongAudioResult`和`Layer4OfflineResult` | 16 kHz；20 ms一帧；默认阈值0.70 |
| 权威Layer 6 | 完整L4/L5及MOS/绝对时间线 | CAMPPlus；B门限；MultiStage最多5簇；MOS择优；静音压缩 | 最终Speaker A～E与审计；DTO保留1～100扩展空间 | 停止后运行并原子替换preview |

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
  → 末尾当前p20 → 20 ms Gate（默认0.60）
  ├─ Gate关闭 → 空方向结果/明确状态
  └─ Gate打开
       → 7麦2–4 kHz增量STFT
       → 200 ms滚动协方差
       → Gate连续OPEN满200 ms后才放行MUSIC候选角
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

### 3.4 Hub与渐进/权威Layer 4-L6内部图

```text
重叠EnhancedAudio窗口
  → 每ID只追加末尾20 ms
  → 缺口补等时静音 / 重复拒绝
  → 可选IMCRA概率响度补偿（实验profile默认关）
  → 实时3200 ms试听上下文 + 完整归档
  ├─ 采集中：L3只signal，producer后台claim 3～15秒连续块
  │    → admission成功才ack
  │    → 48→16 kHz
  │    → 1人旁路 / 2人MF2 + 1秒重叠换序修复
  │    → MarbleNet跨块上下文与稳定帧水位
  │    → 2秒CAMPPlus余量跨块、L6随当前块长刷新revision
  │    → Test UI provisional preview
  └─ stop + drain + preview tail flush + seal
       → 每ID一条完整Layer4LongAudioInput
       → 逐ID验证实时final的范围、SHA、后端、分支及L4/L5水位
       → 验证通过：L4/L5直接转正，不运行模型
       → 验证失败：只对该轨按人数路由补算L4/L5
       → 复用2秒CAMPPlus片段并统一运行最终L6
       → canonical成功后原子替换preview
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
- 默认设备拓扑为`L1 CPU → L2 CPU → L3 DS CPU → 渐进/权威MF2 CUDA → L5/DNSMOS/CAMPPlus/L6 CPU`；后台在首块累计时预载模型并复用渐进/canonical的MF2与CAMPPlus；
- L2/L3/L5阶段队列默认各100窗（约2秒），L4-L6队列默认2块；Hub大块拼接与SHA在主锁外执行；
- `runtime.torch_cpu_threads=1`是当前16逻辑核主机的全链最快配置；不要把隔离单层的多线程或CUDA结果直接当作整链配置；
- `runtime.l3_device/l4_device/l5_device`相互独立，旧配置缺少这些字段时才回退到`preferred_device`；
- L4配置CUDA但设备不可用时，在`allow_cpu_fallback: true`下自动回退CPU；离线分离会明显变慢；
- 灯面朝上，并保持Center→MIC0为0°；逆时针依次为MIC5、MIC4、MIC3、MIC2、MIC1。

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

### 6.2 从采集到渐进预览和最终结果

1. 连接阵列，确认Windows识别设备；启动UI并检查顶部设备、CUDA和模型状态。
2. 点击“启动采集”。等待160 ms窗口累计及IMCRA预热；L2维护200 ms滚动定位历史，且Gate连续OPEN满200 ms后才开始放行MUSIC候选角。
3. 检查左上8路电平。依次轻敲麦克风，确认MIC0–MIC5、Center及HardwareMix映射没有镜像或错位。
4. 在右上查看L2 Gate、360°MUSIC谱、手动MUSIC阶数、实际候选数和方向ID。基线保持DPD关闭、IMCRA白化和ID Tracking开启。
5. 在左下查看Center参考和按`track_id`排列的L3方向轨；基线使用DS。渐进旁路提交首块后，本次采集中锁定L3模式，避免不同算法音频混入同一preview。
6. 需要正式数据时使用“正式录音开始/暂停”；只做临时试听时使用scratch录音。两者不要混作同一资产。
7. 采集满首个配置块后，L4/L5栏显示preview revision和稳定水位，L6栏显示暂定声纹。默认块长4秒，可在Test UI调为3～15秒；L6使用相同刷新周期，奇数秒不会截断2秒声纹证据。
8. ID从权威活动集合消失后先经过1秒防抖；确认结束便在采集中提前提交短尾和逐轨final。点击“停止采集”后只需等待仍活动ID、L2/L3排空和Hub封存。若GPU worker超时未退出，UI会明确报错并禁止启动争用同一GPU的canonical。
9. Test UI逐ID比较实时final或最新逐轨完成检查点与Hub封存源；精确一致的L4/L5直接转正，只有缺失、未完成、降级或契约不一致的轨道运行MF2/MarbleNet补算。全局旁路中止不会使已经安全完成的轨道失效。开始时保留preview，成功后才一次性替换。
10. 最终L6复用已缓存的2秒CAMPPlus证据，统一运行人物聚类、MOS择优与静音压缩。L6不回写L2 ID或角度。

下一轮L3封存触发的自动L4会替换当前UI内的上一批离线结果。Test UI离线缓存不是长期归档。

### 6.3 界面区域与数据来源

| 区域 | 显示内容 | 数据来源 | 会不会控制算法 |
|---|---|---|---|
| 左上L1 | 8路电平、削波、IMCRA状态、预降噪、灯控、录音 | Layer 1与采集源 | 预降噪开关会从后续完整窗生效 |
| 右上L2 | Gate、360°谱、候选/ID、MDL、ID Tracking、DPD/白化设置 | `Layer2PipelineResult` | Gate、阶数和试验开关会更新后续L2配置 |
| 下左L3 | Center参考、各ID增强音频、BF模式 | Hub连续音频与L3诊断 | 首个渐进块前可切换；基线DS |
| 下中L4 | preview/canonical 16 kHz结果、水位、播放、黄色Voice区间 | `Layer4ProcessedAudio/OfflineResult` | 伪实时开启时后端固定YAML默认MF2 |
| 下右L6 | provisional/final Speaker、来源ID、MOS、波形、试听 | L4双候选及L5逐20 ms结果 | 只显示，不反向修改实时结果 |
| 顶部性能栏 | 正式队列及L4-L6状态、完成/错误/丢窗/丢块、缓存 | Runtime公开只读状态 | 只观察，不反压处理链 |

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
| Gate关闭、无方向 | 20 ms概率未达到门限 | L1 IMCRA状态、250–3400 Hz三频带加权概率和Gate阈值 |
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
- 1.3.3详细架构契约：[`../ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)
- 唯一业务配置：[`../config/config.yaml`](../config/config.yaml)
- 各层实现说明：[`../layer1_input/README.md`](../layer1_input/README.md)、[`../layer2_source_detection/README.md`](../layer2_source_detection/README.md)、[`../layer3_direction_signal/README.md`](../layer3_direction_signal/README.md)、[`../layer4_speech_separation/README.md`](../layer4_speech_separation/README.md)、[`../layer5_voice_classifier/README.md`](../layer5_voice_classifier/README.md)
- Test UI说明：[`../gui/dev_test_ui/README.md`](../gui/dev_test_ui/README.md)
- 变更记录：[`../CHANGELOG.md`](../CHANGELOG.md)

发生冲突时，当前代码和严格加载的`config/config.yaml`描述实际运行行为，`ARCHITECTURE_V1.1_TARGET.md`描述1.3.3契约；本文用于把两者组织成可操作的交接说明，不建立第二份配置来源。

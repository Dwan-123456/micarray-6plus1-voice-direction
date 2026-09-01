# 07 波束形成与双人音频恢复

本文属于**未来方案解释**。本章描述可行技术路线和验证顺序，不代表v1.4.3已经实现L3或能够恢复无失真的“原始讲话音频”。

## 1. 当前边界

v1.4.3的最后输出是方向、`track_id`、MUSIC响应和轨迹状态。当前分支没有：

- 波束形成输出；
- 按ID连续音频流；
- 双人分离模型；
- WPE去混响；
- 正式录音或音频文件事务；
- 人声分类和声纹归并。

完整旧L1–L6实现保存在不可变标签`v1.3.6`，可作为历史工程参考，但不能直接复制旧参数到当前精简架构。

## 2. “恢复原始音频”的准确含义

阵列只观测房间中多个源经过传播、混响和噪声叠加后的信号。没有算法能保证从有限7麦观测中精确恢复录音前的干净原始声波。

合理目标应表述为：

- 增强目标方向语音；
- 抑制另一方向讲话和背景噪声；
- 保持目标语音失真在可接受范围；
- 输出连续、可播放、时间轴正确的增强估计。

评价应使用SI-SDR、干扰泄漏、STOI、PESQ、主观听感和实时性，避免使用“无损恢复”表述。

## 3. 预期输入契约

未来L3至少需要：

| 输入 | 来源 |
| --- | --- |
| 连续7路校准音频 | `IngestedAudioBlock/DecisionWindow.physical_samples` |
| session/epoch/sample时间轴 | `IngestCoordinator` |
| 目标`track_id`和角度 | L2 `TrackedDirection` |
| 竞争方向 | 其他活动`TrackedDirection` |
| Gate和声源数 | L2原子快照 |
| 噪声PSD/SPP | L1 `ImcraHopSnapshot` |
| 阵列几何 | `MicGeometry` |

需要保持HardwareMix和native音频作为参考，不把它们混入空间协方差。

## 4. 从DOA到导向矢量

当前MUSIC已经构造自由场导向矢量：

```text
a_m(f,theta) = exp(-j 2 pi f tau_m(theta))
```

传统波束形成可复用同一几何约定，但增强对导向误差更敏感。更可靠的后续方向是从已确认目标片段估计实际相对传递函数（RTF），把房间早期传播和通道频率响应部分纳入目标steering。

## 5. Delay-and-Sum基线

对目标方向补偿理论时延并平均：

```text
y(t) = (1/M) sum_m x_m(t + tau_m(theta_target))
```

频域写法：

```text
w_DS(f) = a(f,theta) / M
Y(f,t) = w_DS(f)^H X(f,t)
```

优点：

- 简单、稳定、计算低；
- 不需要估计协方差；
- 适合作为正确性和实时性基线。

限制：

- 8 cm孔径对低频指向性弱；
- 两源接近时干扰抑制有限；
- 混响和导向误差会降低增益。

## 6. MVDR

MVDR在保持目标方向无失真的约束下，最小化输出功率：

```text
min_w w^H R_n w
subject to w^H a_target = 1
```

闭式解：

```text
w_MVDR = R_n^-1 a / (a^H R_n^-1 a)
```

工程实现应使用`solve(R_n,a)`，避免显式求逆，并使用对角加载：

```text
R_loaded = R_n + lambda * trace(R_n)/M * I
```

关键是`R_n`的定义。在双人同时讲话时，竞争说话人应进入“干扰协方差”，不能只使用安静背景噪声，否则MVDR不会主动压制另一讲话方向。

## 7. LCMV和双方向约束

LCMV允许多个线性约束：

```text
w^H a_target = 1
w^H a_interferer = 0   或 soft null
```

矩阵形式：

```text
w = R^-1 C (C^H R^-1 C)^-1 g
```

其中`C=[a_target,a_interferer]`，`g=[1,0]^T`。对两个分离较大的固定方向，LCMV可比单约束MVDR更明确地抑制竞争源。

硬零陷对DOA误差、移动和混响很敏感。更稳健的方案是根据角距、白噪声增益（WNG）和协方差质量连续调整soft-null强度。

## 8. 目标、干扰和噪声协方差

未来系统需要分开维护：

```text
R_target(f)
R_interferer(f)
R_background(f)
```

可用时频mask估计：

```text
R_s(f) = sum_t mask_s(f,t) X(f,t)X(f,t)^H / sum_t mask_s(f,t)
```

mask来源可按复杂度递进：

1. L1 SPP和P Gate启发式；
2. DOA匹配特征和空间响应；
3. oracle mask上界实验；
4. 轻量DOA-conditioned MaskNet；
5. 更大的神经空间模型作为研究上界。

先做oracle mask实验可以判断瓶颈来自协方差估计，还是阵列孔径/导向失配。

## 9. 两人同时讲话

对两个活动ID A/B，生成两路输出：

```text
y_A = beamform(X, target=A, interferer=B)
y_B = beamform(X, target=B, interferer=A)
```

必须保持：

- 每路输出绑定`session + epoch + track_id`；
- 两路使用同一输入sample范围；
- ID交换时不能静默交换音频文件；
- coasting阶段明确是继续使用预测角、冻结权重还是暂停输出；
- 新ID确认前是否允许低延迟tentative音频必须成为显式策略。

两个方向小于50°时当前L2本身不能稳定公开两峰，L3也无法获得两个可靠目标约束。

## 10. 低频问题

低频波长远大于8 cm孔径，不同麦克风相位差很小。即使MUSIC用2–4 kHz获得正确DOA，直接对全带音频做同一自由场波束形成时，80–1,500 Hz的双源抑制仍可能很弱。

可研究：

- 高频使用MVDR/LCMV；
- 低频采用稳定DS或Wiener/MWF后滤波；
- 根据频率、角距和WNG连续融合；
- 避免在低频强行深零陷造成目标失真。

## 11. 混响和RTF

自由场steering只描述直达声。真实房间的目标协方差包含早期反射和晚期混响。可按顺序研究：

1. loaded MVDR提高数值稳健性；
2. 从高置信单源片段估计RTF；
3. 背景、目标、竞争源协方差分离；
4. WPE去混响；
5. 神经mask与传统波束形成组合。

WPE会增加状态、延迟和计算，必须通过消融确认混响是主要瓶颈后再加入。

## 12. STFT、WOLA和时间轴

未来L3应避免每层独立建立不一致的窗口。推荐：

- 保持48 kHz主时间轴；
- 输出仍按20 ms hop组织；
- STFT可使用960/480或经验证的统一参数；
- 复用已兼容的滚动STFT和预分配buffer；
- 使用满足constant-overlap-add条件的分析/合成窗；
- epoch变化时清空所有OLA、SCM、RTF和权重状态；
- 不跨缺口拼接，缺失策略必须显式记录。

如果L2和L3需要不同FFT，应共享原始时间边界，不能共享不兼容的频谱数组。

## 13. 按ID连续音频

`DecisionWindow`高度重叠，不能每窗把完整160 ms写入ID音轨，否则会重复音频。连续音轨只追加当前窗口新增的20 ms hop，并依据绝对sample：

- 去除重复；
- 检测缺口；
- 决定补静音、断段或标记缺失；
- epoch变化时封存旧段；
- ID过期时完成尾部flush。

该职责应由独立的按ID流拼接层承担，波束形成器只计算一个明确sample区间的增强结果。

## 14. 输出契约建议

未来增强DTO至少包含：

- `session_id`、`stream_epoch`、`track_id`；
- `start_sample`、`end_sample`、`sample_rate`；
- 算法/配置/几何/校准版本；
- 目标角、竞争角和角度来源（observed/predicted）；
- 音频`float32 [N]`；
- 权重状态、协方差质量、WNG和故障原因；
- 输入sequence/window来源。

数组必须finite、C-contiguous、不可变，禁止下游按候选rank重新发明ID。

## 15. 实时实现建议

7×7矩阵适合NumPy/SciPy批处理：

- 按频率batch `solve/eigh/einsum`；
- 缓存steering和固定矩阵；
- 预分配STFT、SCM和输出buffer；
- 权重可低于20 ms频率更新，音频合成仍保持20 ms；
- 避免CPU/GPU逐小矩阵往返；
- 在引入GPU前用细粒度profile确认瓶颈。

神经mask若使用GPU，应让STFT、模型、SCM和beamforming尽量在同一设备上连续运行，再统一回CPU。

## 16. 评价指标

### 音频质量

- SI-SDR / SI-SDRi；
- SDR；
- STOI；
- PESQ；
- 目标语音失真；
- 竞争讲话泄漏；
- 主观听感。

### 方向和ID条件

- 目标/竞争DOA误差；
- 角距45/50/60/90/120/180°分层；
- ID switch和音轨串人；
- tentative/confirmed/coasting分层。

### 实时性

- 每阶段mean/P50/P95/P99/max；
- RTF；
- 队列高水位和drop；
- RAM/VRAM；
- 启动、切换和flush延迟。

所有算法应处理完全相同的mixture，使用paired comparison和置信区间。

## 17. 建议实验顺序

### 阶段A：传统基线

1. DS；
2. loaded MVDR；
3. soft LCMV；
4. 45–180°合成与真实阵列基准；
5. 确认时间轴和ID绑定正确。

### 阶段B：协方差上界

1. oracle target/interference mask；
2. 比较自由场steering与真实RTF；
3. 评估低频分带和WNG；
4. 判断主要瓶颈。

### 阶段C：可部署增强

1. 轻量DOA-conditioned mask；
2. batched SCM + MVDR/LCMV；
3. 必要时WPE；
4. `torch.compile`/ONNX/TensorRT只在profile证明需要后采用。

### 阶段D：研究上界

使用SpatialNet、Beam-Guided TasNet等模型作为质量上界或教师，不直接替代可解释生产基线。

## 18. 验收门槛建议

进入正式Runtime前至少满足：

- 不改变L1/L2现有DTO和20 ms时间轴；
- 所有缓存有硬上限；
- epoch和ID切换无跨段污染；
- 单源输出不比DS基线明显失真；
- 50/60°双源在目标SIR范围内有可重复增益；
- P95/P99满足目标硬件预算；
- 故障能降级或旁路，不阻断采集；
- 自动测试、离线语料和真实阵列测试均通过。

## 19. 当前参考资料

- [归档L3和双源波束形成研究](../v1.4.3_existing_docs/references/README.md)
- [Grondin等2022 ODAS论文](references/26_Grondin_2022_ODAS.pdf)：包含Delay-and-Sum/GSS与嵌入式处理背景。
- 完整旧实现：不可变Git标签`v1.3.6`。

[上一章：当前局限](06-limitations-and-open-issues.md) · [下一章：Runtime、UI与验证](08-runtime-ui-and-verification.md) · [返回项目总导航](../../README.md)

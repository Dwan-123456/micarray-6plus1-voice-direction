# 03 IMCRA、预降噪与P Gate

本文属于**原理解释 + 技术参考**。它解释L1怎样把每20 ms的7路麦克风音频变成噪声PSD、逐频SPP、逐麦P1和阵列P2，以及L2为什么只使用当前20 ms P2控制Gate。

## 1. 处理位置和数据流

```text
校准后的 IngestedAudioBlock [N,8]
        │
        ├─ 前7路 physical microphones
        │       ↓
        │   Layer1Imcra
        │       ├─ noise PSD / smoothed PSD / minima
        │       ├─ posterior SNR / prior SNR / SPP
        │       ├─ P1[7]
        │       └─ P2 = median(P1)
        │
        ├─ 可选 ImcraWienerPreDenoiser
        │
        └─ WindowAssembler → DecisionWindow → ProbabilityGate
```

IMCRA和预降噪只处理逻辑前7路。HardwareMix保留用于显示和诊断，不能影响噪声统计、P1/P2或后续DOA。

## 2. 输入与输出契约

### 2.1 输入

`Layer1Imcra.process()`接收不可变`IngestedAudioBlock`：

- `sample_rate=48000`；
- `samples`为`float32 [N,8]`；
- 包含session、epoch、绝对sample边界、sequence ID和校准身份；
- 输入可以任意分块，内部缓冲后严格按960 samples处理。

### 2.2 每跳输出

每960 samples产生一个`ImcraHopSnapshot`：

| 字段 | shape/类型 | 含义 |
| --- | --- | --- |
| `frequencies_hz` | `[201]` | 0..10,000 Hz，50 Hz步长 |
| `noise_psd` | `[7,201]` | 当前噪声功率估计 |
| `smoothed_psd` | `[7,201]` | 第一遍时频平滑结果 |
| `conditional_smoothed_psd` | `[7,201]` | 排除强语音后的第二遍平滑 |
| `minimum_psd` | `[7,201]` | 第一遍历史最小值 |
| `conditional_minimum_psd` | `[7,201]` | 第二遍历史最小值 |
| `spp` | `[7,201]` | 条件语音存在概率 |
| `speech_absence_probability` | `[7,201]` | 先验语音缺失概率估计 |
| `posterior_snr` | `[7,201]` | 后验SNR |
| `prior_snr` | `[7,201]` | 先验SNR |
| `noise_features` | `[7,4]` | 噪声级、信号级、SNR、P1 |
| `source_probability_per_mic` | `[7]` | P1 |
| `array_source_probability_20ms` | `float或None` | ready时的P2 |

所有数组在DTO中复制到不可写bytes所有者，要求finite、C-contiguous并校验固定shape。

## 3. FFT、分析窗和功率谱

每个20 ms跳包含960个真实样本。v1.4.3直接执行960点RFFT，不补零：

```text
frequency spacing = 48000 / 960 = 50 Hz
RFFT bins = 960 / 2 + 1 = 481
```

算法内部计算0..24 kHz全部RFFT bin，只对外发布0..10 kHz的201个bin。输入先逐通道减去20 ms均值，再乘periodic Hann窗：

```text
Y_m(k,l) = RFFT((x_m[n] - mean(x_m)) * w[n])
power_m(k,l) = |Y_m(k,l)|^2 / sum(w^2)
```

减均值降低直流偏置；窗函数减轻20 ms边界截断造成的频谱泄漏；除以窗能量让功率尺度在不同帧间可比较。

## 4. IMCRA要解决的问题

如果直接把当前功率当作噪声，讲话能量会快速被“学进”噪声底，之后真实语音反而看起来不突出。若只在硬VAD判定静音时更新噪声，低SNR和弱语音又会让VAD不可靠。

IMCRA使用连续软概率更新：

- 通过平滑功率的历史最小值寻找局部噪声基线；
- 估计每个麦、每个频率bin的语音存在概率；
- 语音概率高时减慢该bin的噪声更新；
- 语音概率低时更快跟随环境变化；
- 使用第二遍条件平滑，避免强语音污染最小值追踪。

当前实现对应Israel Cohen 2003 IMCRA论文的工程化版本`cohen_imcra_2003_l1_v11`。

## 5. 第一遍平滑和最小值追踪

### 5.1 频率平滑

对相邻三个bin使用归一化Hann权重：

```text
S_f[k] = 0.25 P[k-1] + 0.50 P[k] + 0.25 P[k+1]
```

两端使用边缘重复，得到`0.75/0.25`。这利用语音在相邻频率bin上的相关性，也降低单bin随机波动。

### 5.2 时间平滑

```text
S(k,l) = alpha_s S(k,l-1) + (1-alpha_s) S_f(k,l)
alpha_s = 0.77
```

### 5.3 分段最小值

每5个20 ms帧关闭一个100 ms子窗口，保存该子窗口最小值；历史保留10个子窗口，对应约1 s最小值范围。偏置系数`Bmin=1.66`补偿有限窗口最小值系统性偏低。

## 6. 后验SNR、先验SNR和语音增益

设当前功率为`P`、噪声PSD为`lambda_d`：

```text
posterior SNR gamma = P / max(lambda_d, eps)
```

先验SNR采用decision-directed递推：

```text
xi = alpha_xi * G_H1(previous)^2 * gamma(previous)
   + (1-alpha_xi) * max(gamma-1, 0)
alpha_xi = 0.81
```

`G_H1`是语音存在假设下的LSA增益，使用指数积分`exp1`计算并限制到`[0,10]`。先验SNR结合上一帧平滑结果与当前瞬时证据，避免概率随单帧剧烈跳动。

## 7. 两遍平滑和先验语音缺失概率

第一遍用当前功率与偏置最小值的比率形成粗略指示：

```text
power / (Bmin * minimum) < gamma0
S     / (Bmin * minimum) < zeta0
gamma0 = 4.60
zeta0  = 1.67
```

只把粗略判为噪声的时频点放入第二遍条件频率平滑。第二遍也使用`alpha_s=0.77`递归并维护自己的最小值。

随后计算：

```text
gamma_min = power / (Bmin * conditional_minimum)
zeta      = first_pass_smoothed / (Bmin * conditional_minimum)
```

当局部功率和瞬时功率都低时，先验语音缺失概率`q_hat`接近1；证据变强时在`gamma_min=1..gamma1`间软过渡，`gamma1=3.0`；超过条件时接近0。

第二遍排除强语音后，讲话期间的最小值更接近真实噪声，尤其适合非平稳噪声、弱语音和低SNR。

## 8. SPP和噪声递归更新

在复高斯STFT模型下，Cohen公式给出条件语音存在概率。实现使用log-odds避免数值溢出：

```text
nu = gamma * xi / (1 + xi)
log_absence_odds = log(q/(1-q)) + log(1+xi) - nu
SPP = 1 / (1 + exp(log_absence_odds))
```

噪声更新平滑系数随SPP变化：

```text
alpha_tilde = alpha_d + (1-alpha_d) * SPP
noise_bar = alpha_tilde * noise_bar + (1-alpha_tilde) * power
noise = beta * noise_bar

alpha_d = 0.66
beta = 1.47
```

SPP越高，`alpha_tilde`越接近1，当前功率进入噪声估计的比例越小。`beta=1.47`是当前偏置补偿参数，与论文在`gamma1=3`时的推导一致。

## 9. 预热、断流和校准变化

新session或新的校准身份会清空全部IMCRA统计，并经过1 s、50个hop后进入`ready`。预热期间仍发布完整谱状态，但`P2=None`，Gate显示`WARMING_UP`。

同一session、同一校准下的sequence/timestamp断流会建立新epoch。IMCRA清空对齐缓冲和sample段信息，但保留已成熟的噪声统计；第一个恢复hop可以继续为`ready`。缺失区间不补零，也不发布虚构概率。

## 10. 从逐频SPP到P1

P1用于构造宽带声源证据。当前Gate频率轴包含250..3400 Hz的64个50 Hz bin，先按三段分配固定总质量：

| 频段 | 离散bin | 总权重 |
| --- | --- | ---: |
| 250–600 Hz | 250、300、…、600 | 30% |
| 600–1600 Hz | 实际使用650..1550 | 30% |
| 1600–3400 Hz | 1600..3400 | 40% |

600 Hz只属于第一段；第二段严格排除两侧边界，避免重复；1600 Hz属于第三段。

第一段30%再按办公室和诊室噪声统计分配：

| Hz | 段内权重 | 全局权重 |
| ---: | ---: | ---: |
| 250 | 0.161502 | 0.0484506 |
| 300 | 0.180000 | 0.0540000 |
| 350 | 0.180000 | 0.0540000 |
| 400 | 0.180000 | 0.0540000 |
| 450 | 0.163814 | 0.0491442 |
| 500 | 0.088159 | 0.0264477 |
| 550 | 0.041165 | 0.0123495 |
| 600 | 0.005360 | 0.0016080 |

中段19个bin等权分配30%，高段37个bin等权分配40%。对第`m`个麦：

```text
P1_m = sum_k weight[k] * SPP_m[k]
```

权重总和严格为1，P1限制到`[0,1]`。

## 11. 为什么强调宽带证据

当前Gate门限为0.80。任何两个完整频段全部饱和，贡献上限也只有0.60或0.70，无法单独打开Gate。系统要求三个宽频段共同出现较强SPP。

这样可压制：

- 低频风扇和HVAC；
- 单个窄带设备音；
- 250 Hz间隔稀疏谐波；
- 只在少数bin突出的敲击或机械声。

代价是窄带真实声源、极弱语音、频谱被设备严重削弱的讲话可能打不开Gate。P Gate是一组工程门限，不是训练得到的人声分类器。

## 12. P2：七麦中位数

```text
P2 = median(P1_0, ..., P1_6)
```

中位数要求至少4个物理麦共同给出较高证据，可降低单麦近场敲击、坏通道或局部气流影响。P2只在IMCRA `ready`时发布。

## 13. Probability Gate

`ProbabilityGate`只读取与当前DOA边界末端严格对齐的当前20 ms P2：

```text
current bounds = [doa_start + 960, doa_end)
OPEN   if P2 >= threshold
CLOSED if P2 <  threshold
```

默认`threshold=0.80`，UI可在`0..1`范围调整，比较包含等号。之前一个20 ms P2可作为诊断信息，但不参与当前Gate平均。

| 状态 | 条件 | MUSIC行为 |
| --- | --- | --- |
| `WARMING_UP` | 当前IMCRA未ready | 不运行 |
| `UNAVAILABLE` | 当前对齐概率缺失 | 不运行 |
| `INVALID` | session/epoch/sample错位或上游无效 | 不运行 |
| `CLOSED` | 当前P2低于门限 | 不运行 |
| `OPEN` | 当前P2大于等于门限 | 允许后续MUSIC计划 |

所有Gate结果携带session、epoch、window ID、decision sample、门限和配置revision，避免旧概率控制新窗口。

## 14. P Gate能过滤什么

它主要为0声源或宽带语音证据不足的窗口提供前置过滤，降低静音时MUSIC必然产生最大峰所造成的误定位。它不能保证：

- OPEN一定是人声；
- CLOSED一定没有人；
- OPEN表示一个或两个源；
- P2代表响度；
- P2可以替代声源数估计或VAD模型。

## 15. IMCRA-Wiener预降噪

预降噪配置：

| 参数 | 值 |
| --- | ---: |
| 默认 | 关闭 |
| frame | 960 samples / 20 ms |
| hop | 480 samples / 10 ms |
| FFT | 960 |
| 频率范围 | 0..10 kHz |
| 窗 | 50%重叠sqrt-Hann |
| 最低增益 | -18 dB |
| 增益平滑 | 0.80 |

ready时，每麦、每频率的目标增益为：

```text
wiener = xi / (1 + xi)
protected_gain = SPP + (1-SPP) * wiener
```

SPP高时增益接近1，保护可能的语音；SPP低时更接近Wiener增益。频率方向再做`0.25/0.5/0.25`平滑，时间方向用0.80平滑并限制到`[-18 dB,0 dB]`。

## 16. WOLA重建和延迟

20 ms帧拆成两个10 ms半帧，sqrt-Hann分析/合成窗在50%重叠下执行WOLA。每个完整输出块需要看到后一个半帧，因此启用后的链路存在一个20 ms块的缓冲关系。epoch变化会先封闭旧块，再清空重叠尾部，禁止跨断流拼接。

只替换逻辑前7路，HardwareMix与`native_samples`保持原样。停止时，仅当预降噪链已经进入延迟模式才flush最后一块，避免默认关闭时重复发布尾块。

## 17. 为什么默认关闭

预降噪会逐麦、逐频率施加不同的时变实数增益。虽然不直接修改相位，但不一致幅度掩码会改变跨麦协方差权重和弱源相对能量，可能影响GCC-PHAT残差峰、MUSIC子空间和第二弱源可见性。

当前项目缺少带角度真值的实机A/B证据来证明预降噪能稳定改善DOA，因此默认关闭。需要开启时，应同时比较：

- 单源角误差和伪峰；
- 双源召回和弱第二源；
- 声源数0/1/2准确性；
- Gate开启率；
- 算法耗时和动态回退周期。

## 18. 主要配置

| 配置 | 值 |
| --- | ---: |
| `layer1_imcra.algorithm_version` | `cohen_imcra_2003_l1_v11` |
| `hop_samples` / `n_fft` | 960 / 960 |
| `spectrum_smoothing` | 0.77 |
| `noise_smoothing` | 0.66 |
| `prior_snr_smoothing` | 0.81 |
| `minimum_subwindow_frames` | 5 |
| `minimum_history_subwindows` | 10 |
| `minimum_bias` | 1.66 |
| `gamma0` / `gamma1` / `zeta0` | 4.60 / 3.00 / 1.67 |
| `bias_compensation` | 1.47 |
| `warmup_seconds` | 1.0 |
| `layer2.probability_gate.threshold` | 0.80 |

## 19. 代码和测试入口

| 内容 | 文件 |
| --- | --- |
| IMCRA实现 | `layer1_input/imcra.py` |
| 人声频段权重 | `layer1_input/speech_spectrum.py` |
| 预降噪 | `layer1_input/pre_denoise.py` |
| IMCRA DTO | `common/data_types.py` |
| Gate | `layer2_source_detection/probability_gate.py` |
| Runtime对齐 | `app/runtime.py`中的`_imcra_probabilities` |
| IMCRA与权重测试 | `tests/test_l1_v03.py` |
| 预降噪测试 | `tests/test_l1_pre_denoise.py` |
| Gate测试 | `tests/test_l2_gate_probability.py` |

自动测试覆盖任意输入分块、精确20 ms输出、201点频率轴、Cohen状态、两遍平滑、宽带权重、HardwareMix隔离、断流恢复、预热、WOLA连续性、Gate包含等号和错窗拒绝。

## 20. 原始论文

- [Cohen 2003 IMCRA原始论文](references/01_Cohen_2003_IMCRA.pdf)
- [参考资料索引](references/README.md)

[上一章：阵列几何](02-array-geometry-and-channel-mapping.md) · [下一章：声源数与NormMUSIC](04-source-counting-and-normmusic-doa.md) · [返回项目总导航](../../README.md)

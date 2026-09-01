# 04 声源数估计与NormMUSIC DOA

本文属于**原理解释 + 技术参考**。它描述L2在一个权威worker中完成的顺序：先评估当前P Gate，再持续推进GCC-PHAT声源数，最后在Gate开启时选择MUSIC阶数、计算空间谱并取峰。

## 1. 总体流程

```text
DecisionWindow [7680,8]
        │
        ├─ 当前20 ms P2 → ProbabilityGate
        │
        ├─ 7路 physical audio
        │      ↓
        │   IncrementalGccPhatSourceCounter
        │      └─ SourceCountSnapshot: 0 / 1 / 2 / warming(None)
        │
        └─ Gate OPEN?
               ├─ 否：不运行MUSIC，轨迹coast/expire
               └─ 是：选择MUSIC order 1或2
                         ↓
                    RollingNormMusicScanner
                         ├─ 逐频7×7协方差
                         ├─ Hermitian EVD
                         ├─ 360° NormMUSIC
                         └─ 最多order个峰
```

Gate、声源数、MUSIC和ID结果都绑定同一`session_id + stream_epoch + window_id + decision_sample`。Runtime把它们组合成一个原子`L2DevUiSnapshot`，UI不从多个异步邮箱拼接相邻窗口。

## 2. 为什么先估计声源数

MUSIC必须知道信号子空间维度`K`。如果真实单源却使用`K=2`，一个噪声/反射模态会被放入信号子空间，空间谱更容易出现虚假第二峰；真实双源却使用`K=1`，第二源能量可能落入噪声子空间而被压低。

当前项目把合法突出方向数限制为0、1、2，并使用轻量GCC-PHAT评估器决定运行时MUSIC使用1阶还是2阶。P Gate负责过滤0声源窗口，因此Gate打开后即使计数为0或仍预热，MUSIC也安全回退到1阶，而不会使用0阶矩阵运算。

## 3. GCC-PHAT基础

对麦克风`i`和`j`，STFT为`X_i(f)`、`X_j(f)`。PHAT归一化互谱：

```text
C_ij(f) = X_i(f) X_j(f)* / max(|X_i(f)| |X_j(f)|, eps)
```

它基本去掉幅度，只保留跨麦相位。对候选lag `tau`做逆傅里叶型合成：

```text
R_ij(tau) = Re sum_f C_ij(f) exp(j 2 pi f tau)
```

若`tau`接近真实到达时间差，多个频率的相位会同向叠加形成峰。7个物理麦产生：

```text
7 × 6 / 2 = 21 microphone pairs
```

所有21对共同投票比单对时延更稳健。

## 4. 当前声源数频带和STFT

| 参数 | 值 |
| --- | ---: |
| backend | `incremental_gcc_phat_deemphasis_v1` |
| 上下文 | 160 ms / 7,680 samples |
| STFT win | 960 samples / 20 ms |
| STFT hop | 480 samples / 10 ms |
| FFT | 1024 |
| 频带 | 2,000..4,000 Hz |
| 角度网格 | 0..359°，1° |
| lag过采样 | 4倍，即0.25 sample |
| 最大帧数 | 15 |

第一窗含15个50%重叠帧。正常每20 ms只新增2帧、移除2帧，避免每窗重复15次FFT。

如果自适应调度计划跳过若干窗口，下一次实算只补算尚未见过的新帧；跨越整个160 ms或stream身份变化时才重建。计划内40–200 ms间隔保留稳定投票，真实非计划缺窗清空投票。

## 5. 几何时延表

对360个方向和21个麦对，程序根据阵列坐标计算理论时延sample数。为减轻小阵列时延只有几个sample造成的离散误差，lag网格使用4倍过采样。

每个理论时延通常落在两个lag点之间，程序预计算`lower index + fraction`并线性插值：

```text
response = gcc[lower] * (1-fraction) + gcc[lower+1] * fraction
spatial_map(theta) = mean(response over 21 pairs)
```

几何、声速或坐标变化会重建时延表。

## 6. 每帧特征

每个20 ms STFT帧：

1. 取前7路并逐麦减均值；
2. 乘periodic Hann；
3. 1024点RFFT；
4. 只取2–4 kHz；
5. 形成21对PHAT互谱；
6. 保存该帧互谱与时域均方功率。

滚动状态保存15帧互谱和总和，空间图使用平均互谱。逐帧互谱仍保留，用于第二源“同时存在”核验，不需要额外FFT。

## 7. 0源判定

先构造原始空间图，找到最大方向`theta_1`，计算：

- 160 ms平均RMS dBFS；
- 第一峰绝对值；
- 第一峰相对整个360°图中位数和MAD的robust-z。

```text
robust_z = (peak - median(map)) / max(1.4826 * MAD(map), 1e-6)
```

任一条件不满足即原始计数为0：

| 条件 | 默认门限 |
| --- | ---: |
| 活动电平 | `>= -70 dBFS` |
| 第一峰 | `>= 0.16` |
| 第一峰robust-z | `>= 2.0` |

RMS避免极低能量随机相位被放大；绝对峰要求麦对具有一致性；robust-z要求第一峰相对空间背景突出。

## 8. 主峰去强调和第二源候选

若第一源成立，对每个麦对的GCC在第一方向理论时延处施加高斯soft notch：

```text
notch = 1 - strength * exp(-0.5 * (lag-first_delay)^2 / width^2)
strength = 0.90
width = 0.25 sample
```

然后重新合成残差空间图。第二候选必须：

- 离第一峰至少50°；
- 是残差图的真实圆周局部极大值；
- 在原始空间图中也有附近局部峰支持；
- 残差峰`>=0.07`；
- 残差robust-z`>=2.0`；
- 残差峰/第一峰`>=0.09`。

soft notch压低主源在其他角度形成的旁瓣，同时尽量保留独立第二方向。

## 9. 逐帧共存验证

160 ms内先后出现两个方向，平均空间图可能同时显示两个峰，但它们并未同时存在。程序直接重用15帧PHAT互谱，分别计算每帧在第一、第二方向的响应。

只有至少3个STFT帧同时满足两个方向响应均`>=0.08`，原始计数才为2。该条件降低“前80 ms来自A、后80 ms来自B”被误判为双源的概率。

## 10. 2-of-3稳定投票

原始计数可能在相邻窗间抖动。最近3次原始判断中，某值至少出现2次后才更新稳定输出：

```text
raw history length = 3
required votes = 2
```

第一次判断通常输出`None`，表示预热；第二次一致后发布0、1或2。`SourceCountSnapshot`不暴露方向、分数或置信度，避免下游误用中间启发式量。

## 11. Gate关闭时仍持续估计

默认声源数估计与P Gate解耦。这样Gate从CLOSED变为OPEN时，计数器已有连续160 ms空间状态，不需要重新等两次投票。

只有用户关闭“启用声源数估计”时才停止并清空状态。关闭会同时关闭MUSIC阶数跟随并恢复固定2阶。

在自适应回退的跳过计算窗口中，Runtime沿用最近计数并更新时间身份；下一个实算窗补齐新增STFT帧。

## 12. 声源数到MUSIC阶数

| 状态 | 跟随关闭 | 跟随开启 |
| --- | ---: | ---: |
| Gate CLOSED/WARMING/INVALID | 不运行MUSIC | 不运行MUSIC |
| Gate OPEN，count `None` | 2 | 1 |
| Gate OPEN，count `0` | 2 | 1 |
| Gate OPEN，count `1` | 2 | 1 |
| Gate OPEN，count `2` | 2 | 2 |
| Gate OPEN，计数故障 | 2 | 1 |

本仓库默认启用计数和阶数跟随。旧外置配置缺少整个`source_counting`段时，Pydantic默认值同样为启用和跟随。

## 13. 历史声源数方法

项目历史中评估过神经网络CountNet类方法，当前v1.4.3已删除模型、依赖和运行入口。用户现场观察认为此前神经网络方案效果不满足需求。

MDL/特征值源数估计也不在当前运行链。历史观察显示工作环境中容易估计偏多。当前`ModelOrderEstimate` DTO仍为兼容下游而保留，但其中`estimated_sources`直接记录当前显式MUSIC阶数，`mdl_age_samples=0`；没有MDL计算。

这些历史结论用于解释设计选择，不构成当前GCC-PHAT方法已完成准确率验收的证据。

项目还测试过把SRP-PHAT直接作为DOA主扫描器。现场结论是单声源方向可以工作，双声源时难以稳定同时输出两个独立方向，因此当前主DOA改用MUSIC。v1.4.3保留的GCC-PHAT只负责轻量声源数估计，不直接产生公开DOA。

## 14. MUSIC的输入协方差

MUSIC使用7路校准物理麦。每个STFT帧在频率`f`形成复向量：

```text
x_f = [X_0(f), ..., X_6(f)]^T
R_frame(f) = x_f x_f^H
```

滚动协方差：

```text
R_x(f) = mean(R_frame(f) over active frames)
```

单个`DecisionWindow`只有160 ms，所以新stream首次重建15帧；连续20 ms窗口每次增加2帧，第二次增量后达到19帧，即完整200 ms。之后每加2帧移除最旧2帧。

Gate关闭或MUSIC阶数为0/None时扫描器reset，下一次正阶MUSIC重新从当前160 ms窗口预热。

## 15. 协方差正则化

有限帧、相关声源和弱频率会使7×7样本协方差病态。当前每个频率使用：

```text
trace_mean = Re(trace(R_x)) / 7
R_reg = (1-shrinkage) R_x
      + shrinkage * trace_mean * I
      + diagonal_loading * max(trace_mean, eigen_floor) * I

shrinkage = 0.05
diagonal_loading = 1e-3
eigen_floor = 1e-10
```

收缩把少量能量拉向各向同性协方差，对角加载保证特征分解数值稳定。它们改善有限样本稳定性，也可能降低极弱第二特征值的可分辨性。

## 16. 特征值分解和子空间

对Hermitian协方差：

```text
R_reg = E Lambda E^H
```

`numpy.linalg.eigh`返回从小到大的特征值。若信号阶数为`K`，最大的`K`个特征向量构成信号子空间`E_s`，其余构成噪声子空间`E_n`。

理想真实方向的导向矢量与噪声子空间近似正交：

```text
P_MUSIC(f,theta) = 1 / (a^H E_n E_n^H a)
```

实现通过总导向能量减去信号子空间投影能量，避免显式构造更大的噪声投影矩阵：

```text
denominator = ||a||^2 - ||E_s^H a||^2
```

数值上限制分母不低于`1e-12`。

## 17. 有效频率bin

1024点FFT在48 kHz下频率间隔为46.875 Hz。选择2–4 kHz后约有43个bin。某bin只有在所有特征值finite且最大特征值高于`1e-10`时有效；有效bin少于12个则当前扫描失败并进入Runtime故障/复用路径。

## 18. NormMUSIC逐频归一化

不同频率的绝对MUSIC谱可能相差很大，直接相加会让少数频率支配结果。当前先对每个频率独立归一化：

```text
P_norm(f,theta) = P(f,theta) / max_theta P(f,theta)
```

随后按4 cm几何固定权重融合：

| 频率 | 权重 |
| --- | ---: |
| 2.0–2.3 kHz | 0.35 |
| 2.3–2.5 kHz | 0.55 |
| 2.5–2.7 kHz | 0.75 |
| 2.7–3.0 kHz | 0.90 |
| 3.0–3.6 kHz | 1.00 |
| 3.6–3.8 kHz | 1.00线性降到0.75 |
| 3.8–4.0 kHz | 0.75线性降到0.45 |

低频孔径信息较弱，高频接近空间采样/阵列模型风险，因此中间频段权重最高。

融合结果再做全角min-max归一化到`[0,1]`，得到公开`SpatialResponse.normalized_scores[360]`。

## 19. 360°取峰

为正确处理0°边界，程序把360点谱复制三份，在中间一圈使用`scipy.signal.find_peaks`：

1. 峰值归一化分数`>=0.35`；
2. prominence `>=0.05`；
3. 按分数从高到低选择；
4. 与已选峰圆周距离`>=50°`；
5. 最多选`effective_order`个。

配置允许`effective_order_limit=1/2/3`和最多3候选；当前Runtime固定2或按计数选择1/2，因此实时公开观测最多2个。

## 20. 连续Gate预热和ID出生

Gate第一次打开时，MUSIC可以立即计算诊断谱，但新ID不能立刻出生。`Layer2Pipeline`要求连续OPEN hop数量达到：

```text
context_ms / 20 ms = 200 / 20 = 10 hops
```

前9个OPEN窗口的谱可显示，但`births_allowed=False`，候选不会进入新轨迹。任何Gate关闭、身份跳跃或MUSIC跳过都会重置连续计数。换句话说，Gate打开后蓝色MUSIC诊断图可以立即更新，新的方向ID需要连续约200 ms声音证据后才允许输出。这是用少量延迟换取更低误报。

这个200 ms门槛用于确认持续活动，并保证MUSIC滚动协方差已经达到19帧。它与单个160 ms `DecisionWindow`是两个不同概念。

当前200 ms滚动窗口更适合静止或较低角速度声源。窗口越长，协方差和谱峰通常越稳定，运动响应和方向切换延迟也越大。快速移动场景可研究160 ms或自适应窗口，但需要重新验证伪峰、双源分辨、ID确认和实时性能。

## 21. 弱声源、归一化和伪峰

逐频归一化会让每个有效频率都贡献形状信息，即使绝对能量很弱。MUSIC在任何非退化协方差上都会有最大值，静音/低SNR也可能产生看似明显的归一化峰。

因此v1.4.3组合使用：

- IMCRA P Gate过滤宽带证据不足窗口；
- 声源数限制MUSIC阶数；
- 协方差正则化；
- 固定频率权重；
- score和prominence；
- 50°硬间距；
- 连续OPEN 200 ms才允许出生；
- tentative需进一步累计观测才confirmed。

这些措施降低误报，仍不能消除反射、相干声源和阵列失配造成的伪峰。

Test UI右侧蓝色闭合曲线就是`normalized_scores[0..359]`的极坐标图。曲线半径越大，表示该角度的阵列相位模式与当前协方差越匹配。它是归一化空间匹配程度，不是声压、音量或“这个方向有人”的概率。

## 22. 输出对象

### `SpatialResponse`

- 四元窗口身份；
- DOA sample边界；
- `theta_degrees=float32 [360]`，严格0..359；
- `raw_scores=float32 [360]`；
- `normalized_scores=float32 [360]`；
- 当前显式order、有效频率数、数值状态和算法版本。

### `CandidateDirection`

- 窗口身份和DOA边界；
- `theta_deg`；
- raw/normalized score。

### `MusicDiagnostics`

- 协方差更新方式、帧数和gap；
- 有效频率数；
- 阶数、候选上限和出生许可；
- covariance/eigensolve/spectrum/total耗时。

## 23. MUSIC不能判断的内容

MUSIC只判断“哪一个理论方向最符合跨麦空间相位结构”。它不能判断：

- 该方向是否为人声；
- 是哪一个人；
- 声源距离和俯仰角；
- 两个同方向的人；
- 输出音频应该如何分离；
- 峰是否来自直达声还是强反射。

## 24. 故障与自适应调度

正常实算周期为20 ms。过载时Runtime可按40、60、…、200 ms减少完整计数/MUSIC/ID实算次数，但仍按20 ms发布重新定时的结果。跳过窗会轻量推进MUSIC协方差；计数沿用最近结果并在下次实算补帧。

特征分解、频率质量或DTO校验异常时，Runtime记录故障并尝试复用同stream且阶数兼容的上一结果；无法安全复用时发布无MUSIC的fault快照。故障不会生成跨窗混合DTO。

## 25. 配置速查

| 配置 | 默认值 |
| --- | ---: |
| `source_counting.enabled` | true |
| `music_order_from_source_count` | true |
| `activity_rms_threshold_dbfs` | -70.0 |
| `first_peak_threshold` / z | 0.16 / 2.0 |
| `residual_peak_threshold` / z | 0.07 / 2.0 |
| `residual_ratio_threshold` | 0.09 |
| `coactivity_frame_threshold` | 0.08 |
| `coactivity_required_frames` | 3 |
| `persistence` | 2 of 3 |
| `layer2.context_ms` | 200 |
| MUSIC FFT / win / hop | 1024 / 960 / 480 |
| MUSIC band | 2,000..4,000 Hz |
| shrinkage / loading | 0.05 / 1e-3 |
| direction threshold | 0.35 |
| peak prominence | 0.05 |
| min distance | 50° |

## 26. 代码和测试入口

| 内容 | 文件 |
| --- | --- |
| 声源数实现 | `source_counting/counter.py` |
| 声源数配置 | `source_counting/configuration.py` |
| 声源数DTO | `source_counting/interface.py` |
| Gate/L2编排 | `layer2_source_detection/pipeline.py` |
| MUSIC | `layer2_source_detection/music.py` |
| MUSIC配置 | `layer2_source_detection/configuration.py` |
| Runtime调度 | `app/runtime.py` |
| 声源数测试 | `tests/test_source_counting.py` |
| MUSIC/跟踪测试 | `tests/test_l2_music_tracking.py` |
| Gate测试 | `tests/test_l2_gate_probability.py` |

自动测试覆盖0/1/2合成方向源、非相关高电平噪声、非同时双方向、近角度双源、增量/重建等价、Gate关闭持续计数、计划/非计划gap、阶数映射、单/双源MUSIC、跨0°、HardwareMix隔离、滚动性能和连续OPEN出生门槛。真实带标注声场准确率仍待专项验收。

## 27. 参考资料

- [Grondin等2022 ODAS论文](references/26_Grondin_2022_ODAS.pdf)：GCC-PHAT、阵列定位和嵌入式工程背景。
- [归档MUSIC研究资料](../v1.4.3_existing_docs/references/README.md)
- [参考资料索引](references/README.md)

[上一章：IMCRA与P Gate](03-imcra-pre-denoise-and-probability-gate.md) · [下一章：方向ID追踪](05-direction-id-tracking.md) · [返回项目总导航](../../README.md)

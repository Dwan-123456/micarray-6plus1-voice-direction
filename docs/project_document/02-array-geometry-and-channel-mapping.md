# 02 阵列几何与通道映射

本文属于**技术参考 + 更换阵列操作指南**。空间算法看到的通道顺序和坐标必须一一对应；任何静默错位都会把相位关系解释成错误方向。

## 1. 三种通道视图

项目同时存在三种通道视图：

| 视图 | shape | 顺序 | 用途 |
| --- | --- | --- | --- |
| 设备原始`native_samples` | `[N,8]` | MA-USB8 Host通道0..7 | 保留采集真值、诊断和校准前数据 |
| 逻辑`samples` | `[N,8]` | MIC0..MIC5、Center、HardwareMix | Runtime公共音频 |
| 物理`physical_samples` | `[N,7]` | MIC0..MIC5、Center | IMCRA、GCC-PHAT和MUSIC |

`HardwareMix`由硬件产生，没有项目定义的独立物理坐标，因此不进入IMCRA、声源数或MUSIC。

## 2. 当前设备映射

`config/config.yaml`中的当前映射为：

```yaml
physical_channel_map: [0, 1, 2, 3, 4, 5, 7]
hardware_mix_channel: 6
logical_channel_map: [0, 1, 2, 3, 4, 5, 7, 6]
```

也就是：

| 设备原始通道 | 项目逻辑通道 | 当前项目解释 |
| ---: | ---: | --- |
| 0 | 0 | MIC0 |
| 1 | 1 | MIC1 |
| 2 | 2 | MIC2 |
| 3 | 3 | MIC3 |
| 4 | 4 | MIC4 |
| 5 | 5 | MIC5 |
| 7 | 6 | Center |
| 6 | 7 | HardwareMix |

`LiveSipeedSource`先把PCM16解码为原始`float32 [N,8]`，再按`logical_channel_map`重排。`ChannelCalibrator`只校准逻辑前7路，最后把第8路HardwareMix原样拼回。

> 重要核验项：项目归档的MA-USB8官方资料把Host CH6描述为硬件波束输出，把CH7描述为保留PEC通道。当前项目通过实机校准把native 7作为Center使用。更换固件、USB板或阵列前，必须重新确认CH7确实提供中央物理麦信号，不能只依赖旧配置名称。

## 3. 6+1物理几何

当前外圈是半径`r=0.04 m`的规则六边形，中心有一只麦克风。`common/geometry.py`中的固定坐标为：

| 物理麦 | 角度 | x / m | y / m |
| --- | ---: | ---: | ---: |
| MIC0 | 0° | 0.040000000 | 0.000000000 |
| MIC1 | 300° | 0.020000000 | -0.034641016 |
| MIC2 | 240° | -0.020000000 | -0.034641016 |
| MIC3 | 180° | -0.040000000 | 0.000000000 |
| MIC4 | 120° | -0.020000000 | 0.034641016 |
| MIC5 | 60° | 0.020000000 | 0.034641016 |
| Center | — | 0.000000000 | 0.000000000 |

几何身份为：

```text
r6plus1_led_face_mic0_posx_ccw_54321_v2
```

这个版本字符串随配置和测试传播，用来阻止坐标约定在没有显式版本变化的情况下被改写。

## 4. 观察面和角度方向

以LED面作为观察面，从该面向下看：

```text
                     MIC4 120°       MIC5 60°
                           \         /
                            \       /
                 MIC3 180° -- Center -- MIC0 0°
                            /       \
                           /         \
                     MIC2 240°       MIC1 300°
```

正角逆时针增加。程序中的`theta=0°`指向MIC0，`90°`位于MIC4和MIC5之间，`180°`指向MIC3。

## 5. 为什么坐标与通道顺序必须严格一致

对麦克风对`(i,j)`，候选方向产生的理论时延差是：

```text
delta_tau_ij(theta) = tau_i(theta) - tau_j(theta)
```

若音频中的通道`i`实际来自另一只麦，而坐标仍使用`r_i`，真实互谱将与错误时延表比较。常见后果包括：

- 整体角度旋转或镜像；
- 单源产生对称伪峰；
- 两源相互交换；
- 不同频率给出不一致方向；
- 轨迹持续出生和消失。

## 6. 校准模型

当前校准模型是：

```text
gain × polarity × integer-sample alignment
```

当前v1.4.3参数为：

| 物理麦 | gain | polarity | delay samples |
| --- | ---: | ---: | ---: |
| MIC0 | 1.069754 | +1 | 2 |
| MIC1 | 1.034156 | +1 | 1 |
| MIC2 | 1.100770 | +1 | 2 |
| MIC3 | 1.056586 | +1 | 0 |
| MIC4 | 1.093528 | +1 | 1 |
| MIC5 | 1.056039 | +1 | 0 |
| Center | 0.673006 | +1 | 1 |

`ChannelCalibrator`在连续块之间保存最长整数延迟所需的历史。校准只修改7个物理麦；HardwareMix和`native_samples`保持设备原值。

### 6.1 当前校准证据

- 2026-08-24报告更新相对增益；
- 2026-08-24室内1 m测量的时延相关性受反射影响，极性和整数时延继续采用2026-08-21更稳定的居中上方测量；
- 原始校准录音不进入Git，只保存版本、参数、哈希和结论。

### 6.2 校准身份如何保护时间轴

配置中的校准对象生成规范化SHA-256。该身份进入每个`IngestedAudioBlock`和`DecisionWindow`。同一epoch内校准身份变化会被拒绝；Coordinator检测到相邻输入校准变化时建立新epoch，防止不同相位基线共享IMCRA、窗口或方向状态。

## 7. 几何如何进入算法

`physical_6plus1_geometry()`返回不可变`MicGeometry`：

```text
positions_m: float64 [7,2]
speed_of_sound_mps: 343.0
version: geometry identity
```

### 7.1 声源数估计

`IncrementalGccPhatSourceCounter`根据几何预计算：

- 7麦的21个无序麦克风对；
- 每个0..359°方向的理论时延；
- 4倍过采样的离散lag网格；
- 方向时延落在相邻lag之间的线性插值系数。

### 7.2 MUSIC

`RollingNormMusicScanner`根据几何和2–4 kHz频率轴构造复数导向张量，shape为：

```text
[frequency_bins, 360 angles, 7 microphones]
```

几何版本、坐标、声速、频带、FFT和配置revision共同组成缓存键。任何一个变化都会重建导向张量。

## 8. 更换同为7麦的阵列

若麦克风数量仍为7、设备仍输出8通道，可按以下顺序修改：

1. 在`config/config.yaml`修改`physical_channel_map`、`logical_channel_map`、半径、声速或几何版本；
2. 在`common/geometry.py`更新7个二维坐标和角度定义；
3. 重新采集校准刺激，生成新的增益、极性、整数延迟和报告；
4. 更新`hardware_calibration_report_hash`和校准版本；
5. 检查`layer1_input/references/`中的硬件图是否仍适用；
6. 更新几何、通道映射和校准测试；
7. 运行全量测试和全角实机验证；
8. 重新标定GCC-PHAT、MUSIC频带、50°间距和门限。

只修改`ring_radius_m`会按比例缩放当前规则6+1坐标。非规则阵列必须直接更新坐标表，不能靠半径参数表达。

## 9. 更换麦克风数量

当前公共契约把物理麦数量严格固定为7。改变数量会影响：

- `HardwareConfig.physical_mic_count`；
- 配置向量长度校验；
- `MicGeometry.positions_m` shape；
- `ImcraHopSnapshot`中所有`[7,*]`数组；
- `DecodedAudio`和`DecisionWindow`的physical切片；
- GCC-PHAT麦克风对数量；
- MUSIC协方差、单位阵和特征子空间维度；
- UI的P1显示列；
- 全部DTO、fixture和测试。

这属于公共schema与跨层架构变更，必须更新完整测试套件和版本说明。

## 10. 更换采集设备或固件

至少确认：

- 采样率确实为48 kHz；
- 输入格式为little-endian interleaved PCM16；
- 每次回调的8通道顺序；
- 所有通道共享同步时钟；
- CH6/CH7的实际信号定义；
- 是否存在固件内延时、AGC、波束形成或其他非线性处理；
- CDC串口、LED命令和音频接口是否仍相互独立。

空间算法应使用未经硬件波束形成的同步原始麦克风通道。经过独立AGC或时变处理的通道会破坏相位和协方差关系。

## 11. 验证清单

### 静态与自动测试

- 逻辑映射是`0..7`的完整排列；
- HardwareMix固定在逻辑最后一列；
- 坐标shape、半径和几何版本正确；
- 校准只修改前7路；
- HardwareMix变化不改变IMCRA、声源数和MUSIC；
- 配置和默认几何身份一致。

### 实机验证

- 单通道敲击确认原始通道顺序；
- 阵列正前方、90°、180°、270°单源验证方向符号；
- 0°附近跨周验证没有359°/0°跳变；
- 每15°或30°扫描全角并记录误差；
- 50°边界双源验证；
- 不同距离、高度、房间和SNR验证；
- 长时运行检查通道漂移、overflow和epoch重置。

## 12. 权威文件

| 内容 | 文件 |
| --- | --- |
| 当前参数 | `config/config.yaml` |
| 几何坐标 | `common/geometry.py` |
| 配置校验 | `common/config.py` |
| 映射实现 | `layer1_input/sources.py` |
| 校准实现 | `layer1_input/calibration.py` |
| 校准配置适配 | `layer1_input/configuration.py` |
| 几何测试 | `tests/test_geometry.py` |
| 映射/校准测试 | `tests/test_l1_v03.py`、`tests/test_ingest_windowing.py` |
| 校准报告 | [2026-08-21](../v1.4.3_existing_docs/L1_HARDWARE_CALIBRATION_2026-08-21.json)、[2026-08-24](../v1.4.3_existing_docs/L1_HARDWARE_CALIBRATION_2026-08-24.json) |

[上一章：基本假设](01-assumptions-models-and-scope.md) · [下一章：IMCRA与P Gate](03-imcra-pre-denoise-and-probability-gate.md) · [返回项目总导航](../../README.md)

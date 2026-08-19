# Layer 3：逐方向音频增强

> 项目1.1.1中，L3消费L2权威`TrackedDirection`，所有方向输入、增强批次和音频输出按`(WindowKey, track_id)`精确对齐，且不自行创建或修补ID。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。本目录现已实现L3音频中心公共契约；内部复数STFT仅用于波束形成，不再跨层输出。

## 公共契约

L3接收同一320 ms、48 kHz的逻辑8通道音频、L2的0～3个平滑候选角度，以及窗口内16个连续20 ms IMCRA结果。方向增强的目标有效频带为**80～8000 Hz**。内部波束形成只使用前7个物理麦；第8路HardwareMix保留在输入接口中，但不参与导向矢量、协方差矩阵或麦对计算。4个及以上候选属于接口错误，不会截断。L3不接收L2内部ID、不再次滤波角度，也不产生仅供UI的预测方向额外批次。

每个候选方向输出：

```text
theta_deg
enhanced_audio: float32 [15360]  # 48 kHz mono
session/epoch/window/decision_sample
algorithm/fallback diagnostics
```

BF用`1-SPP`加权当前7麦STFT外积得到完整空间噪声协方差，并用IMCRA `noise_psd`收缩其对角线。逐频点计算两个导向矢量的空间相关度：`rho<0.3`使用Dual LCMV，`0.3<=rho<0.7`使用soft-null loaded MVDR，`rho>=0.7`使用loaded MVDR；单候选使用loaded MVDR。矩阵病态、求解失败或结果非有限的频点回退DAS。IMCRA缺失、未ready或时间不连续时整窗回退DAS。

IMCRA的先验/后验SNR形成有下限的频点软增益；噪声置信度和4.3 kHz以上混叠保护共同提高diagonal loading。所有阈值均来自唯一配置文件。

公共输入和输出均以音频为中心。复数STFT不再作为跨层主输出，`SpectrogramFeature [33,169]`及其FeatureExtractor从主链删除。

播放器去DC、音量、淡入淡出和峰值归一化只能作用于试听副本，不能改变交给L4或RecordingStore的正式增强音频。

## 有界滚动计算缓存

L3保持现有`n_fft=1024、win_length=960、hop_length=480、center=true、reflect`定义及完整320 ms输出。连续DecisionWindow每前进960点时，新33帧STFT中的1～29号帧逐元素复用上一窗口3～31号帧，只重新计算0、30、31、32号帧。IMCRA插值状态和加权空间协方差分子/分母同步滚动，BF权重仍按当前窗口和候选角度重新求解，不改变算法选择或公共输出。

所有时间相关缓存使用固定容量状态：正常只保留当前320 ms（16个20 ms hop），绝不超过1000 ms（50 hop）。steering vector与空间`p`结果采用16项LRU。session、epoch、sample连续性、STFT配置、IMCRA版本/频率轴、阵列几何或处理设备变化时不得沿用不兼容状态；时间跳跃直接全量重建。缓存张量驻留处理设备，不向其他层发布。

## 契约测试

- 输入8ch、算法只使用7物理麦且不读取HardwareMix做steering；
- Test UI提供三档实时对照：`optimized`、纯`ds_baseline`和
  `constant_beamwidth_baseline`。第三档以30°第一零点波束宽度（FNBW）为逐频点目标，
  根据真实6+1圆阵流形做正则化约束拟合，并用WNG下限保护；物理孔径无法安全实现
  30°的频点退回DS。该档不读取IMCRA或空间可分度p表，只用于对照，不改变正式默认算法；
- 平滑候选数量、顺序、角度和时间身份原样继承，不做第二次追踪；
- 每方向输出固定48 kHz `float32[15360]`、只读且finite；
- 主链不再输出或依赖`[33,169]`；
- 0/1/2候选及`rho`三分支；
- LCMV约束、两种MVDR无失真约束和逐频点DAS降级；
- PSD/SPP确实改变噪声协方差与三种求解权重；
- 音频重建和跨窗口连续性。
- 连续窗口29帧STFT逐元素复用等价、滚动协方差数值等价、时间/流身份失效边界及缓存容量上限。

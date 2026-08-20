# Layer 3：逐方向音频增强

> 本分支实现[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#9-layer-3-改动)的L3公共ID契约；完整1.1.0仍需与L2、L4、UI和Recording分支整合后验收。

权威目标契约见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)。本目录现已实现L3音频中心公共契约；内部复数STFT仅用于波束形成，不再跨层输出。

## 公共契约

L3接收同一160 ms、48 kHz的逻辑8通道音频、L2公开的0～3个`TrackedDirection`，以及窗口内8个连续20 ms IMCRA结果。正式关联键为`(WindowKey, track_id)`；输入元组顺序、`rank`和`theta_deg`均为权威值，L3不得按角度分配、猜测、排序、合并或修补ID。入口拒绝跨窗、重复ID/排名、旧`CandidateDirection`和未获许可的coasting目标。

观测目标可正常处理；未观测目标只有在L2显式设置`allow_l3_prediction=True`时才作为受控短时预测目标处理。长时间coasting轨只属于L2 `active_tracks`时间线，不进入L3 `directions`，因而不产生增强音频。方向增强目标频带为**80～8000 Hz**。内部只使用前7个物理麦，第8路HardwareMix不参与导向矢量、协方差或麦对计算。

每个候选方向输出：

```text
theta_deg
track_id / rank / WindowKey
enhanced_audio: float32 [7680]  # 48 kHz mono
algorithm/fallback diagnostics
```

`BeamformedL3Batch`和`Layer3Output`同样携带`WindowKey`及有序ID元数据。合成出口逐项校验WindowKey、ID集合与顺序、rank、角度和音频数量；任何偏差抛出`Layer3Error`，Runtime记录唯一`FAILED`终态并跳过后续L4，不会发布部分结果。

BF用`1-SPP`加权当前7麦STFT外积得到完整空间噪声协方差，并用IMCRA `noise_psd`收缩其对角线。逐频点计算两个导向矢量的空间相关度：`rho<0.3`使用Dual LCMV，`0.3<=rho<0.7`使用soft-null loaded MVDR，`rho>=0.7`使用loaded MVDR；单候选使用loaded MVDR。矩阵病态、求解失败或结果非有限的频点回退DAS。IMCRA缺失、未ready或时间不连续时整窗回退DAS。

IMCRA的先验/后验SNR形成有下限的频点软增益；噪声置信度和4.3 kHz以上混叠保护共同提高diagonal loading。所有阈值均来自唯一配置文件。

公共输入和输出均以音频为中心。复数STFT不再作为跨层主输出，`SpectrogramFeature [17,169]`及其FeatureExtractor从主链删除。

播放器去DC、音量、淡入淡出和峰值归一化只能作用于试听副本，不能改变交给L4或RecordingStore的正式增强音频。

## 有界滚动计算缓存

L3保持现有`n_fft=1024、win_length=960、hop_length=480、center=true、reflect`定义及完整160 ms输出。连续DecisionWindow每前进960点时，新17帧STFT中的1～13号帧逐元素复用上一窗口3～15号帧，只重新计算0、14、15、16号帧。IMCRA插值状态和加权空间协方差分子/分母同步滚动，BF权重仍按当前窗口和候选角度重新求解，不改变算法选择或公共输出。

所有时间相关缓存使用固定容量状态：正常只保留当前160 ms（8个20 ms hop），绝不超过1000 ms（50 hop）。steering vector与空间`p`结果采用16项LRU。session、epoch、sample连续性、STFT配置、IMCRA版本/频率轴、阵列几何或处理设备变化时不得沿用不兼容状态；时间跳跃直接全量重建。缓存张量驻留处理设备，不向其他层发布。

## 契约测试

- 输入8ch、算法只使用7物理麦且不读取HardwareMix做steering；
- Test UI提供四档实时切换：`optimized`、纯`ds_baseline`、全频独立
  `loaded_mvdr_baseline`和`subband_robust_baseline`。Loaded MVDR档读取同窗IMCRA噪声协方差，
  对每个方向的80～8000 Hz统一执行diagonal-loaded MVDR，不查询空间`p`表、不应用频点后滤波，
  数值不安全频点回退DAS。五频段档读取同窗IMCRA噪声统计但不查询空间`p`表：80～500 Hz
  使用温和干扰感知MVDR和声源专属Wiener增益，500～900 Hz使用WNG约束soft-LCMV，
  900 Hz～1.5 kHz和1.5～4 kHz逐步加强LCMV，4～8 kHz使用防混叠加载MVDR，
  数值不安全频点回退DAS。第一版以自由场steering作为RTF代理，并用当前窗拟合rank-1
  声源SCM；该模式只用于对照，不改变正式默认算法；
- 0～3个公开方向的WindowKey、ID、原始顺序和角度逐项继承，不做第二次追踪；
- 跨0°、重复/缺失ID、跨窗、长coasting、批次出口篡改和失败终态均有契约测试；
- 每方向输出固定48 kHz `float32[7680]`、只读且finite；
- 主链不再输出或依赖`[17,169]`；
- 0/1/2候选及`rho`三分支；
- LCMV约束、两种MVDR无失真约束和逐频点DAS降级；
- PSD/SPP确实改变噪声协方差与自适应求解权重；
- 音频重建和跨窗口连续性。
- 连续窗口13帧STFT逐元素复用等价、滚动协方差数值等价、时间/流身份失效边界及缓存容量上限。

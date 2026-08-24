# Layer 3：逐方向音频增强

> 项目1.3.2中，L3消费L2权威`TrackedDirection`，所有方向输入、增强批次和音频输出按`(WindowKey, track_id)`精确对齐，且不自行创建或修补ID。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。本目录现已实现L3音频中心公共契约；内部复数STFT仅用于波束形成，不再跨层输出。

## 公共契约

L3接收固定160 ms的`DecisionWindow`、L2的0～3个平滑候选角度，并只截取窗口末尾由`timing.downstream_audio_window_ms`指定的40/80/160 ms音频及2/4/8个连续20 ms IMCRA结果；当前配置为40 ms。方向增强的目标有效频带为**80～8000 Hz**。内部波束形成只使用前7个物理麦；第8路HardwareMix保留在输入接口中，但不参与导向矢量、协方差矩阵或麦对计算。4个及以上候选属于接口错误，不会截断。L3不接收L2内部ID、不再次滤波角度，也不产生仅供UI的预测方向额外批次。

每个候选方向输出：

```text
theta_deg
enhanced_audio: float32 [1920/3840/7680]  # 48 kHz mono，由统一配置派生
session/epoch/window/decision_sample
algorithm/fallback diagnostics
```

BF用`1-SPP`加权当前7麦STFT外积得到完整空间噪声协方差，并用IMCRA `noise_psd`收缩其对角线。逐频点计算两个导向矢量的空间相关度：`rho<0.3`使用Dual LCMV，`0.3<=rho<0.7`使用soft-null loaded MVDR，`rho>=0.7`使用loaded MVDR；单候选和三候选使用loaded MVDR。三档diagonal loading按固定retry维批量执行；同一加载协方差的Cholesky分解由LCMV/MVDR多右端复用，两个soft-null目标也批量求解。Hermitian正定矩阵的条件数使用特征值范围校验，分支阈值、首个有效retry和DAS回退语义不变。矩阵病态、求解失败或结果非有限的频点回退DAS。IMCRA缺失、未ready或时间不连续时整窗回退DAS。

IMCRA的先验/后验SNR形成有下限的频点软增益；噪声置信度和4.3 kHz以上混叠保护共同提高diagonal loading。所有阈值均来自唯一配置文件。

公共输入和输出均以音频为中心。复数STFT不再作为跨层主输出；内部特征时间轴随配置为9/17帧，FeatureExtractor不在跨层主链中使用。

播放器去DC、音量、淡入淡出和峰值归一化只能作用于试听副本，不能改变交给L5或RecordingStore的正式增强音频。

CUDA实验接口把prepare、候选BF和ISTFT结果保留在设备上，Runtime仅对已经积压的连续窗口做最多4窗的有界微批，再通过pinned host buffer异步回传完整短音频。finite校验和诊断值格式化延迟到批次同步后执行，输出音频、fallback、diagnostics、track ID及绝对sample契约与同步接口一致；异常批次不得发布。RTX 5060 Laptop GPU的四窗双候选隔离结果仍慢于CPU，因此正式配置继续使用CPU，CUDA路径只用于显式性能实验。

## 有界滚动计算缓存

L3保持现有`n_fft=1024、win_length=960、hop_length=480、center=true、reflect`定义。40/80/160 ms分别形成5/9/17帧STFT；DecisionWindow按绝对sample前进`N`个20 ms hop且仍有配置窗口重叠时，缓存复用对齐内部帧，只重算当前反射边界和新增帧。IMCRA插值状态只搬运新增hop，加权空间协方差分子/分母按相同过期/新增帧集合滚动。BF权重仍按当前窗口和候选角度重新求解，不改变算法选择或公共输出。

所有时间相关缓存使用固定容量状态：正常只保留当前配置的40/80/160 ms（2/4/8个20 ms hop），绝不超过1000 ms（50 hop）。steering vector与空间`p`结果继续采用原有16项LRU。session、epoch、非960 sample对齐、时间倒退、跳跃达到配置窗口长度、STFT配置、IMCRA版本/频率轴、阵列几何或处理设备变化时不得沿用不兼容状态并全量重建。缓存张量驻留处理设备，不向其他层发布。

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
- 每方向输出配置化48 kHz `float32[1920/3840/7680]`、只读且finite；
- 主链不跨层输出或依赖内部`[9/17,169]`特征；
- 0/1/2/3候选及`rho`三分支；
- LCMV约束、两种MVDR无失真约束和逐频点DAS降级；
- PSD/SPP确实改变噪声协方差与自适应求解权重；
- 音频重建和跨窗口连续性。
- 40/80/160 ms三档的STFT逐元素复用、跳窗复用与滚动协方差数值等价、达到配置窗口长度时的失效边界及缓存容量上限。

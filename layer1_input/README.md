# Layer 1：八通道输入、物理映射与IMCRA

> 项目1.2.4的L1输入已按[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#5-layer-1-改动)落地：提供连续、校准后的7麦输入、verified/unverified状态及版本/hash边界；L1不创建ID，MUSIC与公共方向ID由L2负责。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。L1的逻辑8通道、麦克风面坐标和20 ms IMCRA迁移已完成。

## 目标职责

L1读取48 kHz native 8ch音频，完成PCM解码、校准、连续性guard和逻辑重排。Host顺序为`CH0..CH5=MIC0..MIC5、CH6=HardwareMix、CH7=Center`；对外顺序固定为：

```text
[MIC0, MIC1, MIC2, MIC3, MIC4, MIC5, Center, HardwareMix]
```

输出是`float32 [N,8]`。前7路是物理阵列，最后1路是硬件合成总声音；HardwareMix只作为预留接口、显示、录制和后续实验输入，不得加入麦克风几何、SRP/MUSIC或MVDR steering。

## 校准身份边界

每个`IngestedAudioBlock`携带不可变`CalibrationMetadata`，并原样进入`DecisionWindow`。元数据包含`status、version、calibration_hash、correction_model、report_hash`，以及可选的`fractional_delay_asset`和`frequency_response_asset`身份；未来资产仅以`uri、version、sha256`标识，实际补偿算法不在本次实现中。当前校准器继续执行增益、极性和整数sample delay；若配置了尚未支持的未来资产会明确拒绝启动，不能悄悄忽略。

规范化校准配置的SHA-256是处理边界的一部分。校准hash变化会触发新epoch，同一epoch内变更会被WindowAssembler拒绝。没有正式校准身份的兼容输入使用明确的`unverified`身份；Development Test UI显示红色警告，verified状态同时显示版本和hash摘要。

## 长时间连续采集边界

PortAudio回调只复制驱动当前提供的PCM块、分配单调sequence并投递到有界交接队列；RMS电平改为读取`status()`时按需计算，不能占用实时回调。正式主链交接容量为500个20 ms块（10秒），用于吸收Windows调度、GPU或UI造成的短时停顿；持续算力不足仍必须通过队列深度、高水位和drop计数暴露，不能无限积压或伪装为连续。

`status()`公开`input_overflow_count、handoff_drop_count、handoff_queue_depth、handoff_queue_capacity、handoff_queue_high_water`。连续的handoff满队列丢块合并为一次有范围和lost sample数的健康事件，避免同一次拥塞突发对IMCRA反复重置；不连续的独立缺口仍分别增加epoch。任何真实缺失都不补零、不隐藏。

## 麦克风面坐标

官方装配图从灯面俯视，灯面位于麦克风背面。算法统一从朝上的麦克风面观察：中央麦为原点，实际位于底部的MIC0方向为`+x/0°`，角度逆时针增加。MIC0～MIC5依次是`0°、60°、120°、180°、240°、300°`，Center为原点。

官方阵列板资料给出`MIC_D0=MIC0/1`、`MIC_D1=MIC2/3`、`MIC_D2=MIC4/5`、`MIC_D3=Center`；Host的CH映射属于MA-USB8桥接定义，二者必须分别在manifest中记录。

## IMCRA输出

IMCRA已迁入L1，对校准后的7个物理麦按20 ms hop独立更新。实现遵循[Israel Cohen 2003论文](https://doi.org/10.1109/TSA.2003.811544)的双迭代流程：第一轮时频平滑与最小值跟踪产生粗VAD，第二轮排除强语音分量后再次平滑和跟踪最小值，再按论文式(29)、式(7)和式(10)～(12)依次计算先验语音缺失概率、后验SPP和噪声PSD。当前输出频带版本为`cohen_imcra_2003_l1_v3`。

论文表I参数固定为`w=1、αs=0.9、U=8、V=15、D=120、Bmin=1.66、γ0=4.6、γ1=3、ζ0=1.67、α=0.92、αd=0.85、β=1.47`。表I数值原本用于16 kHz实验；本版本把递归参数固定下来，同时适配项目的48 kHz输入、960-sample hop和2048点FFT。`Bmin`与窗、hop和FFT有关，未来若完成统计/实机重标定，必须升级算法版本，不能静默覆盖当前基线。

实现顺序严格为：

1. 式(14)～(16)：第一轮频率/时间平滑`S`及`S_min`；
2. 式(18)、(21)：由`gamma_min`和`zeta`形成粗语音缺失指示；
3. 式(26)～(28)：条件频率/时间平滑和`S_min_tilde`；其中式(28)的`zeta_tilde = S / (Bmin * S_min_tilde)`，分子是第一轮`S`；
4. 式(29)、(7)：计算`q_hat`和后验SPP；
5. 式(32)～(33)：Decision-Directed先验SNR及确定语音存在时的LSA增益；
6. 式(10)～(12)：用SPP调节递归平滑系数并作`β`偏差补偿，得到噪声PSD。

每个结果发布**0～10000 Hz**范围内的427点频率轴、两轮平滑谱、两轮局部最小量、先验/后验SNR、先验语音缺失概率、后验SPP和噪声PSD。每麦噪声特征为`noise_level_db、signal_level_db、snr_db、mean_spp`；前三项使用0～10000 Hz宽频统计，`mean_spp`仍只在500～4000 Hz证据子带聚合。7个`mean_spp`的中位数形成`array_source_probability_20ms ∈ [0,1]`。这是本项目的阵列概率适配器，不属于论文单通道公式。

结果与音频共享session、epoch和绝对sample区间。HardwareMix不参与IMCRA状态或7麦阵列概率聚合。

## 可切换预降噪

`ImcraWienerPreDenoiser`在IMCRA完成原始20 ms音频估计后运行。每个物理麦使用自己的`prior_snr`和`SPP`生成SPP保护Wiener增益，处理0～10000 Hz，最低增益为-18 dB；HardwareMix及10000 Hz以上内容直通。具体处理是：把相邻两个20 ms hop组成40 ms帧，乘平方根Hann窗后执行2048点RFFT；对每麦、每个0～10000 Hz频点的复数STFT系数乘实数增益，因此同时缩放该频点的幅度、保留相位；随后执行IRFFT、再次乘平方根Hann窗，并以20 ms步长50%重叠相加，恢复连续时域音频。当前频带版本为`imcra_wiener_wola_v3`。

Test UI的L1区域提供“IMCRA预降噪”开关。OFF时输出原始LogicalAudio；ON时Runtime等待对应降噪hop完成，将下游音频前7路替换为降噪结果后再生成DecisionWindow。IMCRA预热或无效时使用单位增益。`native_samples`始终保持设备原始数据，第8路HardwareMix始终不修改。

PSD/SPP状态保留0～10000 Hz目标频带；概率证据仍仅从500～4000 Hz聚合，避免新增高频直接改变Gate判决。

不可变`ImcraHopSnapshot`发布字段包括`frequencies_hz、noise_psd、smoothed_psd、conditional_smoothed_psd、minimum_psd、conditional_minimum_psd、spp、speech_absence_probability、posterior_snr、prior_snr、noise_features、noise_level_db、source_probability_per_mic、array_source_probability_20ms`。所有频谱状态形状为`[7,427]`，必须finite且只读；概率必须位于`[0,1]`。

断流、sequence/timestamp/rate异常或epoch切换时必须清空IMCRA状态并重新预热。预热期间发布明确状态，不能把缺失概率写成0。单纯静音或概率下降不改变epoch，也不会让IMCRA重新进入`warming_up`。

## 边界

L1不计算DOA、不执行波束形成、不判断人声，也不分配权威绝对sample。IngestCoordinator仍是唯一时间轴权威；预降噪关闭时向WindowAssembler和RecordingStore分发原始LogicalAudio，开启时分发sample边界相同、前7路已替换的降噪LogicalAudio，同时保留原始`native_samples`。

## 已实现门禁测试

- native→logical 8ch映射、只读/finite/C-contiguous契约；
- verified/unverified校准状态、稳定版本/hash传播、同epoch边界拒绝和未来资产显式拒绝；
- 麦克风面逆时针几何及灯面镜像防错；
- HardwareMix保留但不进入7麦阵列算法；
- IMCRA 20 ms更新、0～10000 Hz PSD/SPP状态、500～4000 Hz概率聚合、重置、预热及概率范围；
- 40 ms/20 ms WOLA连续重建、每麦独立增益、HardwareMix直通、预热旁路和运行时开关；
- 设备、WAV、实时handoff和Ingest时间轴一致性。
- PortAudio回调不执行RMS、连续handoff溢出事件合并、交接队列高水位与输入健康诊断。

Development Test UI左上象限需显示8路电平、IMCRA状态、每麦噪声摘要及20 ms概率；灯控和scratch录音仍复用同一L1输入，不能重开设备。

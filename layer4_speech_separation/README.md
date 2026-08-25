# Layer 4：渐进与采集后双人语音分离

正式20 ms Runtime审计仍只运行到L3、`TrackAudioStreamHub`和`offline_after_l4`占位。`1.3.5`另有隔离的渐进旁路：Hub每个ID累计默认10秒、可调3～15秒后，由后台producer claim连续块；队列接纳成功才推进cursor。停止采集且L3排空后，Hub仍把完整长音频封存为不可变`Layer4LongAudioInput`，再运行权威离线作业。Preview不写DecisionRecord或RecordingStore。

Hub 同步记录每个20 ms hop所属 L2 决策窗的方向输出数量。`l2_direction_count_max_v1`按`min(2, maximum)`路由：整条长音频历史最大值1判为一人，最大值2或3均按当前双人上限处理；原始最大值与实际采用人数同时写入元数据。两种输入均由 L4 公共`Layer4Resampler`从48 kHz降到16 kHz。一人直接旁路形成L4输出并自动进入L5；两人调用所选分离后端。

可选对比后端均为官方开源模型和官方权重：

- `alibabasglab/MossFormer2_SS_16K`，官方 ClearerVoice-Studio 网络，Apache-2.0；
- `JusperLee/TIGER-speech`，官方 TIGER 网络，Apache-2.0。

模型资产保存在`models/mossformer2_ss_16k_v1`与`models/tiger_speech_16k_v1`，manifest固定来源revision、SHA-256、16 kHz和双输出契约。权威长音频适配器使用30秒块/1秒重叠。渐进MF2使用配置块/1秒重叠：首块延迟末尾，后续输入为上一块1秒尾加新块；输出重叠判断匿名排列并交叉淡化。`stable_branch_id`只负责连续身份，A/B按累计匹配分重新排序。

Test UI工作流固定不合并，不再提供“合并”开关。匹配器按`l3_bf_1_4khz_complex_coherence_v3`对完整音频执行512点Hann STFT、160点hop和1～4 kHz逐帧复频谱相干度，但匹配度只用于排序：两条模型原生16 kHz候选均写入临时试听缓存，高分标为A、低分标为B，分数相同时按模型原始顺序稳定排序。两条候选分别进入L5并把逐20 ms结果标在对应波形上；单人轨则保留唯一旁路音频。管线仍保留`merge_candidates`API供非Test UI的历史对照使用。

每条权威L4 16 kHz波形在峰值保护后运行DNSMOS P.835。渐进旁路只在取得9.02秒完整窗口后按30秒间隔采样，尾部冲刷再补一个固定9.02秒尾窗并与周期样本求均值，避免停止成本随录音长度线性增长；只有canonical才对完整分支精算。SIG/BAK/OVRL、采样范围和综合MOS写入元数据，L6消费已保存分数。

L4终端输出固定为16 kHz：同一份波形写入Test UI试听WAV，整批L4完成后由同一后台作业自动直接交给L5，不执行16→48→16 kHz往返重采样。NVIDIA Frame-VAD原始softmax按20 ms帧索引裁齐，输出严格与每320样本一一对应；L5只把逐帧概率和判断返回L4预览条着色。整轨概览概率另用完整序列的连续3帧最大均值汇总，不得覆盖逐帧时间线。

```text
sealed TrackAudioStreamHub long track (48 kHz, ID + angle + L2 count history)
  -> Layer4Resampler (16 kHz)
  -> maximum recorded L2 direction count
  -> 1 speaker: bypass separation -> Layer 5 CNN
  -> 2 speakers: MossFormer2 or TIGER -> exactly two continuous candidates
  -> fixed unmerged workflow: rank both 16 kHz candidates as ID-A / ID-B -> Layer 5 each
```

`ApplicationRuntime.offline_l4_sources`只在sidecar安全停止并完成封存后公开。Test UI采集中显示latest preview revision；伪实时启用时后端固定为YAML默认MF2。完整canonical开始时保留preview，成功后一次性替换，失败时保留并允许重试。

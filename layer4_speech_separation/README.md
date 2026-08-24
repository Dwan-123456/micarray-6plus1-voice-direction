# Layer 4：采集后双人语音分离

Layer 4 是采集结束后的离线层。实时 Runtime 只运行到 L3 和`TrackAudioStreamHub`；L5 CNN 不再消费实时连续片段。停止采集且 L3 队列完全排空后，Hub 把其已经按 ID 去重、交叉淡化、响度补偿并连续拼接的完整长音频封存为不可变`Layer4LongAudioInput`，再由显式离线作业整体提交 L4。RecordingStore WAV 只作为恢复入口，不是主数据链。

Hub 同步记录每个20 ms hop所属 L2 决策窗的方向输出数量。`l2_direction_count_max_v1`按`min(2, maximum)`路由：整条长音频历史最大值1判为一人，最大值2或3均按当前双人上限处理；原始最大值与实际采用人数同时写入元数据。两种输入均由 L4 公共`Layer4Resampler`从48 kHz降到16 kHz。一人直接旁路形成L4输出并自动进入L5；两人调用所选分离后端。

可选对比后端均为官方开源模型和官方权重：

- `alibabasglab/MossFormer2_SS_16K`，官方 ClearerVoice-Studio 网络，Apache-2.0；
- `JusperLee/TIGER-speech`，官方 TIGER 网络，Apache-2.0。

模型资产保存在`models/mossformer2_ss_16k_v1`与`models/tiger_speech_16k_v1`，manifest固定来源revision、SHA-256、16 kHz和双输出契约。长音频按30秒块、1秒重叠推理；适配器在重叠区比较两种匿名排列并修正块间换序，然后交叉淡化为两条完整候选。

匹配器按`l3_bf_1_4khz_complex_coherence_v3`对完整音频执行512点Hann STFT、160点hop和1～4 kHz逐帧复频谱相干度，再按原Hub参考频带能量加权。该统计保留说话人的相位/时序身份，并容忍模型产生的全局极性翻转；末尾不足一帧补零，长录音分批计算以限制内存。音轨短于2秒、最高相干度低于0.50或候选分差小于0.025时，输出回退为原L3参考，避免把短轨或身份不明确的模型伪影送入试听和L5。匹配分数、差值、模型revision及回退原因均进入结果元数据和离线job manifest。

L4终端输出固定为16 kHz：同一份波形写入Test UI试听WAV，整批L4完成后由同一后台作业自动直接交给L5，不执行16→48→16 kHz往返重采样。NVIDIA Frame-VAD原始softmax按20 ms帧索引裁齐，输出严格与每320样本一一对应；L5只把逐帧概率和判断返回L4预览条着色。整轨概览概率另用完整序列的连续3帧最大均值汇总，不得覆盖逐帧时间线。

```text
sealed TrackAudioStreamHub long track (48 kHz, ID + angle + L2 count history)
  -> Layer4Resampler (16 kHz)
  -> maximum recorded L2 direction count
  -> 1 speaker: bypass separation -> Layer 5 CNN
  -> 2 speakers: MossFormer2 or TIGER -> exactly two continuous candidates
  -> 1--4 kHz matcher -> one selected source preserving ID + angle
  -> Layer 5 CNN
```

`ApplicationRuntime.offline_l4_sources`在完全停止后公开封存包，`run_offline_l4()`接受实现`process_sealed()`的编排器。`scripts/run_offline_l4.py`提供已落盘session的恢复/批处理入口。Development Test UI下半区已经提供L3/L4/L5三栏、MossFormer2/TIGER选择和“发送到L4”；整批L4完成后，同一后台任务自动且仅一次运行L5并把逐20 ms结果回写到L4音频条。

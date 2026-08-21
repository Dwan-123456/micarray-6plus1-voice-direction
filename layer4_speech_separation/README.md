# Layer 4：采集后双人语音分离

Layer 4 是采集结束后的离线层。实时 Runtime 只运行到 L3 和`TrackAudioStreamHub`；L5 CNN 不再消费实时连续片段。停止采集且 L3 队列完全排空后，Hub 把其已经按 ID 去重、交叉淡化、响度补偿并连续拼接的完整长音频封存为不可变`Layer4LongAudioInput`，再由显式离线作业整体提交 L4。RecordingStore WAV 只作为恢复入口，不是主数据链。

Hub 同步记录每个20 ms hop所属 L2 决策窗的方向输出数量。`l2_direction_count_max_v1`读取整条长音频历史的最大值：最大值1判为一人，最大值2判为两人；最大值3拒绝进入当前双人L4。两种输入均由 L4 公共`Layer4Resampler`从48 kHz降到16 kHz。一人直接送入同一个 L5 CNN；两人调用所选分离后端。

可选对比后端均为官方开源模型和官方权重：

- `alibabasglab/MossFormer2_SS_16K`，官方 ClearerVoice-Studio 网络，Apache-2.0；
- `JusperLee/TIGER-speech`，官方 TIGER 网络，Apache-2.0。

模型资产保存在`models/mossformer2_ss_16k_v1`与`models/tiger_speech_16k_v1`，manifest固定来源revision、SHA-256、16 kHz和双输出契约。长音频按30秒块、1秒重叠推理；适配器在重叠区比较两种匿名排列并修正块间换序，然后交叉淡化为两条完整候选。

匹配器按`l3_bf_2_4khz_magnitude_cosine_v1`对完整音频执行512点Hann STFT、160点hop和2～4 kHz逐帧幅度余弦相似度，再按原Hub参考频带能量加权。末尾不足一帧会补零，长录音分批计算以限制内存。两个候选只做一次权威选择；同分固定选择索引0。匹配分数、差值、模型revision与最终L5结果均进入离线job manifest。

L4输出送入L5后，NVIDIA Frame-VAD一次读取完整48 kHz长音频并内部降采样到16 kHz。模型的原始softmax人声概率按NVIDIA 20 ms帧索引裁齐到输入hop数量，输出严格与原音频每个960样本一一对应；Test UI按这些逐帧概率着色。整轨概览概率另用完整序列的连续3帧最大均值汇总，不得用概览值覆盖逐帧时间线。离线manifest从`offline_l4_job_v2`开始同时保存逐20 ms概率和判断。

```text
sealed TrackAudioStreamHub long track (48 kHz, ID + angle + L2 count history)
  -> Layer4Resampler (16 kHz)
  -> maximum recorded L2 direction count
  -> 1 speaker: bypass separation -> Layer 5 CNN
  -> 2 speakers: MossFormer2 or TIGER -> exactly two continuous candidates
  -> 2--4 kHz matcher -> one selected source preserving ID + angle
  -> Layer 5 CNN
```

`ApplicationRuntime.offline_l4_sources`在完全停止后公开封存包，`run_offline_l4()`接受实现`process_sealed()`的编排器。`scripts/run_offline_l4.py`提供已落盘session的恢复/批处理入口。UI本次只预留这些接口，不新增页面或控件。

# Layer 4：离线双人语音分离契约

Layer 4 是采集结束后的可选离线层，不属于 L1→L2→L3→L5 的 20 ms 实时 Runtime。只有录音停止、L3 队列排空、按 ID 的连续长音频完成拼接并封存后，外部编排器才可以提交 Layer 4 请求。当前提交只冻结架构与接口，不下载或执行 MossFormer2/TIGER，也不实现讲话人数分类器或任务编排器。

合法输入至多包含两名讲话人。讲话人数分类器将完整 L3 长音频判为一人时直接绕过 Layer 4 进入 Layer 5；判为两人时，每条 L3 ID 长音频分别调用同一个 Layer 4 后端。后端必须接收 16 kHz 单声道音频并一次返回两条匿名、等长、finite `float32` 音频。

每个 L3 ID 的两个候选只保留一个。原 L3 波束长音频先用同一重采样器变为 16 kHz，然后在 2～4 kHz 上按 `l3_bf_2_4khz_magnitude_cosine_v1` 与两个候选比较：512 点 Hann STFT、160 点 hop、逐帧幅度谱余弦相似度、按原 L3 参考频带能量加权。得分较高者继承原 `session_id/stream_epoch/track_id/theta_deg`；另一条仅为未发布候选。分数相同时固定选择索引 0，保证确定性。

长音频可能由模型内部重叠分段，但模型适配器必须先解决分段间匿名输出排列并返回两条连续候选；禁止每 20 ms 重新选择候选。匹配层只对完整、等时长的候选做一次权威选择。匹配分数与差值用于审计，首版不设置拒绝阈值。

正式数据流目标为：

```text
sealed L3 track (48 kHz, ID + angle)
  -> shared high-quality resampler (16 kHz)
  -> future 1/2-speaker classifier
  -> 1 speaker: bypass Layer 4
  -> 2 speakers: MossFormer2 or TIGER adapter -> exactly two candidates
  -> 2--4 kHz matcher -> exactly one selected source preserving ID + angle
  -> Layer 5 CNN
```

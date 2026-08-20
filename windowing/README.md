# WindowAssembler：1.2.1滚动输入契约

> Windowing按[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#6-windowing-改动)实现。直接窗口为160 ms，20 ms发布节拍不变；L2所需的240/320 ms历史由其滚动STFT、协方差与MUSIC状态跨窗口维护。

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。8通道窗口和IMCRA hop对齐已完成迁移。

本层把连续`IngestedAudioBlock [N,8]`组装为每20 ms一个只读`DecisionWindow [7680,8]`。每个窗口保留160 ms完整上下文；`physical_samples`固定为`[7680,7]`，`physical_history(160)`只返回物理麦，超出窗口的请求会被拒绝。HardwareMix不得进入MUSIC。Windowing不裁剪下游音频；L3、L4和Test UI共同使用`timing.downstream_audio_window_ms`从该容器末尾选择80/160 ms，当前为80 ms。

`rolling_state_key=(session_id, stream_epoch, decision_sample)`标识L2滚动状态的时间位置；`rolling_update_start_sample`固定指向最近20 ms的起点，连续后继检查要求同session、同epoch且decision sample增加960。epoch变化立即重置assembler；校准hash变化由IngestCoordinator形成新epoch，同一epoch内校准身份变化被拒绝。

配置`layer2.music.context_ms`只允许160、240或320 ms，`comparison_context_ms`固定覆盖三档，`max_history_ms`固定为320 ms。这些字段控制L2跨窗口滚动历史，不表示WindowAssembler持有320 ms直接窗口，也不生成任何方向ID。

每个窗口对齐末尾40 ms覆盖的两个20 ms IMCRA结果；L2 Probability Gate据此计算：

```text
gate_probability_40ms = (p_previous + p_current) / 2
```

两个概率必须属于同一session/epoch且sample区间连续。缺帧、跨epoch或预热时只发布`WARMING_UP/UNAVAILABLE`，不得拼接旧概率或伪造0。

每个epoch首个endpoint为7680，此后每960 samples生成一个窗口。任意L1切块必须得到相同窗口、概率配对和时间身份；epoch变化立即清空音频与IMCRA缓存并重新预热。

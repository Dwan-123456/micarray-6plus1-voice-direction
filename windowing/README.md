# WindowAssembler：1.1.1滚动输入契约

> 项目1.1.1的Windowing已按[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#6-windowing-改动)实现。窗口大小与20 ms节拍保持不变，320 ms是可用历史上限；滚动STFT、协方差、MUSIC和轨迹coasting由L2实现。

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。8通道窗口和IMCRA hop对齐已完成迁移。

本层把连续`IngestedAudioBlock [N,8]`组装为每20 ms一个只读`DecisionWindow [15360,8]`。每个窗口保留320 ms完整上下文；`physical_samples`固定为`[15360,7]`，`physical_history(160|240|320)`为L2提供只含物理麦的有效历史，HardwareMix不得进入MUSIC。完整8通道窗口继续供现有下游和记录契约使用。

`rolling_state_key=(session_id, stream_epoch, decision_sample)`标识L2滚动状态的时间位置；`rolling_update_start_sample`固定指向最近20 ms的起点，连续后继检查要求同session、同epoch且decision sample增加960。epoch变化立即重置assembler；校准hash变化由IngestCoordinator形成新epoch，同一epoch内校准身份变化被拒绝。

配置`layer2.music.context_ms`只允许160、240或320 ms，`comparison_context_ms`固定覆盖三档，`max_history_ms`固定为320 ms。这些字段冻结输入选择，不表示WindowAssembler实现或缓存MUSIC状态，也不生成任何方向ID。

每个窗口对齐末尾40 ms覆盖的两个20 ms IMCRA结果；L2 Probability Gate据此计算：

```text
gate_probability_40ms = (p_previous + p_current) / 2
```

两个概率必须属于同一session/epoch且sample区间连续。缺帧、跨epoch或预热时只发布`WARMING_UP/UNAVAILABLE`，不得拼接旧概率或伪造0。

每个epoch首个endpoint仍为15360，此后每960 samples生成一个窗口。任意L1切块必须得到相同窗口、概率配对和时间身份；epoch变化立即清空音频与IMCRA缓存并重新预热。

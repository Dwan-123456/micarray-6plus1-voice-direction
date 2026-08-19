# WindowAssembler：v0.3目标窗口契约

> 项目1.1.0待实现改动见[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#6-windowing-改动)：窗口大小与20 ms节拍保持不变，320 ms只是可用历史上限；L2 MUSIC按新增/移出STFT帧增量更新，并在目标机比较160/240/320 ms有效历史。Gate阻断时既有轨迹仍按绝对sample进入coasting/超时。下文描述当前1.0.1实现。

权威目标见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。8通道窗口和IMCRA hop对齐已完成迁移。

本层把连续`IngestedAudioBlock [N,8]`组装为每20 ms一个只读`DecisionWindow [15360,8]`。每个窗口保留320 ms完整上下文，末尾40 ms供L2 SRP，完整窗口供L3增强和L4分类。

每个窗口对齐末尾40 ms覆盖的两个20 ms IMCRA结果；L2 Probability Gate据此计算：

```text
gate_probability_40ms = (p_previous + p_current) / 2
```

两个概率必须属于同一session/epoch且sample区间连续。缺帧、跨epoch或预热时只发布`WARMING_UP/UNAVAILABLE`，不得拼接旧概率或伪造0。

每个epoch首个endpoint仍为15360，此后每960 samples生成一个窗口。任意L1切块必须得到相同窗口、概率配对和时间身份；epoch变化立即清空音频与IMCRA缓存并重新预热。

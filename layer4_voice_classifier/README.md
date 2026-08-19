# Layer 4：公共方向轨上的人声概率

权威目标契约见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md#10-layer-4-改动)。当前 L4 分支已将 `track_id` 加入音频段、检测和阶段结果，并删除 Runtime 中按角度向 L2 回送正式化/续租证据的路径。项目版本与发布标签未提前变更。

L4对每个候选接收L3输出的48 kHz、320 ms单声道增强音频`float32[15360]`，以及与该窗口严格对齐的16个IMCRA `array_source_probability_20ms`。L4先创建独立音频副本，以20 ms为单位执行`imcra_probability_rms_v1`响度补偿，再由模型适配器降采样到16 kHz并执行CNN前处理，输出一个`[0,1]` Voice / Non-Voice概率。L3原始波形、试听和录音始终使用未补偿版本。

响度补偿目标RMS为`-23.0 dBFS`；概率不高于0.30时不补偿、达到0.80时完整补偿，中间线性插值。算法只放大、不主动衰减，并以`-3 dBFS`限制新增增益：若原始输入已经超过该值，则增益为0 dB但不会压低原始峰值。相邻20 ms片段的增益在dB域线性过渡，过渡包络仍逐样本服从峰值保护。概率缺失、IMCRA未预热或静音片段均使用0 dB。

L4不接收`[33,169]`特征。每个输入必须携带L3已绑定的`(WindowKey, track_id, theta_deg, audio)`，同一窗口0～3项的ID必须是唯一正整数。Runtime和L4会校验数量、ID集合、顺序、角度及音频的逐项对应；L4不分配、猜测、合并或重新关联ID。在L3公共DTO分支合并前，有序ID由L3阶段结果携带，进入L4时与有序增强音频逐项绑定；合并后优先直接验证L3公共对象上的ID。

L4不做跨窗口Tracking或身份识别，不确认、续命或删除方向轨，也不反馈改变L2 Gate、几何生命周期或L3音频。未来如恢复语义反馈，只能用完整`(session_id, stream_epoch, track_id)`精确关联，不得按角度匹配。primary/shadow模型可读取同一不可变波形；模型插件仍只返回概率向量，由L4按输入顺序写回权威ID。

L4判断阈值与L2 Gate阈值是两套不同参数。Development Test UI必须使用不同标签和滑动条；拖动L4阈值只重算已缓存概率的标签，不重跑L3或CNN。

## 已实现门禁

- 只接受48 kHz单声道音频并在模型内部降采样；
- 明确拒绝旧`[33,169]`主输入；
- 角度、window和sample身份继承；
- `track_id`在L2/L3/L4阶段结果、`VoiceDetection`、Runtime记录映射和Test UI中同序透传；
- 重复/缺失/错序ID、角度错位、超过3个方向和跨阶段不一致均被拒绝；空批次保持空ID集合；
- 概率finite且位于`[0,1]`，阈值重判不重跑模型；
- primary/shadow读取同一不可变波形批次，artifact加载前校验hash；环境检查执行实际CUDA MarbleNet波形前向。
- 16个20 ms概率按`context_start_sample + i*960`与音频严格对齐；补偿前后RMS、峰值、请求/应用增益及窗口汇总随L4结果记录。

目标域微调、正式概率校准和锁定测试集指标仍待数据集版本完成后实施。

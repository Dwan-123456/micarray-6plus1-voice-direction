# Layer 4：48 kHz音频到人声概率

> 项目1.1.1中，L4音频段、检测和阶段结果贯通`track_id`；L4不按角度创建或修补ID，语义概率反馈只按L2权威身份关联。

权威目标契约见根目录[`ARCHITECTURE_V0.3_TARGET.md`](../ARCHITECTURE_V0.3_TARGET.md)。L4已迁移为独立48 kHz波形输入契约；L3自身的公共契约迁移不属于本层完成范围。

L4对每个候选接收L3输出的48 kHz、320 ms单声道增强音频`float32[15360]`，以及与该窗口严格对齐的16个IMCRA `array_source_probability_20ms`。L4先创建独立音频副本，以20 ms为单位执行`imcra_probability_rms_v1`响度补偿，再由模型适配器降采样到16 kHz并执行CNN前处理，输出一个`[0,1]` Voice / Non-Voice概率。L3原始波形、试听和录音始终使用未补偿版本。

响度补偿目标RMS为`-23.0 dBFS`；概率不高于0.30时不补偿、达到0.80时完整补偿，中间线性插值。算法只放大、不主动衰减，并以`-3 dBFS`限制新增增益：若原始输入已经超过该值，则增益为0 dB但不会压低原始峰值。相邻20 ms片段的增益在dB域线性过渡，过渡包络仍逐样本服从峰值保护。概率缺失、IMCRA未预热或静音片段均使用0 dB。

L4不接收L2内部ID或`[33,169]`特征，方向标签只继承L3携带的平滑角；它不再次滤波、不做跨窗口Tracking或身份识别，也不反馈改变L2 Gate、SRP或L3音频。primary/shadow模型可读取同一不可变波形；只有primary结果进入正式VoiceDetection和DecisionRecord。

L4判断阈值与L2 Gate阈值是两套不同参数。Development Test UI必须使用不同标签和滑动条；拖动L4阈值只重算已缓存概率的标签，不重跑L3或CNN。

## 已实现门禁

- 只接受48 kHz单声道音频并在模型内部降采样；
- 明确拒绝旧`[33,169]`主输入；
- 角度、window和sample身份继承；
- 概率finite且位于`[0,1]`，阈值重判不重跑模型；
- primary/shadow读取同一不可变波形批次，artifact加载前校验hash；环境检查执行实际CUDA MarbleNet波形前向。
- 16个20 ms概率按`context_start_sample + i*960`与音频严格对齐；补偿前后RMS、峰值、请求/应用增益及窗口汇总随L4结果记录。

目标域微调、正式概率校准和锁定测试集指标仍待数据集版本完成后实施。

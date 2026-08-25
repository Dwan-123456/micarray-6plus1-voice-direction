# Layer 5：48 kHz音频到人声概率

> 项目1.3.5开发线中，L5音频段、检测和阶段结果贯通`track_id`；L5不按角度创建或修补ID，语义概率只按L2权威身份关联。

权威目标契约见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)。正式逐窗L5仍只写`offline_after_l4`审计占位。隔离旁路会在L4稳定片段出现后渐进运行MarbleNet；它保留左右各1.6秒上下文并延迟右端，只发布不再受未来音频影响的20 ms帧。停止后的完整L5仍是canonical。

公共服务每窗只追加一个与IMCRA网格严格对齐的20 ms hop，避免40 ms重叠重复；可选执行`imcra_probability_rms_v1`并维护最长3200 ms试听缓冲，同时归档完整长轨。渐进与权威L5都只读取L4原生16 kHz输出。

响度补偿目标RMS为`-23.0 dBFS`；概率不高于0.30时不补偿、达到0.80时完整补偿，中间线性插值。算法只放大、不主动衰减，并以`-3 dBFS`限制新增增益。当前实验profile默认OFF；开关不重建ID、不清空连续上下文。

NVIDIA `Frame_VAD_Multilingual_MarbleNet_v2.0`离线适配器直接接收L4的完整原生16 kHz输出，不再执行48→16 kHz重采样。模型原始softmax输出按NVIDIA帧索引裁齐为与16 kHz输入每320样本严格对应的20 ms概率序列；`center=true`产生的尾部边界帧只在末端丢弃，不用单一概率覆盖整轨。实时Runtime的L5阶段明确记录`offline_after_l4`跳过原因，不执行CNN；离线结果保留完整概率序列、逐帧阈值判断、模型和对齐方式。

L5不接收L2内部ID或`[17,169]`特征，方向标签只继承L3携带的平滑角；它不再次滤波、不做跨窗口Tracking或身份识别，也不反馈改变L2 Gate、MUSIC或L3音频。primary/shadow模型可读取同一不可变波形；只有primary结果形成离线`VoiceDetection`和`Layer4OfflineResult`，不进入实时逐窗DecisionRecord。

L5判断阈值与L2 Gate阈值是两套不同参数。Development Test UI必须使用不同标签和滑动条；拖动L5阈值只重算已缓存概率的标签，不重跑L3或CNN。

每次成功离线L5检测按`(session_id, stream_epoch, track_id)`把第`i`个模型概率写到`[start_sample+i*960,start_sample+(i+1)*960)`，记录概率、Voice/Non-Voice、模型和阈值。整轨概览另取完整概率序列的连续3帧最大均值；它只用于摘要，不得反向覆盖逐20 ms结果。该语义不向L2反馈、不确认或续租ID，也不改变音频。

## 已实现门禁

- 渐进与离线入口都只接受L4输出的16 kHz单声道、完整20 ms hop音频；渐进结果不进入正式逐窗ResultJoiner；
- 明确拒绝旧`[17,169]`主输入；
- 角度、window和sample身份继承；
- 概率finite且位于`[0,1]`，阈值重判不重跑模型；
- primary/shadow读取同一不可变波形批次，artifact加载前校验hash；环境检查执行实际CUDA MarbleNet波形前向。
- 每个20 ms追加hop与自己的IMCRA概率严格对齐；补偿前后RMS、峰值、请求/应用增益及连续上下文汇总随L5结果记录。

目标域微调、正式概率校准和锁定测试集指标仍待数据集版本完成后实施。

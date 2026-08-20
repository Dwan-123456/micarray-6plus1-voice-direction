# Layer 4：48 kHz音频到人声概率

> 项目1.2.2中，L4音频段、检测和阶段结果贯通`track_id`；L4不按角度创建或修补ID，语义概率反馈只按L2权威身份关联。

权威目标契约见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)。L3按统一配置输出40/80/160 ms重叠增强窗（当前40 ms）；正式`TrackAudioStreamHub`在L3与L4之间按`(session_id, stream_epoch, track_id)`把它们变为连续20 ms时间轴。

公共服务每窗只追加一个内部稳定且与IMCRA网格严格对齐的20 ms hop，避免40 ms重叠重复；随后执行`imcra_probability_rms_v1`并维护最长3200 ms连续缓冲。Test UI试听、正式按ID轨音频和CNN读取同一份补偿后波形。重叠L3原始窗只作瞬时输入，不再作为正式音频资产重复保存。

响度补偿目标RMS为`-23.0 dBFS`；概率不高于0.30时不补偿、达到0.80时完整补偿，中间线性插值。算法只放大、不主动衰减，并以`-3 dBFS`限制新增增益。Test UI开关默认ON且可实时切换；开关不重建ID、不清空连续上下文，增益从下一20 ms平滑过渡。

NVIDIA `Frame_VAD_Multilingual_MarbleNet_v2.0`适配器接收可变长度连续48 kHz轨音频，polyphase降采样到16 kHz后一次输出约每20 ms一帧的概率。模型利用最长3200 ms上下文，但窗口标量只聚合最新80 ms内连续3帧，旧语音不得粘住当前判断。这里的“连续/流式”表示按ID维护连续序列并反复提供有界长上下文；MarbleNet本身不是隐藏状态缓存模型。

L4不接收L2内部ID或`[17,169]`特征，方向标签只继承L3携带的平滑角；它不再次滤波、不做跨窗口Tracking或身份识别，也不反馈改变L2 Gate、SRP或L3音频。primary/shadow模型可读取同一不可变波形；只有primary结果进入正式VoiceDetection和DecisionRecord。

L4判断阈值与L2 Gate阈值是两套不同参数。Development Test UI必须使用不同标签和滑动条；拖动L4阈值只重算已缓存概率的标签，不重跑L3或CNN。

## 已实现门禁

- 只接受48 kHz单声道音频并在模型内部降采样；
- 明确拒绝旧`[17,169]`主输入；
- 角度、window和sample身份继承；
- 概率finite且位于`[0,1]`，阈值重判不重跑模型；
- primary/shadow读取同一不可变波形批次，artifact加载前校验hash；环境检查执行实际CUDA MarbleNet波形前向。
- 每个20 ms追加hop与自己的IMCRA概率严格对齐；补偿前后RMS、峰值、请求/应用增益及连续上下文汇总随L4结果记录。

目标域微调、正式概率校准和锁定测试集指标仍待数据集版本完成后实施。

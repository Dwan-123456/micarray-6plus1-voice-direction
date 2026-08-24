# Layer 5：48 kHz音频到人声概率

> 项目1.3.2中，L5音频段、检测和阶段结果贯通`track_id`；L5不按角度创建或修补ID，语义概率反馈只按L2权威身份关联。

权威目标契约见根目录[`ARCHITECTURE_V1.1_TARGET.md`](../ARCHITECTURE_V1.1_TARGET.md)。L3按统一配置输出40/80/160 ms重叠增强窗（当前40 ms）；`TrackAudioStreamHub`按`(session_id, stream_epoch, track_id)`把它们拼接并封存为完整长音频。L5不再实时消费连续片段，只在停机排空、Hub封存并完成L4一/二人路由后执行。

公共服务每窗只追加一个内部稳定且与IMCRA网格严格对齐的20 ms hop，避免40 ms重叠重复；随后执行`imcra_probability_rms_v1`并维护最长3200 ms实时缓冲，同时归档完整补偿后长轨。L3试听与正式按ID轨读取该48 kHz波形；停机后L4把它路由、必要时分离并转换成16 kHz，L5只读取L4最终输出。重叠L3原始窗只作瞬时输入，不再作为正式音频资产重复保存。

响度补偿目标RMS为`-23.0 dBFS`；概率不高于0.30时不补偿、达到0.80时完整补偿，中间线性插值。算法只放大、不主动衰减，并以`-3 dBFS`限制新增增益。Test UI开关默认ON且可实时切换；开关不重建ID、不清空连续上下文，增益从下一20 ms平滑过渡。

NVIDIA `Frame_VAD_Multilingual_MarbleNet_v2.0`离线适配器直接接收L4的完整原生16 kHz输出，不再执行48→16 kHz重采样。模型原始softmax输出按NVIDIA帧索引裁齐为与16 kHz输入每320样本严格对应的20 ms概率序列；`center=true`产生的尾部边界帧只在末端丢弃，不用单一概率覆盖整轨。实时Runtime的L5阶段明确记录`offline_after_l4`跳过原因，不执行CNN；离线结果保留完整概率序列、逐帧阈值判断、模型和对齐方式。

L5不接收L2内部ID或`[17,169]`特征，方向标签只继承L3携带的平滑角；它不再次滤波、不做跨窗口Tracking或身份识别，也不反馈改变L2 Gate、MUSIC或L3音频。primary/shadow模型可读取同一不可变波形；只有primary结果形成离线`VoiceDetection`和`Layer4OfflineResult`，不进入实时逐窗DecisionRecord。

L5判断阈值与L2 Gate阈值是两套不同参数。Development Test UI必须使用不同标签和滑动条；拖动L5阈值只重算已缓存概率的标签，不重跑L3或CNN。

每次成功离线L5检测按`(session_id, stream_epoch, track_id)`把第`i`个模型概率写到`[start_sample+i*960,start_sample+(i+1)*960)`，记录概率、Voice/Non-Voice、模型和阈值。整轨概览另取完整概率序列的连续3帧最大均值；它只用于摘要，不得反向覆盖逐20 ms结果。该语义不向L2反馈、不确认或续租ID，也不改变音频。

## 已实现门禁

- 当前离线入口只接受L4输出的16 kHz单声道、完整20 ms hop音频；兼容实时接口仍保留48 kHz契约但不在1.3.2 Runtime执行；
- 明确拒绝旧`[17,169]`主输入；
- 角度、window和sample身份继承；
- 概率finite且位于`[0,1]`，阈值重判不重跑模型；
- primary/shadow读取同一不可变波形批次，artifact加载前校验hash；环境检查执行实际CUDA MarbleNet波形前向。
- 每个20 ms追加hop与自己的IMCRA概率严格对齐；补偿前后RMS、峰值、请求/应用增益及连续上下文汇总随L5结果记录。

目标域微调、正式概率校准和锁定测试集指标仍待数据集版本完成后实施。

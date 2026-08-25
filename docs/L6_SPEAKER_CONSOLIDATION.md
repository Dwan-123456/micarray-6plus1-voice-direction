# L6 声纹归类、质量择优与时间线拼接

L6 是 L4、L5 完成后的手动离线步骤，不参与实时采集，也不修改 L2 的
`track_id` 和角度。合法录音最多包含三名先后出现的讲话人，同时讲话最多两人。

## 数据流

```text
L3 按 L2 ID/角度形成的长音频
  -> L4 对每条混合音频一拆二
  -> L5 对每条候选逐 20 ms 输出人声概率与 bool
  -> 用户手动运行 L6
  -> 合并短静音间隙，再将连续有声区分为 0.5 s 人物归属窗
  -> 每个归属窗使用同一连续有声区内最多 1.5 s 声纹上下文
  -> CAMPPlus 16 kHz / 192维声纹
  -> 平均链接 AHC + 同时 L2 方向约束，整次录音聚为 0..3 人
  -> 同一讲话人按绝对采集时间线归并
  -> 同一讲话人同一时间的重复音频按质量评分保留较优者
  -> Speaker A / B / C，16 kHz 单声道长音频
```

L4/L5 需要把两个匿名候选都保留到 L6。每条候选必须携带父 L2 ID、父角度、
候选序号、绝对起止采样点、16 kHz 音频、逐 20 ms L5 概率和判定。

L5 的连续 Voice 不等于单个说话人轮次。L6 因此以固定 500 ms 粒度对长有声区做人物归属，
同时以`minimum_embedding_speech_ms=1500`作为CAMPPlus上下文长度，使同一 L2 音频内
先后出现的不同声纹可以在0.5秒边界上分类。上下文不跨越由超过200 ms静音分开的有声区，
避免短残留借用其他语音成为可靠建类样本。
持续重叠不少于 500 ms 的不同 L2 方向会对聚类相似度加约束；高于 0.85
的跨轨泄漏副本仍允许归为同一人。实际语音不足一个声纹窗的短残留只能附着到
已由可靠窗建立的人物类，避免循环填充后产生伪 Speaker ID。

## 质量评分

固定总分为：

```text
Q = 0.30 Q_voice + 0.30 Q_speaker + 0.20 Q_mos
  + 0.10 Q_snr + 0.10 Q_continuity
```

- `Q_voice`：片段 L5 人声概率中位数。
- `Q_speaker`：片段声纹与最终聚类中心的余弦相似度。
- `Q_mos`：同一连续有声区的 DNSMOS P.835 SIG/BAK/OVRL 组合并归一化，其下0.5秒归属片段复用该分数。
- `Q_snr`：与同一候选的 L5 非人声帧噪声 RMS 比较所得分段 SNR。
- `Q_continuity`：削波、近零孔洞及大幅样本突变惩罚。

同一说话人、同一绝对时间发生重叠时，优先选择总分更高的候选；总分接近时，
原 L4 方向匹配胜出的候选获得小幅稳定偏置。输出有效段边缘使用 2 ms 淡入淡出。

## 模型和设备

- 声纹：`iic/speech_campplus_sv_zh_en_16k-common_advanced`，16 kHz，
  80-bin Kaldi fbank，192维归一化声纹。Runtime在CUDA可用时使用最多64条的GPU batch，构建失败或无CUDA时回退CPU。
- 无参考质量：Microsoft DNSMOS P.835 `sig_bak_ovr.onnx`，CPU/ONNX Runtime。
- 同一连续有声区只执行一次DNSMOS，并将结果复用到所有0.5秒子片段。
- 两个模型均由 manifest 和 SHA-256 校验；L6 仅在用户点击后加载并执行。

## 当前公共接口

- `ApplicationRuntime.build_offline_l6_pipeline()`：构建人工触发的 CUDA优先/CPU回退 L6。
- `OfflineLayer6Pipeline.process(tuple[Layer4OfflineResult, ...])`：返回
  `Layer6Result`。
- `Layer6Result.outputs`：0..3 条 `Layer6SpeakerAudio`，包含 Speaker A/B/C、
  16 kHz 波形、绝对 48 kHz 时间线边界、来源 L2 ID 和平均质量分。
- `Layer6Result.fragments`：审计每个候选片段的父 ID、角度、候选序号、声纹类别及
  五项质量分。

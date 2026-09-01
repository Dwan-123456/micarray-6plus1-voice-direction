# IngestCoordinator：v1.4.3唯一时间轴契约

`IngestCoordinator`是当前实时链路中`session_id`、`stream_epoch`和半开绝对sample边界的唯一分配者。它消费L1校准后的逻辑8通道`DecodedAudio`及`InputHealthEvent`，输出不可变`IngestedAudioBlock`；不改变音频内容，不执行IMCRA，也不补零。

`IngestedAudioBlock.samples`固定为finite、C-contiguous、只读的`float32 [N,8]`，顺序为`MIC0..MIC5、Center、HardwareMix`。存在`native_samples`时，它保存校准和逻辑映射前的Host原始8通道float32副本。校准身份随每个block传播，校准版本或哈希变化会建立新epoch。

sequence gap、timestamp异常、设备重启、输入overflow、handoff丢块和校准身份变化在这里转换为epoch重置。一个健康事件及其对应sequence gap只生成一次重置；采集边界会把连续handoff溢出合并为一个缺口事件。新epoch的绝对sample从0重新开始，窗口ID仍由`WindowAssembler`保持单调递增。输入采样率若不是固定48 kHz，Coordinator直接拒绝该帧并报告处理错误，不通过epoch重置继续运行。

当前v1.4.3没有`RecordingStore`或录音消费者。`ApplicationRuntime`把block交给L1 IMCRA、可选预降噪、`WindowAssembler`和1秒内存环。IMCRA hop在Coordinator分配sample区间后生成，并必须与block、epoch和sample边界严格一致。

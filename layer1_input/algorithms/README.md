# Layer 1 解码算法接口

期待输入：设备返回的完整二进制音频块 `bytes`，以及硬件通道数 `int`。

期待输出：`np.ndarray[np.float32]`，形状 `(N, hardware_channels)`，幅值标准化到 `[-1,1)`。

当前实现 `Pcm16InterleavedDecoder` 按 little-endian signed int16 和交错通道解析。新算法继承 `AudioDecoder` 后，可通过 `LiveSipeedSource(config, decoder=my_decoder)` 注入。它不能执行 DOA、跟踪或波束形成。

公共音频为逻辑8通道；L1的`Layer1Imcra`对前7个物理麦运行`cohen_imcra_2003_l1_v6`并发布0～10000 Hz状态及逐频7×7空间噪声协方差，decoder不得针对下游算法改变sample范围、逻辑通道顺序或内容。Layer 3仍可使用多频段MVDR/DAS，但波束形成只读取前7个物理麦，第8路HardwareMix只保留接口。

# CountNet CRNN 16 kHz / 5 s v2

本目录是`faroit/CountNet`官方CRNN的修正推理移植。上游模型使用Keras 1.2.2/Theano；
`scripts/import_countnet.py`把卷积、hard-sigmoid LSTM、Dense权重和标准化器转换为TorchScript。
运行时不依赖旧Keras、Theano、librosa或scikit-learn。

v2修正了旧v1转换器的卷积语义错误：Theano后端执行数学卷积，PyTorch `Conv2d`执行互相关，
因此转换时必须把每个卷积核的两个空间轴翻转一次。使用上游`examples/5_speakers.wav`逐层对照，
修正后原生11类概率相对原Keras/Theano模型的最大绝对误差为`4.77e-7`；旧v1不再作为活动模型。

输入固定为16 kHz、5秒、单声道`float32 [1,80000]`。TorchScript内部执行400点Hann STFT、
160 sample hop、前500帧、逐频标准化及全局L2归一化。原生输出是0～10人的11类logits；项目
公开`P0`、`P1`、`P2+`，其中`P2+`为原生P2～P10之和，并不表示恰好2人。

模型由模拟LibriCount数据训练。数值移植一致只证明实现忠实于上游模型，不代表真实办公室准确率
已经达标；真实麦克风阵列仍需使用标注录音单独验收。

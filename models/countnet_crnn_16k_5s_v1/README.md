# CountNet CRNN 16 kHz / 5 s

本目录是`faroit/CountNet`官方CRNN的推理移植。上游模型使用Keras 1.2.2/Theano，当前项目通过
`scripts/import_countnet.py`把卷积、hard-sigmoid LSTM、Dense权重和标准化器确定性转换为TorchScript；
运行时不依赖旧Keras、Theano、librosa或scikit-learn。

输入固定为16 kHz、5秒、单声道`float32 [1,80000]`。TorchScript内部执行与上游一致的400点
Hann STFT、160 sample hop、前500帧、逐频标准化及全局L2归一化。原生输出是0～10人的11类
logits；项目只公开0/1/2，其中`P2`表示“2人或以上”，由原生P2～P10求和，不能解释为恰好2人。

该模型由模拟LibriCount数据训练，不能把自动测试通过解释为真实办公室准确率验收。重新导入时需使用
manifest固定的上游revision和SHA-256，并同步更新模型hash、配置、测试和CHANGELOG。
